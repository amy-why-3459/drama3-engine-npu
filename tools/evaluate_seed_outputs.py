"""Independent CPU SeedTTS English WER and upstream WavLM SIM evaluation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper.normalizers import EnglishTextNormalizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asr-device', default='cpu', help='Independent ASR device; default CPU FP32')
    parser.add_argument('--asr-model', default='/dataset/models/whisper-large-v3')
    parser.add_argument('--sglang-root', type=Path, default=Path('/home/why/drama3-research/sglang-omni-npu-20261017'))
    args = parser.parse_args()
    run = json.loads((args.run / 'results.json').read_text())
    raw = args.workload.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    references = json.loads(args.manifest.read_text())
    if digest != run['workload_sha256'] or digest != references['workload_sha256']:
        raise ValueError('Workload/manifest/run hashes differ')
    if not run['metrics']['eligible_for_comparison']:
        raise ValueError('Refusing quality evaluation of an incomplete run')
    payloads = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assets = {r['filename']: r for r in json.loads(args.assets.read_text())}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if args.asr_device.startswith('npu:'):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.asr_device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    elif args.asr_device != 'cpu':
        raise ValueError('Quality scorer supports CPU or explicit NPU device')
    processor = WhisperProcessor.from_pretrained(args.asr_model, local_files_only=True)
    asr = WhisperForConditionalGeneration.from_pretrained(
        args.asr_model, local_files_only=True, dtype=torch.float32).to(args.asr_device).eval()
    scorer_path = args.sglang_root / 'benchmarks/metrics/speaker_similarity.py'
    spec = importlib.util.spec_from_file_location('drama3_upstream_speaker_similarity', scorer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.WavLMSpeakerSimilarity(
        finetune_checkpoint=assets['wavlm_large_finetune.pth']['path'],
        wavlm_base=assets['wavlm_large.pt']['path'], device='cpu')
    normalizer = EnglishTextNormalizer()
    rows = []
    for trace in run['traces']:
        index = int(trace['request_id'].rsplit('-', 1)[1]) % len(payloads)
        path = args.run / (trace['request_id'] + '.pcm')
        audio = np.fromfile(path, dtype='<i2').astype(np.float32) / 32768
        if not trace['success'] or len(audio) != trace['audio_samples'] or len(audio) != trace['generated_frames'] * 2048:
            raise ValueError('Incomplete audio/frame accounting: ' + trace['request_id'])
        rate = trace['sample_rate']
        if not len(audio) or len(audio) / rate >= 30:
            raise ValueError('Expected nonempty audio shorter than the ASR window')
        wav = args.output / (trace['request_id'] + '.wav')
        sf.write(wav, audio, rate, subtype='PCM_16')
        waveform = torchaudio.functional.resample(torch.from_numpy(audio), rate, 16000)
        inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        inputs = {name: value.to(args.asr_device) for name, value in inputs.items()}
        with torch.inference_mode():
            tokens = asr.generate(**inputs, language='en', task='transcribe', max_new_tokens=256)
        hypothesis = processor.batch_decode(tokens.cpu(), skip_special_tokens=True)[0]
        reference = payloads[index]['input']
        ref_norm, hyp_norm = normalizer(reference), normalizer(hypothesis)
        errors = jiwer.process_words(ref_norm, hyp_norm)
        sim = scorer.score_batch([references['samples'][index]['audio']], [str(wav)])[0]
        if not np.isfinite(sim):
            raise ValueError('Non-finite speaker similarity')
        row = {'request_id': trace['request_id'], 'sample_id': references['samples'][index]['sample_id'],
               'reference': reference, 'transcript': hypothesis, 'ref_norm': ref_norm, 'hyp_norm': hyp_norm,
               'wer': errors.wer, 'substitutions': errors.substitutions, 'deletions': errors.deletions,
               'insertions': errors.insertions, 'hits': errors.hits, 'sim_percent': sim,
               'clipped_fraction': float(np.mean(np.abs(audio) >= .999))}
        rows.append(row)
        with (args.output / 'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(f"{len(rows)}/{len(run['traces'])} WER={errors.wer:.4f} SIM={sim:.3f}", flush=True)
    words = sum(r['substitutions'] + r['deletions'] + r['hits'] for r in rows)
    if not words:
        raise ValueError('Empty reference corpus')
    summary = {'status': 'MEASURED_NOT_QUALIFIED', 'samples': len(rows),
               'wer_corpus': sum(r['substitutions'] + r['deletions'] + r['insertions'] for r in rows) / words,
               'sim_percent_mean': float(np.mean([r['sim_percent'] for r in rows])),
               'asr_model': args.asr_model, 'asr_device': args.asr_device, 'asr_precision': 'float32',
               'normalizer': 'whisper.normalizers.EnglishTextNormalizer',
               'speaker_assets': assets, 'workload_sha256': digest,
               'speaker_scorer_sha256': hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
               'notes': 'Observed quality only; matched baseline/candidate quality comparison remains required.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
