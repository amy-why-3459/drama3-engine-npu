"""Freeze a Seed-TTS subset with shared FP32 CPU-encoded DRAMA-3 references.

The output is a performance workload, not a full Seed-TTS quality evaluation.
Reference preprocessing follows the pinned SGLang FishReferenceEncodeHook:
mono mean, torchaudio resampling, FP32 codec, no autocast or amplitude clamp.
"""

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from drama3.model.codec import AudioCodec


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.count < 1 or args.threads < 1:
        parser.error('count and threads must be positive')
    rows = [line.split('|') for line in (args.dataset / 'meta.lst').read_text().splitlines()]
    if len(rows) < args.count or any(len(row) != 4 for row in rows[:args.count]):
        raise ValueError('Insufficient rows or unsupported Seed-TTS metadata')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    codec = AudioCodec(args.model / 'codec.pth', 'cpu', torch.float32)
    encoded, records = {}, []
    with (args.output / 'workload.jsonl').open('x') as output:
        for sample_id, transcript, relative, target in rows[:args.count]:
            path = (args.dataset / relative).resolve()
            if not path.is_relative_to(args.dataset.resolve()):
                raise ValueError('Reference escapes dataset directory')
            if path not in encoded:
                samples, rate = sf.read(path, dtype='float32', always_2d=True)
                audio = torch.from_numpy(samples.T.copy()).mean(0, keepdim=True)
                if not torch.isfinite(audio).all() or audio.shape[-1] > 60 * rate:
                    raise ValueError('Invalid or overlong reference audio')
                audio = torchaudio.functional.resample(audio, rate, codec.sample_rate)
                with torch.inference_mode():
                    codes, _ = codec.codec.encode(audio, torch.tensor([audio.shape[-1]]))
                codes = codes[0]
                if (codes.shape[0] != 10 or codes.shape[1] < 1 or codes.min() < 0
                        or codes[0].max() >= 4096 or codes[1:].max() >= 1024):
                    raise ValueError('Codec emitted invalid reference indices')
                encoded[path] = {'codes': codes.tolist(), 'sha256': sha256(path)}
                print(f'ENCODED {relative} frames={codes.shape[1]}', flush=True)
            reference = encoded[path]
            output.write(json.dumps({'input': target, 'ref_text': transcript,
                                     'ref_audio_codes': reference['codes'],
                                     'stream': True, 'response_format': 'pcm',
                                     'max_new_tokens': 500}) + '\n')
            output.flush()
            records.append({'sample_id': sample_id, 'audio': str(path),
                            'audio_sha256': reference['sha256']})
    manifest = {'status': 'PREPARED_NOT_EVALUATED', 'selection': 'first metadata rows',
                'count': len(records), 'unique_references': len(encoded),
                'meta_sha256': sha256(args.dataset / 'meta.lst'),
                'codec_sha256': sha256(args.model / 'codec.pth'),
                'workload_sha256': sha256(args.output / 'workload.jsonl'),
                'reference_precision': 'float32 CPU',
                'required_server_sampling': {'temperature': 1.0, 'top_p': .9, 'top_k': 30},
                'samples': records}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('WORKLOAD_PREPARED', flush=True)


if __name__ == '__main__':
    main()
