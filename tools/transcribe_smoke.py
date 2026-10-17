"""Independent CPU Whisper transcription of a smoke WAV, not a quality verdict."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--asr-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    samples, rate = sf.read(args.audio, dtype='float32')
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Expected finite mono audio')
    processor = WhisperProcessor.from_pretrained(args.asr_model, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.asr_model, local_files_only=True, dtype=torch.float32,
    ).eval()
    waveform = torchaudio.functional.resample(torch.from_numpy(samples), rate, 16000)
    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors='pt',
                       return_attention_mask=True)
    with torch.inference_mode():
        ids = model.generate(**inputs, language='en', task='transcribe', max_new_tokens=128)
    result = {'status': 'DIAGNOSTIC_ONLY', 'audio': str(args.audio),
              'asr_model': str(args.asr_model), 'duration_seconds': len(samples) / rate,
              'rms': float(np.sqrt(np.mean(samples ** 2))),
              'peak': float(np.max(np.abs(samples))),
              'clipped_fraction': float(np.mean(np.abs(samples) >= .999)),
              'transcript': processor.batch_decode(ids, skip_special_tokens=True)[0],
              'notes': 'One ASR transcript does not establish WER, speaker similarity or model parity.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
