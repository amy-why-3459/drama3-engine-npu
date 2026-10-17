"""Pin and download the same WavLM SIM assets used by SGLang's SeedTTS scorer."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

root = Path('/home/why/drama3-research/quality-assets')
root.mkdir(parents=True, exist_ok=True)
records = []
for repository, filename in [('popsoda2002/seedtts-wavlm-sim', 'wavlm_large_finetune.pth'),
                              ('s3prl/converted_ckpts', 'wavlm_large.pt')]:
    revision = HfApi().model_info(repository).sha
    print(f'Downloading {repository}@{revision}/{filename}', flush=True)
    path = Path(hf_hub_download(repository, filename, revision=revision,
                               local_dir=root / repository.replace('/', '--')))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    records.append({'repository': repository, 'revision': revision, 'filename': filename,
                    'path': str(path), 'size': path.stat().st_size, 'sha256': digest.hexdigest()})
    (root / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
print('QUALITY_ASSETS_DOWNLOADED', flush=True)
