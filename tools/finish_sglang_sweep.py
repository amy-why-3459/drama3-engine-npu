"""Stop this campaign's owned server, retain results, then evaluate CPU quality."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    proc = Path(f'/proc/{args.server_pid}')
    command = (proc / 'cmdline').read_bytes()
    if b'sgl-omni' not in command or b'sglang-npu-sweep.yaml' not in command:
        raise ValueError('Refusing to signal a process outside this campaign')
    identity = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    deadline = time.monotonic() + 12 * 3600
    state = {}
    while time.monotonic() < deadline:
        state = json.loads((args.run / 'status.json').read_text())
        if state['status'] != 'RUNNING':
            break
        time.sleep(20)
    else:
        state = {'status': 'FAIL', 'error': '12-hour campaign deadline exceeded'}
    # Verify PID identity again so a reused PID can never be signalled.
    if proc.exists():
        current = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        if identity != current or (proc / 'cmdline').read_bytes() != command:
            raise RuntimeError('Server PID identity changed; refusing shutdown')
        os.kill(args.server_pid, signal.SIGTERM)
        deadline = time.monotonic() + 120
        while proc.exists() and time.monotonic() < deadline:
            time.sleep(2)
        if proc.exists():
            raise RuntimeError('Server did not exit after SIGTERM; inspect retained logs')
    shutil.copytree(args.run, args.evidence / 'sweep')
    telemetry = Path('runs/sglang-sweep-20261017-frames.jsonl')
    if telemetry.exists():
        shutil.copy2(telemetry, args.evidence / 'server-frames.jsonl')
    (args.evidence / 'completion.json').write_text(json.dumps({
        'controller': state, 'owned_server_process_gone': not proc.exists(),
        'root_exit_code': 'REQUIRES_PARENT_VERIFICATION',
        'status': 'INCONCLUSIVE', 'quality': 'STARTING' if state['status'] != 'FAIL' else 'NOT_RUN'
    }, indent=2) + '\n')
    if state['status'] == 'FAIL':
        raise RuntimeError('Sweep failed; server stopped and partial evidence retained')
    for concurrency in (1, 2, 4, 8, 16, 32, 64):
        label = f'c{concurrency}'
        cmd = ['/home/why/drama3-research/venv-quality/bin/python', 'tools/evaluate_seed_outputs.py',
               '--run', str(args.evidence / 'sweep' / label),
               '--workload', 'evidence/campaign-20261016/seed-en-128/workload.jsonl',
               '--manifest', 'evidence/campaign-20261016/seed-en-128/manifest.json',
               '--assets', '/home/why/drama3-research/quality-assets/manifest.json',
               '--output', str(args.evidence / ('quality-' + label))]
        with (args.evidence / ('quality-' + label + '.log')).open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=14400)
    (args.evidence / 'quality-complete.json').write_text(json.dumps({
        'status': 'MEASURED_NOT_QUALIFIED', 'concurrencies': [1, 2, 4, 8, 16, 32, 64],
        'notes': 'No matched vLLM/candidate comparison or 2x validation yet.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
