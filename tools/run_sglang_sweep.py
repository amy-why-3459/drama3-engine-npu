"""Run a retained qualification and concurrency sweep against an owned server.

The caller owns server startup/shutdown. Results remain INCONCLUSIVE until
quality, lifecycle and matched independent baseline/candidate runs are checked.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--frame-telemetry', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8094')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'RUNNING', 'quality': 'NOT_RUN', 'server_lifecycle': 'PENDING',
              'source_revision': args.source_revision, 'physical_devices': [0, 1],
              'workload_sha256': hashlib.sha256(args.workload.read_bytes()).hexdigest(),
              'requests_per_concurrency': 128, 'completed': [], 'current': 'readiness',
              'notes': 'One process/one round; no speedup claim or statistical qualification.'}

    def save():
        tmp = args.output / 'status.tmp'
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(args.output / 'status.json')

    save()
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                with urllib.request.urlopen(args.endpoint + '/health', timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Server did not become ready in 900 seconds')
            time.sleep(2)
        cohorts = [('qualification-c8', 8, 8)] + [(f'c{c}', c, 128) for c in (1, 2, 4, 8, 16, 32, 64)]
        for label, concurrency, count in cohorts:
            status['current'] = label
            save()
            command = [sys.executable, '-m', 'drama3.benchmark',
                       '--endpoint', args.endpoint + '/v1/audio/speech',
                       '--engine', 'sglang-omni', '--adapt-drama3-request',
                       '--source-revision', args.source_revision,
                       '--run-id', 'sglang-20261017-' + label,
                       '--workload', str(args.workload), '--output', str(args.output / label),
                       '--frame-telemetry', str(args.frame_telemetry),
                       '--concurrency', str(concurrency), '--requests', str(count),
                       '--warmup', str(max(8, concurrency)), '--timeout', '1800']
            (args.output / (label + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
            with (args.output / (label + '.log')).open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
            if result.returncode:
                raise RuntimeError(f'{label} client exit {result.returncode}; retained artifacts; sweep stopped')
            measured = json.loads((args.output / label / 'results.json').read_text())
            if not measured['metrics']['eligible_for_comparison']:
                raise RuntimeError(f'{label} incomplete metrics; sweep stopped')
            status['completed'].append({'cohort': label, 'metrics': measured['metrics'], 'client_exit': 0})
            save()
        status['status'] = 'INCONCLUSIVE'
        status['current'] = None
    except Exception as exc:
        status['status'] = 'FAIL'
        status['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    save()


if __name__ == '__main__':
    main()
