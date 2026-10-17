"""Run retained full-weight HTTP smoke and quality checks against an owned service."""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--quality-python', required=True)
    parser.add_argument('--skip-quality', action='store_true', help='Retain requests only; no quality verdict')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--requests', type=int, default=4)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=False)
    status = {'status': 'RUNNING', 'phase': 'readiness', 'source_revision': args.source_revision,
              'server_lifecycle': 'PENDING_PARENT_VERIFICATION', 'performance': 'FUNCTIONAL_SMOKE_ONLY'}

    def save():
        (args.evidence / 'status.json').write_text(json.dumps(status, indent=2) + '\n')

    def run(name, command):
        status['phase'] = name
        save()
        (args.evidence / (name + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
        with (args.evidence / (name + '.log')).open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
        status[name + '_exit'] = result.returncode
        save()
        if result.returncode:
            raise RuntimeError(f'{name} failed with exit {result.returncode}')

    save()
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                with socket.create_connection(('127.0.0.1', args.port), timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Standalone service did not become ready')
                time.sleep(2)
        run('client', [sys.executable, '-m', 'drama3.benchmark', '--endpoint',
                      f'http://127.0.0.1:{args.port}/v1/audio/speech', '--engine', 'drama3-engine',
                      '--source-revision', args.source_revision, '--run-id', args.run_id,
                      '--workload', str(args.workload), '--output', str(args.output),
                      '--frame-telemetry', str(args.telemetry), '--concurrency', str(args.concurrency),
                      '--requests', str(args.requests), '--warmup', str(args.concurrency), '--timeout', '1800'])
        result = json.loads((args.output / 'results.json').read_text())
        if not result['metrics']['eligible_for_comparison']:
            raise RuntimeError('Incomplete acoustic-frame accounting')
        status['metrics'] = result['metrics']
        if args.skip_quality:
            status['status'] = 'REQUESTS_COMPLETE_QUALITY_NOT_RUN'
        else:
            run('quality', [args.quality_python, 'tools/evaluate_seed_outputs.py', '--run', str(args.output),
                           '--workload', str(args.workload), '--manifest', str(args.manifest),
                           '--assets', str(args.assets), '--output', str(args.evidence / 'quality')])
            status['status'] = 'MEASURED_NOT_QUALIFIED'
        status['phase'] = 'await_server_lifecycle'
    except Exception as error:
        status['status'] = 'FAIL'
        status['error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    save()


if __name__ == '__main__':
    main()
