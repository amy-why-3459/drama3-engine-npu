"""Resume pinned HF LFS downloads by verified ranges and check final SHA256."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


class RangeRedirectHandler(urllib.request.HTTPRedirectHandler):
    # Python 3.9 lacks 308 handling; model mirrors use permanent GET redirects.
    def http_error_308(self, request, response, code, message, headers):
        return self.http_error_302(request, response, code, message, headers)

    def redirect_request(self, request, response, code, message, headers, new_url):
        return super().redirect_request(
            request, response, 302 if code == 308 else code, message, headers, new_url
        )


def fetch_range(job):
    url, start, end, total, target = job
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return expected
    temporary = target.with_suffix(".tmp")
    for attempt in range(6):
        try:
            received = temporary.stat().st_size if temporary.exists() else 0
            if received > expected:
                raise ValueError(f"Oversized partial range: {temporary}")
            if received == expected:
                temporary.replace(target)
                return expected
            resume_start = start + received
            deadline = time.monotonic() + 120
            request = urllib.request.Request(
                url + f"&chunk={resume_start}&attempt={attempt}",
                headers={"Range": f"bytes={resume_start}-{end}"},
            )
            opener = urllib.request.build_opener(RangeRedirectHandler())
            with opener.open(request, timeout=20) as response:
                actual = response.headers.get("Content-Range")
                if response.status != 206 or actual != f"bytes {resume_start}-{end}/{total}":
                    raise ValueError(
                        f"Incorrect range response: {response.status} {actual}"
                    )
                with temporary.open("ab") as output:
                    while received < expected:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Range exceeded 120 second budget")
                        chunk = response.read1(min(1024 * 1024, expected - received))
                        if not chunk:
                            raise OSError(f"Truncated range: {received}/{expected}")
                        output.write(chunk)
                        received += len(chunk)
                temporary.replace(target)
                return received
        except Exception as error:
            print(
                f"{target.name}: attempt {attempt + 1}: {type(error).__name__}: {error}",
                flush=True,
            )
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"Failed range {start}-{end}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=24)
    options = parser.parse_args()
    if not 1 <= options.workers <= 64:
        parser.error("workers must be in [1,64]")
    options.destination.mkdir(parents=True, exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    entries = [e for e in json.loads(options.tree.read_text()) if e.get("lfs")]
    jobs = []
    for entry in entries:
        name = entry["path"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid remote path")
        final = options.destination / name
        if final.exists():
            continue
        parts = options.destination / ".parts" / name
        parts.mkdir(parents=True, exist_ok=True)
        previous = options.destination / (name + ".partial")
        if previous.exists():
            with previous.open("rb") as source:
                for start in range(
                    0, previous.stat().st_size // chunk_size * chunk_size, chunk_size
                ):
                    target = parts / f"{start:012d}"
                    block = source.read(chunk_size)
                    if not target.exists():
                        target.write_bytes(block)
        for start in range(0, entry["size"], chunk_size):
            end = min(start + chunk_size, entry["size"]) - 1
            target = parts / f"{start:012d}"
            url = f"{options.endpoint.rstrip('/')}/{options.repo}/resolve/{options.revision}/{name}?download=true"
            jobs.append((url, start, end, entry["size"], target))
    completed = 0
    last = time.monotonic()
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(fetch_range, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            try:
                completed += future.result()
            except Exception as error:
                failures.append(str(error))
                print(f"RANGE_FAILED {error}", flush=True)
            if time.monotonic() - last > 30:
                print(
                    f"Completed {completed / 2**30:.3f} GiB of scheduled ranges",
                    flush=True,
                )
                last = time.monotonic()
    for entry in entries:
        final = options.destination / entry["path"]
        assembled = (
            final if final.exists() else final.with_suffix(final.suffix + ".assembled")
        )
        if not final.exists():
            missing = [
                start
                for start in range(0, entry["size"], chunk_size)
                if not (
                    options.destination / ".parts" / entry["path"] / f"{start:012d}"
                ).is_file()
            ]
            if missing:
                failures.append(f"{entry['path']}: {len(missing)} ranges missing")
                continue
            with assembled.open("wb") as output:
                for start in range(0, entry["size"], chunk_size):
                    with (
                        options.destination / ".parts" / entry["path"] / f"{start:012d}"
                    ).open("rb") as source:
                        shutil.copyfileobj(source, output)
        digest = hashlib.sha256()
        with assembled.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if (
            assembled.stat().st_size != entry["size"]
            or digest.hexdigest() != entry["lfs"]["oid"]
        ):
            raise ValueError(
                f"Size or SHA256 mismatch for {entry['path']}; preserved for investigation"
            )
        if assembled != final:
            assembled.replace(final)
        print(f"VERIFIED {entry['path']} {digest.hexdigest()}", flush=True)
    if failures:
        raise RuntimeError(f"Incomplete checkpoint; resume preserved ranges: {failures}")


if __name__ == "__main__":
    main()
