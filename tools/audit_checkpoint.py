"""Verify every file against a saved Hugging Face tree for a pinned revision."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(directory, tree):
    rows = []
    for entry in tree:
        if entry.get("type") == "directory":
            continue
        name = Path(entry["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Invalid manifest path: {name}")
        path = directory / name
        row = {"path": str(name), "expected_bytes": entry["size"], "verified": False}
        if not path.is_file():
            row["error"] = "missing"
        else:
            row["bytes"] = path.stat().st_size
            lfs = entry.get("lfs")
            digest = hashlib.sha256() if lfs else hashlib.sha1()
            if not lfs:
                digest.update(b"blob " + str(row["bytes"]).encode() + b"\0")
            with path.open("rb") as source:
                for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                    digest.update(block)
            row["algorithm"] = "sha256" if lfs else "git-blob-sha1"
            row["expected_digest"] = lfs["oid"] if lfs else entry["oid"]
            row["actual_digest"] = digest.hexdigest()
            row["verified"] = (row["bytes"] == row["expected_bytes"]
                               and row["actual_digest"] == row["expected_digest"])
            if not row["verified"]:
                row["error"] = "size or hash mismatch"
        rows.append(row)
        print(f"{name}: {'verified' if row['verified'] else row['error']}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.tree.read_bytes()
    rows = audit(args.directory, json.loads(raw))
    complete = bool(rows) and all(row["verified"] for row in rows)
    result = {"model_directory": str(args.directory.resolve()), "revision": args.revision,
              "manifest_sha256": hashlib.sha256(raw).hexdigest(), "files": rows,
              "complete": complete, "status": "PASS" if complete else "INCONCLUSIVE"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
