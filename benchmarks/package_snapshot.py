"""Package a portable source snapshot without Windows/OneDrive permission bits."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh archive path")
    root = Path(__file__).resolve().parents[1]
    paths = [p for folder in ("src", "tests", "benchmarks", "examples")
             for p in (root / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts
             and not any(part.endswith('.egg-info') for part in p.parts)]
    paths += [root / "pyproject.toml"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with tarfile.open(args.output, "w:gz") as archive:
        for path in sorted(paths):
            name = path.relative_to(root).as_posix()
            data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
        manifest = json.dumps(hashes, sort_keys=True, indent=2).encode()
        info = tarfile.TarInfo("SOURCE_SHA256.json")
        info.size, info.mode = len(manifest), 0o644
        archive.addfile(info, io.BytesIO(manifest))
    print(json.dumps({"archive": str(args.output), "files": len(hashes),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
