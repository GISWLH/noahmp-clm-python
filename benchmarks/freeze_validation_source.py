"""Bundle runnable validation sources, fixtures and examples with file hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile


def freeze(output):
    root=Path(__file__).resolve().parents[1]
    paths=set()
    for folder in ('src/noahmp_clm','tests','benchmarks','examples'):
        for p in (root/folder).rglob('*'):
            if (p.is_file() and not any(part in ('__pycache__','.pytest_cache') for part in p.parts)
                    and p.suffix not in ('.pyc','.pyo','.so','.pyd','.dll','.o','.a')):
                paths.add(p)
    for name in ('README.md','LICENSE','pyproject.toml','pytest.ini','conftest.py','setup.cfg'):
        p=root/name
        if p.exists():paths.add(p)
    manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    sidecar=output.with_name(output.name.removesuffix('.tar.gz')+'-source.json')
    sidecar.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with tarfile.open(output,'w:gz') as t:
        for p in sorted(paths):t.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
        t.add(sidecar,arcname='SOURCE_SHA256.json',recursive=False)
    return dict(files=len(paths),manifest_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(freeze(p.parse_args().output)))
