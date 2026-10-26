"""Build unchanged jqe NNUE sources with a read-only observer-FEN C ABI."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'external/devinokeefe-jieqi/engine/src'
OUT = ROOT/'build/nnue-observer'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    relative = ['research/nnue_observer_bridge.cpp', 'scripts/build_nnue_observer.py']
    relative += [str(p.relative_to(ROOT)) for p in sorted(SOURCE.glob('*.h'))]
    relative += [str((SOURCE/name).relative_to(ROOT)) for name in ('nnue.cpp', 'position.cpp', 'tables.cpp')]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in relative}
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-march=native', '-fPIC', '-shared']
    command = ['g++', *flags, '-I', str(SOURCE), str(ROOT/'research/nnue_observer_bridge.cpp'),
               *[str(SOURCE/n) for n in ('nnue.cpp', 'position.cpp', 'tables.cpp')], '-o', str(OUT/'libobserver_nnue.so')]
    subprocess.run(command, check=True)
    assert hashes == {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in relative}
    manifest = {'schema': 1, 'upstream': 'devinokeefe/jieqi', 'commit': '331a6e612a038c4a04cf980f7b9482910fd50078',
                'license': 'MIT', 'flags': flags, 'source_sha256': hashes,
                'library_sha256': hashlib.sha256((OUT/'libobserver_nnue.so').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__': main()
