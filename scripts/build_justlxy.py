"""Build the pinned original web C++ module without modifying its search or bindings."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'external/justlxy-jieqi'
out = root / 'build/justlxy-module'
revision = '9d826d7ff07efea3871417233c2f730997862d13'
actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
if actual != revision or subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD'], text=True).strip():
    raise SystemExit('Require the unchanged pinned justLxy/Jieqi source')
if sys.version_info[:2] != (3, 11):
    raise SystemExit('This validated recipe requires Python 3.11')
commands = [
    ['cmake', '-S', str(source / 'cppjieqi'), '-B', str(out), f'-DPYTHON_EXECUTABLE={sys.executable}'],
    ['cmake', '--build', str(out), '-j', '2'],
]
for command in commands:
    subprocess.run(command, check=True)
modules = list(out.glob('cppjieqi*.so'))
if len(modules) != 1:
    raise SystemExit('Expected exactly one native extension')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
# Public-safe metadata: no host identifiers or environment paths.
manifest = {
    'repository': 'https://github.com/justLxy/Jieqi',
    'revision': revision,
    'license': 'GPL-3.0; upstream LICENSE.md header has a nonstandard date',
    'python': '.'.join(map(str, sys.version_info[:3])),
    'compiler': subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
    'cmake': subprocess.check_output(['cmake', '--version'], text=True).splitlines()[0],
    'module': modules[0].name,
    'module_sha256': sha(modules[0]),
    'bindings_sha256': sha(source / 'cppjieqi/bindings.cpp'),
    'score_sha256': sha(source / 'score.conf'),
    'opening_book_sha256': sha(source / 'kaijuku'),
    'source_modifications': [],
    'scope': 'Build provenance only; private-inventory and timed-match interfaces are not verified.',
}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
