"""Build optional original Yuan rule/legal acceleration outside its pinned checkout."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];up=root/'external/yuan-bai-jieqi-lab';out=root/'build/yuan-native'
if out.exists():raise SystemExit('Use fresh output')
commit='4173aa3ddc72e19227a619ffc4ea80a88b3ba2b4'
if subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()!=commit or subprocess.check_output(['git','-C',up,'diff','HEAD'],text=True).strip():raise SystemExit('Require pinned unchanged Yuan source')
source=out/'source';source.mkdir(parents=True)
for name in ['jieqi.py','ismcts.py','LICENSE']:shutil.copy(up/name,source/name)
shutil.copytree(up/'native',source/'native')
(source/'tools').mkdir()
for name in ['build_rules.py','build_legal.py']:shutil.copy(up/'tools'/name,source/'tools'/name)
for name in ['build_rules.py','build_legal.py']:
 subprocess.run([sys.executable,source/'tools'/name],check=True,cwd=root)
files=list(source.glob('_jieqi*.so'))
if len(files)!=2:raise SystemExit('Expected both native rule/legal modules')
manifest={'schema':1,'repository':'https://github.com/Yuan-Bai/jieqi-lab','commit':commit,'license':'MIT; original LICENSE copied',
 'scope':'Original optional Cython rules and legal-move acceleration; original Python search and all source files unchanged. Does not supply a neural model.',
 'python_version':sys.version.split()[0],'binary_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
 'source_sha256':{str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source/'jieqi.py',source/'ismcts.py',*sorted((source/'native').glob('*.pyx'))]}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
