"""Build the pinned Pika/ABJ-NNUE hybrid with its own declared default release net."""
import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];up=root/'external/pika-hefeng-archive';out=root/'build/pika-hefeng'
commit='88bfb82c3063f69e8cc270cad08f24930da0874f'
if subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()!=commit or subprocess.check_output(['git','-C',up,'diff','HEAD'],text=True).strip():raise SystemExit('Require unchanged pinned source')
if out.exists():raise SystemExit('Use a fresh build')
net=root/'external/AB-JChess-v0.2d-net/abjchess-20261010.nnue'
digest=hashlib.sha256(net.read_bytes()).hexdigest()
if digest!='05c13ac64f5b8ede77cfdc408d399d1f235564ba0999e2c768bebaeedb1db761':raise SystemExit('Wrong declared default net')
source=out/'source';source.mkdir(parents=True)
for name in ['src','scripts']:shutil.copytree(up/name,source/name)
shutil.copy(up/'Copying.txt',out/'LICENSE.upstream')
shutil.copy(net,source/'src'/net.name)
header=source/'src/bitboard.h';text=header.read_text();start=text.index('constexpr Bitboard square_bb(');end=text.index('constexpr bool more_than_one',start)
segment=text[start:end];replacement=segment.replace('constexpr ','inline ')
if segment==replacement:raise SystemExit('Expected declaration repair unavailable')
header.write_text(text[:start]+replacement+text[end:])
manifest={'repository':'https://github.com/hefengfan0615/pikafish-jieqi','commit':commit,'license':'GPL-3.0-or-later source headers and root Copying.txt',
 'repair':'GCC11 declaration-only constexpr-to-inline repair for square_bb and dependent operators, retaining function bodies, search and evaluation.',
 'header_before_sha256':hashlib.sha256(text.encode()).hexdigest(),'header_after_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),
 'network':'abjchess-20261010.nnue','network_sha256':digest,'network_origin':'Its scripts/net.sh explicitly selects the AB-JChess v0.2d release archive; this reuses the already verified identical archive net, not a substituted evaluator.',
 'build_date':'20261026','architecture':'x86-64-avx2','scope':'Build audit of distinct Pika-search/ABJ-NNUE hybrid; no observer compatibility, completed-match or strength claim.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(['make','-C',source/'src','-j2','build','ARCH=x86-64-avx2','COMP=gcc','GIT_SHA='+commit[:8],'GIT_DATE=20261026'],check=True)
manifest['binary_sha256']=hashlib.sha256((source/'src/pikafish').read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
