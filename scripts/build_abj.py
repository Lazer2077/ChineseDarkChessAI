"""Build published AB-JChess v0.2d with a documented GCC11 declaration-only repair."""
import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];upstream=root/'external/AB-JChess-v0.2d';out=root/'build/abj-v02d'
sha=subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()
if sha!='fe35d87c7553fd23e8aab3a1dde7e712f286b416':raise SystemExit('Wrong release source')
if out.exists():raise SystemExit('Output already exists')
source=out/'source';shutil.copytree(upstream/'abjchess',source)
shutil.copy(upstream/'LICENSE',out/'LICENSE.upstream')
p=source/'src/bitboard.h';text=p.read_text();start=text.index('constexpr Bitboard square_bb(');end=text.index('constexpr bool more_than_one',start)
segment=text[start:end];patched=segment.replace('constexpr ','inline ');assert patched!=segment
p.write_text(text[:start]+patched+text[end:])
manifest={'source':'https://github.com/lxsgx23/AB-JChess','tag':'v0.2d','commit':sha,'repair':'Remove constexpr from square_bb and nine dependent operator declarations for GCC11; all function bodies unchanged. No search/evaluation changes.','header_before_sha256':hashlib.sha256(text.encode()).hexdigest(),'header_after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(['make','-C',source,'-j2','build','ARCH=x86-64-avx2','COMP=gcc'],check=True)
manifest['binary_sha256']=hashlib.sha256((source/'src/AB-JChess').read_bytes()).hexdigest();(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
