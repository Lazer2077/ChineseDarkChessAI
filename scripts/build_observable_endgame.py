"""Native NNUE canonicalization only after every on-board identity is public."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',default='build/observable-endgame');p.add_argument('--verify',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parents[1];up=r/'external/devinokeefe-jieqi';out=r/a.out
commit=subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()
if commit!='331a6e612a038c4a04cf980f7b9482910fd50078' or subprocess.check_output(['git','-C',up,'diff','HEAD'],text=True).strip():raise SystemExit('Require pinned unchanged backbone')
if out.exists():raise SystemExit('Use a new output')
source=out/'source';source.mkdir(parents=True)
for name in ['engine','native']:shutil.copytree(up/name,source/name,ignore=shutil.ignore_patterns('build','runs','data','__pycache__'))
shutil.copy(up/'LICENSE',source/'LICENSE.jqe')
f=source/'engine/src/nnue.cpp';s=f.read_text()
for lookup in ['qrow','row']:
 old=f'''        for (int c = 0; c < 2; ++c)
            for (int sl = 0; sl < NUM_SLOTS; ++sl)
                for (int j = 1; j <= pos.urn[c][sl]; ++j) rows[n++] = {lookup}(urn_feature(persp, Color(c), sl, j));'''
 assert s.count(old)==1,lookup
 s=s.replace(old,'''        // No future hidden identity remains: off-board inventory cannot change the game graph.
        if (pos.darkCount[RED] + pos.darkCount[BLACK] > 0) {
'''+old+'\n        }')
old='    if (type_of(pc) == DARK) {\n        const int sl = type_slot(reveal_of(m));'
assert s.count(old)==1
s=s.replace(old,'''    const int beforeDark = pos.darkCount[RED] + pos.darkCount[BLACK];
    const int afterDark = beforeDark - int(type_of(pc) == DARK) - int(cap && type_of(cap) == DARK);
    // A last-dark reveal/capture changes the entire urn feature block at once.
    // Force full refresh at that boundary; later ordinary moves remain incremental.
    if (beforeDark > 0 && afterDark == 0) d.refresh = 0;
'''+old)
f.write_text(s)
f=source/'engine/src/uci.cpp';s=f.read_text().replace('id name jqe 0.1','id name CDCA observable endgame (jqe backbone)');f.write_text(s)
cmd=['cmake','-S',source/'engine','-B',out/'bin','-DCMAKE_BUILD_TYPE=Release']
if a.verify:cmd.append('-DCMAKE_CXX_FLAGS=-DJQ_VERIFY_NNUE')
subprocess.run(cmd,check=True);subprocess.run(['cmake','--build',out/'bin','-j2','--target','jqe'],check=True)
manifest={'backbone':commit,'license':'MIT backbone; GPL project','binary_sha256':hashlib.sha256((out/'bin/jqe').read_bytes()).hexdigest(),'verify_incremental_accumulators':a.verify,'change':'Omit NNUE urn features only when both on-board dark counts are zero; force accumulator refresh on last-dark reveal/capture. Rules, actual position inventories, histories and chance search remain intact. This does not reveal or move a still-dark piece.','nnue_source_sha256':hashlib.sha256((source/'engine/src/nnue.cpp').read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
