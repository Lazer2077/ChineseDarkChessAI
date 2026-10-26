"""Build our native candidate for the original reference project's rule profile.

Opponent source/search stays unmodified. Candidate rule changes are explicit.
"""
import hashlib,json,shutil,subprocess
from pathlib import Path
r=Path(__file__).resolve().parents[1];up=r/'external/devinokeefe-jieqi';out=r/'build/mist-profile'
sha=subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()
if sha!='331a6e612a038c4a04cf980f7b9482910fd50078':raise SystemExit('Wrong backbone')
if out.exists():raise SystemExit('Output exists')
source=out/'source';source.mkdir(parents=True)
for name in ['engine','native']:shutil.copytree(up/name,source/name,ignore=shutil.ignore_patterns('build','runs','data','__pycache__'))
shutil.copy(up/'LICENSE',source/'LICENSE.jqe')
p=source/'engine/src/position.h';s=p.read_text();assert s.count('NO_CAPTURE_LIMIT = 80;')==1;p.write_text(s.replace('NO_CAPTURE_LIMIT = 80;','NO_CAPTURE_LIMIT = 120;'))
p=source/'engine/src/position.cpp';s=p.read_text();old='noCapture = cap ? 0 : int16_t(noCapture + 1);';assert s.count(old)==1;s=s.replace(old,'noCapture = cap || type_of(pc) == DARK ? 0 : int16_t(noCapture + 1);')
start=s.index('int Position::repetition_status() const {');end=s.index('bool Position::has_major(',start)
s=s[:start]+'''int Position::repetition_status() const {
    int count = 0;
    for (int i = int(keyHist.size()) - 1; i >= lastIrrev; i -= 2)
        if (keyHist[i] == key && ++count >= 3) return 2; // last mover loses; side to move wins
    return 0;
}

'''+s[end:];p.write_text(s)
p=source/'engine/src/search.cpp';s=p.read_text();old='if (p.noCapture >= NO_CAPTURE_LIMIT) return p.has_legal_move() ? VALUE_DRAW : mated_in(ply);';assert s.count(old)==1;s=s.replace(old,'if (p.noCapture >= NO_CAPTURE_LIMIT || p.moveHist.size() >= 400) return p.has_legal_move() ? VALUE_DRAW : mated_in(ply);');p.write_text(s)
p=source/'engine/src/uci.cpp';s=p.read_text();s=s.replace('id name jqe 0.1','id name CDCA mist-rule-profile (jqe backbone)');old='else if (cmd == "d") out(engine.pos.pretty());';assert s.count(old)==1;s=s.replace(old,'''else if (cmd == "cdca_rules") out("cdca_rules " + std::to_string(engine.pos.noCapture) + " " + std::to_string(engine.pos.repetition_status()) + " " + std::to_string(engine.pos.moveHist.size()));
        '''+old);p.write_text(s)
subprocess.run(['cmake','-S',source/'engine','-B',out/'bin','-DCMAKE_BUILD_TYPE=Release'],check=True)
subprocess.run(['cmake','--build',out/'bin','-j2','--target','jqe'],check=True)
(out/'manifest.json').write_text(json.dumps({'backbone':sha,'license':'MIT backbone, GPL project','profile':'Mist: third public repetition last mover loses; quiet reset on capture/reveal; 120 quiet or 400 total plies draw; mate/stalemate precedence','candidate_binary_sha256':hashlib.sha256((out/'bin/jqe').read_bytes()).hexdigest(),'opponent_modified':False},indent=2)+'\n')
