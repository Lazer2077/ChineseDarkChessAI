"""Pinned native selfplay actor; exclude truncated/error games from RL labels."""
import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];up=root/'external/devinokeefe-jieqi';out=root/'build/data-actor'
sha=subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()
if sha!='331a6e612a038c4a04cf980f7b9482910fd50078' or subprocess.check_output(['git','-C',up,'diff','HEAD'],text=True).strip():raise SystemExit('Unexpected backbone source')
if out.exists():raise SystemExit('Output already exists')
source=out/'source';source.mkdir(parents=True)
for name in ['engine','native']:shutil.copytree(up/name,source/name,ignore=shutil.ignore_patterns('build','runs','data','__pycache__'))
shutil.copy(up/'LICENSE',source/'LICENSE.jqe')
p=source/'engine/src/selfplay.cpp';s=p.read_text()
def replace(old,new):
 global s
 if s.count(old)!=1:raise RuntimeError('Patch anchor mismatch: '+old[:50])
 s=s.replace(old,new)
replace('std::vector<TrainRecord>& recs) {','std::vector<TrainRecord>& recs, bool& valid) {\n    valid = true;')
replace('if (v.noCapture >= NO_CAPTURE_LIMIT || ply >= cfg.maxPly) { result = 0; break; }','if (v.noCapture >= NO_CAPTURE_LIMIT) { result = 0; break; }\n        if (ply >= cfg.maxPly) { valid = false; break; }')
replace('std::fprintf(stderr, "selfplay: apply failed: %s\\n", err.c_str());','std::fprintf(stderr, "selfplay: apply failed: %s\\n", err.c_str());\n            valid = false;')
replace('    for (std::size_t i = first; i < recs.size(); ++i)','    if (!valid) recs.resize(first);\n    for (std::size_t i = first; i < recs.size(); ++i)')
replace('std::atomic<int> games{0}, redWins{0}, blackWins{0}, draws{0};','std::atomic<int> games{0}, redWins{0}, blackWins{0}, draws{0}, rejected{0};')
replace('const int r = play_one(cfg, mix(cfg.seed * 0x100000001B3ull + uint64_t(g)), S, recs);','bool valid = false;\n            const int r = play_one(cfg, mix(cfg.seed * 0x100000001B3ull + uint64_t(g)), S, recs, valid);\n            if (!valid) ++st.rejected;')
replace('(r > 0 ? st.redWins : r < 0 ? st.blackWins : st.draws)++;','if (valid) (r > 0 ? st.redWins : r < 0 ? st.blackWins : st.draws)++;')
replace('    std::fclose(f);','    std::fclose(f);\n    std::printf("info string selfplay rejected %d\\n", st.rejected.load());')
p.write_text(s)
subprocess.run(['cmake','-S',source/'engine','-B',out/'bin','-DCMAKE_BUILD_TYPE=Release'],check=True)
subprocess.run(['cmake','--build',out/'bin','-j2','--target','jqe'],check=True)
(out/'manifest.json').write_text(json.dumps({'backbone_commit':sha,'license':'MIT backbone; project GPL-3.0','binary_sha256':hashlib.sha256((out/'bin/jqe').read_bytes()).hexdigest(),'selfplay_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'search':'unchanged upstream search','changes':'Exclude all records of games truncated at maxply or failed observed-move application; report rejection count.'},indent=2)+'\n')
