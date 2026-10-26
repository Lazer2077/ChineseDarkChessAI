"""Reproducibly build experimental CDCA changes over a fixed, unmodified MIT backbone."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',default='build/adaptive');p.add_argument('--jobs',type=int,default=2);a=p.parse_args()
root=Path(__file__).resolve().parents[1];upstream=root/'external/devinokeefe-jieqi';out=(root/a.out).resolve()
if out.exists():raise SystemExit('Output exists; use a new build directory')
sha=subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()
if sha!='331a6e612a038c4a04cf980f7b9482910fd50078' or subprocess.check_output(['git','-C',upstream,'diff','HEAD'],text=True).strip():raise SystemExit('Backbone differs from pinned source')
source=out/'source';source.mkdir(parents=True)
for name in ['engine','native']:
    shutil.copytree(upstream/name,source/name,ignore=shutil.ignore_patterns('build','runs','data','__pycache__'))
shutil.copy(upstream/'LICENSE',source/'LICENSE.jqe')
s=source/'engine/src/search.cpp';text=s.read_text()
def edit(old,new):
    global text
    if text.count(old)!=1:raise RuntimeError('Patch anchor mismatch: '+old[:80])
    text=text.replace(old,new)
edit('#include "search.h"','#include "search.h"\n#include "chance_allocation.h"')
edit('int P_ChanceK = 430;', 'int P_CDCAAdaptive = 1;\nint P_CDCAThreshold = 50;\nint P_CDCAExtra = 0;\nint P_ChanceK = 430;')
edit('const bool skipRep = P_ChanceMixSkip &&', 'const bool skipRep = !P_CDCAAdaptive && P_ChanceMixSkip &&')
edit('        if (repByProbe) {', '''        int deepOrder[NUM_SLOTS];
        for (int i = 0; i < n; ++i) deepOrder[i] = i;
        if (P_CDCAAdaptive && pvProbe && !repByProbe && rhoPct == 0) {
            const auto allocation = cdca::allocate(cnt, e0, n, P_CDCAThreshold / 1000.0);
            if (allocation.active) {
                for (int i = 0; i < n; ++i) deepOrder[i] = allocation.order[i];
                cvI = deepOrder[0];
                if (P_CDCAExtra) top = std::max(top, std::min(n, 2));
            }
        }
        if (repByProbe) {''')
edit('for (int j = top > 1 ? std::min(top, n) - 1 : -1; j >= 0; --j) {', 'for (int rank = top > 1 ? std::min(top, n) - 1 : -1; rank >= 0; --rank) {\n            const int j = deepOrder[rank];')
edit('static std::vector<Param> p = {', 'static std::vector<Param> p = {\n        {"CDCAAdaptive", &P_CDCAAdaptive, 0, 1}, {"CDCAThreshold", &P_CDCAThreshold, 0, 1000}, {"CDCAExtra", &P_CDCAExtra, 0, 1},')
s.write_text(text)
shutil.copy(root/'native/chance_allocation.h',source/'engine/src/chance_allocation.h')
uci=source/'engine/src/uci.cpp';uci.write_text(uci.read_text().replace('id name jqe 0.1','id name CDCA adaptive-v1 (jqe backbone)').replace('id author jqe authors','id author ChineseDarkChessAI; jqe backbone by Devin O\'Keefe'))
subprocess.run(['cmake','-S',source/'engine','-B',out/'bin','-DCMAKE_BUILD_TYPE=Release'],check=True)
subprocess.run(['cmake','--build',out/'bin','-j',str(a.jobs),'--target','jqe'],check=True)
manifest={'backbone':'https://github.com/devinokeefe/jieqi','commit':sha,'backbone_license':'MIT','binary_sha256':hashlib.sha256((out/'bin/jqe').read_bytes()).hexdigest(),'algorithm':'probe-variance identity allocation v1','header_sha256':hashlib.sha256((root/'native/chance_allocation.h').read_bytes()).hexdigest(),'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'defaults':{'CDCAAdaptive':1,'CDCAThreshold':50,'CDCAExtra':0}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
