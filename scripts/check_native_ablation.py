"""Verify experimental code is behavior-identical to its backbone when disabled."""
import json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];net=root/'external/devinokeefe-jieqi/engine/nets/it014.jqn'
rows={}
for name,binary,opts in [('base',root/'external/devinokeefe-jieqi/engine/build/jqe',''),('disabled',root/'build/adaptive/bin/jqe','setoption name CDCAAdaptive value 0\n'),('adaptive',root/'build/adaptive/bin/jqe','')]:
    p=subprocess.run([binary],input=f'setoption name EvalFile value {net}\n{opts}bench 5\nquit\n',text=True,capture_output=True,check=True)
    rows[name]=[line for line in (p.stdout+p.stderr).splitlines() if line.startswith(('bench ','Nodes searched:'))]
assert rows['base']==rows['disabled'],'Disabled experiment differs from backbone'
assert rows['base']!=rows['adaptive'],'Experiment did not change benchmark search'
report={'depth':5,'positions':len(rows['base'])-1,'disabled_matches':True,'enabled_changes_search':True,'rows':rows}
(root/'runs/native-ablation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
