"""Build unchanged licensed Muses-family research source; no engine promotion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external/miaosisari-jieqi-mcts'
COMMIT='c506cdd03693a0cb34797e2ed8194a372b48528e'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()!=COMMIT:raise ValueError('Wrong research source pin')
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip():raise ValueError('Upstream source is modified')
    license_text=(SOURCE/'LICENSE').read_text()
    if 'GNU GENERAL PUBLIC LICENSE' not in license_text or 'Version 3' not in license_text:raise ValueError('Expected GPLv3 source license')
    out=ROOT/'build/jieqi-mcts-c506cdd0';out.mkdir(parents=True,exist_ok=False)
    include=Path(torch.__file__).parent/'include';headers=include/'pybind11'
    if not (headers/'pybind11.h').exists():raise ValueError('Installed modern pybind11 headers unavailable')
    source_files=[SOURCE/'main.cpp',*sorted((SOURCE/'board').glob('*.cpp'))]
    target=out/('cppjieqi'+sysconfig.get_config_var('EXT_SUFFIX'))
    # Keep all game/search sources unchanged. The vendored older binding headers
    # are replaced solely by the installed headers for the current Python ABI.
    command=[os.environ.get('CXX','c++'),'-shared','-fPIC','-std=c++17','-O2',
        '-I'+str(include),'-I'+sysconfig.get_paths()['include'],*[str(p) for p in source_files],'-o',str(target)]
    with (out/'compiler.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    record={'schema':1,'status':'compiled' if not result.returncode else 'compile-failed',
        'repository':'https://github.com/miaosiSari/JieqiMCTS','commit':COMMIT,'license':'GPL-3.0',
        'license_sha256':sha(SOURCE/'LICENSE'),'python_abi':sysconfig.get_config_var('SOABI'),
        'compiler':subprocess.check_output([command[0],'--version'],text=True).splitlines()[0],
        'source_sha256':{str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.suffix in ['.h','.cpp'] and 'pybind11' not in p.relative_to(SOURCE).parts},
        'binding_header_sha256':{str(p.relative_to(include)):sha(p) for p in sorted(headers.rglob('*.h'))},
        'build_profile':'Original unchanged C++ sources; installed pybind11 headers instead of vendored headers; no search/rule changes.',
        'scope':'Build feasibility only. No UCI adapter, strict observation/history/rule compatibility, full MCTS self-play controller, native match or strength established.',
        'compiler_exit_code':result.returncode}
    if not result.returncode:record['module_sha256']=sha(target)
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['source_sha256','binding_header_sha256']}),flush=True)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
