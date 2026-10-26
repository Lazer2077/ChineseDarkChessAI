"""Build the attributed optional node-expansion backend in a separate directory."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'external/yuan-bai-jieqi-lab';out=ROOT/'build/yuan-search'
    if out.exists():raise ValueError('Use a fresh build directory')
    commit='4173aa3ddc72e19227a619ffc4ea80a88b3ba2b4'
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=commit or subprocess.check_output(['git','-C',str(source),'diff','HEAD'],text=True).strip():
        raise ValueError('Require unchanged pinned upstream source')
    copied=out/'source';copied.mkdir(parents=True)
    files=[*source.glob('*.py'),source/'LICENSE',source/'native/jieqi_search.pyx',source/'tools/build_search.py']
    inputs={}
    for file in files:
        name=file.relative_to(source);target=copied/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file,target);inputs[str(name)]=hashlib.sha256(file.read_bytes()).hexdigest()
    subprocess.run([sys.executable,str(copied/'tools/build_search.py')],cwd=ROOT,check=True)
    binaries=list(copied.glob('_jieqi_search*.so'))
    if len(binaries)!=1:raise ValueError('Expected one native search extension')
    for name,h in inputs.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=h or hashlib.sha256((copied/name).read_bytes()).hexdigest()!=h:
            raise ValueError('Build source changed')
    manifest={'schema':1,'repository':'https://github.com/Yuan-Bai/jieqi-lab','commit':commit,
        'license':'MIT; original LICENSE copied','scope':'Original optional Cython PUCT expand method only; Python constructor and original search unchanged. Build is outside original checkout. Not an algorithmic innovation or strength result.',
        'python_version':sys.version.split()[0],
        'source_sha256':inputs,'binary_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in binaries}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
