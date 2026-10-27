"""Observable-entropy strata for existing frozen teacher-retention evaluation."""
import inspect
from pathlib import Path
import sys
import textwrap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import diagnose_teacher_retention as original
from research.observation_uncertainty import record_uncertainty


def main():
    source=textwrap.dedent(inspect.getsource(original.main))
    def replace(needle,value):
        nonlocal source
        if source.count(needle)!=1:raise ValueError('Retention extension site changed')
        source=source.replace(needle,value)
    replace('    full=((raw[:,:90]&15)!=8).all(1)',
        """    full=((raw[:,:90]&15)!=8).all(1)
    uncertainty=_record_uncertainty(raw)['normalized_entropy']
    strata=[('all',np.ones(len(raw),dtype=bool)),('partial',~full),('public',full),
            ('type-certain',uncertainty==0),('entropy0-.25',(uncertainty>0)&(uncertainty<=.25)),
            ('entropy.25-.5',(uncertainty>.25)&(uncertainty<=.5)),
            ('entropy.5-.75',(uncertainty>.5)&(uncertainty<=.75)),('entropy.75-1',uncertainty>.75)]""")
    replace("        for label,mask in [('all',np.ones(len(raw),dtype=bool)),('partial',~full),('public',full)]:",
            '        for label,mask in strata:')
    replace('            ids=np.flatnonzero(mask);subset=logs[mask]',
        """            if not mask.any():
                row['strata'][label]={'positions':0}
                continue
            ids=np.flatnonzero(mask);subset=logs[mask]""")
    environment=dict(original.__dict__,_record_uncertainty=record_uncertainty)
    exec(compile(source,'<cdca-observable-entropy-retention>','exec'),environment)
    environment['main']()


if __name__=='__main__':main()
