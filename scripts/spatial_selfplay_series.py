"""Original fixed RL iteration recipe with decoder-aware entrypoints."""
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import neural_selfplay_series as original
from research.neural_replay import sha


def install():
    if sha(ROOT / 'scripts/neural_selfplay_series.py') != 'ee01c4b6770292bd97efd020fb3c9432666ca2c742288665350e4ae6f41cf7a4':
        raise ValueError('Pinned original self-play controller changed')
    source = textwrap.dedent(inspect.getsource(original.main))

    def replace(needle, value, count=1):
        nonlocal source
        if source.count(needle) != count:
            raise ValueError('Self-play controller patch site changed')
        source = source.replace(needle, value)

    replace("generation={**plan['generation'],'schema':1", "generation={**plan['generation'],'policy_head':plan['policy_head'],'schema':1")
    replace("training={**plan['training'],'schema':1", "training={**plan['training'],'policy_head':plan['policy_head'],'schema':1")
    replace("'scripts/generate_neural_selfplay.py'", "'scripts/generate_spatial_selfplay.py'")
    replace("'scripts/train_neural_selfplay.py'", "'scripts/train_spatial_selfplay.py'")
    replace("'adapters/neural_policy_gpu_uci.sh'", "'adapters/spatial_policy_gpu_uci.sh'")
    replace("'--candidate=adapters/neural_policy_gpu_uci.sh'", "'--candidate=adapters/spatial_policy_gpu_uci.sh'")
    replace("            tp=iteration/'training-plan.json';save_json(tp,training)",
            "            tp=iteration/'training-plan.json'")
    needle = "            run('training',[sys.executable,'scripts/train_spatial_selfplay.py','--plan',str(tp)],iteration/'training.log')"
    replace(needle, """            # Bind all completed replay artifacts before writing the training plan once.
            for file in sorted(replay.iterdir()):
                if file.suffix in ('.npz','.json'):
                    training['sha256'][str(file.relative_to(ROOT))]=sha(file)
            save_json(tp,training)
""" + needle)
    namespace = dict(original.__dict__)
    exec(compile(source, '<cdca-spatial-selfplay-controller>', 'exec'), namespace)
    return namespace['main'], hashlib.sha256(source.encode()).hexdigest()


if __name__ == '__main__':
    main, _ = install()
    main()
