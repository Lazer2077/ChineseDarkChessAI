"""Original native-prefix training arithmetic, replacing only its policy-target loader."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from train_native_terminal_selfplay import install
from research.pruned_policy_targets import load_split


if __name__ == '__main__':
    main, digest = install()
    if digest != 'ab43b9582ed5b508b58c979b9eb1abf3f25c67635c173e6531d7c6196c068d8f':
        raise ValueError('Native-prefix trainer arithmetic changed')
    main.__globals__['_native_load_split'] = load_split
    main()
