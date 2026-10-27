"""One fully public check witness; opening agreement cannot certify full rules."""
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import sha,save_json
from research.native_opponent_policy import FixedDepthOpponent


def main():
    output=ROOT/'results/20261027-jieqi-mcts-public-check-probe.json'
    if output.exists():raise ValueError('Refuse to overwrite rule probe')
    folder=ROOT/'build/jieqi-mcts-c506cdd0';manifest=json.loads((folder/'manifest.json').read_text())
    binary=next(folder.glob('cppjieqi*.so'))
    if manifest['status']!='compiled' or sha(binary)!=manifest['module_sha256']:raise ValueError('Research module changed')
    spec=importlib.util.spec_from_file_location('cppjieqi',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ranks=['....k....','.........','.........','.........','.........','....r....','.........','.........','.........','R...K....']
    fen='4k4/9/9/9/9/4r4/9/9/9/R3K4 w - 0 1'
    game=module.God();game.Initialize(random=False,turn=True,board=''.join(ranks))
    source=sorted(module.UCCI(src,dst) for src,dst in game.GenMoves());hint=game.Hint(depth=1)
    teacher=ROOT/'external/devinokeefe-jieqi/engine/build/jqe';net=ROOT/'external/devinokeefe-jieqi/engine/nets/it014.jqn'
    with FixedDepthOpponent(teacher,net,depth=1) as native:
        native._send('position fen '+fen);native._send('moves');native._send('isready');lines=native._until('readyok')
        lists=[line.split()[1:] for line in lines if line=='moves' or line.startswith('moves ')]
        if len(lists)!=1:raise ValueError('Missing native legal-list echo')
    legal=sorted(lists[0])
    report={'schema':1,'status':'completed','commit':manifest['commit'],'module_sha256':sha(binary),
        'referee_binary_sha256':sha(teacher),'fen':fen,'god_generated_moves':source,'native_legal_moves':legal,
        'source_only':sorted(set(source)-set(legal)),'native_only':sorted(set(legal)-set(source)),
        'sets_equal':source==legal,'depth1_hint':hint,'depth1_hint_native_legal':hint in legal,
        'scope':'Single fully public red-in-check fixture. Compare original God.GenMoves API with unchanged native UCI moves; FEN history reset is explicit. No dark identities, full-history equivalence, engine-strength or defeat claim. A differing generated set does not imply every AIHint choice is illegal.'}
    save_json(output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
