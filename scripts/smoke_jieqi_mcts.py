"""Opening-only API check for pinned Muses-family C++/MCTS scaffold."""
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import fresh_game,sha,save_json
from research.observer_expert import ObserverHistoryExpert
from research.teacher_leaf import observer_fen


def main():
    out=ROOT/'results/20261027-jieqi-mcts-build-smoke.json'
    if out.exists():raise ValueError('Refuse to overwrite build smoke')
    folder=ROOT/'build/jieqi-mcts-c506cdd0';manifest=json.loads((folder/'manifest.json').read_text())
    module_path=next(folder.glob('cppjieqi*.so'))
    if manifest['status']!='compiled' or sha(module_path)!=manifest['module_sha256']:raise ValueError('Compiled research artifact changed')
    spec=importlib.util.spec_from_file_location('cppjieqi',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    god=module.God();god.Initialize(random=False)
    legal={module.UCCI(src,dst) for src,dst in god.GenMoves()};hint=god.Hint(depth=1)
    if hint not in legal or len(legal)!=44:raise ValueError('Opening API smoke differs')
    state,belief,_=fresh_game(8320001)
    with ObserverHistoryExpert(ROOT/'external/devinokeefe-jieqi/engine/build/jqe',ROOT/'external/devinokeefe-jieqi/engine/nets/it014.jqn',depth=1) as native:
        snapshot=native.position_snapshot([],observer_fen(belief.info_state(state,state.side)))
    if legal!=set(snapshot['legal_moves']):raise ValueError('Opening legal sets differ')
    searcher=module.Searcher();searcher.Initialize(random=False);before=searcher.get().n_visits
    root_hint=searcher.ai(depth=1);after=searcher.get().n_visits
    if root_hint[2] not in legal or not root_hint[3] or before!=0 or after!=0:raise ValueError('Direct AIHint smoke differs')
    report={**manifest,'status':'opening-smoke-passed','build_manifest_sha256':sha(folder/'manifest.json'),
        'opening_legal_count':len(legal),'opening_native_legal_sets_equal':True,
        'god_depth1_hint':hint,'node_depth1_hint':root_hint[2],
        'node_visits_before_after_direct_ai_hint':[before,after],
        'backprop_method_exposed_in_python':hasattr(searcher,'bp'),
        'scope':'Unchanged pinned C++ game/search sources build with installed binding headers. God and Node expose callable depth1AIBoard4 hints; Searcher.ai does not perform MCTS visit backups in this opening smoke. One opening legal-set comparison only. No full-game UCI, strict private-observation/history/rule compatibility, strength, independent-family or defeat claim.'}
    save_json(out,report);print(json.dumps({k:report[k] for k in ['status','opening_legal_count','god_depth1_hint','node_depth1_hint','backprop_method_exposed_in_python']}),flush=True)


if __name__=='__main__':main()
