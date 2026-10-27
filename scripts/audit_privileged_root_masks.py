"""Bounded training-corpus audit of true-deal and hidden-inventory root masks."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from research.neural_replay import sha,save_json
from research.policy_value import AZ
from research.private_inventory import PrivateInventory
from research.privileged_training import fresh_fixed_game,actor_projection,training_pair
from wait_privileged_corpus import main as wait_main


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args()
    path=ROOT/args.plan;plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Mask audit plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Mask audit source changed: '+name)
    verify();sys.argv=[sys.argv[0],'--plan',plan['corpus_plan']];wait_main()
    cp=json.loads((ROOT/plan['corpus_plan']).read_text())
    manifest=json.loads((ROOT/cp['out']/'manifest.json').read_text())
    entries={x['index']:x for x in manifest['episodes']}
    report={'schema':1,'status':'running','scope':plan['scope'],'plan_sha256':digest,
        'games':0,'positions':0,'ban_positions':0,'eligible_unknown_inventory_roots':0,
        'compatible_inventory_worlds':0,'actual_vs_projected_mask_differences':0,
        'inventory_sensitive_roots':0,'selected_not_universally_legal':0,'counterexamples':[]}
    out=ROOT/plan['out']
    if out.exists():raise ValueError('Refuse to overwrite mask audit')
    for index in cp['splits']['training']:
        entry=entries[index];file=ROOT/entry['file']
        if sha(file)!=entry['npz_sha256']:raise ValueError('Training episode changed')
        with np.load(file,allow_pickle=False) as arrays:data={k:arrays[k] for k in arrays.files}
        state,belief,rng=fresh_fixed_game(entry['seed'])
        for ply,raw_move in enumerate(data['moves']):
            raw,truth=training_pair(state,belief)
            if not np.array_equal(raw,data['records'][ply]) or not np.array_equal(truth,data['dark_truth'][ply]):
                raise ValueError('Seeded training pair mismatch')
            move=tuple(map(int,raw_move));actual=set(state.legal_moves())
            projected=set(actor_projection(state).legal_moves())
            if move not in actual or move not in projected:raise ValueError('Recorded action illegal in actual/projected state')
            report['positions']+=1
            if actual!=projected:
                report['actual_vs_projected_mask_differences']+=1
                report['counterexamples'].append({'episode':index,'ply':ply,'kind':'true-assignment-map',
                    'actual_only':[AZ.jq.format_move_uci(m) for m in sorted(actual-projected)],
                    'projected_only':[AZ.jq.format_move_uci(m) for m in sorted(projected-actual)]})
            if any(state._banned_for_side()):
                report['ban_positions']+=1;actor=state.side;info=belief.info_state(state,actor)
                if info.unknown_removals[actor]>0 and info.pool_total_known[actor]>0:
                    report['eligible_unknown_inventory_roots']+=1
                    if info.unknown_removals[1-actor]!=0:raise ValueError('Unexpected unknown enemy inventory')
                    inv=PrivateInventory(tuple(info.pool_known[actor][t] for t in AZ.jq.POOL_TYPES),info.unknown_removals[actor])
                    masks=set();intersection=None
                    for hidden in inv.log_probabilities:
                        pools=[row.copy() for row in info.pool_known]
                        for t,count in zip(AZ.jq.POOL_TYPES,hidden):pools[actor][t]-=count
                        missing=info.unknown_removals.copy();missing[actor]=0
                        world=replace(info,pool_known=pools,unknown_removals=missing).determinize(random.Random(0))
                        mask=frozenset(world.legal_moves());masks.add(mask)
                        intersection=set(mask) if intersection is None else intersection&mask
                        report['compatible_inventory_worlds']+=1
                    if frozenset(projected) not in masks:raise ValueError('Projected actual inventory missing from enumeration')
                    if len(masks)>1:
                        report['inventory_sensitive_roots']+=1
                        report['selected_not_universally_legal']+=move not in intersection
                        report['counterexamples'].append({'episode':index,'ply':ply,'kind':'unknown-own-inventory',
                            'mask_variants':len(masks),'selected_not_universally_legal':move not in intersection})
            undo=AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng)
            if (undo.moved_reveal,undo.captured_reveal)!=tuple(map(int,data['reveals'][ply])):
                raise ValueError('Fixed-deal reveal mismatch')
        if state.game_outcome()!=(entry['outcome'],entry['reason']):raise ValueError('Wrong terminal outcome')
        report['games']+=1
        if report['games']%32==0:verify();save_json(out,report)
    verify();report['status']='completed';save_json(out,report)


if __name__=='__main__':main()
