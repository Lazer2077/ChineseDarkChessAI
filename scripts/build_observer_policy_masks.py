"""Actor-information-only legal masks for existing audited TRAIN trajectories."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import fresh_game,load_episode,sha,save_json
from research.policy_value import AZ,records_from_info
from research.policy_anchor import observer_legal_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args()
    path=Path(args.plan);plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Mask-build plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Mask-build input changed: '+name)
    verify();out=ROOT/plan['out'];out.mkdir(parents=True,exist_ok=False)
    folder=ROOT/plan['episodes'];manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Require complete generation')
    entries={x['index']:x for x in manifest['episodes']}
    masks=[];offsets=[0];rows=[];legal_count=visited_count=full=0
    for index in plan['indices']:
        data,meta=load_episode(folder/f'game-{index:06d}.npz')
        expected=dict(entries[index]);expected.pop('index')
        if meta!=expected:raise ValueError('Episode/manifest mismatch')
        state,belief,rng=fresh_game(meta['seed'])
        for ply,action in enumerate(data['moves']):
            info=belief.info_state(state,state.side)
            if not np.array_equal(records_from_info(info)[0],data['records'][ply]):
                raise ValueError('Observer record mismatch')
            mask=observer_legal_mask(info,data['visit_counts'][ply])
            move=tuple(map(int,action))
            if not mask[AZ.move_to_index(move)]:raise ValueError('Selected move outside observer mask')
            masks.append(np.packbits(mask,bitorder='little'))
            legal_count+=int(mask.sum());visited_count+=int((data['visit_counts'][ply]>0).sum())
            full+=not ((data['records'][ply,:90]&15)==8).any()
            undo=AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng)
            if (undo.moved_reveal,undo.captured_reveal)!=tuple(map(int,data['reveals'][ply])):
                raise ValueError('Actual seeded replay differs')
        offsets.append(len(masks));rows.append({'index':index,'seed':meta['seed'],
            'positions':len(data['records']),'episode_npz_sha256':meta['npz_sha256']})
        if len(rows)%32==0:verify();print(json.dumps({'games':len(rows),'positions':len(masks)}),flush=True)
    np.savez_compressed(out/'masks.npz',mask_bits=np.asarray(masks,dtype=np.uint8),
        episode_indices=np.asarray(plan['indices'],dtype=np.uint32),offsets=np.asarray(offsets,dtype=np.uint32))
    verify();save_json(out/'report.json',{'schema':1,'status':'completed','scope':plan['scope'],
        'plan_sha256':digest,'mask_arrays_sha256':sha(out/'masks.npz'),'games':len(rows),'positions':len(masks),
        'actions':AZ.MOVE_SIZE,'bitorder':'little','fully_public_positions':int(full),
        'total_observer_legal_actions':legal_count,'total_visited_actions':visited_count,
        'all_stored_visits_observer_legal':True,'episodes':rows})


if __name__=='__main__':main()
