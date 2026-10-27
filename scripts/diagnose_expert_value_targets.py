"""Frozen descriptive value-target and exact-input-alias study; no fitting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import load_episode,fresh_game,sha,save_json
from research.policy_value import AZ,encode_records
from research.neural_search import load_model
from research.observer_expert import observer_token
from research.observation_uncertainty import record_uncertainty
from research.expert_value_diagnostic import search_score_proxy,observer_input_keys


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    path=ROOT/a.plan;plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Value-target diagnostic recipe changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen value diagnostic input changed: '+name)
    verify();out=ROOT/plan['out'];out.mkdir(parents=True,exist_ok=False)
    annotation=json.loads((ROOT/plan['annotation_report']).read_text())
    if annotation['status']!='completed':raise ValueError('Require completed expert annotations')
    entries={r['game']:r for r in annotation['episodes']}
    annotation_plan=ROOT/plan['annotation_plan'];recipe=json.loads(annotation_plan.read_text())
    if annotation['plan_sha256']!=sha(annotation_plan):raise ValueError('Annotation recipe differs')
    expected=[i for i in recipe['selection_validation_games'] if entries[i]['status']=='compatible-roots']
    if plan['games']!=expected:raise ValueError('Require exactly the accepted original selection games')
    raw=[];z=[];proxy=[];games=[];actions=[];histories=[];sources=[]
    for game in plan['games']:
        row=entries[game]
        if row['status']!='compatible-roots':raise ValueError('Quarantined game requested')
        source=ROOT/plan['episodes']/f'game-{game:06d}.npz';data,metadata=load_episode(source)
        label_file=ROOT/plan['labels']/f'game-{game:06d}-labels.json'
        if sha(source)!=row['source_npz_sha256'] or sha(label_file)!=row['labels_sha256']:raise ValueError('Source/label digest differs')
        labels=json.loads(label_file.read_text())
        if labels['source_npz_sha256']!=sha(source) or len(labels['labels'])!=metadata['positions']:raise ValueError('Wrong expert roots')
        state,belief,rng=fresh_game(metadata['seed']);observed=[[],[]]
        for ply,label in enumerate(labels['labels']):
            if label['ply']!=ply or label['actor']!=state.side:raise ValueError('Wrong expert perspective')
            move=tuple(map(int,data['moves'][ply]))
            if label['learner_move']!=AZ.jq.format_move_uci(move):raise ValueError('Wrong source history')
            proxy.append(search_score_proxy(label['score'],plan['score_scale']))
            actions.append(label['expert_move']);histories.append(tuple(observed[state.side]));games.append(game)
            undo=AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng)
            for viewer in (AZ.jq.RED,AZ.jq.BLACK):observed[viewer].append(observer_token(move,undo,viewer))
        raw.append(data['records']);z.append(data['values'])
        sources.append({'game':game,'npz_sha256':sha(source),'positions':metadata['positions'],'labels_sha256':sha(label_file)})
    raw=np.concatenate(raw);z=np.concatenate(z);proxy=np.asarray(proxy,dtype=np.float64);games=np.asarray(games,dtype=np.int32)
    uncertainty=record_uncertainty(raw)['normalized_entropy'];full=((raw[:,:90]&15)!=8).all(1)
    strata={'all':np.ones(len(raw),dtype=bool),'partial':~full,'public':full,'type-certain':uncertainty==0,
            'entropy0-.25':(uncertainty>0)&(uncertainty<=.25),'entropy.25-.5':(uncertainty>.25)&(uncertainty<=.5),
            'entropy.5-.75':(uncertainty>.5)&(uncertainty<=.75),'entropy.75-1':uncertainty>.75}
    def summarize(values,mask):
        selected=games[mask]
        if not len(selected):return {'positions':0,'games':0}
        return {'positions':int(mask.sum()),'games':len(np.unique(selected)),
            'position_mean':float(values[mask].mean()),
            'game_mean':float(np.mean([values[mask][selected==g].mean() for g in np.unique(selected)]))}
    groups=defaultdict(list)
    for i,key in enumerate(observer_input_keys(raw)):groups[key].append(i)
    conflicts=[];minimum_ce=0.;minimum_mse=0.
    for ids in groups.values():
        _,counts=np.unique([actions[i] for i in ids],return_counts=True);probs=counts/counts.sum()
        minimum_ce-=len(ids)*float((probs*np.log(probs)).sum())
        minimum_mse+=float(((z[ids]-z[ids].mean())**2).sum())
        if len(counts)>1:conflicts.append(ids)
    alias={'positions':len(raw),'unique_inputs':len(groups),'repeated_input_groups':sum(len(v)>1 for v in groups.values()),
        'groups_with_multiple_projected_histories':sum(len({histories[i] for i in ids})>1 for ids in groups.values()),
        'conflicting_expert_action_groups':len(conflicts),'positions_in_conflicting_groups':sum(map(len,conflicts)),
        'minimum_empirical_policy_ce_for_this_finite_encoded_sample':minimum_ce/len(raw),
        'minimum_empirical_terminal_mse_for_this_finite_encoded_sample':minimum_mse/len(raw),
        'scope':'Exact duplicates in this finite reused selection sample only; empirical fitted-sample minima are not Bayes-risk bounds, proof of history sufficiency, or generalization estimates.'}
    report={'schema':1,'status':'running','scope':plan['scope'],'plan_sha256':digest,'sources':sources,
        'teacher_score_proxy':'cp -> tanh(cp/860); nonzero mate -> sign(mate). Fixed monotone proxy, not calibrated value or game-theoretic truth.',
        'alias_diagnostic':alias,'source_terminal_vs_teacher_search_proxy_mse':{k:summarize((z-proxy)**2,m) for k,m in strata.items()},'models':[]}
    save_json(out/'report.json',report);predictions=[]
    torch.set_num_threads(1);torch.set_num_interop_threads(1);device=torch.device(plan['device'])
    for model in plan['models']:
        net,meta=load_model(ROOT/model['directory'],device);values=[]
        with torch.inference_mode():
            for start in range(0,len(raw),plan['batch']):
                _,v=net(encode_records(torch.from_numpy(raw[start:start+plan['batch']]).to(device)));values.append(v.cpu().numpy())
        value=np.concatenate(values).astype(np.float64);predictions.append(value)
        row={'label':model['label'],'model_sha256':meta['npz_sha256'],'strata':{k:{
             'source_learner_terminal_mse':summarize((value-z)**2,m),
             'teacher_search_proxy_mse':summarize((value-proxy)**2,m),
             'mean_absolute_prediction':summarize(np.abs(value),m)} for k,m in strata.items()}}
        report['models'].append(row);verify();save_json(out/'report.json',report)
        print(json.dumps({'label':row['label'],'all':row['strata']['all']}),flush=True);del net
    with (out/'predictions.npz').open('wb') as file:
        np.savez_compressed(file,predictions=np.asarray(predictions),source_terminal=z,teacher_search_proxy=proxy,games=games,normalized_entropy=uncertainty)
    verify();report.update(status='completed',prediction_arrays_sha256=sha(out/'predictions.npz'));save_json(out/'report.json',report)


if __name__=='__main__':main()
