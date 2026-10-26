import {legalMoves, move, publicState, other} from '../vendor/mist-jieqi/engine.js';
import {remaining, values} from '../vendor/mist-jieqi/ai.js';
export const VERSION='action-features-v1';
const kinds=['K','R','N','B','A','C','P'];
export function features(view,m){
 const p=view.board[m.from],q=view.board[m.to],pool=q?.hidden?remaining(view,q.side):[];
 const capture=q?(q.hidden?pool.reduce((s,t)=>s+values[t],0)/Math.max(1,pool.length):values[q.type]):0;
 return [1,...kinds.map(t=>Number((p.hidden?p.role:p.type)===t)),Number(p.hidden),Number(!!q),Number(!!q?.hidden),Math.min(capture,2000)/1000,(m.to%9-4)/4,(Math.floor(m.to/9)-4.5)/4.5,(Math.floor(m.from/9)-Math.floor(m.to/9))*(p.side==='r'?1:-1)/9];
}
export const DIM=15;
export function distribution(view,moves,model){
 const logits=moves.map(m=>{const f=features(view,m);return model?f.reduce((s,x,i)=>s+x*model.policy[i],0):f[11]*2+f[8]*.1;});
 const max=Math.max(...logits),e=logits.map(x=>Math.exp(x-max)),z=e.reduce((a,b)=>a+b,0);return e.map(x=>x/z);
}
export function draw(probs,rng){let x=rng();for(let i=0;i<probs.length;i++){x-=probs[i];if(x<=0)return i;}return probs.length-1;}
export function policy(view,rng,model){const ms=legalMoves(view);return ms[draw(distribution(view,ms,model),rng)];}
function material(view,side){return view.board.reduce((s,p)=>{if(!p)return s;const pool=p.hidden?remaining(view,p.side):[];const v=p.hidden?pool.reduce((a,t)=>a+values[t],0)/Math.max(1,pool.length):values[p.type];return s+(p.side===side?1:-1)*v;},0);}
// Root-only belief Monte Carlo baseline. This is NOT POMCP/ISMCTS.
// Opponent rollout decisions are reconstructed from that opponent's observation.
export function search(view,{rng,model=null,simulations=64,horizon=8}={}){
 if(view.viewer!==view.turn)throw Error('Search requires acting-player observation');
 const ms=legalMoves(view);if(!ms.length)throw Error('No legal actions');
 const n=ms.map(()=>0),q=ms.map(()=>0),prior=distribution(view,ms,model);
 for(let it=0;it<Math.max(simulations,ms.length);it++){
  let a=n.findIndex(x=>x===0);
  if(a<0){let best=-Infinity;for(let i=0;i<ms.length;i++){const u=q[i]/n[i]+1.5*prior[i]*Math.sqrt(it)/(1+n[i]);if(u>best){best=u;a=i;}}}
  let s=sampleWorld(view,rng);s=move(s,ms[a]);
  // Root observer knows their captures; other player's unobserved capture
  // identities must be sampled too, not copied from the true simulator.
  for(let t=0;t<horizon&&!s.result;t++){
   const obs=publicState(s,s.turn),m=policy(obs,rng,model);if(!m)break;s=move(s,m);
  }
  const reward=s.result?(s.result.winner===null?0:s.result.winner===view.turn?1:-1):Math.tanh(material(publicState(s,view.turn),view.turn)/1800);
  n[a]++;q[a]+=reward;
 }
 const z=n.reduce((a,b)=>a+b,0),pi=n.map(x=>x/z);
 return {move:ms[n.indexOf(Math.max(...n))],moves:ms,pi,features:ms.map(m=>features(view,m)),visits:n};
}

export function sampleWorld(view,rng){
 const s=structuredClone(view);
 for(const side of ['r','b']){
  const pool=remaining(view,side);
  const slots=[...s.board,...s.captured].filter(p=>p?.side===side&&p.hidden);
  if(pool.length!==slots.length)throw Error('Inconsistent hidden inventory');
  for(let i=pool.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[pool[i],pool[j]]=[pool[j],pool[i]];}
  slots.forEach(p=>p.type=pool.pop());
 }
 return s;
}
