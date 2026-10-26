import {seeded} from '../vendor/mist-jieqi/ai.js';
export function pairedEvidence(rows,expectedGames,{confirmation=false}={}) {
 if(!Number.isInteger(expectedGames)||expectedGames<2||expectedGames%2)throw Error('Require an even expected sample size');
 for(const r of rows)if(!['candidate','mist-native'].includes(r.red)||![0,.5,1,null].includes(r.result)||!Number.isInteger(r.pair)||r.seed===undefined)throw Error('Malformed match evidence');
 const complete=rows.filter(r=>r.result!==null),voids=rows.length-complete.length;
 const scores=complete.map(r=>r.red==='candidate'?r.result:1-r.result);
 const pairs=new Map();
 for(const r of complete){if(!pairs.has(r.pair))pairs.set(r.pair,[]);pairs.get(r.pair).push(r);}
 const paired=[];
 for(const group of pairs.values()){
  if(group.length!==2||new Set(group.map(r=>r.red)).size!==2||new Set(group.map(r=>String(r.seed))).size!==1)continue;
  paired.push(group.reduce((sum,r)=>sum+(r.red==='candidate'?r.result:1-r.result),0)/2);
 }
 const mean=x=>x.reduce((s,v)=>s+v,0)/x.length;
 let bootstrap=null,bounded=null;
 if(paired.length){const r=Math.sqrt(Math.log(40)/(2*paired.length)),m=mean(paired);bounded=[Math.max(0,m-r),Math.min(1,m+r)];}
 if(paired.length>=20){
  const rng=seeded(20261026),boot=[];
  for(let k=0;k<10000;k++){let s=0;for(let j=0;j<paired.length;j++)s+=paired[Math.floor(rng()*paired.length)];boot.push(s/paired.length);}
  boot.sort((a,b)=>a-b);bootstrap=[boot[249],boot[9749]];
 }
 const technical=complete.filter(r=>r.technical).map(r=>r.reason);
 const valid=rows.length===expectedGames&&complete.length===expectedGames&&paired.length===expectedGames/2&&!voids&&!technical.length;
 return {games:complete.length,expected_games:expectedGames,wins:scores.filter(x=>x===1).length,draws:scores.filter(x=>x===.5).length,losses:scores.filter(x=>x===0).length,voids,score:scores.length?mean(scores):null,complete_pairs:paired.length,paired_bootstrap_95:bootstrap,paired_hoeffding_95:bounded,technical_results:technical,valid_technical_run:valid,single_match_superiority_gate:confirmation&&valid&&paired.length>=100&&bootstrap[0]>.5&&bounded[0]>.5,promotion_supported:false};
}
