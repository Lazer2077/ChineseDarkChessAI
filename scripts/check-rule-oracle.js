import fs from 'node:fs';
import {initial,publicState,legalMoves,move} from '../vendor/mist-jieqi/engine.js';
import {seeded} from '../vendor/mist-jieqi/ai.js';
import {UciEngine,fen,parseMove} from '../src/uci.js';
const binary=process.argv[2];if(!binary)throw Error('Pass jqe binary path');
const e=new UciEngine(binary),rng=seeded(20261026),key=m=>`${m.from}:${m.to}`;
let positions=0,mismatches=[];
try{
 await e.start();
 for(let g=0;g<20;g++){
  let s=initial(rng);
  for(let ply=0;ply<120&&!s.result;ply++){
   const v=publicState(s,s.turn),ours=legalMoves(v);
   e.process.stdin.write(`position fen ${fen(v)}\n`);
   const line=await e.wait('moves',s=>s==='moves'||s.startsWith('moves '));
   const theirs=line.split(' ').slice(1).filter(Boolean).map(x=>parseMove(x)),a=new Set(ours.map(key)),b=new Set(theirs.map(key));positions++;
   const onlyOurs=[...a].filter(x=>!b.has(x)),onlyTheirs=[...b].filter(x=>!a.has(x));
   if(onlyOurs.length||onlyTheirs.length){mismatches.push({game:g,ply,fen:fen(v),onlyOurs,onlyTheirs,observation:v});break;}
   if(!ours.length)break;s=move(s,ours[Math.floor(rng()*ours.length)]);
  }
 }
}finally{e.close();}
const report={oracle:binary,positions,mismatches,scope:'Legal geometry only, not adjudication. Independent implementation agreement is not a formal rule specification.'};
fs.mkdirSync('runs/rule-oracle',{recursive:true});fs.writeFileSync('runs/rule-oracle/report.json',JSON.stringify(report,null,2));console.log(JSON.stringify({positions,mismatches:mismatches.length}));if(mismatches.length)process.exitCode=1;
