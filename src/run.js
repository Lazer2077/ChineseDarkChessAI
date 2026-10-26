import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {initial, publicState, move, positionKey, legalMoves} from '../vendor/mist-jieqi/engine.js';
import {seeded, analyze} from '../vendor/mist-jieqi/ai.js';
import {search,draw,VERSION} from './research.js';
const args=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const games=Number(args.games??2),seed=Number(args.seed??1),limit=Number(args.plies??400),simulations=Number(args.simulations??64),horizon=Number(args.horizon??8);
if(!Number.isInteger(games)||games<1||!Number.isInteger(limit)||limit<1||simulations<1||horizon<0)throw Error('Invalid budget');
const out=args.out??'runs/pilot';fs.mkdirSync(out,{recursive:true});
if(fs.existsSync(`${out}/manifest.json`))throw Error('Output already exists; use a fresh directory');
const model=args.model?JSON.parse(fs.readFileSync(args.model)):null;
if(model&&(!Array.isArray(model.policy)||model.policy.length!==15||!model.policy.every(Number.isFinite)))throw Error('Invalid policy weights');
if(model&&model.version!==VERSION)throw Error('Model schema mismatch');
const manifest={schema:1,version:VERSION,args,seed,games,limit,simulations,horizon,node:process.version,upstream:'da1ab8ce11659110f19422213c5811cbfab77cb6',started:new Date().toISOString(),modelSha256:args.model?createHash('sha256').update(fs.readFileSync(args.model)).digest('hex'):null,commit:(()=>{try{return execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();}catch{return null;}})()};
fs.writeFileSync(`${out}/manifest.json`,JSON.stringify(manifest,null,2));
let results=[];
for(let g=0;g<games;g++){
 // Paired layout seeds; candidate alternates colours. Separate search RNGs.
 const layoutSeed=seed+Math.floor(g/2),candidate=g%2===0?'r':'b';
 let s=initial(seeded(layoutSeed));s.positions=[positionKey(s)];const records=[];
 const rngs={r:seeded(seed+g*2+100000),b:seeded(seed+g*2+100001)};
 while(!s.result&&s.ply<limit){
  const started=performance.now();const v=publicState(s,s.turn),rng=rngs[s.turn];let m,rec=null;
  if(args.opponent&&s.turn!==candidate){
   if(args.opponent==='upstream')m=analyze(v,{seed:seed+g*1000+s.ply,timeMs:Number(args.timeMs??100),samples:8,maxDepth:8}).move;
   else if(args.opponent==='random'){const ms=legalMoves(v);m=ms[Math.floor(rng()*ms.length)];}
   else throw Error('Unknown opponent');
  }else{rec=search(v,{rng,model,simulations,horizon});m=args.opponent?rec.move:rec.moves[draw(rec.pi,rng)];}
  if(!legalMoves(v).some(x=>x.from===m?.from&&x.to===m?.to))throw Error('Illegal move');
  records.push({elapsedMs:performance.now()-started,ply:s.ply,side:s.turn,observation:v,move:m,...(rec?{features:rec.features,pi:rec.pi}:{})});s=move(s,m);
 }
 const result=s.result??{winner:null,text:'truncated',truncated:true};
 for(const rec of records){rec.value=result.truncated?null:result.winner===null?0:result.winner===rec.side?1:-1;}
 fs.writeFileSync(`${out}/game-${g}.json`,JSON.stringify({layoutSeed,candidate,result,records}));
 results.push({game:g,layoutSeed,candidate,result,plies:s.ply});console.log(JSON.stringify(results.at(-1)));
}
fs.writeFileSync(`${out}/summary.json`,JSON.stringify({results,completed:results.filter(x=>!x.result.truncated).length,truncated:results.filter(x=>x.result.truncated).length},null,2));
