// Original Mist rules, complete private histories, and actual game clocks.
// Both engines use the same external move-time allocation policy.
import fs from 'node:fs';import path from 'node:path';import {createHash} from 'node:crypto';import {execFileSync} from 'node:child_process';
import {initial,legalMoves,move,positionKey} from '../vendor/mist-jieqi/engine.js';
import {seeded} from '../vendor/mist-jieqi/ai.js';
import {UciEngine,parseMove} from '../src/uci.js';
import {timeControl,moveBudget,charge} from '../src/match_clock.js';
import {pairedEvidence} from '../src/paired_evidence.js';
const a=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const games=Number(a.games??2),seed=Number(a.seed??7700001),out=a.out??'runs/mist-clock-smoke',tc=a.tc??'15+0.1',clock=timeControl(tc),stage=a.stage??'development';
if(!Number.isInteger(games)||games<2||games%2||!Number.isSafeInteger(seed)||!['development','confirmation'].includes(stage))throw Error('Invalid games, seed or stage');
if(stage==='confirmation'&&games<200)throw Error('Confirmation requires at least 100 layout pairs');
const candidate=path.resolve('build/mist-profile/bin/jqe'),baseline=path.resolve('adapters/mist_native.sh'),net=path.resolve(a.net??'models/experimental/warm-20261026.jqn');
const hash=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const revision=execFileSync('git',['-C','external/mist-jieqi','rev-parse','HEAD'],{encoding:'utf8'}).trim();
if(revision!=='da1ab8ce11659110f19422213c5811cbfab77cb6')throw Error('Wrong original opponent revision');
execFileSync('git',['-C','external/mist-jieqi','diff','--exit-code','HEAD'],{stdio:'pipe'});
if(hash('vendor/mist-jieqi/engine.js')!==hash('external/mist-jieqi/engine.js'))throw Error('Rule source differs from original opponent');
const profile=JSON.parse(fs.readFileSync('build/mist-profile/manifest.json'));
if(hash(candidate)!==profile.candidate_binary_sha256)throw Error('Candidate profile binary mismatch');
const sourcePaths=['scripts/mist_clock_match.mjs','src/match_clock.js','src/paired_evidence.js','src/history.js','src/uci.js','adapters/mist_native.mjs','adapters/mist_native.sh','vendor/mist-jieqi/engine.js','vendor/mist-jieqi/ai.js','external/mist-jieqi/native/bridge.mjs'];
const manifest={schema:1,profile:'mist-da1ab8c',games,seed,tc,threads:4,stage,status:'running',opponentRevision:revision,candidateNetSha256:hash(net),candidateBinarySha256:hash(candidate),opponentBinarySha256:hash('external/mist-jieqi/.cache/native/mist-core'),candidateProfile:profile,sourceHashes:Object.fromEntries(sourcePaths.map(p=>[p,hash(p)])),timeManagement:'Both engines receive go movetime floor(min(60000, max(1, bank-20), bank/20+0.8*increment)), at least 1ms. Full wall time from position submission to bestmove is charged; zero grace, increment only after an on-time legal move. Startup/public-position warmup excluded.',scope:'A frozen candidate under one shared external time manager; no global promotion or independent-training-seed claim.'};
if(fs.existsSync(out))throw Error('Use a fresh run directory');fs.mkdirSync(out,{recursive:true});
const saveManifest=()=>fs.writeFileSync(`${out}/manifest.json`,JSON.stringify(manifest,null,2)+'\n');saveManifest();
const fixedFiles={...manifest.sourceHashes,[net]:manifest.candidateNetSha256,[candidate]:manifest.candidateBinarySha256,'external/mist-jieqi/.cache/native/mist-core':manifest.opponentBinarySha256};
const verifyFrozen=()=>{for(const [file,digest]of Object.entries(fixedFiles))if(hash(file)!==digest){manifest.status='invalidated';manifest.error='Frozen source or artifact changed';saveManifest();throw Error(manifest.error);}};
const square=i=>String.fromCharCode(97+i%9)+String(9-Math.floor(i/9));
const init=async(binary,options)=>{
 const e=new UciEngine(binary,{options,timeoutMs:30000});
 try{await e.start();await e.wait('position startpos\ngo movetime 1',s=>s.startsWith('bestmove '),30000);e.process.stdin.write('ucinewgame\n');await e.wait('isready',s=>s==='readyok');return e;}
 catch(error){e.close();throw error;}
};
const results=[];
for(let g=0;g<games;g++){
 verifyFrozen();
 const candSide=g%2?'b':'r',layoutSeed=seed+Math.floor(g/2),engines={},banks={r:clock.baseMs,b:clock.baseMs};
 let s=initial(seeded(layoutSeed));s.positions=[positionKey(s)];
 const histories={r:[],b:[]},records=[];let failure=null,technical=null,attempt=null;
 try{
  engines[candSide]=await init(candidate,{Threads:4,Hash:64,EvalFile:net});
  engines[candSide==='r'?'b':'r']=await init(baseline,{Threads:4});
  while(!s.result){
   const side=s.turn,e=engines[side],before=banks[side],budget=moveBudget(before,clock.incrementMs),started=performance.now();
   attempt={side,bankBeforeMs:before,budgetMs:budget};
   e.process.stdin.write('position startpos'+(histories[side].length?' moves '+histories[side].join(' '):'')+'\n');
   let answer;
   try{answer=await e.wait(`go movetime ${budget}`,line=>line.startsWith('bestmove '),Math.max(1,Math.ceil(before)));}
   catch(error){
    attempt.elapsedMs=performance.now()-started;attempt.error=String(error);
    technical={side,reason:attempt.elapsedMs>=before?'TIME_FORFEIT':'ENGINE_FAILURE'};break;
   }
   const elapsed=performance.now()-started;attempt.elapsedMs=elapsed;attempt.answer=answer;
   const charged=charge(before,elapsed,clock.incrementMs);
   if(charged.flagged){technical={side,reason:'TIME_FORFEIT'};break;}
   let m;
   try{m=parseMove(answer.split(/\s+/)[1]);}catch{technical={side,reason:'ILLEGAL_MOVE'};break;}
   if(!legalMoves(s).some(x=>x.from===m.from&&x.to===m.to)){technical={side,reason:'ILLEGAL_MOVE'};break;}
   banks[side]=charged.remainingMs;
   const p=s.board[m.from],q=s.board[m.to],letter=(type,owner)=>owner==='r'?type:type.toLowerCase();
   const base=square(m.from)+square(m.to),reveal=p.hidden?letter(p.type,p.side):'',capture=q?.hidden?letter(q.type,q.side):'';
   histories[side].push(base+reveal+capture);histories[side==='r'?'b':'r'].push(base+reveal+'x');
   records.push({m:base,r:p.hidden?p.type:null,c:q?.hidden?q.type:null,side,elapsedMs:elapsed,budgetMs:budget,bankBeforeMs:before,bankAfterMs:banks[side]});
   s=move(s,m);attempt=null;
  }
 }catch(error){failure=String(error);}
 finally{for(const e of Object.values(engines))e.close();}
 const result=failure?null:technical?(technical.side==='r'?0:1):s.result.winner===null?.5:s.result.winner==='r'?1:0;
 const row={game:g,pair:Math.floor(g/2),seed:String(layoutSeed),red:candSide==='r'?'candidate':'mist-native',black:candSide==='b'?'candidate':'mist-native',result,reason:failure??technical?.reason??s.result.text,technical:Boolean(technical),failedAttempt:attempt,plies:records.length,moves:records};
 fs.appendFileSync(`${out}/${failure?'void':'games'}.jsonl`,JSON.stringify(row)+'\n');results.push(row);
 manifest.completedGames=g+1;saveManifest();
 console.log(JSON.stringify({game:g,pair:row.pair,red:row.red,result:row.result,reason:row.reason,plies:row.plies}));
}
verifyFrozen();
const summary=pairedEvidence(results,games,{confirmation:stage==='confirmation'});
fs.writeFileSync(`${out}/summary.json`,JSON.stringify(summary,null,2)+'\n');
manifest.status='completed';saveManifest();console.log(JSON.stringify(summary));
