// Match under the pinned original reference's exact game adjudication profile.
// Truth is held only here; engines receive observer-specific histories.
import fs from 'node:fs';import path from 'node:path';import {createHash} from 'node:crypto';
import {initial,legalMoves,move,positionKey} from '../vendor/mist-jieqi/engine.js';
import {seeded} from '../vendor/mist-jieqi/ai.js';
import {UciEngine} from '../src/uci.js';
const a=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const games=Number(a.games??2),budget=Number(a.movetime??1000),seed=Number(a.seed??7400001),out=a.out??'runs/mist-profile-smoke';
if(games<2||games%2||!(budget>0))throw Error('Require an even game count and positive budget');
if(fs.existsSync(out))throw Error('Use a fresh run directory');fs.mkdirSync(out,{recursive:true});
const candidate=path.resolve(a.candidate??'build/mist-profile/bin/jqe'),baseline=path.resolve('adapters/mist_native.sh'),net=path.resolve(a.net??'runs/native-warm-pilot/best.jqn');
const hash=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const manifest={schema:1,profile:'mist-da1ab8c',games,budget,seed,threads:4,candidateNetSha256:hash(net),candidateBinarySha256:hash(candidate),opponentBinarySha256:hash('external/mist-jieqi/.cache/native/mist-core'),adapterSha256:hash('adapters/mist_native.mjs'),candidateProfile:JSON.parse(fs.readFileSync('build/mist-profile/manifest.json')),stage:'development',scope:'Not an all-engine superiority claim. Candidate uses a trained derivative of the attributed jqe net.'};
fs.writeFileSync(`${out}/manifest.json`,JSON.stringify(manifest,null,2));
const square=i=>String.fromCharCode(97+i%9)+String(9-Math.floor(i/9));
const init=async(binary,options)=>{const e=new UciEngine(binary,{options,timeoutMs:30000});try{await e.start();await e.wait('position startpos\ngo movetime 1',s=>s.startsWith('bestmove '),30000);e.process.stdin.write('ucinewgame\n');await e.wait('isready',s=>s==='readyok');return e;}catch(error){e.close();throw error;}};
const results=[];
for(let g=0;g<games;g++){
 const candSide=g%2?'b':'r',layoutSeed=seed+Math.floor(g/2),engines={};let s=initial(seeded(layoutSeed));s.positions=[positionKey(s)];const histories={r:[],b:[]},records=[];let failure=null;
 try{
  engines[candSide]=await init(candidate,{Threads:4,Hash:64,EvalFile:net});engines[candSide==='r'?'b':'r']=await init(baseline,{Threads:4});
  while(!s.result){
   const side=s.turn,e=engines[side],started=performance.now();
   e.process.stdin.write('position startpos'+(histories[side].length?' moves '+histories[side].join(' '):'')+'\n');
   const answer=await e.wait(`go movetime ${budget}`,s=>s.startsWith('bestmove '),budget*3+10000),elapsed=performance.now()-started;
   const text=answer.split(/\s+/)[1],m=(await import('../src/uci.js')).parseMove(text);
   if(!legalMoves(s).some(x=>x.from===m.from&&x.to===m.to))throw Error('Illegal move from '+side);
   if(elapsed>budget+1000)throw Error(`Timing overrun by ${side}: ${elapsed}`);
   const p=s.board[m.from],q=s.board[m.to],letter=(type,side)=>side==='r'?type:type.toLowerCase(),base=square(m.from)+square(m.to),reveal=p.hidden?letter(p.type,p.side):'',capture=q?.hidden?letter(q.type,q.side):'';
   histories[side].push(base+reveal+capture);histories[side==='r'?'b':'r'].push(base+reveal+'x');
   records.push({m:base,r:p.hidden?p.type:null,c:q?.hidden?q.type:null,side,elapsedMs:elapsed,info:e.log.filter(line=>line.startsWith('info depth ')).at(-1)??null});s=move(s,m);
  }
 }catch(error){failure=String(error);}
 finally{for(const e of Object.values(engines))e.close();}
 const row={pair:Math.floor(g/2),seed:String(layoutSeed),red:candSide==='r'?'candidate':'mist-native',black:candSide==='b'?'candidate':'mist-native',result:failure?null:s.result.winner===null?.5:s.result.winner==='r'?1:0,reason:failure??s.result.text,plies:records.length,moves:records};
 fs.appendFileSync(`${out}/${failure?'void':'games'}.jsonl`,JSON.stringify(row)+'\n');results.push(row);console.log(JSON.stringify({game:g,pair:row.pair,red:row.red,result:row.result,reason:row.reason,plies:row.plies}));
}
const completed=results.filter(r=>r.result!==null),scores=completed.map(r=>r.red==='candidate'?r.result:1-r.result);
const summary={games:completed.length,expected_games:games,wins:scores.filter(x=>x===1).length,draws:scores.filter(x=>x===.5).length,losses:scores.filter(x=>x===0).length,voids:games-completed.length,score:scores.length?scores.reduce((x,y)=>x+y,0)/scores.length:null,promotion_supported:false};
fs.writeFileSync(`${out}/summary.json`,JSON.stringify(summary,null,2));console.log(JSON.stringify(summary));
