import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {remaining} from '../vendor/mist-jieqi/ai.js';
export function fen(view){
 if(view.viewer!==view.turn)throw Error('UCI FEN must use actor observation');
 const rows=[];
 for(let y=0;y<10;y++){
  let row='',empty=0;
  for(let x=0;x<9;x++){
   const p=view.board[y*9+x];if(!p){empty++;continue;}
   if(empty){row+=empty;empty=0;}
   const t=p.hidden?'X':p.type;row+=p.side==='r'?t:t.toLowerCase();
  }
  if(empty)row+=empty;rows.push(row);
 }
 let pools='';for(const side of ['r','b']){const pool=remaining(view,side);for(const t of ['R','A','C','P','N','B']){const n=pool.filter(x=>x===t).length;if(n)pools+=(side==='r'?t:t.toLowerCase())+n;}}
 return `${rows.join('/')} ${view.turn==='r'?'w':'b'} ${pools||'-'} ${view.quiet} ${Math.floor(view.ply/2)+1}`;
}
export function parseMove(text){
 const m=/^([a-i])([0-9])([a-i])([0-9])/.exec(text);if(!m)throw Error(`Invalid UCI move: ${text}`);
 return {from:(9-Number(m[2]))*9+m[1].charCodeAt(0)-97,to:(9-Number(m[4]))*9+m[3].charCodeAt(0)-97};
}
export class UciEngine{
 constructor(binary,{args=[],options={Hash:64,Threads:1},timeoutMs=10000}={}){
  this.process=spawn(binary,args,{stdio:['pipe','pipe','pipe']});this.pending=null;this.log=[];this.error=null;this.options=options;this.timeoutMs=timeoutMs;
  this.lines=createInterface({input:this.process.stdout});
  this.lines.on('line',line=>{this.log.push(line);if(this.log.length>200)this.log.shift();if(this.pending?.matches(line)){const p=this.pending;this.pending=null;clearTimeout(p.timer);p.resolve(line);}});
  this.process.stderr.on('data',()=>{});
  const fail=error=>{this.error=error;if(this.pending){clearTimeout(this.pending.timer);this.pending.reject(error);this.pending=null;}};
  this.process.on('error',fail);this.process.on('exit',(code,signal)=>fail(Error(`UCI exited ${code}/${signal}`)));
 }
 wait(command,matches,timeout=this.timeoutMs){
  if(this.error)return Promise.reject(this.error);if(this.pending)return Promise.reject(Error('Concurrent UCI commands'));
  return new Promise((resolve,reject)=>{
   const timer=setTimeout(()=>{this.pending=null;this.process.kill('SIGKILL');reject(Error(`UCI timeout: ${command}`));},timeout);
   this.pending={matches,resolve,reject,timer};this.process.stdin.write(command+'\n');
  });
 }
 async start(){await this.wait('uci',s=>s==='uciok');for(const [name,value]of Object.entries(this.options))this.process.stdin.write(`setoption name ${name} value ${value}\n`);await this.wait('isready',s=>s==='readyok');}
 async choose(view,timeMs=100){
  // Reconstruct from actor-visible FEN. History-dependent engine adjudication
  // is unavailable in this adapter; the external referee remains authoritative.
  this.process.stdin.write(`position fen ${fen(view)}\n`);
  const line=await this.wait(`go movetime ${timeMs}`,s=>s.startsWith('bestmove '),Math.max(this.timeoutMs,timeMs*3));
  return parseMove(line.split(/\s+/)[1]);
 }
 close(){this.process.stdin.end('quit\n');this.process.kill();this.lines.close();}
}
