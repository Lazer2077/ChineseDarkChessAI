// UCI transport around the UNMODIFIED GPL mist-jieqi C++ search + original bridge.
import path from 'node:path';import {pathToFileURL} from 'node:url';import {createInterface} from 'node:readline';
import {observationFromHistory} from '../src/history.js';
import {legalMoves} from '../vendor/mist-jieqi/engine.js';
const dir=path.resolve(process.env.MIST_JIEQI_DIR??new URL('../external/mist-jieqi',import.meta.url).pathname);
const {Native,fixture,observationSeed}=await import(pathToFileURL(path.join(dir,'native/bridge.mjs')).href);
let native=new Native();await native.ready;let tokens=[];
const square=i=>String.fromCharCode(97+i%9)+String(9-Math.floor(i/9));
const send=s=>process.stdout.write(s+'\n');
try {
 for await(const line of createInterface({input:process.stdin})){
  const cmd=line.trim();
  if(cmd==='uci'){send('id name mist-jieqi native (pinned external source)');send('option name Threads type spin default 4 min 4 max 4');send('uciok');}
  else if(cmd==='isready')send('readyok');
  else if(cmd==='ucinewgame')tokens=[];
  else if(cmd.startsWith('position ')){
   const fields=cmd.split(/\s+/),at=fields.indexOf('moves');
   // This adapter supports the standard start FEN plus complete own history.
   // Refuse arbitrary snapshots rather than fabricating their private inventory.
   if(fields[1]!=='startpos'&&fields.slice(2,at<0?undefined:at).join(' ')!=='xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1')throw Error('Adapter requires standard start and full history');
   tokens=at<0?[]:fields.slice(at+1);
  } else if(cmd.startsWith('go ')){
   const parts=cmd.split(/\s+/),at=parts.indexOf('movetime');if(at<0)throw Error('This adapter requires go movetime');
   const budget=Number(parts[at+1]);if(!(budget>0&&budget<=60000))throw Error('Invalid move budget');
   const view=observationFromHistory(tokens);const result=await native.request(fixture(view,{seed:observationSeed(view),samples:8}),{mode:'SEARCH',depth:8,budget});
   if(!legalMoves(view).some(m=>m.from===result.move?.from&&m.to===result.move?.to))throw Error('Native returned an illegal move');
   send(`info depth ${result.depth} nodes ${result.nodes} score cp ${Math.round(result.score)}`);send('bestmove '+square(result.move.from)+square(result.move.to));
  } else if(cmd==='quit')break;
  else if(cmd.startsWith('setoption name Threads value ')&&cmd.split(/\s+/).at(-1)!=='4')throw Error('Native baseline requires four-thread allowance');
 }
}finally{await native.close();}
