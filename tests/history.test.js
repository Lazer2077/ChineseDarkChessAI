import test from 'node:test';import assert from 'node:assert/strict';
import {observationFromHistory} from '../src/history.js';
import {initial,legalMoves,move,publicState,positionKey} from '../vendor/mist-jieqi/engine.js';
import {seeded} from '../vendor/mist-jieqi/ai.js';
const square=i=>String.fromCharCode(97+i%9)+String(9-Math.floor(i/9));
test('observer history matches independent true simulator across reveals and private captures',()=>{
 const rng=seeded(71);let captures=0,checked=0;
 for(let g=0;g<8;g++){
  let s=initial(rng);s.positions=[positionKey(s)];const history={r:[],b:[]};
  for(let ply=0;ply<100&&!s.result;ply++){
   const view=observationFromHistory(history[s.turn]);assert.deepEqual(view,publicState(s,s.turn));checked++;
   const moves=legalMoves(s),m=moves[Math.floor(rng()*moves.length)],p=s.board[m.from],q=s.board[m.to];
   const base=square(m.from)+square(m.to),letter=(type,side)=>side==='r'?type:type.toLowerCase();
   const revealed=p.hidden?letter(p.type,p.side):'';
   history[p.side].push(base+revealed+(q?.hidden?letter(q.type,q.side):''));
   history[p.side==='r'?'b':'r'].push(base+revealed+'x');
   if(q?.hidden)captures++;
   s=move(s,m);
  }
 }
 assert.ok(checked>=100);assert.ok(captures>0);
});
test('rejects private victim identity on opponent move and missing public reveal',()=>{
 assert.throws(()=>observationFromHistory(['a3a4']));
 // Red cannon captures black dark horse; black is not entitled to victim identity.
 assert.throws(()=>observationFromHistory(['b2b9Pr'],'b'),/Opponent private/);
});
