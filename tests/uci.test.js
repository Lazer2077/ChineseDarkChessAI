import test from 'node:test';import assert from 'node:assert/strict';
import {initial,publicState} from '../vendor/mist-jieqi/engine.js';
import {seeded} from '../vendor/mist-jieqi/ai.js';
import {fen,parseMove,UciEngine} from '../src/uci.js';
test('initial FEN and board coordinates agree with Pikafish convention',()=>{
 const s=initial(seeded(1));assert.equal(fen(publicState(s,'r')),'xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1');
 assert.deepEqual(parseMove('a3a4P'),{from:54,to:45});assert.deepEqual(parseMove('i9i8'),{from:8,to:17});assert.throws(()=>parseMove('(none)'));
});
test('FEN inventory follows private capture knowledge',()=>{
 const s=initial(seeded(1)),i=s.board.findIndex(p=>p?.hidden&&p.side==='b'),p=s.board[i];s.board[i]=null;s.captured.push({...p,capturedBy:'r'});
 const a=fen(publicState(s,'r'));s.turn='b';const b=fen(publicState(s,'b'));assert.notEqual(a.split(' ')[2],b.split(' ')[2]);
});
test('UCI startup error rejects without hanging',async()=>{
 const e=new UciEngine('/nonexistent/jieqi-test-engine');try{await assert.rejects(e.start());}finally{e.close();}
});
