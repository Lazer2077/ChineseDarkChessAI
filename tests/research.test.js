import test from 'node:test';import assert from 'node:assert/strict';
import {initial,publicState,legalMoves,move,positionKey} from '../vendor/mist-jieqi/engine.js';
import {seeded,knowledge} from '../vendor/mist-jieqi/ai.js';
import {features,DIM,sampleWorld,search} from '../src/research.js';
test('hidden identities do not affect observations or search',()=>{
 const s=initial(seeded(1)),t=structuredClone(s),slots=t.board.filter(p=>p?.hidden&&p.side==='b');[slots[0].type,slots[1].type]=[slots[1].type,slots[0].type];
 const a=publicState(s,'r'),b=publicState(t,'r');assert.deepEqual(a,b);
 assert.deepEqual(search(a,{rng:seeded(4),simulations:2,horizon:1}),search(b,{rng:seeded(4),simulations:2,horizon:1}));
});
test('private captures visible only to captor and sampled consistently',()=>{
 const s=initial(seeded(5)),i=s.board.findIndex(p=>p?.hidden&&p.side==='b'),p=s.board[i];s.board[i]=null;s.captured.push({...p,capturedBy:'r'});
 assert.equal(publicState(s,'b').captured[0].type,undefined);assert.equal(publicState(s,'r').captured[0].type,p.type);
 for(const side of ['r','b']){const v=publicState(s,side);assert.ok(knowledge(v,'b').consistent);const w=sampleWorld(v,seeded(7));assert.ok(w.captured[0].type);assert.deepEqual(publicState(w,side),v);}
});
test('legal seeded selfplay, feature schema, deterministic repeat',()=>{
 let s=initial(seeded(3));s.positions=[positionKey(s)];
 for(let i=0;i<20&&!s.result;i++){const v=publicState(s,s.turn),ms=legalMoves(v);assert.equal(features(v,ms[0]).length,DIM);const r=search(v,{rng:seeded(i),simulations:1,horizon:0});assert.ok(ms.some(m=>m.from===r.move.from&&m.to===r.move.to));assert.ok(Math.abs(r.pi.reduce((a,b)=>a+b)-1)<1e-9);s=move(s,r.move);}
});
