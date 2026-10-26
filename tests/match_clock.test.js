import test from 'node:test';import assert from 'node:assert/strict';
import {timeControl,moveBudget,charge} from '../src/match_clock.js';
import {pairedEvidence} from '../src/paired_evidence.js';
test('actual elapsed time is charged including overhead; increment cannot rescue a flag',()=>{
 assert.deepEqual(timeControl('15+0.1'),{baseMs:15000,incrementMs:100});
 assert.deepEqual(charge(1000,450,100),{flagged:false,remainingMs:650});
 assert.deepEqual(charge(1000,1000,100),{flagged:true,remainingMs:0});
 assert.deepEqual(charge(1000,1001,100),{flagged:true,remainingMs:0});
 assert.equal(moveBudget(15000,100),830);
 assert.equal(moveBudget(10,100),1);
 assert.equal(moveBudget(1e9,0),60000);
 for(const bad of ['0+1','-1+0','1+-2','NaN+0','15','15+0 trailing'])assert.throws(()=>timeControl(bad));
});
const games=n=>Array.from({length:n},(_,i)=>({pair:Math.floor(i/2),seed:String(Math.floor(i/2)),red:i%2?'mist-native':'candidate',result:i%2?0:1,technical:false}));
test('paired gate rejects tiny perfect samples, missing pairs and technical wins',()=>{
 assert.equal(pairedEvidence(games(16),16,{confirmation:true}).single_match_superiority_gate,false);
 const full=games(200),good=pairedEvidence(full,200,{confirmation:true});
 assert.equal(good.single_match_superiority_gate,true);assert.equal(good.promotion_supported,false);
 full[0].technical=true;full[0].reason='TIME_FORFEIT';
 assert.equal(pairedEvidence(full,200,{confirmation:true}).single_match_superiority_gate,false);
 assert.equal(pairedEvidence(games(199),200,{confirmation:true}).valid_technical_run,false);
 const wrong=games(200);wrong[1].seed='wrong-layout';
 assert.equal(pairedEvidence(wrong,200,{confirmation:true}).valid_technical_run,false);
 assert.equal(pairedEvidence(games(200),200).single_match_superiority_gate,false);
});
