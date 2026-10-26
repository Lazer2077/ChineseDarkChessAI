// Rebuild one actor's observation from only that actor's UCI move history.
import {initial,publicState,legalMoves,move,positionKey} from '../vendor/mist-jieqi/engine.js';
import {knowledge,seeded} from '../vendor/mist-jieqi/ai.js';
import {parseMove} from './uci.js';
export function observationFromHistory(tokens,viewer=tokens.length%2?'b':'r') {
 if(!['r','b'].includes(viewer))throw Error('Invalid observer');
 let s=publicState(initial(seeded(0)),viewer);s.positions=[positionKey(s)];
 for(const token of tokens){
  const m=parseMove(token),p=s.board[m.from],q=s.board[m.to];
  if(!p||p.side!==s.turn||!legalMoves(s).some(x=>x.from===m.from&&x.to===m.to))throw Error(`Illegal observed move: ${token}`);
  let rest=token.slice(4).replace(/[xX-]/g,'');
  const decode=(ch,side)=>{
   if(!ch||!/[RNBACP]/.test(ch.toUpperCase())||(side==='r')!==(ch===ch.toUpperCase()))throw Error('Invalid reveal/capture identity');return ch.toUpperCase();
  };
  if(p.hidden){p.type=decode(rest[0],p.side);rest=rest.slice(1);}
  if(q?.hidden&&rest){
   if(p.side!==viewer)throw Error('Opponent private capture identity in observer history');
   q.type=decode(rest[0],q.side);rest=rest.slice(1);
  }
  if(rest)throw Error('Unexpected move suffix');
  s=move(s,m,false);s.positions.push(positionKey(s));
  const visible=publicState(s,viewer);
  for(const side of ['r','b'])if(!knowledge(visible,side).consistent)throw Error('Impossible observed inventory');
 }
 if(s.turn!==viewer)throw Error('Observer is not the acting side');
 return publicState(s,viewer);
}
