// Shared external time management for engines exposing only go movetime.
export function timeControl(text) {
 const m=/^(\d+(?:\.\d+)?)\+(\d+(?:\.\d+)?)$/.exec(text);
 if(!m||Number(m[1])<=0)throw Error('Require positive seconds+nonnegative increment');
 return {baseMs:Number(m[1])*1000,incrementMs:Number(m[2])*1000};
}
export function moveBudget(remainingMs,incrementMs) {
 if(!(remainingMs>0)||!(incrementMs>=0))throw Error('Invalid clock');
 // Same controller for both engines; no time is added to the actual bank.
 return Math.max(1,Math.floor(Math.min(60000,Math.max(1,remainingMs-20),remainingMs/20+.8*incrementMs)));
}
export function charge(remainingMs,elapsedMs,incrementMs) {
 if(!(elapsedMs>=0)||!(remainingMs>0)||!(incrementMs>=0))throw Error('Invalid clock charge');
 const flagged=elapsedMs>=remainingMs;
 return {flagged,remainingMs:flagged?0:remainingMs-elapsedMs+incrementMs};
}
