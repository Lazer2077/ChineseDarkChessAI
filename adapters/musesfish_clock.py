"""Game-clock front end for the unchanged pinned jqe Musesfish timed shim.

Only translates UCI clock commands. It does not change search, observer history,
private capture suffixes, root masking, or the original 30ms preparation reserve.
The underlying native-root-mask/timed-iteration profile is disclosed separately.
"""
import math
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
START_FEN='xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1'


def move_budget(command, ply):
    parts=command.split()
    if not parts or parts[0]!='go':raise ValueError('Require a go command')
    def number(key,default=None):
        if key not in parts:return default
        try:value=float(parts[parts.index(key)+1])
        except (ValueError,IndexError):raise ValueError('Invalid clock field') from None
        if not math.isfinite(value) or value<0:raise ValueError('Invalid clock value')
        return value
    explicit=number('movetime')
    if explicit is not None:return max(1,min(60000,int(explicit)))
    bank=number('wtime' if ply%2==0 else 'btime')
    inc=number('winc' if ply%2==0 else 'binc',0)
    if bank is None:raise ValueError('This profile requires a finite game clock or movetime')
    return max(1,min(60000,int(min(bank/25+.75*inc,bank/4))))


def position_ply(command):
    parts=command.split()
    if len(parts)<2 or parts[0]!='position':raise ValueError('Require position')
    end=parts.index('moves') if 'moves' in parts else len(parts)
    if parts[1]=='fen':
        if ' '.join(parts[2:end])!=START_FEN:raise ValueError('Only standard public start is supported')
    elif parts[1]!='startpos' or end!=2:raise ValueError('Unsupported position')
    return len(parts)-end-1 if end<len(parts) else 0


def main():
    env=dict(os.environ,MUSESFISH_DIR=str(ROOT/'external/musesfish'))
    child=subprocess.Popen([sys.executable,str(ROOT/'external/devinokeefe-jieqi/adapters/musesfish_uci/shim.py')],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,env=env)
    def send(text):child.stdin.write(text+'\n');child.stdin.flush()
    def until(prefix):
        for line in child.stdout:
            print(line.rstrip('\n'),flush=True)
            if line.startswith(prefix):return
        raise RuntimeError('Pinned Musesfish shim exited before '+prefix)
    ply=0
    try:
        for line in sys.stdin:
            command=line.strip()
            if not command:continue
            if command=='quit':break
            if command.startswith('position '):ply=position_ply(command)
            if command=='ucinewgame':ply=0
            if command.split()[0]=='go':command='go movetime '+str(move_budget(command,ply))
            send(command)
            if command=='uci':until('uciok')
            elif command=='isready':until('readyok')
            elif command.startswith('go '):until('bestmove')
    finally:
        if child.poll() is None:
            try:send('quit');child.wait(timeout=2)
            except (BrokenPipeError,subprocess.TimeoutExpired):child.kill();child.wait()


if __name__=='__main__':main()
