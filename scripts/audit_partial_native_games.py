"""Retain a63/64screen with one zero-ply void; never certify a complete match."""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).with_name('audit_native_games.py')
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != 'bbab817dcd47d51b50e741bcff8b5439a4996254ebe78f67682ba2b713d9cff7':
    raise ValueError('Pinned complete-match audit changed')
text = SOURCE.read_text()


def replace(old, new):
    global text
    if text.count(old) != 1:
        raise ValueError('Pinned partial-audit patch site changed')
    text = text.replace(old, new)


replace("if meta['status']!='completed' or len(games)!=meta['args']['games']:raise SystemExit('Incomplete match')",
        """expected_games=meta['args']['games']
void_raw=(run/'void.jsonl').read_bytes()
void_rows=[json.loads(line) for line in void_raw.splitlines()]
if (meta['status']!='completed' or expected_games!=64 or len(games)!=63 or len(void_rows)!=1
    or any(v['plies']!=0 or v['moves'] or not v['reason'].startswith('harness error:') for v in void_rows)):
 raise SystemExit('Require exactly63complete games plus one retained zero-ply void')""")
replace("seeds=[str(rng.getrandbits(63)) for _ in range(len(games)//2)]",
        """seeds=[str(rng.getrandbits(63)) for _ in range(expected_games//2)]
void_slots=set()
for v in void_rows:
 if (not 0<=v['pair']<len(seeds) or v['seed']!=seeds[v['pair']]
     or {v['red'],v['black']}!={'candidate','baseline'}):raise ValueError('Wrong void layout or colors')
 void_slots.add((v['pair'],v['red']))""")
replace("for i in range(len(seeds)):\n if (i,'candidate') not in seen or (i,'baseline') not in seen:raise ValueError('Missing color swap')",
        """expected_slots={(i,color) for i in range(len(seeds)) for color in ['candidate','baseline']}
if expected_slots-seen!=void_slots or seen&void_slots:raise ValueError('Unaccounted or repeated attempt slot')
complete_pairs=sum((i,'candidate') in seen and (i,'baseline') in seen for i in range(len(seeds)))""")
replace("'complete_pairs':len(seeds)",
        "'complete_pairs':complete_pairs,'expected_games':expected_games,'void_attempts':len(void_rows),"
        "'partial_match':True,'all_planned_attempt_slots_accounted':True,"
        "'missing_color_slots':sorted(void_slots),'void_raw_sha256':hashlib.sha256(void_raw).hexdigest(),"
        "'source_audit_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()")
replace("'scope':'Post-match replay using pinned native referee;",
        "'scope':'PARTIAL MATCH:63completed games plus one zero-ply void,31complete pairs. No full64-game or technical-clean certification. Original strict per-move replay remains intact. Post-match replay using pinned native referee;")
exec(compile(text, '<cdca-partial-native-audit>', 'exec'), dict(globals(), __name__='__main__'))
