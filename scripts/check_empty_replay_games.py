"""Certify games ending before their first searched/recorded position.

Replay the pinned actor's exact seeded random prefix using its native rules.
Escape immediately before the first search, so this is not a new self-play run.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--build',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(a.out);build=Path(a.build)
if out.exists() or build.exists():raise SystemExit('Use fresh evidence/build paths')
source=root/'build/data-actor/source/engine/src/selfplay.cpp';text=source.read_text()
needle='            const SearchResult r = S[side]->go(v, lim, quiet);'
if text.count(needle)!=1:raise SystemExit('Actor search context changed')
patched=text.replace(needle,'            return 2; // This game reaches a recorded search position.\n'+needle)
patched+='''
#include <iostream>
int main(int argc,char**argv) {
 if(argc!=4)return 2;
 jq::init_tables();jq::Config cfg;cfg.seed=std::stoull(argv[1]);cfg.games=std::stoi(argv[2]);cfg.randomPlies=std::stoi(argv[3]);
 jq::Searcher red(1,1),black(1,1);jq::Searcher* searchers[2]={&red,&black};
 int empty=0;
 for(int g=0;g<cfg.games;++g) {
  std::vector<jq::TrainRecord> records;bool valid=false;
  int result=jq::play_one(cfg,jq::mix(cfg.seed*0x100000001B3ull+uint64_t(g)),searchers,records,valid);
  if(!valid||!records.empty())return 3;
  if(result!=2){std::cout<<g<<" "<<result<<"\\n";++empty;}
 }
 std::cerr<<"empty "<<empty<<" / "<<cfg.games<<"\\n";
}
'''
build.mkdir(parents=True);cpp=build/'prefix_probe.cpp';cpp.write_text(patched)
library=root/'build/data-actor/bin/libjqe_core.a';binary=build/'prefix_probe'
subprocess.run(['g++','-std=c++17','-O2','-pthread','-I',str(source.parent),str(cpp),str(library),'-o',str(binary)],check=True)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
report={'schema':1,'scope':'Exact native actor seeded random-prefix replay, escaping immediately before first search. Certifies zero-row games, not new generated data, new labels or playing strength.','actor_source_sha256':sha(source),'native_core_sha256':sha(library),'probe_source_sha256':sha(cpp),'probe_binary_sha256':sha(binary),'controller_sha256':sha(root/'scripts/continual_learning.py'),'random_opening_max_plies':8,'datasets':{}}
for filename in a.data:
 path=Path(filename);meta=json.loads(path.with_suffix('.complete.json').read_text());state=json.loads((path.parent/'state.json').read_text())
 if sha(path)!=meta['sha256'] or meta['rejected_games'] or meta['recipe']['actor']!=sha(root/'build/data-actor/bin/jqe') or state['pins']['controller']!=report['controller_sha256']:
  raise SystemExit('Generation provenance differs')
 run=subprocess.run([str(binary.resolve()),str(meta['recipe']['seed']),str(meta['recipe']['games']),'8'],text=True,capture_output=True,check=True)
 empty=[{'game_index':int(line.split()[0]),'red_result':int(line.split()[1])} for line in run.stdout.splitlines()]
 report['datasets'][meta['sha256']]={'generated_games':meta['recipe']['games'],'seed':meta['recipe']['seed'],'zero_search_record_games':empty,'nonempty_games':meta['recipe']['games']-len(empty)}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
