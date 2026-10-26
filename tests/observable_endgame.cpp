#include "position.h"
#include "nnue.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace jq;
int main(int argc, char** argv) {
 if(argc!=2) return 2;
 init_tables();std::string error;
 if(!nnue::load(argv[1],&error))throw std::runtime_error(error);
 int full=0,boundaries=0,ordinary=0;
 std::string fen;
 while(std::getline(std::cin,fen)) {
  Position p;if(!p.set_fen(fen,&error))throw std::runtime_error(error);
  for(bool quantized : {false,true}) {
   nnue::set_quantized(quantized);
   if(nnue::quantized()!=quantized)throw std::runtime_error("requested inference mode unavailable");
   if(p.darkCount[RED]+p.darkCount[BLACK]==0) {
    Position canonical=p;std::memset(canonical.urn,0,sizeof canonical.urn);canonical.urnTotal[0]=canonical.urnTotal[1]=0;
    if(nnue::evaluate(p)!=nnue::evaluate(canonical))throw std::runtime_error("dead inventory affects canonical value");
    ++full;
   }
   Move moves[MAX_MOVES];int n=p.gen_legal(moves);
   for(int i=0;i<n;i++) {
    const auto piece=p.board[from_sq(moves[i])];
    for(int slot=0;slot<(type_of(piece)==DARK?NUM_SLOTS:1);slot++) {
     if(type_of(piece)==DARK&&!p.urn[p.stm][slot])continue;
     Move move=make_move(from_sq(moves[i]),to_sq(moves[i]),type_of(piece)==DARK?slot_type(slot):0);
     const int before=p.darkCount[RED]+p.darkCount[BLACK];
     nnue::Accumulator prior,incremental,reference;nnue::refresh(p,prior);
     nnue::Dirty dirty;nnue::make_dirty(p,move,dirty);StateInfo saved;p.do_move(move,saved);
     bool last=before>0&&p.darkCount[RED]+p.darkCount[BLACK]==0;
     if(last&&dirty.refresh<0)throw std::runtime_error("missing last-dark accumulator refresh");
     if(dirty.refresh>=0)nnue::refresh(p,incremental);else nnue::update(prior,incremental,dirty);
     nnue::refresh(p,reference);
     for(Color side:{RED,BLACK})if(std::abs(int(nnue::evaluate(incremental,side))-int(nnue::evaluate(reference,side)))>1)throw std::runtime_error("incremental/reference mismatch");
     last?++boundaries:++ordinary;p.undo_move(move,saved);
    }
   }
  }
 }
 if(!full||!boundaries||!ordinary)throw std::runtime_error("required regimes not exercised");
 std::cout<<"{\"full_public_checks\":"<<full<<",\"last_dark_transitions\":"<<boundaries<<",\"ordinary_updates\":"<<ordinary<<",\"float_and_quantized\":true}"<<std::endl;
}
