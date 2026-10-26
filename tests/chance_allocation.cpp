#include "../native/chance_allocation.h"
#include <cassert>
#include <cmath>
int main() {
    int counts[] = {9,1};double values[] = {.5,.9};
    auto a=cdca::allocate(counts,values,2,.05);
    assert(a.active && a.order[0]==1 && std::abs(a.mean-.54)<1e-10);
    int reversed[]={1,9};double rv[]={.9,.5};
    auto b=cdca::allocate(reversed,rv,2,.05);assert(b.order[0]==0);assert(std::abs(a.deviation-b.deviation)<1e-12);
    double same[]={.2,.2};assert(!cdca::allocate(counts,same,2,0).active);
    assert(!cdca::allocate(counts,values,2,.5).active);
    int single[]={1};double sv[]={.9};assert(!cdca::allocate(single,sv,1,0).active);
    bool invalid=false;try { cdca::allocate(single,sv,0,0); }catch(const std::invalid_argument&) {invalid=true;}assert(invalid);
}
