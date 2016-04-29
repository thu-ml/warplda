#include "lda.hpp"

void LDA::loadBinary(std::string fname)
{
    if (!g.Load(fname))
        throw std::runtime_error(std::string("Load Binary failed : ") + fname);
	printf("Bigraph loaded from %s, %u documents, %u unique tokens, %lu total words\n", fname.c_str(), g.NU(), g.NV(), g.NE());
}
