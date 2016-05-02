#pragma once

#include <string>
#include "Bigraph.hpp"

class LDA
{
protected:
    Bigraph g;
public:
    LDA() {}
    virtual void loadBinary(std::string prefix);
    virtual void estimate(int K, float alpha, float beta, int niter) = 0;
    virtual void inference(int niter) = 0;
    virtual void loadModel(std::string prefix) = 0;
    virtual void storeModel(std::string prefix) = 0;
    virtual void loadZ(std::string prefix) = 0;
    virtual void storeZ(std::string prefix) = 0;
    virtual void writeInfo(std::string vocab, std::string info) = 0;
};

template <unsigned MH>
class WarpLDA;
