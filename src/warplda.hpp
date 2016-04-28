#pragma once

#include "lda.hpp"

class WarpLDA : public LDA
{
public:
    virtual void load(std::string prefix) override;
    virtual void estimate(int K, float alpha, float beta) override;
    virtual void inference(int K, float alpha, float beta) override;
    virtual void loadModel(std::string prefix) override;
    virtual void storeModel(std::string prefix) const override;
    virtual void loadZ(std::string prefix) override;
    virtual void storeZ(std::string prefix) override;
    virtual void writeInfo(std::string prefix) override;
};
