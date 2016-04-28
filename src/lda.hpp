#pragma once

class LDA
{
public:
    virtual void load(std::string prefix) = 0;
    virtual void estimate(int K, float alpha, float beta);
    virtual void inference(int K, float alpha, float beta);
    virtual void loadModel(std::string prefix);
    virtual void storeModel(std::string prefix) const;
    virtual void loadZ(std::string prefix);
    virtual void storeZ(std::string prefix);
    virtual void writeInfo(std::string prefix);
};
