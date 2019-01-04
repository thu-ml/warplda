#pragma once

#include <memory>
#include "lda.hpp"
#include "HashTable.hpp"
#include "Xorshift.hpp"
#include "Utils.hpp"
#include "Shuffle.hpp"
#include "alias_urn.h"

template <unsigned MH>
class WarpLDA;

template <unsigned MH>
class WarpLDA : public LDA
{
public:
    WarpLDA();
    virtual void estimate(int K, float alpha, float beta, int niter, int perplexity_interval) override;
    virtual void inference(int niter, int perplexity_interval) override;
    virtual void loadModel(std::string prefix) override;
    virtual void storeModel(std::string prefix) override;
    virtual void loadZ(std::string prefix) override;
    virtual void storeZ(std::string prefix) override;
    virtual void writeInfo(std::string vocab, std::string info, uint32_t ntop) override;

private:
    struct TData
    {
        TTopic newk[MH];
        TTopic oldk;
    };
    TTopic K;
    float alpha, beta, alpha_bar, beta_bar;
    int niter, perplexity_interval;
    NumaArray<TCount> nnz_d;
    NumaArray<TCount> nnz_w;
    NumaArray<TCount> ck;
    std::unique_ptr<Shuffle<TData>> shuffle;
    XorShift generator;
    std::vector<HashTable<TTopic, TCount>> cwk_model;
    std::vector<AliasUrn> cwk_urns;
    std::vector<double> cwk_sums;
    AliasUrn global_urn;
    double global_sum;
    double total_log_likelihood;        // p(w | \alpha, \beta)
    template <bool testMode = false>
    double perplexity();
    template <bool testMode = false>
    void initialize();
    template <bool testMode = false>
    void accept_d_propose_w();
    template <bool testMode = false>
    void accept_w_propose_d();
    void reduce_ck();
    struct LocalBuffer{
        std::vector<TCount> ck_new;
        HashTable<TTopic, TCount> cxk_sparse;
        std::vector<TData> local_data;
        double log_likelihood;
        XorShift generator;
        uint32_t Rand32() { return generator.Rand32(); }
        LocalBuffer(TTopic K, TDegree maxdegree)
        : ck_new(K), cxk_sparse(logceil(K)), local_data(maxdegree)
        {
        }
        void Init();
    };
    std::vector<std::unique_ptr<LocalBuffer>> local_buffers;
};

extern template class WarpLDA<1>;
