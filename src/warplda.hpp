#pragma once

#include <memory>
#include "lda.hpp"
#include "HashTable.hpp"
#include "Xorshift.hpp"
#include "Utils.hpp"
#include "Shuffle.hpp"

template <unsigned MH>
class WarpLDA;

template <unsigned MH>
class WarpLDA : public LDA
{
public:
    WarpLDA();
    virtual void estimate(int K, float alpha, float beta, int niter) override;
    virtual void inference(int K, float alpha, float beta, int niter) override;
    virtual void loadModel(std::string prefix) override;
    virtual void storeModel(std::string prefix) const override;
    virtual void loadZ(std::string prefix) override;
    virtual void storeZ(std::string prefix) override;
    virtual void writeInfo(std::string vocab, std::string info) override;

private:
    struct TData
    {
        TTopic newk[MH];
        TTopic oldk;
    };
    TTopic K;
    float alpha, beta, alpha_bar, beta_bar;
    int niter;
    NumaArray<TCount> nnz_d;
    NumaArray<TCount> nnz_w;
    std::vector<TCount> ck;
    std::vector<TCount> ck_new;
    std::unique_ptr<Shuffle<TData>> shuffle;
    XorShift generator;
    double total_jll;
    double total_log_likelihood;
    double lw, ld, lk;
    void initialize();
    void accept_d_propose_w();
    void accept_w_propose_d();
    struct LocalBuffer{
        uint32_t r[MH], rn[MH], rk[MH];
        TTopic sparse_dense_th;
        std::vector<TCount> ck_new;
        std::vector<int> keys;
        std::vector<float> values;
        HashTable<TTopic, TCount> cxk_sparse;
        HashTable<TTopic, TCount> new_cxk_sparse;
        std::vector<TCount> cxk_dense;
        std::vector<TData> local_data;
        float log_likelihood;
        XorShift generator;
        //AliasUrn urn;
        float total_jll;
        void Generate3()
        {
            generator.MakeBuffer(r, sizeof(uint32_t) * MH);
            generator.MakeBuffer(rn, sizeof(uint32_t) * MH);
            generator.MakeBuffer(rk, sizeof(uint32_t) * MH);
        }
        void Generate1()
        {
            generator.MakeBuffer(r, sizeof(uint32_t) * MH);
        }
        LocalBuffer(TTopic K, TDegree maxdegree)
        : sparse_dense_th(K/4), ck_new(K), cxk_sparse(logceil(K)), cxk_dense(K), local_data(maxdegree)
        {
        }
    };
    std::vector<std::unique_ptr<LocalBuffer>> local_buffers;
};

extern template class WarpLDA<1>;
