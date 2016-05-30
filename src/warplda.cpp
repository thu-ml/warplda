#include <exception>
#include <random>
#include <unordered_map>
#include "warplda.hpp"
#include "Vocab.hpp"
#include "clock.hpp"
#include <exception>
#include <atomic>

const int LOAD_FACTOR = 4;

template <unsigned MH>
WarpLDA<MH>::WarpLDA() : LDA()
{

}

template <unsigned MH>
void WarpLDA<MH>::reduce_ck()
{
    #pragma omp parallel for
    for (TTopic i = 0; i < K; i++)
    {
        TCount s = 0;
        for (auto& buffer : local_buffers)
            s += buffer->ck_new[i];
        ck[i] = s;
    }
}

template <unsigned MH>
template <bool testMode>
void WarpLDA<MH>::initialize()
{
    shuffle.reset(new Shuffle<TData>(g));
	alpha_bar = alpha * K;
	beta_bar = beta * g.NV();

	if (!testMode)
        ck.Assign(K, 0);

	nnz_d.Assign(g.NU());
	nnz_w.Assign(g.NV());

	TDegree max_degree_u = 0;

	local_buffers.resize(omp_get_max_threads());

    #pragma omp parallel for reduction(max:max_degree_u)
	for (TUID i = 0; i < g.NU(); i++)
    {
		max_degree_u = std::max(max_degree_u, g.DegreeU(i));
		nnz_d[i] = g.DegreeU(i);
    }
    #pragma omp parallel for
	for (TVID i = 0; i < g.NV(); i++)
		nnz_w[i] = g.DegreeV(i);

    #pragma omp parallel
    {
        #pragma omp critical
        local_buffers[omp_get_thread_num()].reset(new LocalBuffer(K, max_degree_u));
    }

	shuffle->VisitByV([&](TVID v, TDegree N, const TUID* lnks, TData* data)
	{
        LocalBuffer* buffer = local_buffers[omp_get_thread_num()].get();
		TCount* ck_new = buffer->ck_new.data();
		for (TDegree i = 0; i < N; i++)
		{
			TTopic k = buffer->generator() % K;
			data[i].oldk = k;
            for (unsigned mh=0; mh<MH; mh++)
                data[i].newk[mh] = k;
			++ck_new[k];
		}
	});
    if (!testMode)
        reduce_ck();

	printf("Initialization finished.\n");
}


template<unsigned MH>
void WarpLDA<MH>::LocalBuffer::Init()
{
    std::fill(ck_new.begin(), ck_new.end(), 0);
    log_likelihood = 0;
}

template<unsigned MH>
template <bool testMode>
void WarpLDA<MH>::accept_d_propose_w()
{
    #pragma omp parallel
    {
        local_buffers[omp_get_thread_num()]->Init();
    }

	shuffle->VisitByV([&](TVID v, TDegree N, const TUID* lnks, TData* data)
    {
		LocalBuffer *local_buffer = local_buffers[omp_get_thread_num()].get();
		TCount* ck_new = local_buffer->ck_new.data();

        HashTable<TTopic, TCount>* cxk;
        if (testMode)
            cxk = &cwk_model[v];
        else
        {
            cxk = &local_buffer->cxk_sparse;
            cxk->Rebuild(logceil(std::min(K, nnz_w[v] * LOAD_FACTOR)));
            for (TDegree i=0; i<N; i++) {
                cxk->Put(data[i].oldk)++;
            }
        }

        // Perplexity
        float lgammabeta = lgamma(beta);
        for (auto entry: cxk->table)
            if (entry.key != cxk->EMPTY_KEY)
                local_buffer->log_likelihood += lgamma(beta+entry.value) - lgammabeta;

        int remove_current = !testMode;
        for (TDegree i = 0; i < N; i++)
        {
            TTopic oldk = data[i].oldk;
            TTopic originalk = data[i].oldk;
            float b = cxk->Get(oldk)+beta-remove_current;
            float d = ck[oldk]+beta_bar-remove_current;
            //#pragma simd
            for (unsigned mh=0; mh<MH; mh++) {
                TTopic newk = data[i].newk[mh];
                float a, c;
                if (testMode) {
                    a = cxk->Get(newk)+beta - (newk==originalk);
                    c = ck[newk]+beta_bar - (newk==originalk);
                } else {
                    a = cxk->Get(newk)+beta;
                    c = ck[newk]+beta_bar;
                }
                float ad = a*d;
                float bc = b*c;
                bool accept = local_buffer->Rand32() *bc < ad * std::numeric_limits<uint32_t>::max();
                if (accept) {
                    oldk = newk;
                    b = a; d = c;
                }
            }
            data[i].oldk = oldk;
            if (!testMode)
                ck_new[oldk]++;
        }
        nnz_w[v] = cxk->NKey();

        double new_topic = K*beta / (K*beta + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * new_topic;

        // TODO this propose is incorrect!
        // TODO use alias table
        if (!testMode) {
            for (TDegree i = 0; i < N; i++)
            {
                for (unsigned mh=0; mh<MH; mh++)
                {
                    uint32_t r= local_buffer->Rand32();
                    uint32_t rk = local_buffer->Rand32() % K;
                    uint32_t rn = local_buffer->Rand32() % N;
                    data[i].newk[mh] = r < new_topic_th ? rk : data[rn].oldk;
                }
            }
        } else {
            unsigned global_th = std::numeric_limits<uint32_t>::max()
                * (global_sum / (global_sum + cwk_sums[v]));
            double one_scale = 1.0 / ((double)std::numeric_limits<uint32_t>::max()+1);
            for (TDegree i = 0; i < N; i++)
            {
                for (unsigned mh=0; mh<MH; mh++)
                {
                    uint32_t r = local_buffer->Rand32();
                    if (r < global_th)
                        data[i].newk[mh] = global_urn.DrawSample(local_buffer->Rand32(), local_buffer->Rand32() * one_scale);
                    else
                        data[i].newk[mh] = cwk_urns[v].DrawSample(local_buffer->Rand32(), local_buffer->Rand32() * one_scale);
                }
            }
        }
    });

    if (!testMode)
        reduce_ck();

    for (TTopic i = 0; i < K; i++)
        total_log_likelihood += -lgamma(ck[i]+beta_bar) + lgamma(beta_bar);

    for (auto &buffer : local_buffers)
        total_log_likelihood += buffer->log_likelihood;
}

template<unsigned MH>
template <bool testMode>
void WarpLDA<MH>::accept_w_propose_d()
{
    #pragma omp parallel
    {
        local_buffers[omp_get_thread_num()]->Init();
    }
	shuffle->VisitURemoteData([&](TUID d, TDegree N, const TVID* lnks, RemoteArray64<TData> &data)
    {
		LocalBuffer *local_buffer = local_buffers[omp_get_thread_num()].get();
        TCount* ck_new = local_buffers[omp_get_thread_num()]->ck_new.data();
        TData * local_data = local_buffer->local_data.data();

        auto& cxk = local_buffer->cxk_sparse;
        cxk.Rebuild(logceil(std::min(K, nnz_d[d] * LOAD_FACTOR)));

        for (TDegree i=0; i<N; i++) {
            local_data[i] = data[i];
            cxk.Put(data[i].oldk)++;
        }

        // Perplexity
        float lgammaalpha = lgamma(alpha);
        for (auto entry: cxk.table)
            if (entry.key != cxk.EMPTY_KEY)
                local_buffer->log_likelihood += lgamma(alpha+entry.value) - lgammaalpha;

        local_buffer->log_likelihood -= lgamma(alpha_bar+N) - lgamma(alpha_bar);

        for (TDegree i = 0; i < N; i++)
        {
            TTopic oldk = local_data[i].oldk;
            TTopic originalk = local_data[i].oldk;
            float b = cxk.Get(oldk)+alpha-1;
            float d = ck[oldk]+beta_bar-1;
            //#pragma simd
            for (unsigned mh=0; mh<MH; mh++)
            {
                TTopic newk = local_data[i].newk[mh];
                float a = cxk.Get(newk)+alpha-(newk==originalk);
                float c = ck[newk]+beta_bar-(newk==originalk);
                float ad = a*d;
                float bc = b*c;
                bool accept = local_buffer->Rand32() *bc < ad * std::numeric_limits<uint32_t>::max();
                if (accept) {
                    oldk = newk;
                    b = a; d = c;
                }
            }
            if (!testMode)
                ck_new[oldk]++;
            local_data[i].oldk = oldk;
        }
        nnz_d[d] = cxk.NKey();

        double new_topic = alpha_bar / (alpha_bar + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * new_topic;
        for (TDegree i = 0; i < N; i++)
        {
            data[i].oldk = local_data[i].oldk;
            for (unsigned mh=0; mh<MH; mh++) {
                uint32_t r = local_buffer->Rand32();
                uint32_t rk = local_buffer->Rand32() % K;
                uint32_t rn = local_buffer->Rand32() % N;
                data[i].newk[mh] = r < new_topic_th ? rk : local_data[rn].oldk;
            }
        }
    });
    if (!testMode)
        reduce_ck();
    for (auto& buffer : local_buffers)
		total_log_likelihood += buffer->log_likelihood;
}

template <unsigned MH>
void WarpLDA<MH>::estimate(int _K, float _alpha, float _beta, int _niter, int _perperplexity_interval)
{
    this->K = _K;
    this->alpha = _alpha;
    this->beta = _beta;
    this->niter = _niter;
    initialize();

	for (int i = 0; i < niter; i++)
    {
		Clock clk;
		clk.start();

        total_log_likelihood = 0;
        accept_d_propose_w();
        accept_w_propose_d();

        double ppl = 0;
        bool eval_perplexity = _perperplexity_interval != -1 && i % _perperplexity_interval == 0;
		// Evaluate perplexity p(w_d | \hat\theta, \hat\phi)
        if (eval_perplexity)
            ppl = perplexity();

        double tm = clk.timeElapsed();
		printf("Iteration %d, %f s, %.2f Mtokens/s, log_likelihood (per token) %lf", i, tm, (double)g.NE()/tm/1e6, total_log_likelihood/g.NE());
        if (eval_perplexity) printf(" perplexity %lf\n", ppl);
        else printf("\n");
		fflush(stdout);
	}
}

template <unsigned MH>
void WarpLDA<MH>::inference(int niter, int _perperplexity_interval)
{
    initialize<true>();
	for (int i = 0; i < niter; i++)
    {
		Clock clk;
		clk.start();

        accept_d_propose_w<true>();
        accept_w_propose_d<true>();

        double tm = clk.timeElapsed();

        double ppl = 0;
        bool eval_perplexity = _perperplexity_interval != -1 && i % _perperplexity_interval == 0;
		// Evaluate perplexity p(w_d | \hat\theta, \hat\phi)
        if (eval_perplexity)
            ppl = perplexity<true>();

		// Evaluate likelihood p(w_d | \hat\theta, \hat\phi)
		printf("Iteration %d, %f s, %.2f Mtokens/s ", i, tm, (double)g.NE()/tm/1e6);
        if (eval_perplexity) printf(" perplexity %lf\n", ppl);
        else printf("\n");
		fflush(stdout);
	}
}

template <unsigned MH>
void WarpLDA<MH>::loadModel(std::string fmodel)
{
    std::ifstream fin(fmodel);
    if (!fin)
        throw std::runtime_error(std::string("Failed to load model file : ") + fmodel);
    TVID nv = 0;
    fin >> nv >> K >> alpha >> beta;
    cwk_model.clear();
    cwk_model.resize(nv);
    cwk_urns.resize(nv);
    cwk_sums.resize(nv);
    ck.Assign(K);
	alpha_bar = alpha * K;
	beta_bar = beta * g.NV();

    for (TVID v = 0; v < g.NV(); v++)
    {
        auto& cwk = cwk_model[v];
        TTopic nkey;
        fin >> nkey;
        cwk.Rebuild(logceil(nkey * LOAD_FACTOR));
        for (TDegree i = 0; i < nkey; i++) {
            TTopic k;
            TCount c;
            fin >> k; fin.ignore(); fin >> c;
            cwk.Put(k) = c;
            ck[k] += c;
        }
    }
    fin.close();

    std::vector<TTopic> kds;
    std::vector<double> probs(K);
    global_sum = 0;
    for (TTopic k = 0; k < K; k++)
        global_sum += probs[k] = beta / (ck[k] + beta_bar);
    global_urn.BuildAlias(probs, generator.Rand32());
    for (TVID v = 0; v < g.NV(); v++)
    {
        kds.clear();
        probs.clear();
        auto &cwk = cwk_model[v];
        cwk_sums[v] = 0;
        for (auto &entry: cwk.table) if (entry.key != cwk.EMPTY_KEY) {
            kds.push_back(entry.key);
            double p = entry.value / (ck[entry.key] + beta_bar);
            probs.push_back(p);
            cwk_sums[v] += p;
        }
        cwk_urns[v].BuildAlias(probs, generator.Rand32());
        cwk_urns[v].SetKeys(kds);
    }
}

template <unsigned MH>
void WarpLDA<MH>::storeModel(std::string fmodel)
{
    cwk_model.clear();
    cwk_model.resize(g.NV());
	shuffle->VisitByV([&](TVID v, TDegree N, const TUID* lnks, TData* data)
    {
        auto& cxk = cwk_model[v];
        cxk.Rebuild(logceil(std::min(K, nnz_w[v] * LOAD_FACTOR)));
        for (TDegree i=0; i<N; i++) {
            cxk.Put(data[i].oldk)++;
        }
    });
    std::ofstream fou(fmodel);
    if (!fou)
        throw std::runtime_error(std::string("Failed to store model file : ") + fmodel);
    fou << g.NV() << ' ' << K << ' ' << alpha << ' ' << beta << std::endl;
    for (TVID v = 0; v < g.NV(); v++)
    {
        auto& cwk = cwk_model[v];
        fou << cwk.NKey() << '\t';

        for (auto entry: cwk.table)
        if (entry.key != cwk.EMPTY_KEY)
            fou << entry.key << ":" << entry.value << ' ';
        fou << std::endl;
    }
    fou.close();
}
template <unsigned MH>
void WarpLDA<MH>::loadZ(std::string filez)
{
    throw std::runtime_error("This method is not implemented");
    //std::ifstream fin(filez);
	//shuffle->VisitURemoteDataSequential([&](TUID d, TDegree N, const TVID* lnks, RemoteArray64<TData> &data)
    //{
    //    for (unsigned i = 0; i < N; i++)
    //    {
    //        fin >> data[i].oldk;
    //    }
    //});
    //fin.close();
}

template <unsigned MH>
void WarpLDA<MH>::storeZ(std::string filez)  {
    std::ofstream fou(filez);
	shuffle->VisitURemoteDataSequential([&](TUID d, TDegree N, const TVID* lnks, RemoteArray64<TData> &data)
    {
        fou << N << ' ';
        for (unsigned i = 0; i < N; i++)
        {
            fou << lnks[i] << ':' << data[i].oldk << ' ';
        }
        fou << '\n';
    });
    fou.close();
}

template <unsigned MH>
void WarpLDA<MH>::writeInfo(std::string vocab_fname, std::string info, uint32_t ntop)
{
    Vocab vocab;
    if (!vocab.load(vocab_fname))
        throw std::runtime_error(std::string("Failed to load vocab file : ") + vocab_fname);

	std::vector<std::vector<std::vector<std::pair<double, TVID>>>> result; //result[thread][k][10](value, word)
	result.resize(omp_get_max_threads());
    #pragma omp parallel
		result[omp_get_thread_num()].resize(K);
	shuffle->VisitByV([&](TVID v, TDegree N, const TUID* lnks, TData* data){
		int tid = omp_get_thread_num();
		auto &result_local = result[tid];
		std::unordered_map<TTopic, TCount> cnt;
		for (TDegree i = 0; i < N; i++)
		{
			cnt[data[i].oldk]++;
		}
		for (auto t : cnt)
		{
			TTopic k = t.first;
			TCount c = t.second; //ckw
			auto &r = result_local[k];
			double value = double(c + beta)/(ck[k]+beta_bar);
			//printf("ckw %d %d = %lf\n", k, v, value);
			std::pair<double, TVID> p(value, v);
			r.push_back(p);
			std::push_heap(r.begin(), r.end(), std::greater<std::pair<double, TVID>>());
			if (r.size() > ntop)
			{
				std::pop_heap(r.begin(), r.end(), std::greater<std::pair<double, TVID>>());
				r.pop_back();
			}
		}
	});
	std::ofstream fou1(info+".full.txt");
	std::ofstream fou2(info+".words.txt");
	std::vector<std::vector<std::pair<double, TVID>>> ans;
	for (TTopic k = 0; k < K; k++)
	{
		ans.resize(K);
		auto &a = ans[k];
		for (unsigned tid = 0; tid < result.size(); tid++)
		{
			auto &r = result[tid][k];
			for (auto &p : r)
			{
				a.push_back(p);
				std::push_heap(a.begin(), a.end(), std::greater<std::pair<double, TVID>>());
				if (a.size() > ntop)
				{
					std::pop_heap(a.begin(), a.end(), std::greater<std::pair<double,int>>());
					a.pop_back();
				}
			}
		}
		std::sort(a.rbegin(), a.rend());
		fou1 << k << "\t";
		fou1 << ck[k] << "\t";
		for (auto &p : a)
		{
			std::string word = vocab.getWordById(p.second);
			fou1 << '('<< p.first << ',' << word << ") ";
			fou2 << word << " ";
		}
		fou1 << std::endl;
		fou2 << std::endl;
	}
	fou1.close();
	fou2.close();

}

template <unsigned MH>
template <bool testMode>
double WarpLDA<MH>::perplexity()
{
    std::vector<std::vector<std::pair<int, int>>> cdk(g.NU());
	shuffle->VisitURemoteData([&](TUID d, TDegree N, const TVID* lnks, RemoteArray64<TData> &data) {
		LocalBuffer *local_buffer = local_buffers[omp_get_thread_num()].get();

        auto& cxk = local_buffer->cxk_sparse;
        cxk.Rebuild(logceil(std::min(K, nnz_d[d] * LOAD_FACTOR)));

        for (TDegree i=0; i<N; i++)
            cxk.Put(data[i].oldk)++;

        int Kd = 0;
        for (auto &entry: cxk.table)
            if (entry.key != cxk.EMPTY_KEY)
                Kd++;

        auto &local_cdk = cdk[d];
        local_cdk.resize(Kd);
        Kd = 0;
        for (auto &entry: cxk.table)
            if (entry.key != cxk.EMPTY_KEY)
                local_cdk[Kd++] = std::make_pair(entry.key, entry.value);
    });

    #pragma omp parallel
    {
        local_buffers[omp_get_thread_num()]->Init();
    }

	shuffle->VisitByV([&](TVID v, TDegree N, const TUID* lnks, TData* data)
	{
		LocalBuffer *local_buffer = local_buffers[omp_get_thread_num()].get();

        auto &cxk = !testMode ? local_buffer->cxk_sparse : cwk_model[v];
        double alpha_term = 0;
        if (!testMode) {
            cxk.Rebuild(logceil(std::min(K, nnz_w[v] * LOAD_FACTOR)));
            for (TDegree i=0; i<N; i++) {
                cxk.Put(data[i].oldk)++;
            }
        }
        for (TTopic k = 0; k < K; k++)
            alpha_term += alpha * (cxk.Get(k) + beta) / (ck[k] + beta_bar);

        for (TDegree i=0; i<N; i++) {
            double prob = 0;
            double L = 0;
            auto &local_cdk = cdk[lnks[i]];
            for (auto &entry: local_cdk) {
                L += entry.second;
                prob += entry.second * (cxk.Get(entry.first) + beta) / (ck[entry.first] + beta_bar);
                if ((cxk.Get(entry.first) + beta) / (ck[entry.first] + beta_bar) > 1) {
                    std::cout << cxk.Get(entry.first) << ' ' << beta << ' ' << ck[entry.first] << ' ' << beta_bar << std::endl;
                    throw std::runtime_error("what?");
                }
            }
            prob += alpha_term;
            prob /= (L + alpha_bar);
            local_buffer->log_likelihood += log(prob);
        }
    });
    double log_likelihood = 0;
    for (auto &buffer : local_buffers)
        log_likelihood += buffer->log_likelihood;

    return exp(-log_likelihood / g.NE());
}

template class WarpLDA<1>;
