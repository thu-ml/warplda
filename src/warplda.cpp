#include <exception>
#include <random>
#include "warplda.hpp"
#include "clock.hpp"

const int LOAD_FACTOR = 4;

template <unsigned MH>
WarpLDA<MH>::WarpLDA() : LDA()
{

}

template <unsigned MH>
void WarpLDA<MH>::initialize()
{
    shuffle.reset(new Shuffle<TData>(g));
	alpha_bar = alpha * K;
	beta_bar = beta * g.NV();

	ck.Assign(K, 0);
	ck_new.Assign(K, 0);

	nnz_d.Assign(g.NU());
	nnz_w.Assign(g.NV());

	TDegree max_degree_u = 0;

	local_buffers.resize(omp_get_max_threads());

#pragma omp parallel for
	for (TUID i = 0; i < g.NU(); i++)
		nnz_d[i] = g.DegreeU(i);
#pragma omp parallel for
	for (TVID i = 0; i < g.NV(); i++)
		nnz_w[i] = g.DegreeV(i);
	// Initialize cwk, cdk
#pragma omp parallel for reduction(max:max_degree_u)
	for (TUID i = 0; i < g.NU(); i++)
		max_degree_u = std::max(max_degree_u, g.DegreeU(i));
#pragma omp parallel
	for (uint32_t i = 0; i < local_buffers.size(); i++) {
#pragma omp barrier
		if ((int)i == omp_get_thread_num())
			local_buffers[i].reset(new LocalBuffer(K, max_degree_u));
	}

	std::uniform_int_distribution<TTopic> distribution(0, K-1);

#pragma omp parallel for
	for (uint32_t i = 0; i < local_buffers.size(); i++) {
		local_buffers[i]->ck_new.assign(K, 0);
	}
#pragma omp parallel for
	for (TVID v = g.Vbegin(); v < g.Vend(); v++)
	{
		std::vector<TCount> &ck_new = local_buffers[omp_get_thread_num()]->ck_new;
		TDegree N = g.DegreeV(v);
		TData* data = shuffle->DataV(v);
		for (TDegree i = 0; i < N; i++)
		{
			TTopic k = distribution(generator);
			data[i].oldk = k;
            for (unsigned mh=0; mh<MH; mh++)
                data[i].newk[mh] = k;
			++ck_new[k];
		}
	}

#pragma omp parallel for
	for (TTopic i = 0; i < K; i++)
	{
		TCount s = 0;
		for (TDegree j = 0; j < local_buffers.size(); j++)
			s += local_buffers[j]->ck_new[i];
		ck[i] = s;
	}

	printf("Initialization finished.\n");
}

template <unsigned MH>
void WarpLDA<MH>::estimate(int _K, float _alpha, float _beta, int _niter)
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

        total_jll = 0;
        accept_d_propose_w();
        accept_w_propose_d();

        double tm = clk.timeElapsed();

		// Evaluate likelihood p(w_d | \hat\theta, \hat\phi)

		double jperplexity = exp(-total_log_likelihood / g.NE());

		printf("Iteration %d, %f s, %.2f Mtokens/s, log_likelihood %lf jperplexity %lf ld %f lw %f lk %f jll %f\n", i, tm, (double)g.NE()/tm/1e6, total_log_likelihood, jperplexity, ld, lw, lk, exp(-total_jll/g.NE()));
		fflush(stdout);
	}
}

template<unsigned MH>
void WarpLDA<MH>::accept_d_propose_w()
{
#pragma omp parallel for
	for (unsigned i = 0; i < local_buffers.size(); i++)
    {
		local_buffers[i]->ck_new.assign(K, 0);
		local_buffers[i]->log_likelihood = 0;
        local_buffers[i]->total_jll = 0;
	}
	shuffle->VisitByV([&](TVID v, TDegree N, const TUID* lnks, TData* data)
    {
		LocalBuffer *local_buffer = local_buffers[omp_get_thread_num()].get();
		TCount* ck_new = local_buffer->ck_new.data();

        auto& cxk = local_buffer->cxk_sparse;
        cxk.Rebuild(logceil(std::min(K, nnz_w[v] * LOAD_FACTOR)));

        for (TDegree i=0; i<N; i++) {
            cxk.Put(data[i].oldk)++;
        }
        // Perplexity
        float lgammabeta = lgamma(beta);
        for (auto entry: cxk.table)
            if (entry.key != cxk.EMPTY_KEY)
                local_buffer->log_likelihood += lgamma(beta+entry.value) - lgammabeta;

        // p(x, z | phi, theta)
        for (TDegree i=0; i<N; i++) {
            TTopic k = data[i].oldk;
            float phi = (cxk.Get(k)+beta)/(ck[k]+beta_bar);
            local_buffer->total_jll += log(phi);
        }

        for (TDegree i = 0; i < N; i++)
        {
            TTopic oldk = data[i].oldk;
            TTopic originalk = data[i].oldk;
            float b = cxk.Get(oldk)+beta-1;
            float d = ck[oldk]+beta_bar-1;
            //#pragma simd
            for (unsigned mh=0; mh<MH; mh++) {
                TTopic newk = data[i].newk[mh];
                float a = cxk.Get(newk)+beta - (newk==originalk);
                float c = ck[newk]+beta_bar - (newk==originalk);
                float ad = a*d;
                float bc = b*c;
                bool accept = local_buffer->Rand32() *bc < ad * std::numeric_limits<uint32_t>::max();
                if (accept) {
                    data[i].oldk = newk;
                    b = a; d = c;
                }
            }
            ck_new[data[i].oldk]++;
        }
        nnz_w[v] = cxk.NKey();

        double new_topic = K*beta / (K*beta + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * new_topic;

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
  });

#pragma omp parallel for
	for (TTopic i = 0; i < K; i++)
	{
		int s = 0;
		for (unsigned j = 0; j < local_buffers.size(); j++)
			s += local_buffers[j]->ck_new[i];
		ck[i] = s;
	}
    lk = ld = lw = 0;
    for (unsigned i = 0; i < local_buffers.size(); i++)
    {
        lw += local_buffers[i]->log_likelihood;
        total_jll += local_buffers[i]->total_jll;
    }
    for (TTopic i = 0; i < K; i++) {
        //lk -= ck[i] * log(beta_bar + ck[i]);
        lk -= lgamma(ck[i]+beta_bar) - lgamma(beta_bar);
    }
    total_log_likelihood = lw + lk;
}

template<unsigned MH>
void WarpLDA<MH>::accept_w_propose_d()
{
#pragma omp parallel for
	for (unsigned i = 0; i < local_buffers.size(); i++)
    {
		local_buffers[i]->ck_new.assign(K, 0);
		local_buffers[i]->log_likelihood = 0;
        local_buffers[i]->total_jll = 0;
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

        for (TDegree i=0; i<N; i++) {
            TTopic k = data[i].oldk;
            float theta = (cxk.Get(k)+alpha) / (N+alpha_bar);
            local_buffer->total_jll += log(theta);
        }

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
#pragma omp parallel for
	for (TTopic i = 0; i < K; i++)
	{
		TCount s = 0;
		for (unsigned j = 0; j < local_buffers.size(); j++)
			s += local_buffers[j]->ck_new[i];
		ck[i] = s;
	}
	ld = 0;
	for (unsigned i = 0; i < local_buffers.size(); i++)
	{
		ld += local_buffers[i]->log_likelihood;
		total_jll += local_buffers[i]->total_jll;
	}
	total_log_likelihood += ld;
}

template <unsigned MH>
void WarpLDA<MH>::inference(int K, float alpha, float beta, int niter)  {}
template <unsigned MH>
void WarpLDA<MH>::loadModel(std::string prefix)  {}
template <unsigned MH>
void WarpLDA<MH>::storeModel(std::string prefix) const  {}
template <unsigned MH>
void WarpLDA<MH>::loadZ(std::string prefix)  {}
template <unsigned MH>
void WarpLDA<MH>::storeZ(std::string prefix)  {}
template <unsigned MH>
void WarpLDA<MH>::writeInfo(std::string vocab, std::string info) {}

template class WarpLDA<1>;
