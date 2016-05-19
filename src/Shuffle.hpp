#pragma once

#include "omp.h"

#include <algorithm>
#include <unordered_set>
#include <utility>
#include <iostream>

#include "Types.hpp"
#include "Bigraph.hpp"
#include "Partition.hpp"
#include "NumaArray.hpp"

template <class T>
class Shuffle
{
public:
	Shuffle(Bigraph &g) : g_(g) { Init(); }
	~Shuffle() {}
public:
	T* DataV(TVID v) { return &v_data_vec_[g_.VIdx(v)]; }

	template <class Function>
	void VisitURemoteData(Function f)
	{
		#pragma omp parallel
		{
			int thread_id = omp_get_thread_num();
			NumaInfo info(thread_id, g_.NU());
			for (TUID u = info.beg; u < info.end; u+= info.step)
			{
				TDegree N = g_.DegreeU(u);
				const TVID* lnks = g_.EdgeOfU(u);
				RemoteArray64<T> data = RemoteArray64<T>(DataV(0), &v2u_shuffle_pos_[g_.UIdx(u)]);
				f(u, N, lnks, data);
			}
		}
	}
	template <class Function>
	void VisitURemoteDataSequential(Function f)
	{
		for (TUID u = g_.Ubegin(); u < g_.Uend(); u++)
		{
			TDegree N = g_.DegreeU(u);
			const TVID* lnks = g_.EdgeOfU(u);
			RemoteArray64<T> data = RemoteArray64<T>(DataV(0), &v2u_shuffle_pos_[g_.UIdx(u)]);
			f(u, N, lnks, data);
		}
	}
	template <class Function>
	void VisitByV(Function f)
	{
		#pragma omp parallel for
		for (TVID v = g_.Vbegin(); v < g_.Vend(); v++)
		{
			TDegree N = g_.DegreeV(v);
			const TUID* lnks = g_.EdgeOfV(v);
			T* data = DataV(v);
			f(v, N, lnks, data);
		}
	}
	static void shuffle_gather( NumaArray<T> &src_data, NumaArray<T> &tar_data, NumaArray<TEID>& shuffle_pos)
	{
		#pragma omp parallel for //schedule(static, 256)
		for (TEID i = 0; i < shuffle_pos.size(); i++)
		tar_data[i] = src_data[shuffle_pos[i]];
	}
	static void shuffle_scatter( NumaArray<T> &src_data, NumaArray<T> &tar_data, NumaArray<TEID>& shuffle_pos)
	{
		#pragma omp parallel for //schedule(static, 256)
		for (TEID i = 0; i < shuffle_pos.size(); i++)
		tar_data[shuffle_pos[i]] = src_data[i];
	}

private:
	void Init()
	{
		cnt_u_data = g_.EdgeOfU(g_.Uend()) - g_.EdgeOfU(g_.Ubegin());
		cnt_v_data = g_.EdgeOfV(g_.Vend()) - g_.EdgeOfV(g_.Vbegin());

		v_data_vec_.Assign(cnt_v_data);
		InitShuffle(g_.AdjV(), g_.AdjU(), v2u_shuffle_pos_);
	}

	template <class TAdjsrc, class TAdjtar>
	static void InitShuffle(TAdjsrc &src_adj, TAdjtar &tar_adj,
		NumaArray<TEID>& shuffle_pos)
		{
			using TSrc = typename TAdjsrc::TSrc;
			using TDst = typename TAdjsrc::TDst;

			shuffle_pos.Assign(tar_adj.NumEdges());

			int64_t threshold = (1<<20) / sizeof(TEID);

			std::vector<TEID> src_off(src_adj.NumVertices(), 0);
			for (TSrc i = 1; i < src_adj.NumVertices(); i++)
			src_off[i] = src_off[i-1] + src_adj.Degree(i-1);

			TDst tar_id = 0;
			TEID pos = 0;
			while (tar_id < tar_adj.NumVertices())
			{
				TDst tar_end_id = tar_id;
				while (tar_end_id < tar_adj.NumVertices() && tar_adj.Edges(tar_end_id) - tar_adj.Edges(tar_id) < threshold)
				{
					TDst id = tar_end_id;
					const TSrc* lnks = tar_adj.Edges(id);
					for (TDegree k = 0; k < tar_adj.Degree(id); k++)
					{
						shuffle_pos[pos++] = src_off[lnks[k]]++;
					}
					tar_end_id++;
				}
				tar_id = tar_end_id;
			}
		}
	public:
		Bigraph &g_;
		NumaArray<T> v_data_vec_;
		NumaArray<TEID> v2u_shuffle_pos_;
		size_t cnt_u_data;
		size_t cnt_v_data;
	};
