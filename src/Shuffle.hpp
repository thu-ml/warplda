#pragma once

#include <algorithm>
#include <unordered_set>
#include <utility>
#include <iostream>

#include "Types.hpp"
#include "Bigraph.hpp"
#include "Partition.hpp"
#include "NumaArray.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
template <class T>
class Shuffle
{
#if 0
		struct Meta
		{
			Int beg, end, gid;
		};
		using NAII = NumaArray<std::pair<Int, Int>>;
#endif
	public:
		Shuffle(Bigraph &g) : g_(g) { Init(); }
		~Shuffle()
		{
		}
	public:
//		T* DataU(TUID u) { return &u_data_vec_[g_.UIdx(u)]; }
		T* DataV(TVID v) { return &v_data_vec_[g_.VIdx(v)]; }
#if 0
		template <class Function>
			void VisitByU(Function f)
			{
#pragma omp parallel for
				for (TUID u = g_.Ubegin(); u < g_.Uend(); u++)
				{
					TDegree N = g_.DegreeU(u);
					const TVID* lnks = g_.EdgeOfU(u);
					T* data = DataU(u);
					f(u, N, lnks, data);
				}
			}
#endif

#if 0
		template <class Function>
			void VisitUGatherFromV(Function f)
			{
#pragma omp parallel
				{
					int thread_id = omp_get_thread_num();
					NumaInfo info(thread_id, g_.NU());
//					printf("VisitUScatterToV info beg = %d  step = %d  end = %d\n", info.beg, info.step, info.end);
					for (TUID u = info.beg; u < info.end; u+= info.step)
					{
						TUID N = g_.DegreeU(u);
						const TVID* lnks = g_.EdgeOfU(u);
						RemoteArray<T> data_old = RemoteArray<T>(DataV(0), &v2u_shuffle_pos_[g_.UIdx(u)]);
						T* data_new = DataU(u);
						f(u, N, lnks, data_old, data_new);
					}
				}
			}
#endif
		template <class Function>
			void VisitURemoteData(Function f)
			{
#pragma omp parallel
				{
					int thread_id = omp_get_thread_num();
					NumaInfo info(thread_id, g_.NU());
//					printf("VisitUScatterToV info beg = %d  step = %d  end = %d\n", info.beg, info.step, info.end);
					for (TUID u = info.beg; u < info.end; u+= info.step)
					{
						TDegree N = g_.DegreeU(u);
						const TVID* lnks = g_.EdgeOfU(u);
						RemoteArray64<T> data = RemoteArray64<T>(DataV(0), &v2u_shuffle_pos_[g_.UIdx(u)]);
						f(u, N, lnks, data);
					}
				}
			}
#if 0
		template <class Function>
			void VisitUScatterToV(Function f)
			{
#pragma omp parallel
				{
					int thread_id = omp_get_thread_num();
					NumaInfo info(thread_id, g_.NU());
//					printf("VisitUScatterToV info beg = %d  step = %d  end = %d\n", info.beg, info.step, info.end);
					for (TUID u = info.beg; u < info.end; u+= info.step)
					{
						TDegree N = g_.DegreeU(u);
						const TVID* lnks = g_.EdgeOfU(u);
						const T* data = DataU(u);
						RemoteArray<T> data_new = RemoteArray<T>(DataV(0), &v2u_shuffle_pos_[g_.UIdx(u)]);
						f(u, N, lnks, data, data_new);
					}
				}
			}
#endif
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
#if 0
		static void shuffle_gather2( NumaArray<T> &src_data, NumaArray<T> &tar_data, NumaArray<Int>& shuffle_pos, std::vector<Meta>& meta)
		{
#pragma omp parallel for schedule(static, 1)
			for (int i = 0 ; i < meta.size(); i++)
				for (int k = meta[i].beg; k <= meta[i].end; k++)
					tar_data[k] = src_data[shuffle_pos[k]];
		}
		static void shuffle_scatter2( NumaArray<T> &src_data, NumaArray<T> &tar_data, NumaArray<Int>& shuffle_pos, std::vector<Meta>& meta)

		{
#pragma omp parallel for schedule(static, 1)
			for (int i = 0 ; i < meta.size(); i++)
				for (int k = meta[i].beg; k <= meta[i].end; k++)
					tar_data[shuffle_pos[k]] = src_data[k];
		}
#endif

#if 0
		void UtoV()
		{
			Clock t2;
			t2.start();
			//shuffle_gather(u_data_vec_, v_data_vec_, u2v_shuffle_pos_);
			shuffle_scatter(u_data_vec_, v_data_vec_, v2u_shuffle_pos_);
			//shuffle_scatter2(u_data_vec_, v_data_vec_, v2u_shuffle_pos_, v2u_meta_);
			//shuffle_gather2(u_data_vec_, v_data_vec_, u2v_shuffle_pos_, u2v_meta_);
			double speed = (u_data_vec_.size() * sizeof(T) *2 + v2u_shuffle_pos_.size() * sizeof(TEID)) /1073741824.0/ t2.timeElapsed() ;
			printf("%d : shuffle UtoV time2 %lf\tSpeed %lf GB/s\n", Global::rank, t2.timeElapsed(), speed);
		}
		void VtoU()
		{
			Clock t2;
			t2.start();
			shuffle_gather(v_data_vec_, u_data_vec_, v2u_shuffle_pos_);
	//		shuffle_gather2(v_data_vec_, u_data_vec_, v2u_shuffle_pos_, v2u_meta_);
			//shuffle_scatter(u_data_vec_, v_data_vec_, u2v_shuffle_pos_);
			double speed = (u_data_vec_.size() * sizeof(T) *2 + v2u_shuffle_pos_.size() * sizeof(TEID)) /1073741824.0/ t2.timeElapsed() ;
			printf("%d : shuffle VtoU time2 %lf\tSpeed %lf GB/s\n", Global::rank, t2.timeElapsed(), speed);
		}
#endif
	private:
		void Init()
		{
			cnt_u_data = g_.EdgeOfU(g_.Uend()) - g_.EdgeOfU(g_.Ubegin());
			cnt_v_data = g_.EdgeOfV(g_.Vend()) - g_.EdgeOfV(g_.Vbegin());


//			u_data_vec_.Assign(cnt_u_data);
			v_data_vec_.Assign(cnt_v_data);


			//InitShuffle(g_.AdjU(), g_.AdjV(), u2v_shuffle_pos_);
			InitShuffle(g_.AdjV(), g_.AdjU(), v2u_shuffle_pos_);

			//InitShuffle2(g_.AdjU(), g_.AdjV(), u2v_shuffle_pos_, u2v_meta_);
			//InitShuffle2(g_.AdjV(), g_.AdjU(), v2u_shuffle_pos_, v2u_meta_);
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
#if 0
		static void InitShuffle2(AdjList &src_adj, AdjList &tar_adj,
				NumaArray<Int>& shuffle_pos,
			std::vector<Meta>& meta)
		{
			shuffle_pos.Assign(tar_adj.NumEdges());

			size_t threshold = (Global::chunk<<20) / sizeof(Int);

			std::vector<Int> src_off(src_adj.NumVertices(), 0);
			for (Int i = 1; i < src_adj.NumVertices(); i++)
				src_off[i] = src_off[i-1] + src_adj.Degree(i-1);

			std::vector<Int> tar_off(tar_adj.NumVertices(), 0);
			for (Int i = 1; i < tar_adj.NumVertices(); i++)
				tar_off[i] = tar_off[i-1] + tar_adj.Degree(i-1);
			std::vector<Int> tar_off_old = tar_off;

			size_t delta = Global::chunk;
			for (Int tar_id = 0; tar_id < tar_adj.NumVertices(); tar_id++)
			{
				Meta p;
				p.gid = -1;
				for (Int k = 0; k < tar_adj.Degree(tar_id); k++)
				{
					const Int* lnks = tar_adj.Edges(tar_id);
					Int src_id = lnks[k];
					if (p.gid == src_id / delta)
					{
						p.end++;
					}else
					{
						if (p.gid != -1)
						{
							meta.push_back(p);
						}
						p.gid = src_id / delta;
						p.beg = p.end = tar_off[tar_id];
					}
					shuffle_pos[tar_off[tar_id]++] = src_off[lnks[k]]++;
				}
				if (p.gid != -1)
				{
					meta.push_back(p);
				}
			}
			std::cout << "meta size : " << meta.size() << std::endl;
			std::sort(meta.begin(), meta.end(), [&](const Meta &a, const Meta &b) {return a.gid < b.gid;} );
		}
#endif


    public:
		Bigraph &g_;
//		NumaArray<T> u_data_vec_;
		NumaArray<T> v_data_vec_;
//		NumaArray<TEID> u2v_shuffle_pos_;
		NumaArray<TEID> v2u_shuffle_pos_;
#if 0
		std::vector<Meta> u2v_meta_;
		std::vector<Meta> v2u_meta_;
#endif
		size_t cnt_u_data;
		size_t cnt_v_data;

};
