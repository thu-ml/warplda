#pragma once

#include "Types.hpp"
//#include "Partition.h"
#include "NumaArray.hpp"

template <class TSrc_, class TDst_, class TEdge_, class TDegree_>
class AdjList
{
	//template<class T> using NumaArray=NumaArray1<T>;
	public:
		using TSrc = TSrc_;
		using TDst = TDst_;
		using TEdge = TEdge_;
		using TDegree = TDegree_;
	public:
		AdjList(){}
		bool Load(std::string);
		TSrc NumVertices() { return n_; }
		TSrc NumVerticesLocal() { return n_local_; }
		TEdge NumEdges() { return ne_; }
		TDegree Degree(TSrc id) { return idx_[id + 1] - idx_[id]; }
		TEdge Idx(TSrc id) { return idx_[id]; }
		const TDst* Edges(TSrc id) { return &lnk_[idx_[id]]; }
		TSrc Begin() { return beg_; }
		TSrc End() { return end_; }

		template <class Function>
			void Visit(Function f)
			{
				for (TSrc id = beg_; id < end_; id++)
				{
					f(id, Degree(id), lnk_ + idx_[id]);
				}
			}

		TEdge * idx_;
		TDst * lnk_;
	private:
		//Partition p_;
		TSrc n_;
		TSrc n_local_;
		TEdge ne_;
		TSrc beg_;
		TSrc end_;
		NumaArray<TEdge> idx_vec_;
		NumaArray<TDst> lnk_vec_;
};
