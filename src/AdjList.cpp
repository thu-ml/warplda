#include "AdjList.hpp"
#include "Utils.hpp"

#include <fstream>
#include <iostream>


template <class T>
void readvec(NumaArray<T> &a, std::ifstream &f, uint64_t idx_beg, uint64_t count)
{
	a.Assign(count);
	f.seekg(idx_beg * sizeof(T), std::ios::beg);
	f.read((char*)&a[0], count * sizeof(T));
}

template <class TSrc, class TDst, class TEdge, class TDegree>
bool AdjList<TSrc, TDst, TEdge, TDegree>::Load(std::string name)
{
	std::ifstream fidx(name + ".idx", std::ios::binary);
	std::ifstream flnk(name + ".lnk", std::ios::binary);

	if (!fidx || !flnk)
		return false;
	n_ = filesize(fidx) / sizeof(TEdge) - 1;

//	p_ = Partition(1, n_);

	beg_ = 0; //p_.Startid(0);
	end_ = n_; //p_.Startid(1);
	n_local_ = end_ - beg_;

	readvec(idx_vec_, fidx, beg_ , end_ - beg_ + 1);

	idx_ = &idx_vec_[0] - beg_;

	readvec(lnk_vec_, flnk, idx_[beg_], idx_[end_] - idx_[beg_]);

	lnk_ = &lnk_vec_[0] - idx_[beg_];

	ne_ = filesize(flnk) / sizeof(TDst);

	return true;
}

template bool AdjList<uint32_t, uint32_t, TEID, TDegree>::Load(std::string name);
