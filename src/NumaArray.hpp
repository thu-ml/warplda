#pragma once
#include "Partition.hpp"
#include "omp.h"
#include <numa.h>
#include <map>
#include "Types.hpp"

template <class T>
class RemoteArray
{
	using TIdx = int;
	public:
		RemoteArray() : arr_(nullptr), idx_(nullptr){}
		RemoteArray(T* arr, const TIdx* idx) : arr_(arr), idx_(idx){}
		void Assign(T* arr, const TIdx* idx) { arr_ = arr; idx_ = idx; }
		T& operator[](size_t pos) { return arr_[idx_[pos]]; }
		const T& operator[](size_t pos) const { return arr_[idx_[pos]]; }
	private:
		T* arr_;
		const TIdx* idx_;
};


template <class T>
class RemoteArray64
{
	using TIdx = uint64_t;
	public:
		RemoteArray64() : arr_(nullptr), idx_(nullptr){}
		RemoteArray64(T* arr, TIdx* idx) : arr_(arr), idx_(idx){}
		void Assign(T* arr, TIdx* idx) { arr_ = arr; idx_ = idx; }
		T& operator[](size_t pos) { return arr_[idx_[pos]]; }
		const T& operator[](size_t pos) const { return arr_[idx_[pos]]; }
	private:
		T* arr_;
		TIdx* idx_;
};


class NumaInfo
{
private:
	struct info_t
	{
		std::map<int, std::map<int, int>> info;
		std::map<int, int> ord;
		std::map<int, int> numa_id;
		info_t();
	};
	static info_t info;
public:
	NumaInfo(int thread_id, size_t n);
	size_t beg, end, step;
};

template <class T>
class NumaArray
{
	public:
		NumaArray() : arr_(nullptr) {}
		~NumaArray() { Free(); }
		NumaArray(size_t n, T v = T())
		{
			Assign(n, v);
		}
		T* data() { return arr_; }
		const T* data() const { return arr_; }
		void Assign(size_t n, T v = T())
		{
			Free();
			arr_ = new T[n];
#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < n; i++)
				arr_[i] = v;
			size_ = n;
		}
		void Free()
		{
			if (arr_)
				delete[] arr_;
		}
		size_t size() { return size_; }
		T& operator[](size_t pos) { return arr_[pos]; }
		const T& operator[](size_t pos) const { return arr_[pos]; }
	private:
		T* arr_;
		size_t size_;
};


#if 0
template <class T>
class NumaArray1
{
	public:
		NumaArray1() : arr_(nullptr) {}
		~NumaArray1() { Free(); }
		NumaArray1(size_t n, T v = T())
		{
			Assign(n, v);
		}
		void Assign(size_t n, T v = T())
		{
			Free();
			arr_ = new T[n];
//#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < n; i++)
				arr_[i] = v;
			size_ = n;
		}
		void Free()
		{
			if (arr_)
				delete[] arr_;
		}
		size_t size() { return size_; }
		T& operator[](size_t pos) { return arr_[pos]; }
		const T& operator[](size_t pos) const { return arr_[pos]; }
	private:
		T* arr_;
		size_t size_;
};

#endif
