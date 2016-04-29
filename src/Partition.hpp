#pragma once

class Partition
{
	public:
		//Partition(){}
		Partition(int size, int n)
			: size_(size)
			, n_(n)
			, d_(n / size)
			, r_(n % size)
			{}
		int Startid(int par)
		{
			return d_ * par + (par < r_ ? par : r_ );
		}
		int Parid(int idx)
		{
			return idx / (d_ + 1) < r_ ? idx / (d_ + 1) :  (idx - r_) / d_;
		}
	private:
		int size_;
		int n_;
		int d_;
		int r_;
};
