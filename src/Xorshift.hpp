#pragma once
#include <cstdint>
#include <limits>

class XorShift
{
	uint64_t s[16];
	int p;
	uint64_t x; /* The state must be seeded with a nonzero value. */

	uint64_t xorshift1024star(void) {
		uint64_t s0 = s[ p ];
		uint64_t s1 = s[ p = (p+1) & 15 ];
		s1 ^= s1 << 31; // a
		s1 ^= s1 >> 11; // b
		s0 ^= s0 >> 30; // c
		return ( s[p] = s0 ^ s1 ) * UINT64_C(1181783497276652981);
	}
	uint64_t xorshift128plus(void) {
		uint64_t x = s[0];
		uint64_t const y = s[1];
		s[0] = y;
		x ^= x << 23; // a
		x ^= x >> 17; // b
		x ^= y ^ (y >> 26); // c
		s[1] = x;
		return x + y;
	}
	uint64_t xorshift64star(void) {
		x ^= x >> 12; // a
		x ^= x << 25; // b
		x ^= x >> 27; // c
		return x * UINT64_C(2685821657736338717);
	}
	public:

	using result_type=uint64_t;

	XorShift() : p(0), x((uint64_t)std::rand() * RAND_MAX + std::rand()){
		for (int i = 0; i < 16; i++)
		{
			s[i] = xorshift64star();
		}
	}
	uint64_t operator()(){
		return xorshift128plus();
		//return xorshift64star();
	}
	uint32_t Rand32(){
		return (uint32_t)xorshift128plus();
	}
	void MakeBuffer(void *p, size_t len)
	{
		int N = len / sizeof(uint32_t);
		uint32_t *arr = (uint32_t *)p;
		for (int i = 0; i < N; i++)
			arr[i] = (*this)();
		int M = len % sizeof(uint32_t);
		if (M > 0)
		{
			uint32_t k = (*this)();
			memcpy(arr + N, &k, M);
		}
	}
	uint64_t max() {return std::numeric_limits<uint64_t>::max();}
	uint64_t min() {return std::numeric_limits<uint64_t>::min();}
};

