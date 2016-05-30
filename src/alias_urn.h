#ifndef __ALIAS_URN
#define __ALIAS_URN

#include <vector>
#include <limits>
#include <random>
#include <cassert>
#include <iostream>
#include "Types.hpp"
using std::vector;
using std::cout;
using std::endl;

class AliasUrn {
	public:
		struct AliasEntry {
			int LoKey, HiKey;
			long long p;
            AliasEntry(): LoKey(0), HiKey(0), p(0) {}
            AliasEntry(int l, int h, int p): LoKey(l), HiKey(h), p(p) {}
		};

		virtual void BuildAlias(const vector<double> &p, uint32_t u)
		{
			if (p.empty()) 
			{
				table.resize(0);
				return;
			}
			std::uniform_real_distribution<float> u01;
			long long nMax = std::numeric_limits<long long>::max();

			size = p.size();
			this->binSize = nMax / size;
			table.resize(size);

			float totalMass = 0;
			for (size_t i=0; i<p.size(); i++) 
				totalMass += p[i];
			totalMass = 1. / totalMass;

			// Partition Lo and Hi
			// Lo: p < binSize
			// Hi: p >= binSize
			long long remaining = nMax;
			int pLo = 0;
			int pHi = size - 1;
			for (int i=0; i<size; i++)
			{
				AliasEntry entry;
				entry.p = nMax * p[i] * totalMass;
				remaining -= entry.p;
				if (entry.p < binSize) {
					entry.LoKey = i; //entry.HiKey = -1;
					table[pLo++] = entry;
				}
				else {
					entry.HiKey = i; //entry.LoKey = -1;
					table[pHi--] = entry;
				}
			}
			assert(pHi + 1 == pLo);
			int nHi = size - pHi - 1;
		//	for (auto &entry: table) cout << (float)entry.p/nMax << " ";
		//	cout << "======" << endl;

			// Put the remaining mass to a random bin
			if (nHi > 0) {
                //int pos = pHi + 1 + u % nHi;
                //if (pos < 0 || pos > table.size()) {
                //    cout << pos << " " << size << " " << table.size() << endl;
                //}
				//table[pHi + 1 + u % nHi].p += remaining;
				table.back().p += remaining;
			} else {
				table.back().p += remaining;
				assert(table.back().p >= binSize);
				pHi--;
			}

			int uHi = pHi + 1;

			// Build alias table
			// while lo is not empty, pick up a lo and a hi
			// create a table entry
			// put the remaining of hi back
			for (int i=0; i<size; i++)
			{
				//if (i+1 < size)
				//	assert(i < uHi);
				assert(i<size);
				assert(uHi<size);
				auto &loEntry = table[i];
				auto &hiEntry = table[uHi];
				loEntry.HiKey = hiEntry.HiKey;
				hiEntry.p -= (binSize - loEntry.p);
				if (hiEntry.p < binSize) {
					hiEntry.LoKey = hiEntry.HiKey; uHi++;
				}
				//for (auto &entry: table)
				//	cout << entry.LoKey << " " << entry.HiKey << " " << (float)entry.p / nMax << endl;
				//cout << "-----" << endl;
			}
			table.back().LoKey = table.back().HiKey;
			//cout << "++++++++++++++" << endl;
		}

		void SetKeys(const vector<unsigned int> &keys)
		{
			for (auto &entry: table)
			{
				entry.LoKey = keys[entry.LoKey];
				entry.HiKey = keys[entry.HiKey];
			}
		}

		int DrawSample(size_t rSize, float u2)
		{
			if (table.empty()) assert(0);
			int bin = rSize % size;
			auto &entry = table[bin];
			long long pos = u2 * binSize;

			return pos<entry.p ? entry.LoKey : entry.HiKey;
		}
	

  public:
		std::uniform_real_distribution<float> u01;
		vector<AliasEntry> table;

		long long size;
		long long binSize;
};

/*class ParallelAliasUrn : public AliasUrn {
    // User should guarantee only one ParallelAliasUrn.Build is called at the same time
    public:
		void BuildAlias(const vector<float> &p, uint32_t u);
};*/

#endif
