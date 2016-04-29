#pragma once

#include <vector>

template <class TKey, class TValue>
class HashTable
{
	public:
		using key_type = TKey;
		using value_type = TValue;
		using reference = value_type&;
		using const_reference = const value_type&;
		const key_type EMPTY_KEY = key_type(-1);

		//std::vector<uint64_t> count_step[2];
	public: 	// Jianfei: hack
		struct Entry {
			key_type key;
			value_type value;
			int32_t l, r;
		};
		std::vector<Entry> table;
		std::vector<bool> keyset;
		uint32_t sizeFactor;
		uint32_t sizeMask;
		uint32_t nKey;

		template <bool AddKey>
		uint32_t findkey(key_type key)
		{
			//printf("find<%d> %d sz %d\n", AddKey, key, table.size());
			uint32_t pos = key & sizeMask;
			if (table[pos].key == key)
			{

		//		count_step[AddKey][0]++;
			//printf("return find<%d> %d = %d\n", AddKey, key, pos);
				return pos;
			}else if (table[pos].key == EMPTY_KEY)
			{
				if (AddKey)
				{
					table[pos] = Entry{key, value_type(), -1, -1};
					nKey++;
			//printf("return find<%d> %d = %d\n", AddKey, key, pos);
		//			count_step[AddKey][0]++;
					return pos;
				}else
				{
					//printf("%d : return find<%d> %d = %d\n", GlobalMPI::rank, AddKey, key, -1);
					//		count_step[AddKey][0]++;
					return -1;
				}
			}else
			{
				//int32_t *p = nullptr;
				int32_t father = -1;
				bool isleft = true;
				int i = 0;
				while (pos != (uint32_t)-1 && table[pos].key != key)
				{
					i++;
					//printf("find<%d> %d at %d (%d, %d,%d))\n", AddKey, key, pos, table[pos].key, table[pos].l, table[pos].r);
					if (key < table[pos].key)
					{
						if (AddKey)
						{
							//p = &table[pos].l;
							father = pos;
							isleft = true;
						}
						pos = table[pos].l;
					}else
					{
						if (AddKey)
						{
							//		p = &table[pos].r;
							father = pos;
							isleft = false;
						}
						pos = table[pos].r;
					}
					//printf("next pos = %d\n", pos);
				}
				//		count_step[AddKey][i]++;
				if (pos == (uint32_t)-1)
				{
					//printf("empty pos\n");
					int current = table.size();
					if (AddKey)
					{
						if (isleft)
							table[father].l = current;
						else
							table[father].r = current;
						//printf(" p = %p\n", p);
						//	*p = table.size();
						//printf(" pushback\n");
						table.push_back(Entry{key, value_type(), -1, -1});
						nKey++;
						//printf("return find<%d> %d = %d\n", AddKey, key, *p);
						//return *p;
						return current;
					}else
					{
						//printf("%d : return find<%d> %d = %d\n", GlobalMPI::rank, AddKey, key, -1);
						return -1;

					}
				}else
				{
			//printf("return find<%d> %d = %d\n", AddKey, key, pos);
					return pos;
				}
			}
		}

	public:
		HashTable(size_t sizeFactor = 0)
		{
		//	count_step[0].assign(100, 0);
	//		count_step[1].assign(100, 0);
			Rebuild(sizeFactor);
		}
        HashTable& operator = (const HashTable &from)
        {
            table = from.table;
            keyset = from.keyset;
            sizeFactor = from.sizeFactor;
            sizeMask = from.sizeMask;
            nKey = from.nKey;
            return *this;
        }
		reference Put(key_type key)
		{
			uint32_t pos = findkey<true>(key);
			return table[pos].value;
		}
		value_type Get(key_type key)
		{
			uint32_t pos = findkey<false>(key);
			if (pos == (uint32_t)-1)
				return value_type();
			else
				return table[pos].value;
		}
		void Rebuild(size_t sizeFactor)
		{
			sizeMask = (1<<sizeFactor)-1;
			nKey = 0;
			table.resize(1<<sizeFactor);
			for (auto &x : table)
			{
				x.key = EMPTY_KEY;
				x.value = value_type();
			}
		}
		uint32_t NKey() const
		{
			return nKey;
		}
};
