#include "NumaArray.hpp"


NumaInfo::info_t::info_t()
{
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int nid = numa_node_of_cpu(tid);
		#pragma omp critical
		{
			numa_id[tid] = nid;
			ord[tid] = info[nid].size();
			info[nid][tid] = info[nid].size();
			//printf("NumaInfo::Init thread id %d at numa %d ord = %d\n", tid, nid, info[nid][tid]);
		}
	}
}

NumaInfo::NumaInfo(int thread_id, size_t n)
{
	Partition p(info.info.size(), n);
	beg = p.Startid(info.numa_id[thread_id])+info.ord[thread_id];
	end = p.Startid(info.numa_id[thread_id]+1);
	step = info.info[info.numa_id[thread_id]].size();
}

NumaInfo::info_t NumaInfo::info;
