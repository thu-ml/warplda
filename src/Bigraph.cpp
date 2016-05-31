#include "Bigraph.hpp"

#include <algorithm>
#include <fstream>


Bigraph::Bigraph()
{
}

bool Bigraph::Load(std::string name)
{
	if (u.Load(name + ".u") && v.Load(name + ".v"))
	{
		return true;
		#if 0
		std::ifstream fwordid(name + ".wordid", std::ios::binary);
		if (fwordid){
			word_id.resize(NV());
			if (fwordid.read((char*)&word_id[0], NV() * sizeof(TVID)))
				return true;
		}
		#endif
	}
	return false;
}

void Bigraph::Generate(std::string name, std::vector<std::pair<TUID, TVID>>& edge_list, TVID nv)
{
	TUID nu = 0;
	for (auto &e : edge_list)
		nu = std::max(nu, e.first);
	nu = nu + 1;

    if (nv == 0) {
	    for (auto &e : edge_list)
		    nv = std::max(nv, e.second);
	    nv = nv + 1;
    }

#if 0
	std::vector<TUID> pu(nu);
	std::vector<TVID> pv(nv);
	for (TUID i = 0; i < nu; i++)
		pu[i] = i;
	for (TVID i = 0; i < nv; i++)
		pv[i] = i;
	std::random_shuffle(pu.begin(), pu.end());
	std::random_shuffle(pv.begin(), pv.end());
	for (auto &e : edge_list)
	{
		e.first = pu[e.first];
		e.second = pv[e.second];
	}

	std::vector<TVID> word_id(nv);
	for (TVID i = 0; i < nv; i++)
		word_id[pv[i]] = i;
	std::ofstream fwordid(name + ".wordid", std::ios::binary);
	fwordid.write((char*)&word_id[0], nv * sizeof(TVID));
	fwordid.close();
#endif

	std::ofstream fuidx(name + ".u.idx", std::ios::binary);
	std::ofstream fulnk(name + ".u.lnk", std::ios::binary);
	std::ofstream fvidx(name + ".v.idx", std::ios::binary);
	std::ofstream fvlnk(name + ".v.lnk", std::ios::binary);

	std::sort(edge_list.begin(), edge_list.end(), [](const std::pair<TUID, TVID> & a, const std::pair<TUID, TVID> & b){ return a.first < b.first || (a.first == b.first && a.second < b.second); });

	TEID off = 0;
	fuidx.write((char*)&off, sizeof(off));
	for (TUID i = 1; i <= nu; i++)
	{
		while (off < edge_list.size() && edge_list[off].first < i)
		{
			auto tar = edge_list[off].second;
			fulnk.write((char*)&tar, sizeof(tar));
			off++;
		}
		fuidx.write((char*)&off, sizeof(off));
	}

	std::sort(edge_list.begin(), edge_list.end(), [](const std::pair<TUID, TVID> & a, const std::pair<TUID, TVID> & b){ return a.second < b.second || (a.second == b.second && a.first < b.first); });

	off = 0;
	fvidx.write((char*)&off, sizeof(off));
	for (TVID i = 1; i <= nv; i++)
	{
		while (off < edge_list.size() && edge_list[off].second < i)
		{
			auto src = edge_list[off].first;
			fvlnk.write((char*)&src, sizeof(src));
			off++;
		}
		fvidx.write((char*)&off, sizeof(off));
	}

	fuidx.close();
	fulnk.close();
	fvidx.close();
	fvlnk.close();
}
