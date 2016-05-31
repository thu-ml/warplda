#pragma once

#include "AdjList.hpp"

class Bigraph
{
	public:
		using TAdjU = AdjList<TUID, TVID, TEID, TDegree>;
		using TAdjV = AdjList<TVID, TUID, TEID, TDegree>;

	private:
		TAdjU u;
		TAdjV v;
		#if 0
		std::vector<TVID> word_id;
		#endif
	public:
		Bigraph();
		bool Load(std::string);
		TUID NU() { return u.NumVertices(); }
		TVID NV() { return v.NumVertices(); }
		TEID NE() { return u.NumEdges(); }

		static void Generate(std::string, std::vector<std::pair<TUID, TVID>> &, TVID nv = 0);

		template <class Function>
			void VisitU(Function f)
			{
				u.Visit(f);
			}

		template <class Function>
			void VisitV(Function f)
			{
				v.Visit(f);
			}

		TUID DegreeU(TUID uid)
		{
			return u.Degree(uid);
		}
		TVID DegreeV(TVID vid)
		{
			return v.Degree(vid);
		}
		const TVID* EdgeOfU(TUID uid)
		{
			return u.Edges(uid);
		}
		const TUID* EdgeOfV(TVID vid)
		{
			return v.Edges(vid);
		}
		TAdjU & AdjU() { return u; }
		TAdjV & AdjV() { return v; }

		TEID UIdx(TUID uid) { return u.Idx(uid); }
		TEID VIdx(TVID vid) { return v.Idx(vid); }

		TUID Ubegin() { return u.Begin(); }
		TUID Uend() { return u.End(); }

		TVID Vbegin() { return v.Begin(); }
		TVID Vend() { return v.End(); }
//		TVID WordId(TVID vid) { return word_id[vid]; }
};
