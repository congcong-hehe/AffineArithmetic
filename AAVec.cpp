// 2022/1/16

#include "AAVec.h"
#include <iterator>

int AANumVec::static_ser = 0;

AANumVec::AANumVec(const double low, const double high)
{
	center_value = (low + high) / 2.0;
	sers.emplace_back(static_ser++);
	vals.emplace_back((high - low) / 2.0);
}

AANumVec::AANumVec(const AANumVec& bb)
{
	center_value = bb.center_value;
	sers = bb.sers;
	vals = bb.vals;
}

AANumVec::~AANumVec()
{
	
}

AANumVec& AANumVec:: operator += (const AANumVec& bb)
{
	center_value += bb.center_value;
	int index_aa = 0;
	int index_bb = 0;
	while (index_aa != size() && index_bb != bb.size())
	{
		int ser_aa = sers[index_aa];
		int ser_bb = bb.sers[index_bb];
		if (ser_aa == ser_bb)
		{
			vals[index_aa] += bb.vals[index_bb];
			index_aa++;
			index_bb++;
		}
		else if (ser_aa < ser_bb)
		{
			index_aa++;
		}
		else
		{
			vals.insert(vals.begin() + index_aa, bb.vals[index_bb]);
			sers.insert(sers.begin() + index_aa, bb.sers[index_bb]);
			index_bb++;
			index_aa += 2;
		}
	}
	if (index_aa == sers.size() && index_bb != bb.sers.size())
	{
		copy(bb.sers.begin() + index_bb, bb.sers.end(), back_inserter(sers));
		copy(bb.vals.begin() + index_bb, bb.vals.end(), back_inserter(vals));
	}
	return *this;
}

void AANumVec::add(const AANumVec& aa, const AANumVec& bb, AANumVec& ans) const
{
	ans.center_value = aa.center_value + bb.center_value;

	int index_aa = 0;
	int index_bb = 0;
	while (index_aa != aa.size() && index_bb != bb.size())
	{
		int ser_aa = aa.sers[index_aa];
		int ser_bb = bb.sers[index_bb];
		if (ser_aa == ser_bb)
		{
			ans.sers.emplace_back(ser_aa);
			ans.vals.emplace_back(aa.vals[index_aa] + bb.vals[index_bb]);
			index_aa++;
			index_bb++;
		}
		else if (ser_aa > ser_bb)
		{
			ans.sers.emplace_back(ser_bb);
			ans.vals.emplace_back(bb.vals[index_bb]);
			index_bb++;
		}
		else
		{
			ans.sers.emplace_back(ser_aa);
			ans.vals.emplace_back(aa.vals[index_aa]);
			index_aa++;
		}
	}

	if (index_aa != aa.size())
	{
		copy(aa.sers.begin() + index_aa, aa.sers.end(), back_inserter(ans.sers));
		copy(aa.vals.begin() + index_aa, aa.vals.end(), back_inserter(ans.vals));
	}
	if (index_bb != bb.size())
	{
		copy(bb.sers.begin() + index_bb, bb.sers.end(), back_inserter(ans.sers));
		copy(bb.vals.begin() + index_bb, bb.vals.end(), back_inserter(ans.vals));
	}

}

void AANumVec::sub(const AANumVec& aa, const AANumVec& bb, AANumVec& ans) const
{
	ans.center_value = aa.center_value - bb.center_value;

	int index_aa = 0;
	int index_bb = 0;
	while (index_aa != aa.size() && index_bb != bb.size())
	{
		int ser_aa = aa.sers[index_aa];
		int ser_bb = bb.sers[index_bb];
		if (ser_aa == ser_bb)
		{
			ans.sers.emplace_back(ser_aa);
			ans.vals.emplace_back(aa.vals[index_aa] - bb.vals[index_bb]);
			index_aa++;
			index_bb++;
		}
		else if (ser_aa > ser_bb)
		{
			ans.sers.emplace_back(ser_bb);
			ans.vals.emplace_back(-bb.vals[index_bb]);
			index_bb++;
		}
		else
		{
			ans.sers.emplace_back(ser_aa);
			ans.vals.emplace_back(aa.vals[index_aa]);
			index_aa++;
		}
	}

	if (index_aa != aa.size())
	{
		copy(aa.sers.begin() + index_aa, aa.sers.end(), back_inserter(ans.sers));
		copy(aa.vals.begin() + index_aa, aa.vals.end(), back_inserter(ans.vals));
	}
	if (index_bb != bb.size())
	{
		copy(bb.sers.begin() + index_bb, bb.sers.end(), back_inserter(ans.sers));
		for (int i = index_bb; i < bb.size(); ++i)
		{
			ans.vals.emplace_back(-bb.vals[i]);
		}
	}

}

void AANumVec::mul(const AANumVec& aa, const AANumVec& bb, AANumVec& ans) const
{
	AANumVec tempa = aa;
	tempa.center_value = 0;
	tempa *= bb.center_value;
	AANumVec tempb = bb;
	tempb.center_value = 0;
	tempb *= aa.center_value;

	add(tempa, tempb, ans);
	ans.center_value = aa.center_value * bb.center_value;

	double u = 0, v = 0;
	for (int i = 0; i < aa.size(); ++i)
	{
		u += abs(aa.vals[i]);
	}
	for (int i = 0; i < bb.size(); ++i)
	{
		v += abs(bb.vals[i]);
	}

	ans.sers.emplace_back(static_ser++);
	ans.vals.emplace_back(u * v);
}

AANumVec::AANumVec(const Interval& interval)
{
	center_value = (interval.low + interval.high) / 2.0;
	sers.emplace_back(static_ser++);
	vals.emplace_back((interval.high - interval.low) / 2.0);
}