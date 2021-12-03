#pragma once

#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iterator>

using namespace std;

// 对放射算数之间的运算又加了一层拷贝，提高了运算效率
// 196ms
class AANum
{
public:
	vector<int> sers;
	vector<double> vals;

	double center_value = 0;
	static int static_ser;

	inline int size() const { return sers.size(); }

	void print()
	{
		printf("center : %f\n", center_value);
		for (int i = 0; i < size(); ++i)
		{
			printf("%d : %f\n", sers[i], vals[i]);
		}
	}

	AANum()
	{

	}

	AANum(const double low, const double high)
	{
		center_value = (low + high) / 2.0;
		sers.emplace_back(static_ser++);
		vals.emplace_back((high - low) / 2.0);
	}

	void ToIA(double* low, double* high)
	{
		double val_count = 0;
		for (int i = 0; i < size(); ++i)
		{
			val_count += vals[i];
		}
		*low = center_value - val_count;
		*high = center_value + val_count;
	}

	// 深拷贝
	AANum& operator = (const AANum& bb)
	{
		center_value = bb.center_value;
		sers = vector<int>(bb.sers);
		vals = vector<double>(bb.vals);
		return *this;
	}

	inline AANum operator + (const double p) const
	{
		AANum ans(*this);
		ans.center_value += p;
		return ans;
	}

	inline AANum operator - (const double p) const
	{
		AANum ans(*this);
		ans.center_value -= p;
		return ans;
	}

	inline AANum& operator += (const double p)
	{
		center_value += p;
		return *this;
	}

	inline AANum& operator -= (const double p)
	{
		center_value -= p;
		return *this;
	}

	inline AANum& operator *= (const double p)
	{
		center_value *= p;
		for (int i = 0; i < size(); ++i)
		{
			vals[i] *= p;
		}
		return *this;
	}

	inline AANum operator - ()
	{
		AANum ans(*this);
		ans.center_value = -ans.center_value;
		for (int i = 0; i < size(); ++i)
		{
			vals[i] = -vals[i];
		}
		return ans;
	}

	inline AANum operator * (const double p) const
	{
		AANum ans(*this);
		ans *= p;
		return ans;
	}

	AANum operator + (const AANum& bb) const
	{
		AANum ans;
		add(*this, bb, ans);
		return ans;
	}

	AANum operator - (const AANum& bb) const
	{
		AANum ans;
		sub(*this, bb, ans);
		return ans;
	}

	AANum operator * (const AANum& bb) const
	{
		AANum ans;
		mul(*this, bb, ans);
		return ans;
	}

private:

	void add(const AANum& aa, const AANum& bb, AANum& ans) const
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
				ans.vals.emplace_back(bb.vals[index_aa]);
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

	void sub(const AANum& aa, const AANum& bb, AANum& ans) const
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
				ans.vals.emplace_back(bb.vals[index_aa]);
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

	void mul(const AANum& aa, const AANum& bb, AANum& ans) const
	{
		AANum tempa = aa;
		tempa.center_value = 0;
		tempa *= bb.center_value;
		AANum tempb = bb;
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

};

int AANum::static_ser = 0;