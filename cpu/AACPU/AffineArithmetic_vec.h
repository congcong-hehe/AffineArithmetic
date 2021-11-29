#pragma once

#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iterator>

using namespace std;

class AANum
{
public:
	// sers中的值代表一个唯一的代表元， 保证升序排列
	vector<int> sers;
	vector<double> vals;

	double center_value = 0;

	inline int size() const { return sers.size(); }

	void print()
	{
		printf("center : %f\n", center_value);
		for (int i = 0; i < size(); ++i)
		{
			printf("%d : %f\n", sers[i], vals[i]);
		}
	}
};

class Affine 
{
public:
	int aa_ser;

	Affine() : aa_ser(0) {}
	~Affine() {}

	inline void add(const AANum& aa, const double p, AANum& ans)
	{
		ans = aa;
		ans.center_value += p;
	}

	inline void sub(const AANum& aa, const double p, AANum& ans)
	{
		ans = aa;
		ans.center_value -= p;
	}

	inline void mul(const AANum& aa, const double p, AANum& ans)
	{
		ans = aa;
		ans.center_value *= p;
		for (int i = 0; i < ans.size(); ++i)
		{
			ans.vals[i] *= p;
		}
	}

	void neg(const AANum& aa, AANum& ans)
	{
		ans = aa;
		ans.center_value = - ans.center_value;
		for (int i = 0; i < ans.size(); ++i)
		{
			ans.vals[i] = -ans.vals[i];
		}
	}

	void add(const AANum& aa, const AANum& bb, AANum& ans)
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

	void sub(const AANum& aa, const AANum& bb, AANum& ans)
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
				ans.vals.emplace_back(- bb.vals[index_bb]);
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

	void mul(const AANum& aa, const AANum& bb, AANum& ans)
	{
		AANum tempa = aa;
		tempa.center_value = 0;
		mulSelf(tempa, bb.center_value);
		AANum tempb = bb;
		tempb.center_value = 0;
		mulSelf(tempb, aa.center_value);

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

		ans.sers.emplace_back(aa_ser++);
		ans.vals.emplace_back(u * v);
	}

	inline void addSelf(AANum& aa, const double p) 
	{
		aa.center_value += p;
	}

	void subSelf(AANum& aa, const double p)
	{
		aa.center_value -= p;
	}

	void mulSelf(AANum& aa, const double p)
	{
		aa.center_value *= p;
		for (int i = 0; i < aa.size(); ++i)
		{
			aa.vals[i] *= p;
		}
	}

	void negSelf(AANum& aa)
	{
		aa.center_value = -aa.center_value;
		for (int i = 0; i < aa.size(); ++i)
		{
			aa.vals[i] = -aa.vals[i];
		}
	}

	void IAToAA(const double low, const double high, AANum& ans)
	{
		ans.center_value = (low + high) / 2.0;
		ans.sers.emplace_back(aa_ser++);
		ans.vals.emplace_back((high - low) / 2.0);
	}

	void AAToIA(const AANum& aa, double& low, double& high)
	{
		double val_count = 0;
		for (int i = 0; i < aa.size(); ++i)
		{
			val_count += aa.vals[i];
		}
		low = aa.center_value - val_count;
		high = aa.center_value + val_count;
	}
};
