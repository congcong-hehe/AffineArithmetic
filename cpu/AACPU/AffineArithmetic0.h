#pragma once

#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iterator>
#include <memory>

using namespace std;

// ��������ʹ�������=����ʹ��ǳ������Ϊ����ans = a + b�������м���һ������� ���Բ�Ҫʹ�ÿ������캯������һ�������ʼ��
// debug 306ms
class AANum
{
public:
	// sers�е�ֵ����һ��Ψһ�Ĵ���Ԫ�� ��֤��������
	shared_ptr<vector<int>> sers;
	shared_ptr<vector<double>> vals;

	double center_value = 0;

	static int static_ser;

public:

	AANum()
	{
		sers = make_shared<vector<int>>();
		vals = make_shared<vector<double>>();
	}

	AANum(const double low, const double high)
	{
		sers = make_shared<vector<int>>();
		vals = make_shared<vector<double>>();
		center_value = (low + high) / 2.0;
		sers->emplace_back(static_ser++);
		vals->emplace_back((high - low) / 2.0);
	}

	AANum(const AANum& bb)
	{
		center_value = bb.center_value;
		sers = make_shared<vector<int>>(*bb.sers);
		vals = make_shared<vector<double>>(*bb.vals);
	}

	~AANum()
	{

	}

	inline int size() const { return sers->size(); }

	void print()
	{
		printf("center : %f\n", center_value);
		for (int i = 0; i < size(); ++i)
		{
			printf("%d : %f\n", (*sers)[i], (*vals)[i]);
		}
	}

public:

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
			(*vals)[i] *= p;
		}
		return *this;
	}

	inline AANum operator - ()
	{
		AANum ans(*this);
		ans.center_value = -ans.center_value;
		for (int i = 0; i < size(); ++i)
		{
			(*vals)[i] = -(*vals)[i];
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
		const AANum &aa = *this;
		ans.center_value = aa.center_value + bb.center_value;

		int index_aa = 0;
		int index_bb = 0;
		while (index_aa != aa.size() && index_bb != bb.size())
		{
			int ser_aa = (*aa.sers)[index_aa];
			int ser_bb = (*bb.sers)[index_bb];
			if (ser_aa == ser_bb)
			{
				ans.sers->emplace_back(ser_aa);
				ans.vals->emplace_back((*aa.vals)[index_aa] + (*bb.vals)[index_bb]);
				index_aa++;
				index_bb++;
			}
			else if (ser_aa > ser_bb)
			{
				ans.sers->emplace_back(ser_bb);
				ans.vals->emplace_back((*bb.vals)[index_bb]);
				index_bb++;
			}
			else
			{
				ans.sers->emplace_back(ser_aa);
				ans.vals->emplace_back((*aa.vals)[index_aa]);
				index_aa++;
			}
		}

		if (index_aa != aa.size())
		{
			copy(aa.sers->begin() + index_aa, aa.sers->end(), back_inserter(*ans.sers));
			copy(aa.vals->begin() + index_aa, aa.vals->end(), back_inserter(*ans.vals));
		}
		if (index_bb != bb.size())
		{
			copy(bb.sers->begin() + index_bb, bb.sers->end(), back_inserter(*ans.sers));
			copy(bb.vals->begin() + index_bb, bb.vals->end(), back_inserter(*ans.vals));
		}
		return ans;
	}

	AANum operator - (const AANum& bb) const
	{
		AANum ans;
		const AANum& aa = *this;
		ans.center_value = aa.center_value - bb.center_value;

		int index_aa = 0;
		int index_bb = 0;
		while (index_aa != aa.size() && index_bb != bb.size())
		{
			int ser_aa = (*aa.sers)[index_aa];
			int ser_bb = (*bb.sers)[index_bb];
			if (ser_aa == ser_bb)
			{
				ans.sers->emplace_back(ser_aa);
				ans.vals->emplace_back((*aa.vals)[index_aa] - (*bb.vals)[index_bb]);
				index_aa++;
				index_bb++;
			}
			else if (ser_aa > ser_bb)
			{
				ans.sers->emplace_back(ser_bb);
				ans.vals->emplace_back(-(*bb.vals)[index_bb]);
				index_bb++;
			}
			else
			{
				ans.sers->emplace_back(ser_aa);
				ans.vals->emplace_back((*aa.vals)[index_aa]);
				index_aa++;
			}
		}

		if (index_aa != aa.size())
		{
			copy(aa.sers->begin() + index_aa, aa.sers->end(), back_inserter(*ans.sers));
			copy(aa.vals->begin() + index_aa, aa.vals->end(), back_inserter(*ans.vals));
		}
		if (index_bb != bb.size())
		{
			copy(bb.sers->begin() + index_bb, bb.sers->end(), back_inserter(*ans.sers));
			for (int i = index_bb; i < bb.size(); ++i)
			{
				ans.vals->emplace_back(-(*bb.vals)[i]);
			}
		}
		return ans;
	}

	AANum operator * (const AANum& bb) const
	{
		const AANum& aa = *this;
		AANum tempa = *this;
		tempa.center_value = 0;
		tempa *= bb.center_value;
		AANum tempb = bb;
		tempb.center_value = 0;
		tempb *= aa.center_value;

		AANum ans = tempa + tempb;
		ans.center_value = aa.center_value * bb.center_value;

		double u = 0, v = 0;
		for (int i = 0; i < aa.size(); ++i)
		{
			u += abs((*aa.vals)[i]);
		}
		for (int i = 0; i < bb.size(); ++i)
		{
			v += abs((*bb.vals)[i]);
		}

		ans.sers->emplace_back(static_ser++);
		ans.vals->emplace_back(u * v);

		return ans;
	}

	void ToIA(double* low, double* high)
	{
		double val_count = 0;
		for (int i = 0; i < size(); ++i)
		{
			val_count += abs((*vals)[i]);
		}
		*low = center_value - val_count;
		*high = center_value + val_count;
	}
	
	// ǳ����
	AANum& operator = (const AANum& bb)
	{
		center_value = bb.center_value;
		sers = bb.sers;
		vals = bb.vals;
		return *this;
	}
};

int AANum::static_ser = 0;