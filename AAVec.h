// 2022/1/16
#pragma once

#include "Interval.h"
#include <stdio.h>
#include <math.h>
#include <vector>

using namespace std;

class AANumVec
{
public:
	vector<int> sers;
	vector<double> vals;

	double center_value = 0;

	AANumVec() : center_value(0) {}
	~AANumVec();
	AANumVec(const double low, const double high);
	AANumVec(const Interval& interval);
	AANumVec(const AANumVec& bb);
	inline void del();
	inline void print() const;
	inline void fromIA(const Interval& interval);
	inline void fromIA(const double low, const double high);
	inline void toIA(Interval& interval);
	void toIA(double* low, double* high);
	inline void reverse();
	inline AANumVec& operator += (const double p);
	inline AANumVec& operator -= (const double p);
	inline AANumVec& operator *= (const double p);
	inline AANumVec operator + (const double p) const;
	inline AANumVec operator - (const double p) const;
	inline AANumVec operator * (const double p) const;
	inline AANumVec operator + (const AANumVec& bb) const;
	inline AANumVec operator - (const AANumVec& bb) const;
	inline AANumVec operator * (const AANumVec& bb) const;
	AANumVec& operator += (const AANumVec& bb);

	static void reset() { static_ser = 0; }
	int size() const { return sers.size(); }
private:
	void add(const AANumVec& aa, const AANumVec& bb, AANumVec& ans) const;
	void sub(const AANumVec& aa, const AANumVec& bb, AANumVec& ans) const;
	void mul(const AANumVec& aa, const AANumVec& bb, AANumVec& ans) const;

	static int static_ser;
};

inline void AANumVec::del()
{
	center_value = 0.0;
	sers.clear();
	vals.clear();
}

inline void AANumVec::print() const
{
	printf("center : %f\n", center_value);
	for (int i = 0; i < size(); ++i)
	{
		printf("%d : %f\n", sers[i], vals[i]);
	}
}

inline void AANumVec::reverse()
{
	center_value = -center_value;
	for (int i = 0; i < size(); ++i)
	{
		vals[i] = -vals[i];
	}
}

inline void AANumVec::fromIA(const double low, const double high)
{
	sers.clear();
	vals.clear();
	center_value = (low + high) / 2.0;
	sers.emplace_back(static_ser++);
	vals.emplace_back((high - low) / 2.0);
}

inline void AANumVec::toIA(double* low, double* high)
{
	double val_count = 0;
	for (int i = 0; i < size(); ++i)
	{
		val_count += abs(vals[i]);
	}
	*low = center_value - val_count;
	*high = center_value + val_count;
}

inline AANumVec& AANumVec::operator += (const double p)
{
	center_value += p;
	return *this;
}

inline AANumVec& AANumVec::operator -= (const double p)
{
	center_value -= p;
	return *this;
}

inline AANumVec& AANumVec::operator *= (const double p)
{
	center_value *= p;
	for (int i = 0; i < size(); ++i)
	{
		vals[i] *= p;
	}
	return *this;
}

inline AANumVec AANumVec::operator + (const double p) const
{
	AANumVec ans(*this);
	ans.center_value += p;
	return ans;
}

inline AANumVec AANumVec::operator - (const double p) const
{
	AANumVec ans(*this);
	ans.center_value -= p;
	return ans;
}

inline AANumVec AANumVec::operator * (const double p) const
{
	AANumVec ans(*this);
	ans *= p;
	return ans;
}

inline AANumVec AANumVec::operator + (const AANumVec& bb) const
{
	AANumVec ans;
	add(*this, bb, ans);
	return ans;
}

inline AANumVec AANumVec::operator - (const AANumVec& bb) const
{
	AANumVec ans;
	sub(*this, bb, ans);
	return ans;
}

inline AANumVec AANumVec::operator * (const AANumVec& bb) const
{
	AANumVec ans;
	mul(*this, bb, ans);
	return ans;
}

inline void AANumVec::fromIA(const Interval& interval)
{
	fromIA(interval.low, interval.high);
}

inline void AANumVec::toIA(Interval& interval)
{
	toIA(&interval.low, &interval.high);
}