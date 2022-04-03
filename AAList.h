// 2022/1/16

#pragma once

#include "Interval.h"
#include <stdio.h>
#include <math.h>

using namespace std;

struct AANumNodeList
{
	int ser;
	double val;
	AANumNodeList* next;
	AANumNodeList() : ser(0), val(0), next(nullptr) {}
	AANumNodeList(int _ser, double _val, AANumNodeList* _next) : ser(_ser), val(_val), next(_next) {}
	AANumNodeList(const AANumNodeList* node) : ser(node->ser), val(node->val), next(node->next) {}
	~AANumNodeList() {}
};

class AANumList
{
public:
	AANumNodeList* head;
	double center_value;

	AANumList() : head(nullptr), center_value(0) {}
	~AANumList();
	AANumList(const double low, const double high);
	AANumList(const Interval& interval);
	AANumList(const AANumList& bb);
	inline void del();
	inline void print() const;
	inline void fromIA(const Interval& interval);
	inline void fromIA(const double low, const double high);
	inline void toIA(Interval& interval);
	void toIA(double* low, double* high);
	inline void reverse();
	inline AANumList& operator += (const double p);
	inline AANumList& operator -= (const double p);
	inline AANumList& operator *= (const double p);
	inline AANumList operator + (const double p) const;
	inline AANumList operator - (const double p) const;
	inline AANumList operator * (const double p) const;
	inline AANumList operator + (const AANumList& bb) const;
	inline AANumList operator - (const AANumList& bb) const;
	inline AANumList operator * (const AANumList& bb) const;
	inline AANumList& operator = (const AANumList& bb);
	AANumList& operator += (const AANumList& bb);

	static void reset() { static_ser = 0; }

private:
	AANumNodeList* add(const AANumList& aa, const AANumList& bb, AANumList& ans) const;
	AANumNodeList* sub(const AANumList& aa, const AANumList& bb, AANumList& ans) const;
	void mul(const AANumList& aa, const AANumList& bb, AANumList& ans) const;
	AANumNodeList* copyList(AANumNodeList* src) const;

	static int static_ser;
};

inline void AANumList::del()
{
	AANumNodeList* cur = head, * pre = head;
	while (nullptr != cur)
	{
		cur = cur->next;
		delete pre;
		pre = cur;
	}
	head = nullptr;
	center_value = 0.0;
}

inline void AANumList::print() const
{
	printf("center : %f\n", center_value);
	AANumNodeList* cur = head;
	while (nullptr != cur)
	{
		printf("%d : %lf\n", cur->ser, cur->val);
		cur = cur->next;
	}
}

inline void AANumList::reverse()
{
	center_value = -center_value;
	AANumNodeList* cur = head;
	while (nullptr != cur)
	{
		cur->val = -cur->val;
		cur = cur->next;
	}
}

inline void AANumList::fromIA(const double low, const double high)
{
	del();
	center_value = (low + high) / 2.0;
	head = new AANumNodeList(static_ser++, (high - low) / 2.0, nullptr);
}

inline void AANumList::toIA(double* low, double* high)
{
	double val_count = 0;
	AANumNodeList* cur = head;
	while (nullptr != cur)
	{
		val_count += fabs(cur->val);
		cur = cur->next;
	}
	*low = center_value - val_count;
	*high = center_value + val_count;
}

inline AANumList& AANumList::operator += (const double p)
{
	center_value += p;
	return *this;
}

inline AANumList& AANumList::operator -= (const double p)
{
	center_value -= p;
	return *this;
}

inline AANumList& AANumList::operator *= (const double p)
{
	center_value *= p;
	AANumNodeList* cur = head;
	while (nullptr != cur)
	{
		cur->val *= p;
		cur = cur->next;
	}
	return *this;
}

inline AANumList AANumList::operator + (const double p) const
{
	AANumList ans;
	ans = *this;
	ans.center_value += p;
	return ans;
}

inline AANumList AANumList::operator - (const double p) const
{
	AANumList ans;
	ans = *this;
	ans.center_value -= p;
	return ans;
}

inline AANumList AANumList::operator * (const double p) const
{
	AANumList ans;
	ans = *this;
	ans.center_value *= p;
	AANumNodeList* cur = ans.head;
	while (nullptr != cur)
	{
		cur->val *= p;
		cur = cur->next;
	}
	return ans;
}

inline AANumList AANumList::operator + (const AANumList& bb) const
{
	AANumList ans;
	add(*this, bb, ans);
	return ans;
}

inline AANumList AANumList::operator - (const AANumList& bb) const
{
	AANumList ans;
	sub(*this, bb, ans);
	return ans;
}

inline AANumList AANumList::operator * (const AANumList& bb) const
{
	AANumList ans;
	mul(*this, bb, ans);
	return ans;
}

inline AANumList& AANumList::operator = (const AANumList& bb)
{
	del();
	center_value = bb.center_value;
	head = copyList(bb.head);
	return *this;
}

inline void AANumList::fromIA(const Interval &interval)
{
	del();
	center_value = (interval.low + interval.high) / 2.0;
	head = new AANumNodeList(static_ser++, (interval.high - interval.low) / 2.0, nullptr);
}

inline void AANumList::toIA(Interval& interval)
{
	toIA(&interval.low, &interval.high);
}





