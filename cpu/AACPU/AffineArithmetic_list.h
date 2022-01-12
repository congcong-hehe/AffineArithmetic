#pragma once

#include <stdio.h>
#include <math.h>

// 链表实现，单线程

struct AANumNode
{
	int ser;
	double val;
	AANumNode* next;
	AANumNode() : ser(0), val(0), next(nullptr) {}
	AANumNode(int _ser, double _val, AANumNode* _next) : ser(_ser), val(_val), next(_next) {}
	AANumNode(const AANumNode* node) : ser(node->ser), val(node->val), next(node->next) {}
	~AANumNode() {}
};

class AANum
{
public:
	AANumNode* head;
	double center_value;

	AANum() : head(nullptr), center_value(0) {}
	~AANum();
	AANum(const double low, const double high);
	AANum(const AANum& bb);
	void del();
	void print() const;
	void fromIA(const double low, const double high);
	void toIA(double* low, double* high);
	void reverse();
	AANum& operator += (const double p);
	AANum& operator -= (const double p);
	AANum& operator *= (const double p);
	AANum operator + (const double p) const;
	AANum operator - (const double p) const;
	AANum operator * (const double p) const;
	AANum operator + (const AANum& bb) const;
	AANum operator - (const AANum& bb) const;
	AANum operator * (const AANum& bb) const;
	AANum& operator = (const AANum& bb);
	AANum& operator += (const AANum& bb);

private:
	AANumNode* add(const AANum& aa, const AANum& bb, AANum& ans) const;
	AANumNode* sub(const AANum& aa, const AANum& bb, AANum& ans) const;
	void mul(const AANum& aa, const AANum& bb, AANum& ans) const;
	AANumNode* copyList(AANumNode* src) const;

};

int d_static_ser = 0;

AANum::AANum(const double low, const double high)
{
	center_value = (low + high) / 2.0;
	head = new AANumNode(d_static_ser++, (high - low) / 2.0, nullptr);
}

AANum::AANum(const AANum& bb)
{
	center_value = bb.center_value;
	head = copyList(bb.head);
}

AANum::~AANum()
{
	del();
}

void AANum::del()
{
	AANumNode* cur = head, * pre = head;
	while (nullptr != cur)
	{
		cur = cur->next;
		delete pre;
		pre = cur;
	}
	head = nullptr;
	center_value = 0.0;
}

void AANum::print() const
{
	printf("center : %f\n", center_value);
	AANumNode* cur = head;
	while (nullptr != cur)
	{
		printf("%d : %lf\n", cur->ser, cur->val);
		cur = cur->next;
	}
}

void AANum::reverse()
{
	center_value = -center_value;
	AANumNode* cur = head;
	while (nullptr != cur)
	{
		cur->val = -cur->val;
		cur = cur->next;
	}
}

void AANum::fromIA(const double low, const double high)
{
	del();
	center_value = (low + high) / 2.0;
	head = new AANumNode(d_static_ser++, (high - low) / 2.0, nullptr);
}

void AANum::toIA(double* low, double* high)
{
	double val_count = 0;
	AANumNode* cur = head;
	while (nullptr != cur)
	{
		val_count += fabs(cur->val);
		cur = cur->next;
	}
	*low = center_value - val_count;
	*high = center_value + val_count;
}

AANum& AANum::operator += (const double p)
{
	center_value += p;
	return *this;
}

AANum& AANum::operator -= (const double p)
{
	center_value -= p;
	return *this;
}

AANum& AANum::operator *= (const double p)
{
	center_value *= p;
	AANumNode* cur = head;
	while (nullptr != cur)
	{
		cur->val *= p;
		cur = cur->next;
	}
	return *this;
}

AANum AANum::operator + (const double p) const
{
	AANum ans;
	ans = *this;
	ans.center_value += p;
	return ans;
}

AANum AANum::operator - (const double p) const
{
	AANum ans;
	ans = *this;
	ans.center_value -= p;
	return ans;
}

AANum AANum::operator * (const double p) const
{
	AANum ans;
	ans = *this;
	ans.center_value *= p;
	AANumNode* cur = ans.head;
	while (nullptr != cur)
	{
		cur->val *= p;
		cur = cur->next;
	}
	return ans;
}

AANum AANum::operator + (const AANum& bb) const
{
	AANum ans;
	add(*this, bb, ans);
	return ans;
}

AANum AANum::operator - (const AANum& bb) const
{
	AANum ans;
	sub(*this, bb, ans);
	return ans;
}

AANum AANum::operator * (const AANum& bb) const
{
	AANum ans;
	mul(*this, bb, ans);
	return ans;
}

AANum& AANum::operator = (const AANum& bb)
{
	del();
	center_value = bb.center_value;
	head = copyList(bb.head);
	return *this;
}

AANum& AANum:: operator += (const AANum& bb)
{
	center_value += bb.center_value;
	AANumNode* cur_bb = bb.head;
	AANumNode* cur = head;
	AANumNode* pre = new AANumNode;
	AANumNode* temp = pre;
	pre->next = cur;
	while (nullptr != cur_bb && nullptr != cur)
	{
		if (cur_bb->ser == cur->ser)
		{
			cur->val += cur_bb->val;
			pre = cur;
			cur = cur->next;
			cur_bb = cur_bb->next;
		}
		else if (cur_bb->ser > cur->ser)
		{
			pre = cur;
			cur = cur->next;
		}
		else
		{
			pre->next = new AANumNode(cur_bb->ser, cur_bb->val, cur);
			pre = pre->next;
			cur_bb = cur_bb->next;
		}
	}
	if (nullptr == cur && nullptr != cur_bb)
	{
		while (nullptr != cur_bb)
		{
			pre->next = new AANumNode(cur_bb);
			pre = pre->next;
			cur_bb = cur_bb->next;
		}
	}
	head = temp->next;
	delete temp;
	return *this;
}

AANumNode* AANum::add(const AANum& aa, const AANum& bb, AANum& ans) const
{
	ans.center_value = aa.center_value + bb.center_value;

	AANumNode* cur_aa = aa.head;
	AANumNode* cur_bb = bb.head;

	AANumNode* head = new AANumNode;
	AANumNode* cur = head;

	while (nullptr != cur_aa && nullptr != cur_bb)
	{
		int ser_aa = cur_aa->ser;
		int ser_bb = cur_bb->ser;
		if (ser_aa == ser_bb)
		{
			AANumNode* temp = new AANumNode(ser_aa, cur_aa->val + cur_bb->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_aa = cur_aa->next;
			cur_bb = cur_bb->next;
		}
		else if (ser_aa > ser_bb)
		{
			AANumNode* temp = new AANumNode(ser_bb, cur_bb->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_bb = cur_bb->next;
		}
		else
		{
			AANumNode* temp = new AANumNode(ser_aa, cur_aa->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_aa = cur_aa->next;
		}
	}
	if (nullptr != cur_aa) cur->next = copyList(cur_aa);
	if (nullptr != cur_bb) cur->next = copyList(cur_bb);
	while (nullptr != cur->next) cur = cur->next;
	ans.head = head->next;
	delete head;
	return cur;
}

AANumNode* AANum::sub(const AANum& aa, const AANum& bb, AANum& ans) const
{
	ans.center_value = aa.center_value - bb.center_value;

	AANumNode* cur_aa = aa.head;
	AANumNode* cur_bb = bb.head;
	AANumNode* head = new AANumNode;
	AANumNode* cur = head;

	while (nullptr != cur_aa && nullptr != cur_bb)
	{
		int ser_aa = cur_aa->ser;
		int ser_bb = cur_bb->ser;
		if (ser_aa == ser_bb)
		{
			AANumNode* temp = new AANumNode(ser_aa, cur_aa->val - cur_bb->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_aa = cur_aa->next;
			cur_bb = cur_bb->next;
		}
		else if (ser_aa > ser_bb)
		{
			AANumNode* temp = new AANumNode(ser_bb, -cur_bb->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_bb = cur_bb->next;
		}
		else
		{
			AANumNode* temp = new AANumNode(ser_aa, cur_aa->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_aa = cur_aa->next;
		}
	}
	if (nullptr != cur_aa) cur->next = copyList(cur_aa);
	if (nullptr != cur_bb)
	{
		AANumNode* temp = new AANumNode(cur_bb->ser, -cur_bb->val, nullptr);
		cur->next = temp;
		cur = temp;
		cur_bb = cur_bb->next;
	}
	while (nullptr != cur->next) cur = cur->next;
	ans.head = head->next;
	delete head;
	return cur;
}

void AANum::mul(const AANum& aa, const AANum& bb, AANum& ans) const
{
	AANum tempa = aa;
	tempa.center_value = 0;
	tempa *= bb.center_value;
	AANum tempb = bb;
	tempb.center_value = 0;
	tempb *= aa.center_value;

	AANumNode* cur_ans = add(tempa, tempb, ans);
	ans.center_value = aa.center_value * bb.center_value;

	double u = 0, v = 0;
	AANumNode* cur = aa.head;
	while (nullptr != cur)
	{
		u += fabs(cur->val);
		cur = cur->next;
	}
	cur = bb.head;
	while (nullptr != cur)
	{
		v += fabs(cur->val);
		cur = cur->next;
	}

	AANumNode* temp = new AANumNode(d_static_ser++, u * v, nullptr);
	cur_ans->next = temp;
}

AANumNode* AANum::copyList(AANumNode* src) const
{
	if (nullptr == src)
	{
		return nullptr;
	}
	AANumNode* head = new AANumNode(src->ser, src->val, nullptr);
	AANumNode* cur_dis = head;
	src = src->next;
	while (nullptr != src)
	{
		AANumNode* temp = new AANumNode(src->ser, src->val, nullptr);
		cur_dis->next = temp;
		cur_dis = cur_dis->next;
		src = src->next;
	}
	return head;
}
