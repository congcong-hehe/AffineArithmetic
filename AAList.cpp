// 2022/1/16

#include "AAList.h"

int AANumList::static_ser = 0;

AANumList::AANumList(const double low, const double high)
{
	center_value = (low + high) / 2.0;
	head = new AANumNodeList(static_ser++, (high - low) / 2.0, nullptr);
}

AANumList::AANumList(const AANumList& bb)
{
	center_value = bb.center_value;
	head = copyList(bb.head);
}

AANumList::~AANumList()
{
	del();
}

AANumList& AANumList:: operator += (const AANumList& bb)
{
	center_value += bb.center_value;
	AANumNodeList* cur_bb = bb.head;
	AANumNodeList* cur = head;
	AANumNodeList* pre = new AANumNodeList;
	AANumNodeList* temp = pre;
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
			pre->next = new AANumNodeList(cur_bb->ser, cur_bb->val, cur);
			pre = pre->next;
			cur_bb = cur_bb->next;
		}
	}
	if (nullptr == cur && nullptr != cur_bb)
	{
		while (nullptr != cur_bb)
		{
			pre->next = new AANumNodeList(cur_bb);
			pre = pre->next;
			cur_bb = cur_bb->next;
		}
	}
	head = temp->next;
	delete temp;
	return *this;
}

AANumNodeList* AANumList::add(const AANumList& aa, const AANumList& bb, AANumList& ans) const
{
	ans.center_value = aa.center_value + bb.center_value;

	AANumNodeList* cur_aa = aa.head;
	AANumNodeList* cur_bb = bb.head;

	AANumNodeList* head = new AANumNodeList;
	AANumNodeList* cur = head;

	while (nullptr != cur_aa && nullptr != cur_bb)
	{
		int ser_aa = cur_aa->ser;
		int ser_bb = cur_bb->ser;
		if (ser_aa == ser_bb)
		{
			AANumNodeList* temp = new AANumNodeList(ser_aa, cur_aa->val + cur_bb->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_aa = cur_aa->next;
			cur_bb = cur_bb->next;
		}
		else if (ser_aa > ser_bb)
		{
			AANumNodeList* temp = new AANumNodeList(ser_bb, cur_bb->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_bb = cur_bb->next;
		}
		else
		{
			AANumNodeList* temp = new AANumNodeList(ser_aa, cur_aa->val, nullptr);
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

AANumNodeList* AANumList::sub(const AANumList& aa, const AANumList& bb, AANumList& ans) const
{
	ans.center_value = aa.center_value - bb.center_value;

	AANumNodeList* cur_aa = aa.head;
	AANumNodeList* cur_bb = bb.head;
	AANumNodeList* head = new AANumNodeList;
	AANumNodeList* cur = head;

	while (nullptr != cur_aa && nullptr != cur_bb)
	{
		int ser_aa = cur_aa->ser;
		int ser_bb = cur_bb->ser;
		if (ser_aa == ser_bb)
		{
			AANumNodeList* temp = new AANumNodeList(ser_aa, cur_aa->val - cur_bb->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_aa = cur_aa->next;
			cur_bb = cur_bb->next;
		}
		else if (ser_aa > ser_bb)
		{
			AANumNodeList* temp = new AANumNodeList(ser_bb, -cur_bb->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_bb = cur_bb->next;
		}
		else
		{
			AANumNodeList* temp = new AANumNodeList(ser_aa, cur_aa->val, nullptr);
			cur->next = temp;
			cur = temp;
			cur_aa = cur_aa->next;
		}
	}
	if (nullptr != cur_aa) cur->next = copyList(cur_aa);
	if (nullptr != cur_bb)
	{
		AANumNodeList* temp = new AANumNodeList(cur_bb->ser, -cur_bb->val, nullptr);
		cur->next = temp;
		cur = temp;
		cur_bb = cur_bb->next;
	}
	while (nullptr != cur->next) cur = cur->next;
	ans.head = head->next;
	delete head;
	return cur;
}

void AANumList::mul(const AANumList& aa, const AANumList& bb, AANumList& ans) const
{
	AANumList tempa = aa;
	tempa.center_value = 0;
	tempa *= bb.center_value;
	AANumList tempb = bb;
	tempb.center_value = 0;
	tempb *= aa.center_value;

	AANumNodeList* cur_ans = add(tempa, tempb, ans);
	ans.center_value = aa.center_value * bb.center_value;

	double u = 0, v = 0;
	AANumNodeList* cur = aa.head;
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

	AANumNodeList* temp = new AANumNodeList(static_ser++, u * v, nullptr);
	cur_ans->next = temp;
}

AANumNodeList* AANumList::copyList(AANumNodeList* src) const
{
	if (nullptr == src)
	{
		return nullptr;
	}
	AANumNodeList* head = new AANumNodeList(src->ser, src->val, nullptr);
	AANumNodeList* cur_dis = head;
	src = src->next;
	while (nullptr != src)
	{
		AANumNodeList* temp = new AANumNodeList(src->ser, src->val, nullptr);
		cur_dis->next = temp;
		cur_dis = cur_dis->next;
		src = src->next;
	}
	return head;
}

AANumList::AANumList(const Interval& interval)
{
	center_value = (interval.low + interval.high) / 2.0;
	head = new AANumNodeList(static_ser++, (interval.high - interval.low) / 2.0, nullptr);
}