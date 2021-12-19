// 2021/12/5
// make by hyq

#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

struct AANumNode
{
	int ser;
	double val;
	AANumNode* next;
	__device__  AANumNode() : ser(0), val(0), next(nullptr) {}
	__device__  AANumNode(int _ser, double _val, AANumNode* _next) : ser(_ser), val(_val), next(_next) {}
	__device__ ~AANumNode() {}
};

__device__ int d_static_ser = 0;

class AANum
{
public:
	AANumNode* head;
	double center_value;

	__device__  AANum() : head(nullptr), center_value(0) {}
	__device__ ~AANum() {}

	__device__  AANum(const double low, const double high)
	{
		center_value = (low + high) / 2.0;
		head = new AANumNode(d_static_ser ++, (high - low) / 2.0, nullptr);
	}

	// 深拷贝
	__device__  AANum(const AANum& bb)
	{
		center_value = bb.center_value;
		head = copyList(bb.head);
	}

	__device__ void del()
	{
		AANumNode* cur = head, *pre = head;
		while (nullptr != cur)
		{
			cur = cur->next;
			delete pre;
			pre = cur;
		}
		head = nullptr;
	}

	__device__ void print() const
	{
		printf("center : %f\n", center_value);
		AANumNode* cur = head;
		while (nullptr != cur)
		{
			printf("%d : %lf\n", cur->ser, cur->val);
			cur = cur->next;
		}
	}

	__device__ void FromIA(const double low, const double high)
	{
		del();
		center_value = (low + high) / 2.0;
		head = new AANumNode(d_static_ser ++, (high - low) / 2.0, nullptr);
	}

	__device__ void ToIA(double* low, double* high)
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

	__device__ AANum& operator += (const double p)
	{
		center_value += p;
		return *this;
	}

	__device__ AANum& operator -= (const double p)
	{
		center_value -= p;
		return *this;
	}

	__device__ AANum& operator *= (const double p)
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

	__device__ AANum operator + (const double p) const
	{
		AANum ans;
		ans = *this;
		ans.center_value += p;
		return ans;
	}

	__device__ AANum operator - (const double p) const
	{
		AANum ans;
		ans = *this;
		ans.center_value -= p;
		return ans;
	}

	__device__ AANum operator * (const double p) const
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

	__device__ AANum operator + (const AANum& bb) const
	{
		AANum ans;
		add(*this, bb, ans);
		return ans;
	}

	__device__ AANum operator - (const AANum& bb) const
	{
		AANum ans;
		sub(*this, bb, ans);
		return ans;
	}

	__device__ AANum operator * (const AANum& bb) const
	{
		AANum ans;
		mul(*this, bb, ans);
		return ans;
	}

	__device__ AANum& operator = (const AANum& bb)
	{
		center_value = bb.center_value;
		head = copyList(bb.head);
		return *this;
	}

private:
	__device__ AANumNode* add(const AANum& aa, const AANum& bb, AANum& ans) const
	{
		ans.center_value = aa.center_value + bb.center_value;

		AANumNode* cur_aa = aa.head;
		AANumNode* cur_bb = bb.head;

		// 使用尾插法创建新的链表，创建一个临时的头节点，方便对所有的节点进行统一的插入操作，最后删除
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
		ans.head = head->next;
		delete head;
		while (nullptr != cur->next) cur = cur->next;
		return cur;
	}

	// 返回最后指向节点的指针，方便乘法使用
	__device__ AANumNode* sub(const AANum& aa, const AANum& bb, AANum& ans) const
	{
		ans.center_value = aa.center_value - bb.center_value;

		AANumNode* cur_aa = aa.head;
		AANumNode* cur_bb = bb.head;

		// 使用尾插法创建新的链表，创建一个临时的头节点，方便对所有的节点进行统一的插入操作，最后删除
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
		ans.head = head->next;
		delete head;
		while (nullptr != cur->next) cur = cur->next;
		return cur;
	}

	__device__ void mul(const AANum& aa, const AANum& bb, AANum& ans) const
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

	__device__ AANumNode* copyList(AANumNode* src) const
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

};