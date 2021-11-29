#pragma once
#pragma once

#include <memory>
using namespace std;

class AANum  //definition of Affine Arithmetic Number
{
public:
	int ser;  // the serial number of the element of AANum
	double val; // the value of the element of AANum
	shared_ptr<AANum> nxt; // the pointer pointed to the next element of AANum
};

class Affine
{
public:
	int m_iAANumSer;

	Affine()
	{
		m_iAANumSer = 0;
	}

	~Affine()
	{
		m_iAANumSer = 0;
	}

	//Function: return a + b
	shared_ptr<AANum> Add(shared_ptr<AANum> a, shared_ptr<AANum> b)
	{
		shared_ptr<AANum> c = CopyAANum(a);

		shared_ptr<AANum> bcur = b;
		while (bcur)
		{
			InsertAANum(c, bcur);
			bcur = bcur->nxt;
		}

		return c;
	}

	shared_ptr<AANum> AddSelf(shared_ptr<AANum> a, shared_ptr<AANum> b)
	{
		shared_ptr<AANum> bcur = b;
		while (bcur)
		{
			InsertAANum(a, bcur);
			bcur = bcur->nxt;
		}

		return a;
	}

	//Function: return a - b
	shared_ptr<AANum> Sub(shared_ptr<AANum> a, shared_ptr<AANum> b)
	{
		shared_ptr<AANum> c = CopyAANum(a);
		Neg(b);

		shared_ptr<AANum> bcur = b;
		while (bcur)
		{
			InsertAANum(c, bcur);
			bcur = bcur->nxt;
		}

		return c;
	}

	shared_ptr<AANum> SubSelf(shared_ptr<AANum> a, shared_ptr<AANum> b)
	{
		Neg(b);

		shared_ptr<AANum> bcur = b;
		while (bcur)
		{
			InsertAANum(a, bcur);
			bcur = bcur->nxt;
		}

		return a;
	}

	//Function: return c = a*b
	shared_ptr<AANum> Mul(shared_ptr<AANum> a, shared_ptr<AANum> b)
	{
		double a0 = a->val;
		double b0 = b->val;

		shared_ptr<AANum> tempa = CopyAANum(a);
		shared_ptr<AANum> tempb = CopyAANum(b);
		tempa->val = tempb->val = 0;

		shared_ptr<AANum> cur = tempa->nxt;
		while (cur)
		{
			cur->val = b0 * cur->val;
			cur = cur->nxt;
		}
		cur = tempb->nxt;
		while (cur)
		{
			cur->val = a0 * cur->val;
			cur = cur->nxt;
		}

		shared_ptr<AANum> c = make_shared<AANum>();
		c->ser = 0;
		c->val = a0 * b0;
		c->nxt = nullptr;

		cur = tempa->nxt;
		while (cur)
		{
			InsertAANum(c, cur);
			cur = cur->nxt;
		}

		cur = tempb->nxt;
		while (cur)
		{
			InsertAANum(c, cur);
			cur = cur->nxt;
		}

		double u = 0, v = 0;
		cur = a->nxt;
		while (cur)
		{
			u = u + fabs(cur->val);
			cur = cur->nxt;
		}
		cur = b->nxt;
		while (cur)
		{
			v = v + fabs(cur->val);
			cur = cur->nxt;
		}
		m_iAANumSer++;
		shared_ptr<AANum> ne = make_shared<AANum>();
		ne->ser = m_iAANumSer;
		ne->val = u * v;
		ne->nxt = nullptr;
		InsertAANum(c, ne);

		return c;
	}


	//Function: Copy the AANum src to dst = CopyAANum(src)
	shared_ptr<AANum> CopyAANum(shared_ptr<AANum> src)
	{
		shared_ptr<AANum> head = make_shared<AANum>();

		head->ser = src->ser;
		head->val = src->val;
		head->nxt = nullptr;

		shared_ptr<AANum> scur, dcur, dold;
		scur = src->nxt;
		dold = head;
		while (scur)
		{
			dcur = make_shared<AANum>();
			dcur->ser = scur->ser;
			dcur->val = scur->val;
			dcur->nxt = nullptr;
			dold->nxt = dcur;
			dold = dcur;
			scur = scur->nxt;
		}

		return head;
	}

	// Function: Insert the element elem to the link head
	void InsertAANum(shared_ptr<AANum> head, shared_ptr<AANum> elem)
	{
		if (elem->val == 0) return;

		shared_ptr<AANum> cur, old;
		cur = head;
		old = head;
		while (cur)
		{
			if (elem->ser > cur->ser)
			{
				old = cur;
				cur = cur->nxt;
			}
			else if (elem->ser == cur->ser)
			{
				cur->val = cur->val + elem->val;
				if (cur->val == 0 && cur != head)
				{
					old->nxt = cur->nxt;
					//delete cur;
				}
				break;
			}
			else if (elem->ser < cur->ser)
			{
				shared_ptr<AANum> nele = make_shared<AANum>();
				nele->ser = elem->ser;
				nele->val = elem->val;
				nele->nxt = cur;
				old->nxt = nele;
				break;
			}
		}

		if ((cur == nullptr) && (elem->ser > old->ser))
		{
			shared_ptr<AANum> nele = make_shared<AANum>();
			nele->ser = elem->ser;
			nele->val = elem->val;
			nele->nxt = nullptr;
			old->nxt = nele;
		}

	}

	//Function: Make the AA number a to be negative
	void Neg(shared_ptr<AANum> a)
	{
		shared_ptr<AANum> cur;
		cur = a;
		while (cur)
		{
			cur->val = -cur->val;
			cur = cur->nxt;
		}

	}

	//Function: Convert the Affine Number to Interval Number
	void getInterval(shared_ptr<AANum> head, double* low, double* high)
	{
		shared_ptr<AANum> cur;
		double ksi = 0;
		cur = head->nxt;
		while (cur)
		{
			ksi = ksi + fabs(cur->val);
			cur = cur->nxt;
		}
		double xl, xr;
		xl = head->val; xr = head->val;
		xl = xl - ksi;
		xr = xr + ksi;

		*low = xl;
		*high = xr;
	}


	//Function: Convert Interval Number to Affine Number
	shared_ptr<AANum> IAConvtAA(double low, double hgh)
	{
		shared_ptr<AANum> x = make_shared<AANum>();
		x->ser = 0;
		x->val = (low + hgh) / 2;
		x->nxt = nullptr;

		shared_ptr<AANum> x1 = make_shared<AANum>();
		m_iAANumSer = m_iAANumSer + 1;

		x1->ser = m_iAANumSer;
		x1->val = (hgh - low) / 2;
		x1->nxt = nullptr;
		x->nxt = x1;

		return x;
	}

	//Function: Remove the zero element in the AA number
	void RemvZero(shared_ptr<AANum> head)
	{
		if (head == nullptr) return;

		shared_ptr<AANum> old = head;
		shared_ptr<AANum> cur = head->nxt;
		while (cur)
		{
			if (cur->val == 0)
			{
				old->nxt = cur->nxt;
				cur = old->nxt;
			}
			else
			{
				old = cur;
				cur = cur->nxt;
			}

		}

	}


	//Function: Return a*b
	shared_ptr<AANum> Mul(double a, shared_ptr<AANum> b)
	{
		shared_ptr<AANum> head;
		if (a == 0)
		{
			head = make_shared<AANum>();
			head->val = 0;
			head->ser = 0;
			head->nxt = nullptr;
			return head;
		}

		head = CopyAANum(b);
		shared_ptr<AANum> cur;
		cur = head;
		while (cur)
		{
			cur->val = a * cur->val;
			cur = cur->nxt;
		}

		return head;
	}

	//Function: Return a + b
	shared_ptr<AANum> Add(shared_ptr<AANum> a, double b)
	{
		shared_ptr<AANum> head = CopyAANum(a);

		head->val = head->val + b;
		return head;
	}

	shared_ptr<AANum> AddSelf(shared_ptr<AANum> a, double b)
	{
		a->val += b;
		return a;
	}

	shared_ptr<AANum>  SubSelf(shared_ptr<AANum> a, double b)
	{
		a->val -= b;
		return a;
	}

	//Function: Return b - a
	shared_ptr<AANum> Sub(shared_ptr<AANum> a, double b)
	{
		shared_ptr<AANum> head = CopyAANum(a);
		head->val = head->val - b;
		return head;
	}

	//Function: Return zero AA number
	shared_ptr<AANum> Zero()
	{
		shared_ptr<AANum> head = make_shared<AANum>();
		head->ser = 0;
		head->val = 0;
		head->nxt = nullptr;
		return head;
	}
};
