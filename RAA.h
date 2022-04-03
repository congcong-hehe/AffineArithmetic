// 2022/1/19

#pragma once

#include "Interval.h"
#include <stdio.h>

// reduced Affine Arithmetic 
class RAA4
{
public:
	double x, y, z, w;

	RAA4() : x(0), y(0), z(0), w(0) {}
	RAA4(double _x, double _y, double _z, double _w) : x(_x), y(_y), z(_z), w(_w) {}
	RAA4(const RAA4 &raa) : x(raa.x), y(raa.y), z(raa.z), w(raa.w) {}

	// tag == true:u tag == fasle : v
	RAA4(const double low, const double high, bool tag)
	{
		x = (low + high) / 2;
		w = 0;
		if (true == tag)
		{
			y = (high - low) / 2;
			z = 0;
		}
		else
		{
			y = 0;
			z = (high - low) / 2;
		}
	}

	void del()
	{
		x = y = z = w = 0;
	}

	void reverse()
	{
		x = -x; y = -y; z = -z;
	}

	RAA4(const Interval& interval, bool tag)
	{
		double low = interval.low;
		double high = interval.high;
		x = (low + high) / 2;
		w = 0;
		if (true == tag)
		{
			y = (high - low) / 2;
			z = 0;
		}
		else
		{
			y = 0;
			z = (high - low) / 2;
		}
	}

	void operator += (const double p)
	{
		x += p;
	}

	void operator *= (const double p)
	{
		x *= p; y *= p; z *= p; w *= abs(p);
	}

	RAA4 operator + (const double p) const
	{
		RAA4 ans(*this);
		ans += p;
		return ans;
	}

	RAA4 operator * (const double p) const
	{
		RAA4 ans(*this);
		ans *= p;
		return ans;
	}

	void operator += (const RAA4& raa)
	{
		x += raa.x; y += raa.y; z += raa.z; w += raa.w;
	}

	RAA4 operator + (const RAA4& raa)
	{
		RAA4 ans(*this);
		ans += raa;
		return ans;
	}

	RAA4 operator * (const RAA4& raa)
	{
		RAA4 ans;
		ans.x = x * raa.x;
		ans.y = x * raa.y + y * raa.x;
		ans.z = x * raa.z + z * raa.x;
		ans.w = abs(x) * raa.w + abs(raa.x) * w + (abs(y) + abs(z) + w) * (abs(raa.y) + abs(raa.z) + raa.w);
		return ans;
	}

	void toIA(double* low, double* high)
	{
		double rad = abs(y) + abs(z) + w;
		*low = x - rad;
		*high = x + rad;
	}

	void print()
	{
		double low, high;
		toIA(&low, &high);
		printf("%f %f , %f %f %f %f\n", low, high, x, y, z, w);
	}
	
private:
	double abs(const double a)
	{
		return a >= 0 ? a : -a;
	}
};

// reduced Affine Arithmetic 
class RAA3
{
public:
	double x, y, z;

	RAA3() : x(0), y(0), z(0) {}
	RAA3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
	RAA3(const RAA3& raa) : x(raa.x), y(raa.y), z(raa.z) {}

	RAA3(const double low, const double high)
	{
		x = (low + high) / 2;
		y = (high - low) / 2;
		z = 0;
	}

	void del()
	{
		x = y = z = 0;
	}

	void reverse()
	{
		x = -x; y = -y;
	}

	RAA3(const Interval& interval)
	{
		double low = interval.low;
		double high = interval.high;
		x = (low + high) / 2;
		y = (high - low) / 2;
		z = 0;
	}

	void operator += (const double p)
	{
		x += p;
	}

	void operator *= (const double p)
	{
		x *= p; y *= p; z *= abs(p);
	}

	RAA3 operator + (const double p) const
	{
		RAA3 ans(*this);
		ans += p;
		return ans;
	}

	RAA3 operator * (const double p) const
	{
		RAA3 ans(*this);
		ans *= p;
		return ans;
	}

	void operator += (const RAA3& raa)
	{
		x += raa.x; y += raa.y; z += raa.z; 
	}

	RAA3 operator + (const RAA3& raa)
	{
		RAA3 ans(*this);
		ans += raa;
		return ans;
	}

	RAA3 operator * (const RAA3& raa)
	{
		RAA3 ans;
		ans.x = x * raa.x;
		ans.y = x * raa.y + y * raa.x ;
		ans.z = abs(x) * raa.z + abs(raa.x) * z + (abs(y) + z) * (abs(raa.y) + raa.z);
		return ans;
	}

	void toIA(double* low, double* high)
	{
		double rad = abs(y) + z;
		*low = x - rad;
		*high = x + rad;
	}

	void print()
	{
		double low, high;
		toIA(&low, &high);
		printf("%f %f , %f %f %f\n", low, high, x, y, z);
	}

private:
	double abs(const double a)
	{
		return a >= 0 ? a : -a;
	}
};