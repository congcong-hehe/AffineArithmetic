// 2022/1/4

#pragma once

struct Interval
{
	double low, high;
	Interval(double _low, double _high) : low(_low), high(_high) {}
	Interval() : low(0), high(0) {}
	Interval(const Interval &ia) : low(ia.low), high(ia.high) {}

	inline Interval& operator += (const double p);
	inline Interval& operator -= (const double p);
	inline Interval& operator *= (const double p);
	inline Interval operator + (const double p) const;
	inline Interval operator - (const double p) const;
	inline Interval operator * (const double p) const;
	inline Interval operator + (const Interval& ia) const;
	inline Interval operator - (const Interval& ia) const;
	inline Interval operator * (const Interval& ia) const;
	inline Interval& operator += (const Interval& ia);
	inline Interval operator -();
	void reset() { low = 0; high = 0; }
	double center() const { return (low + high) / 2; }
};

inline Interval& Interval::operator += (const double p)
{
	low += p; high += p;
	return *this;
}

inline Interval& Interval::operator -= (const double p)
{
	low -= p; high -= p;
	return *this;
}

inline Interval& Interval::operator *= (const double p)
{
	low *= p; high *= p;
	return *this;
}

inline Interval Interval::operator + (const double p) const
{
	Interval ans(*this);
	ans += p;
	return ans;
}

inline Interval Interval::operator - (const double p) const
{
	Interval ans(*this);
	ans -= p;
	return ans;
}

inline Interval Interval::operator * (const double p) const
{
	Interval ans(*this);
	ans *= p;
	return ans;
}

inline Interval Interval::operator + (const Interval& ia) const
{
	Interval ans(*this);
	ans.low += ia.low;
	ans.high += ia.high;
	return ans;
}

inline Interval Interval::operator - (const Interval& ia) const
{
	Interval ans;
	ans.low = low - ia.high;
	ans.high = high - ia.low;
	return ans;
}

inline Interval Interval::operator * (const Interval& ia) const
{
	Interval ans;

	double num = low * ia.low;
	ans.low = ans.high = num;

	num = low * ia.high;
	ans.low = ans.low < num ? ans.low : num;
	ans.high = ans.high > num ? ans.high : num;

	num = high * ia.low;
	ans.low = ans.low < num ? ans.low : num;
	ans.high = ans.high > num ? ans.high : num;

	num = high * ia.high;
	ans.low = ans.low < num ? ans.low : num;
	ans.high = ans.high > num ? ans.high : num;

	return ans;
}

inline Interval& Interval::operator += (const Interval& ia)
{
	low += ia.low;
	high += ia.high;
	return *this;
}

inline Interval Interval::operator -()
{
	Interval ans(*this);
	ans.low = -ans.low;
	ans.high = -ans.high;
	return ans;
}
