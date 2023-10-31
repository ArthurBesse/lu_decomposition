#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <numeric>

using ld = long double;

template<size_t N>
using vec = std::array<ld, N>;

template<size_t N>
using mtx = std::array<vec<N>, N>;

template<typename F>
ld bisection(ld a, ld b, F f, ld precision = 0.0003)
{
	
	ld actual_precision = b - a;
	while(true)
	{
		ld mid = a + (b - a) / 2;
		if (f(mid) == 0)
			return mid;
		if (f(mid) < 0)
			a = mid;
		else
			b = mid;

		actual_precision /= 2;
		if (actual_precision < precision)
			return mid;
	}
}

template<typename F>
ld fixed_point(ld x0, F f, ld precision = 0.0003)
{
	ld x1 = x0;
	ld x2 = x0;
	while(true)
	{
		if (abs(x2 - x1) < precision)
			return x2;
		x1 = std::exchange(x2, f(x2));
	}
}

template<typename F, typename DF>
ld newton(ld x0, F f, DF df, ld precision = 0.0003)
{
	ld x1 = x0;
	ld x2 = x1 - (f(x1) / df(x1));
	while (true)
	{
		if (abs(x2 - x1) < precision)
			return x2;
		x1 = std::exchange(x2, x2 - (f(x2) / df(x2)));
	}
}

template<typename F>
ld secant(ld a, ld b, F f, ld precision = 0.0003)
{
	ld x1 = a;
	ld x2 = b;
	while (true)
	{
		if (abs(x2 - x1) < precision)
			return x2;
		x1 = std::exchange(x2, x2 - ((f(x1) / (f(x2) - f(x1))) * (x2 - x1)));
	}
}

template<size_t N>
vec<N> lu_decomposition(mtx<N> const& a, vec<N> const& b)
{
	
	mtx<N> l = a;
	mtx<N> u = a;
	{
		vec<N> temp;
		temp.fill(0);
		l.fill(temp);
		u.fill(temp);
	}

	for (size_t i = 0; i < l.size(); ++i)
	{
		for (size_t j = 0; j <= i; ++j)
		{
			typename mtx<N>::value_type::value_type temp = 0;
			for (size_t k = 0; k < j; ++k)
				temp += l[i][k] * u[k][j];
			l[i][j] = a[i][j] - temp;
		}

		for (size_t j = i + 1; j < u[i].size(); ++j)
		{
			typename mtx<N>::value_type::value_type temp = 0;
			for (size_t k = 0; k < i; ++k)
				temp += l[i][k] * u[k][j];
			u[i][j] = (a[i][j] - temp) / l[i][i];
		}
		u[i][i] = 1;
	}


	vec<N> y;
	y[0] = b[0] / l[0][0];

	for (size_t i = 1; i < y.size(); ++i)
	{
		typename vec<N>::value_type temp = 0;
		for (size_t j = 0; j < i; ++j)
			temp += l[i][j] * y[j];
		y[i] = (b[i] - temp) / l[i][i];
	}

	const size_t n = N;

	vec<N> x;
	x[n - 1] = y[n - 1] / u[n - 1][n - 1];

	for (int i = static_cast<int>(n) - 2; i >= 0; --i)
	{
		typename vec<N>::value_type temp = 0;
		for (int j = i + 1; j < static_cast<int>(n); ++j)
			temp += u[i][j] * x[j];
		x[i] = (y[i] - temp) / u[i][i];
	}
	return x;
}

template<size_t N>
vec<N> gauss(mtx<N> const& a, vec<N> const& b, ld precision = 0.0003)
{
	vec<N> x1;
	x1.fill(0);
	vec<N> x2;
	x2.fill(0);

	while(true)
	{
		for (size_t i = 0; i < x2.size(); ++i)
		{
			typename vec<N>::value_type temp = 0;

			for (size_t j = 0; j < i; ++j)
				temp -= a[i][j] * x1[j];

			for (size_t j = i + 1; j < x1.size(); ++j)
				temp -= a[i][j] * x1[j];
			temp += b[i];
			x2[i] = temp / a[i][i];
		}

		ld actual_precision = 0;

		for (size_t i = 0; i < x2.size(); ++i)
			actual_precision = max(actual_precision, abs(x2[i] - x1[i]));

		if (actual_precision < precision)
			return x2;
		x1 = x2;
	}
}


template<size_t N>
vec<N> gauss_seidel(mtx<N> const& a, vec<N> const& b, ld precision = 0.0003)
{
	vec<N> x1;
	x1.fill(0);
	vec<N> x2;
	x2.fill(0);

	while (true)
	{
		for (size_t i = 0; i < x2.size(); ++i)
		{
			typename vec<N>::value_type temp = 0;

			for (size_t j = 0; j < i; ++j)
				temp -= a[i][j] * x2[j];

			for (size_t j = i + 1; j < x1.size(); ++j)
				temp -= a[i][j] * x1[j];
			temp += b[i];
			x2[i] = temp / a[i][i];
		}

		ld actual_precision = 0;

		for (size_t i = 0; i < x2.size(); ++i)
			actual_precision = max(actual_precision, abs(x2[i] - x1[i]));

		if (actual_precision < precision)
			return x2;
		x1 = x2;
	}
}


int main(int, char**)
{
	auto f = [](ld x)
		{
			return 0.5*x - 2.354612;
		};

	vec<3> b;
	b = { 2, 1, 3};

	mtx<3> a;
	a[0] = { 4, 3, 9};
	a[1] = { 6, 3, -8};
	a[2] = { 3, -1, 4 };

	const auto res = lu_decomposition(a, b);
	
	for (const auto& e : res)
		std::cout << e << ' ';

}