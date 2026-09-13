#pragma once

#include <vector>

#include "common/global_var.hpp"

namespace polynomial
{
    // Inputs are evaluations at 0, 1 and 2, rather than monomial coefficients.
    inline F evaluate_quadratic_samples(const F &y0, const F &y1, const F &y2, const F &x)
    {
        return (y0 * (x - F(1)) * (x - F(2)) - F(2) * y1 * x * (x - F(2)) +
                y2 * x * (x - F(1))) / F(2);
    }
}

class linear_poly;

class cubic_poly {
public:
	F a, b, c, d;
	cubic_poly();
	cubic_poly(const F &, const F &, const F &, const F &);
	cubic_poly operator + (const cubic_poly &) const;
    cubic_poly operator * (const F &) const;
	F eval(const F &) const;
	void clear();
};

class quadratic_poly {
public:
	F a, b, c;
	quadratic_poly();
	quadratic_poly(const F &, const F &, const F &);
	quadratic_poly operator + (const quadratic_poly &) const;
	quadratic_poly operator + (const linear_poly &) const;
	cubic_poly operator * (const linear_poly &) const;
	quadratic_poly operator * (const F &) const;
	F eval(const F &) const;
	void clear();
};

class linear_poly {
public:
	F a, b;
	linear_poly();
	linear_poly(const F &, const F &);
	linear_poly(const F &);
	linear_poly operator + (const linear_poly &) const;
	quadratic_poly operator * (const linear_poly &) const;
	linear_poly operator * (const F &) const;
	F eval(const F &) const;
	void clear();
};
