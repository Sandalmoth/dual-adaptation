#include <cmath>
#include <fstream>
#include <iostream>

#include "dual_adaptation_process.h"


struct RateBeta {
  RateBeta(double s, double c, double w, double u, double m)
    : a(s*c)
    , b(s - a)
    , c(c)
    , w(w)
    , u(u)
    , m(m) {
    factor = pow(a, a) * pow(b, b) * pow(a+b, -(a + b));
  }
  double a, b, c, w, u, m;
  double factor;
  double operator()(double x) {
    if (x <= u - c*w || x >= u - (c - 1)*w)
      return 0.0;
    double y = pow(x/w - u/w + c, a) * pow(1 - (x/w - u/w + c), b);
    y = m*y/factor;
    return y;
  }
};


int main() {
  RateBeta rate(20, 0.8, 3, 0.0, 1.0);
  DAP<RateBeta> dap(rate, 2701);
  dap.set_death_rate(0.2);
  dap.set_noise_sigma(0.1);
  dap.add_cell(0.0);
  auto result = dap.simulate(10000);
  std::cout << result.first << '\t' << result.second << std::endl;
}
