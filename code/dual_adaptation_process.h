#ifndef __DUAL_ADAPTATION_PROCESS_H__
#define __DUAL_ADAPTATION_PROCESS_H__


#include <cstddef>
#include <random>
#include <vector>
#include <utility>


template <typename TRate>
class DAP {
private:
  std::mt19937 rng;

  double death_rate = 0.0;
  double noise_sigma = 0.0;

  std::vector<double> cells;

  TRate rate;


public:
  DAP (TRate rate)
    : rate(rate) {
    std::random_device rd;
    rng.seed(rd());
  }
  DAP (TRate rate, typename decltype(rng)::result_type seed)
    : rate(rate) {
    rng.seed(seed);
  }


  void set_death_rate(double dr) { death_rate = dr; }

  void set_noise_sigma(double s) { noise_sigma = s; }

  void add_cell (double parameter) { cells.push_back(parameter); }


  std::pair<bool, double> simulate(size_t n_end) {
    auto noise = [&]() {
      return std::normal_distribution<double>(0.0, noise_sigma)(rng);
    };



    return std::make_pair(false, noise());
  }

};


#endif
