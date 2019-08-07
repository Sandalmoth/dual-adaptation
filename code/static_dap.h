#ifndef STATIC_DUAL_ADAPTATION_PROCESS_H_
#define STATIC_DUAL_ADAPTATION_PROCESS_H_


#include <cstddef>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>
#include <utility>


// The static DAP synchronously kills a cell with every birth
// thus keeping a perfectly constant population size


template <typename TRate>
class SDAP {
private:
  TRate rate; // functor implementing operator()(double) -> double
  std::mt19937& rng;

  double noise_sigma = 0.0;

  std::vector<double> cells;


public:
  SDAP (TRate rate, std::mt19937& rng)
    : rate(rate)
    , rng(rng) { }

  void set_noise_sigma(double s) { noise_sigma = s; }

  void add_cell (double parameter) { cells.push_back(parameter); }


  auto simulate(size_t n_end, double t_end) {

    // simulate with gillespies algorithm until
    // we reach n_end cells total

    std::normal_distribution<double> noise(0.0, noise_sigma);

    // precalculate growth rates for all cells
    std::vector<double> growth_rates;
    growth_rates.reserve(cells.size());
    for (auto& cell: cells) {
      growth_rates.push_back(rate(cell));
    }
    // maintain running sum of growth rates
    // otherwise we would have to sum at the start of each step
    double total_growth_rate = std::accumulate(growth_rates.begin(),
                                               growth_rates.end(),
                                               0.0);

    double time = 0;
    size_t max_cells = cells.size();

    while (cells.size() > 0 && cells.size() < n_end && time < t_end) {

      // advance time depending on total event rate (birth or death)
      double total_rate = total_growth_rate;

      time += std::exponential_distribution<double>(total_rate)(rng);

      // select event cell proportional to birth rates
      int event_cell = 0;
      double rate_cumulative = 0.0;
      double rate_cumulative_target = std::uniform_real_distribution<double>(0, total_growth_rate)(rng);
      for (size_t i = 0; i < cells.size(); ++i) {
        rate_cumulative += growth_rates[i];
        if (rate_cumulative > rate_cumulative_target) {
          event_cell = i;
          break;
        }
      }

      // Produce a new cell as a mutant of the old one
      double new_cell = cells[event_cell] + noise(rng);
      double new_cell_rate = rate(new_cell);
      // maintain running growth rate sum
      total_growth_rate += new_cell_rate;
      // add to list
      cells.push_back(new_cell);
      growth_rates.push_back(new_cell_rate);
      // update max cells if we broke the record
      if (cells.size() > max_cells)
        max_cells = cells.size();

      // death event

      // select random cell
      // (proportional to death rates, but the rates are all equal)
      event_cell = std::uniform_int_distribution<int>(0, cells.size() - 1)(rng);

      // minimize vector shuffling operations by swapping
      // to end before removing the dying cell
      std::swap(cells[event_cell], cells.back());
      std::swap(growth_rates[event_cell], growth_rates.back());
      // maintain running growth rate sum
      total_growth_rate -= growth_rates.back();
      // remove from list
      cells.pop_back();
      growth_rates.pop_back();

    }

    // return if we reach n_end cells, the time it took, and the greatest number of cells reached
    return std::make_tuple(cells.size() == n_end, time, max_cells);
  }

};


#endif
