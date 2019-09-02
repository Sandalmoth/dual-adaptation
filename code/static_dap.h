#ifndef STATIC_DUAL_ADAPTATION_PROCESS_H_
#define STATIC_DUAL_ADAPTATION_PROCESS_H_


#include <cstddef>
#include <numeric>
#include <random>
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

  double get_parameter_mean() {
    return std::accumulate(cells.begin(), cells.end(), 0.0)/static_cast<double>(cells.size());
  }

  double get_parameter_stdev(double mean) {
    return sqrt(std::accumulate(cells.begin(), cells.end(), 0.0, [&](auto sum, auto parameter){
          return sum + (mean - parameter)*(mean - parameter);
        })/(static_cast<double>(cells.size() - 1)));
  }

public:
  SDAP (TRate rate, std::mt19937& rng)
    : rate(rate)
    , rng(rng) { }

  void set_noise_sigma(double s) { noise_sigma = s; }

  void add_cell (double parameter) { cells.push_back(parameter); }


  auto simulate(const double* time_axis, size_t time_points, double* result_mean, double* result_stdev) {

    // simulate with gillespies algorithm until
    // we have a measure for each point in time_axis

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

    size_t t_point = 0;

    while (t_point < time_points) {

      // advance time depending on total event rate (birth or death)
      double total_rate = total_growth_rate;

      time += std::exponential_distribution<double>(total_rate)(rng);

      // if we passed the current time in time_axis, record data
      // we might have skipped several time points,
      // keep saving values until we are caught up
      while (time > time_axis[t_point]) {
        result_mean[t_point] = get_parameter_mean();
        result_stdev[t_point] = get_parameter_stdev(result_mean[t_point]);
        ++t_point;
        if (t_point ==  time_points) {
          break;
        }
      }

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

    // return nothing, as the result was written directly to memory pointers
  }

};


#endif
