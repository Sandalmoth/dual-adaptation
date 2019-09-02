#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <tclap/CmdLine.h>
#include <H5Cpp.h>

#include "cpptoml.h" // TODO Create a conda package, for now link to source dir

#include "static_dap.h"
#include "logistic_dap.h"


const std::string VERSION = "0.0.0";

const double RESULT_MEAN_FILL = 0.0;
const double RESULT_STDEV_FILL = 0.0;


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


enum class MODE { STATIC, LOGISTIC };


struct Arguments {
  std::string infile;
  std::string outfile;
  std::string paramfile;
  int verbosity;
  MODE mode;
};


struct Parameters {
  size_t time_points_up;
  size_t time_points_down;
  size_t parameter_points;
  size_t number_of_simulations;
  size_t cores_per_node;
  size_t population_size;
  double death_rate;
  double interaction_death_rate;
  double max_time;
  double noise_function_sigma;
  double rate_function_center;
  double rate_function_shape;
  double rate_function_width;
  double rate_function_max;
  double rate_function_ratio;
  double rate_function_optimum_normal;
  double rate_function_optimum_treatment;
  std::vector<long> rng_seeds;
};


double* read_vector(H5::DataSet &ds, H5::PredType pt) {
  // figure out size of data to be read
  H5::DataSpace sp = ds.getSpace();
  hsize_t dims;
  sp.getSimpleExtentDims(&dims);
  // construct vector with enough room for data
  double* v = new double[dims];
  // read data
  hsize_t start = 0;
  sp.selectHyperslab(H5S_SELECT_SET, &dims, &start);
  H5::DataSpace sp_mem(1, &dims);
  ds.read(v, pt, sp_mem, sp);
  return v;
}


double* read_vector_2d(H5::DataSet &ds, H5::PredType pt, size_t row_dim) {
  // I'm confident this can be done much better with HDF5 trickery
  // But it's been a while since I used it
  // And I don't feel like reading the manual
  // find size of data to be read
  H5::DataSpace sp = ds.getSpace();
  hsize_t dims[2];
  sp.getSimpleExtentDims(dims);
  size_t column_dim = (row_dim + 1) % 2;
  // allocate vector
  double *v = new double[dims[0]*dims[1]];
  // copy row by row
  for (size_t i = 0; i < dims[row_dim]; ++i) {
    // find area of dataset to copy
    hsize_t start[2] = {0, 0};
    if (row_dim == 0) start[0] = i;
    else start[1] = i;
    hsize_t count[2] = {1, 1};
    if (row_dim == 0) count[1] = dims[column_dim];
    else count[0] = dims[column_dim];
    // read data
    sp.selectHyperslab(H5S_SELECT_SET, count, start);
    H5::DataSpace sp_mem(1, &dims[column_dim]);
    ds.read(&v[i*dims[column_dim]], pt, sp_mem, sp);
  }
  return v;
}


int main(int argc, char** argv) {

  auto total_timer = [start = std::chrono::system_clock::now()] {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()
                                                                 - start).count();
  };

  Arguments arguments;
  Parameters parameters;

  // MPI Broadcast doesn't work too nicely with std::vectors
  // (especially vector<vector>)
  // so that's why C arrays
  double* time_axis_up = nullptr;
  double* time_axis_down = nullptr;
  double* parameter_axis = nullptr;
  double* parameter_density_up = nullptr;
  double* parameter_density_down = nullptr;

  try {
    TCLAP::CmdLine cmd("MPI DAP (static size variants) simulator", ' ', VERSION);
		TCLAP::ValueArg<std::string> a_input_file("i", "input-file", "Path to input file",
                                              true, "", "*.hdf5", cmd);
		TCLAP::ValueArg<std::string> a_output_file("o", "output-file", "Path to output file",
                                               true, "", "*.hdf5", cmd);
		TCLAP::ValueArg<std::string> a_parameter_file("p", "parameter-file", "Path to parameter file",
                                                  true, "", "*.toml", cmd);
    std::vector<std::string> allowed_modes;
    allowed_modes.push_back("static");
    allowed_modes.push_back("logistic");
    TCLAP::ValuesConstraint<std::string> vc_allowed_modes(allowed_modes);
    TCLAP::ValueArg<std::string> a_mode("m", "mode", "Simulation mode", false, "undefined", "static|logistic", cmd);
    TCLAP::MultiSwitchArg a_verbosity("v", "verbosity", "Increase verbosity of output", cmd);

    cmd.parse(argc, argv);

    arguments.infile = a_input_file.getValue();
    arguments.outfile = a_output_file.getValue();
    arguments.paramfile = a_parameter_file.getValue();
    arguments.verbosity = a_verbosity.getValue();
    std::string mode = a_mode.getValue();
    if (mode == "static") {
      arguments.mode = MODE::STATIC;
    } else if (mode == "logistic") {
      arguments.mode = MODE::LOGISTIC;
    } else {
      std::cerr << "Set simulation mode with -m" << std::endl;
      return 1;
    }

  } catch (TCLAP::ArgException &e) {
    std::cerr << "TCLAP Error: " << e.error() << std::endl << "\targ: " << e.argId() << std::endl;
    return 1;
 }

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Load parameter density data in one process
  if (world_rank == 0) {
    try {

      H5::H5File infile(arguments.infile, H5F_ACC_RDONLY);
      H5::Group gp_parameter_density = infile.openGroup("parameter_density");

      if (arguments.verbosity > 1) std::cout << "Loading time axis (up)" << std::endl;
      H5::DataSet ds_time_axis_up = gp_parameter_density.openDataSet("time_axis_up");
      time_axis_up = read_vector(ds_time_axis_up, H5::PredType::NATIVE_DOUBLE);

      if (arguments.verbosity > 1) std::cout << "Loading time axis (down)" << std::endl;
      H5::DataSet ds_time_axis_down = gp_parameter_density.openDataSet("time_axis_down");
      time_axis_down = read_vector(ds_time_axis_down, H5::PredType::NATIVE_DOUBLE);

      if (arguments.verbosity > 1) std::cout << "Loading parameter axis" << std::endl;
      H5::DataSet ds_parameter_axis = gp_parameter_density.openDataSet("parameter_axis");
      parameter_axis = read_vector(ds_parameter_axis, H5::PredType::NATIVE_DOUBLE);

      if (arguments.verbosity > 1) std::cout << "Loading parameter density (up)" << std::endl;
      H5::DataSet ds_parameter_density_up = gp_parameter_density.openDataSet("parameter_density_up");
      parameter_density_up = read_vector(ds_parameter_density_up, H5::PredType::NATIVE_DOUBLE);

      if (arguments.verbosity > 1) std::cout << "Loading parameter density (down)" << std::endl;
      H5::DataSet ds_parameter_density_down = gp_parameter_density.openDataSet("parameter_density_down");
      parameter_density_down = read_vector(ds_parameter_density_down, H5::PredType::NATIVE_DOUBLE);

    } catch (H5::Exception &e) {
      std::cerr << "HDF5 Error:\n\t";
      e.printErrorStack();
      return 1;
    }

    // load simulation parameters from file
    // TODO add proper checking for missing parameters
    auto parameters_toml = cpptoml::parse_file(arguments.paramfile);
    parameters.time_points_up =
      parameters_toml->get_as<size_t>("time_points_up").value_or(-1);
    parameters.time_points_down =
      parameters_toml->get_as<size_t>("time_points_down").value_or(-1);
    parameters.parameter_points =
      parameters_toml->get_as<size_t>("parameter_points").value_or(-1);
    parameters.number_of_simulations =
      parameters_toml->get_as<size_t>("mpi_statics_number_of_simulations").value_or(-1);
    parameters.cores_per_node =
      parameters_toml->get_as<size_t>("mpi_cores_per_node").value_or(-1);
    parameters.population_size =
      parameters_toml->get_as<size_t>("mpi_statics_population_size").value_or(-1);
    parameters.death_rate =
      parameters_toml->get_as<double>("mpi_death_rate").value_or(-1);
    parameters.interaction_death_rate =
      parameters_toml->get_as<double>("mpi_statics_interaction_death_rate").value_or(-1);
    parameters.max_time =
      parameters_toml->get_as<double>("mpi_max_time").value_or(-1);
    parameters.noise_function_sigma =
      parameters_toml->get_as<double>("mpi_noise_function_sigma").value_or(-1);
    parameters.rate_function_center =
      parameters_toml->get_as<double>("mpi_rate_function_center").value_or(-1);
    parameters.rate_function_shape =
      parameters_toml->get_as<double>("mpi_rate_function_shape").value_or(-1);
    parameters.rate_function_width =
      parameters_toml->get_as<double>("mpi_rate_function_width").value_or(-1);
    parameters.rate_function_max =
      parameters_toml->get_as<double>("mpi_rate_function_max").value_or(-1);
    parameters.rate_function_ratio =
      parameters_toml->get_as<double>("mpi_rate_function_ratio").value_or(-1);
    parameters.rate_function_optimum_normal =
      parameters_toml->get_as<double>("optimum_normal").value_or(-1);
    parameters.rate_function_optimum_treatment =
      parameters_toml->get_as<double>("optimum_treatment").value_or(-1);
    auto seeds = parameters_toml->get_array_of<long>("mpi_rng_seeds");
    parameters.rng_seeds.assign(seeds->begin(), seeds->end());

    if (parameters.rng_seeds.size() < world_size*parameters.cores_per_node) {
      std::cerr << "Not enough rng seeds. Provide at least one per thread (= MPI nodes * cores per node)\n";
      return 1;
    }
  }

  if (world_rank == 0 && arguments.verbosity)
    std::cout << total_timer() << "\tSuccessfully parsed input data" << std::endl;

  // Broadcast parameter density and simulation parameters
  // Consider broadcasting one struct instead
  //   (since this happens only once, performance effects are probably negligible)
  MPI_Bcast(&parameters.time_points_up, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.time_points_down, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.parameter_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.number_of_simulations, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.cores_per_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.population_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.death_rate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.interaction_death_rate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.max_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.noise_function_sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_center, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_shape, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_width, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_ratio, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_optimum_normal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_optimum_treatment, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  { // reduce scope of some variables
    int num_seeds;
    if (world_rank == 0) {
      num_seeds = parameters.rng_seeds.size();
    }
    MPI_Bcast(&num_seeds, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
      parameters.rng_seeds.resize(num_seeds);
    }
    MPI_Bcast(parameters.rng_seeds.data(), num_seeds, MPI_LONG, 0, MPI_COMM_WORLD);
  }

  // Processes that aren't rank 0 and didn't load from file need to allocate space
  if (world_rank != 0) {
    time_axis_up = new double[parameters.time_points_up];
    time_axis_down = new double[parameters.time_points_down];
    parameter_axis = new double[parameters.parameter_points];
    parameter_density_up = new double[parameters.parameter_points];
    parameter_density_down = new double[parameters.parameter_points];
  }

  MPI_Bcast(time_axis_up, parameters.time_points_up, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(time_axis_down, parameters.time_points_down, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(parameter_axis, parameters.parameter_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(parameter_density_up, parameters.parameter_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(parameter_density_down, parameters.parameter_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0 && arguments.verbosity)
    std::cout << total_timer() << "\tSuccessfully broadcasted input data" << std::endl;

  if (world_rank == 0 && arguments.verbosity > 1)
    std::cout << "MPI world size: " << world_size
              << "\nThreads per node: " << parameters.cores_per_node << std::endl;

  // Construct rate functions since they are reused without further changes
  RateBeta rate_up(parameters.rate_function_shape,
                   parameters.rate_function_center,
                   parameters.rate_function_width,
                   parameters.rate_function_optimum_treatment,
                   parameters.rate_function_max*parameters.rate_function_ratio);
  RateBeta rate_down(parameters.rate_function_shape,
                     parameters.rate_function_center,
                     parameters.rate_function_width,
                     parameters.rate_function_optimum_normal,
                     parameters.rate_function_max);

  // Allocate enough space for results from this mpi process
  double* result_mean_up = new double[parameters.number_of_simulations*parameters.time_points_up/world_size];
  double* result_mean_down = new double[parameters.number_of_simulations*parameters.time_points_down/world_size];
  double* result_stdev_up = new double[parameters.number_of_simulations*parameters.time_points_up/world_size];
  double* result_stdev_down = new double[parameters.number_of_simulations*parameters.time_points_down/world_size];

#pragma omp parallel num_threads(parameters.cores_per_node)
  {
    // Seed the rng for this thread
    int thread_rank = omp_get_thread_num();
    std::mt19937 rng; // TODO read seeds from file in some way
    std::vector<long> seeds;
    for (size_t i = thread_rank + world_rank*world_size;
         i < parameters.rng_seeds.size();
         i += parameters.cores_per_node*world_size) {
      seeds.push_back(parameters.rng_seeds[i]);
    }
    std::seed_seq rng_seed(seeds.begin(), seeds.end());
    rng.seed(rng_seed);

    // Set up parameter distribution for simulations
    std::piecewise_linear_distribution<double>
      parameter_distribution_up(parameter_axis,
                                parameter_axis + parameters.parameter_points,
                                parameter_density_up);
    std::piecewise_linear_distribution<double>
      parameter_distribution_down(parameter_axis,
                                parameter_axis + parameters.parameter_points,
                                parameter_density_down);

#pragma omp for schedule(static)
    for (size_t i = 0; i < parameters.number_of_simulations/world_size; ++i) {

      // TODO Consider using polymorphism to reduce code repetition
      switch (arguments.mode) {
      case MODE::STATIC:
        {
          SDAP<RateBeta> sdap_up(rate_up, rng);
          sdap_up.set_noise_sigma(parameters.noise_function_sigma);
          for (size_t j = 0; j < parameters.population_size; ++j) {
            double first_parameter = parameter_distribution_up(rng);
            sdap_up.add_cell(first_parameter);
          }
          sdap_up.simulate(time_axis_up, parameters.time_points_up,
                           result_mean_up + parameters.time_points_up*i,
                           result_stdev_up + parameters.time_points_up*i);
          SDAP<RateBeta> sdap_down(rate_down, rng);
          sdap_down.set_noise_sigma(parameters.noise_function_sigma);
          for (size_t j = 0; j < parameters.population_size; ++j) {
            double first_parameter = parameter_distribution_down(rng);
            sdap_down.add_cell(first_parameter);
          }
          sdap_down.simulate(time_axis_down, parameters.time_points_down,
                           result_mean_down + parameters.time_points_down*i,
                           result_stdev_down + parameters.time_points_down*i);
        }
        break;
      case MODE::LOGISTIC:
        break;
      default:
        std::cerr << "Unknown mode. Set a mode with -m" << std::endl;
        break;
      }
    }
  } // end omp segment

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0 && arguments.verbosity)
    std::cout << total_timer() << "\tSimulations finished" << std::endl;

  // TODO use parallel HDF5? maybe h5cpp library?
  if (world_rank == 0) {
    // one process creates the file and sets up the datasets
    // TODO abstract into function
    H5::H5File outfile(arguments.outfile, H5F_ACC_TRUNC);
    H5::Group gp_result = outfile.createGroup("result");
    hsize_t dims_up[2] = {parameters.time_points_up, parameters.number_of_simulations};
    H5::DataSpace sp_up(2, dims_up);
    H5::DSetCreatPropList pl_mean_up;
    H5::DSetCreatPropList pl_stdev_up;
    pl_mean_up.setFillValue(H5::PredType::NATIVE_DOUBLE, &RESULT_MEAN_FILL);
    pl_stdev_up.setFillValue(H5::PredType::NATIVE_DOUBLE, &RESULT_STDEV_FILL);
    pl_mean_up.setDeflate(5);
    pl_stdev_up.setDeflate(5);
    hsize_t chunk_dims_up[2] {parameters.time_points_up,
        parameters.number_of_simulations/world_size/parameters.cores_per_node};
    pl_mean_up.setChunk(2, chunk_dims_up);
    pl_stdev_up.setChunk(2, chunk_dims_up);
    gp_result.createDataSet("mean_up", H5::PredType::NATIVE_DOUBLE, sp_up, pl_mean_up);
    gp_result.createDataSet("stdev_up", H5::PredType::NATIVE_DOUBLE, sp_up, pl_stdev_up);
    hsize_t dims_down[2] = {parameters.time_points_down, parameters.number_of_simulations};
    H5::DataSpace sp_down(2, dims_down);
    H5::DSetCreatPropList pl_mean_down;
    H5::DSetCreatPropList pl_stdev_down;
    pl_mean_down.setFillValue(H5::PredType::NATIVE_DOUBLE, &RESULT_MEAN_FILL);
    pl_stdev_down.setFillValue(H5::PredType::NATIVE_DOUBLE, &RESULT_STDEV_FILL);
    pl_mean_down.setDeflate(5);
    pl_stdev_down.setDeflate(5);
    hsize_t chunk_dims_down[2] {parameters.time_points_down,
        parameters.number_of_simulations/world_size/parameters.cores_per_node};
    pl_mean_down.setChunk(2, chunk_dims_down);
    pl_stdev_down.setChunk(2, chunk_dims_down);
    gp_result.createDataSet("mean_down", H5::PredType::NATIVE_DOUBLE, sp_down, pl_mean_down);
    gp_result.createDataSet("stdev_down", H5::PredType::NATIVE_DOUBLE, sp_down, pl_stdev_down);
  }

  if (world_rank == 0 && arguments.verbosity)
    std::cout << total_timer() << "\tCreated output file" << std::endl;

  try {

    MPI_Barrier(MPI_COMM_WORLD);

    // Iterate over all MPI processes taking turns to write result
    for (int i = 0; i < world_size; ++i) {
      if (world_rank == i) {

        H5::H5File outfile(arguments.outfile, H5F_ACC_RDWR);
        H5::Group gp_result = outfile.openGroup("result");
        H5::DataSet ds_mean_up = gp_result.openDataSet("mean_up");
        H5::DataSet ds_stdev_up = gp_result.openDataSet("stdev_up");
        H5::DataSet ds_mean_down = gp_result.openDataSet("mean_down");
        H5::DataSet ds_stdev_down = gp_result.openDataSet("stdev_down");
        H5::DataSpace sp_mean_up = ds_mean_up.getSpace();
        H5::DataSpace sp_stdev_up = ds_stdev_up.getSpace();
        H5::DataSpace sp_mean_down = ds_mean_down.getSpace();
        H5::DataSpace sp_stdev_down = ds_stdev_down.getSpace();
        for (size_t j = 0; j < parameters.number_of_simulations/world_size; ++j) {
          {
            hsize_t start[2] {0, world_rank*parameters.number_of_simulations/world_size + j};
            hsize_t count[2] {parameters.time_points_up, 1};
            sp_mean_up.selectHyperslab(H5S_SELECT_SET, count, start);
            sp_stdev_up.selectHyperslab(H5S_SELECT_SET, count, start);
          }
          hsize_t dims_mem[1] = {parameters.number_of_simulations*parameters.time_points_up/world_size};
          hsize_t start[1] = {j*parameters.time_points_up};
          hsize_t count[1] = {parameters.time_points_up};
          H5::DataSpace sp_mem(1, dims_mem);
          sp_mem.selectHyperslab(H5S_SELECT_SET, count, start);
          ds_mean_up.write(result_mean_up, H5::PredType::NATIVE_DOUBLE, sp_mem, sp_mean_up);
          ds_stdev_up.write(result_stdev_up, H5::PredType::NATIVE_DOUBLE, sp_mem, sp_stdev_up);
        }
        for (size_t j = 0; j < parameters.number_of_simulations/world_size; ++j) {
          {
            hsize_t start[2] {0, world_rank*parameters.number_of_simulations/world_size + j};
            hsize_t count[2] {parameters.time_points_down, 1};
            sp_mean_down.selectHyperslab(H5S_SELECT_SET, count, start);
            sp_stdev_down.selectHyperslab(H5S_SELECT_SET, count, start);
          }
          hsize_t dims_mem[1] = {parameters.number_of_simulations*parameters.time_points_down/world_size};
          hsize_t start[1] = {j*parameters.time_points_down};
          hsize_t count[1] = {parameters.time_points_down};
          H5::DataSpace sp_mem(1, dims_mem);
          sp_mem.selectHyperslab(H5S_SELECT_SET, count, start);
          ds_mean_down.write(result_mean_down, H5::PredType::NATIVE_DOUBLE, sp_mem, sp_mean_down);
          ds_stdev_down.write(result_stdev_down, H5::PredType::NATIVE_DOUBLE, sp_mem, sp_stdev_down);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

  } catch (H5::Exception &e) {
    std::cerr << "HDF5 Error:\n\t";
    e.printErrorStack();
    return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0 && arguments.verbosity)
    std::cout << total_timer() << "\tWrote result" << std::endl;

  // Free dynamic memory
  delete time_axis_up;
  delete time_axis_down;
  delete parameter_axis;
  delete parameter_density_up;
  delete parameter_density_down;
  delete result_mean_up;
  delete result_stdev_up;
  delete result_mean_down;
  delete result_stdev_down;

  MPI_Finalize();

}
