#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <tclap/CmdLine.h>
#include <H5Cpp.h>

#include "cpptoml.h" // TODO Create a conda package, for now link to source dir

#include "dual_adaptation_process.h"


std::string VERSION = "0.0.2";


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


struct Arguments {
  std::string infile;
  std::string outfile;
  std::string paramfile;
};


struct Parameters {
  size_t time_points;
  size_t parameter_points;
  size_t simulations_per_time_point;
  size_t cores_per_node;
  size_t max_population_size;
  double noise_function_sigma;
  double rate_function_center;
  double rate_function_shape;
  double rate_function_width;
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

  Arguments arguments;
  Parameters parameters;

  // MPI Broadcast doesn't work too nicely with std::vectors
  // (especially vector<vector>)
  // so that's why C arrays
  double* time_axis = nullptr;
  double* parameter_axis = nullptr;
  double* parameter_density = nullptr;

  try {
    TCLAP::CmdLine cmd("MPI dual adaptation process simulator", ' ', VERSION);
		TCLAP::ValueArg<std::string> a_input_file("i", "input-file", "Path to input file",
                                              true, "", "*.hdf5", cmd);
		TCLAP::ValueArg<std::string> a_output_file("o", "output-file", "Path to output file",
                                               true, "", "*.hdf5", cmd);
		TCLAP::ValueArg<std::string> a_parameter_file("p", "parameter-file", "Path to parameter file",
                                                  true, "", "*.toml", cmd);

    cmd.parse(argc, argv);

    arguments.infile = a_input_file.getValue();
    arguments.outfile = a_output_file.getValue();
    arguments.paramfile = a_parameter_file.getValue();

  } catch (TCLAP::ArgException &e) {
    std::cerr << "TCLAP Error: " << e.error() << std::endl << "\targ: " << e.argId() << std::endl;
    return 1;
 }

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::cout << world_rank << ' ' << world_size << ' ' << std::endl;

  // Load parameter density data in one process
  if (world_rank == 0) {
    try {

      H5::H5File infile(arguments.infile, H5F_ACC_RDONLY);
      H5::Group gp_parameter_density = infile.openGroup("parameter_density");

      std::cout << "Loading time axis" << std::endl;
      H5::DataSet ds_time_axis = gp_parameter_density.openDataSet("time_axis");
      time_axis = read_vector(ds_time_axis, H5::PredType::NATIVE_DOUBLE);

      std::cout << "Loading parameter axis" << std::endl;
      H5::DataSet ds_parameter_axis = gp_parameter_density.openDataSet("parameter_axis");
      parameter_axis = read_vector(ds_parameter_axis, H5::PredType::NATIVE_DOUBLE);

      std::cout << "Loading parameter density" << std::endl;
      H5::DataSet ds_parameter_density = gp_parameter_density.openDataSet("parameter_density");
      parameter_density = read_vector_2d(ds_parameter_density, H5::PredType::NATIVE_DOUBLE, 1);

    } catch (H5::Exception &e) {
      std::cerr << "HDF5 Error:\n\t";
      e.printErrorStack();
      return 1;
    }

    // load simulation parameters from file
    // TODO add proper checking for missing parameters
    auto parameters_toml = cpptoml::parse_file(arguments.paramfile);
    parameters.time_points =
      parameters_toml->get_as<size_t>("time_points_up").value_or(-1);
    parameters.parameter_points =
      parameters_toml->get_as<size_t>("parameter_points").value_or(-1);
    parameters.simulations_per_time_point =
      parameters_toml->get_as<size_t>("mpi_simulations_per_time_point").value_or(-1);
    parameters.cores_per_node =
      parameters_toml->get_as<size_t>("mpi_cores_per_node").value_or(-1);
    parameters.max_population_size =
      parameters_toml->get_as<size_t>("mpi_max_population_size").value_or(-1);
    parameters.noise_function_sigma =
      parameters_toml->get_as<double>("mpi_noise_function_sigma").value_or(-1);
    parameters.rate_function_center =
      parameters_toml->get_as<double>("mpi_rate_function_center").value_or(-1);
    parameters.rate_function_shape =
      parameters_toml->get_as<double>("mpi_rate_function_shape").value_or(-1);
    parameters.rate_function_width =
      parameters_toml->get_as<double>("mpi_rate_function_width").value_or(-1);
  }

  // Broadcast parameter density and simulation parameters
  // Consider broadcasting one struct instead
  //   (since this happens only once, performance effects are probably negligible)
  MPI_Bcast(&parameters.time_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.parameter_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.simulations_per_time_point, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.cores_per_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.max_population_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.noise_function_sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_center, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_shape, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters.rate_function_width, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Processes that aren't rank 0 and didn't load from file need to allocate space
  if (world_rank != 0) {
    time_axis = new double[parameters.time_points];
    parameter_axis = new double[parameters.parameter_points];
    parameter_density = new double[parameters.parameter_points*parameters.time_points];
  }

  MPI_Bcast(time_axis, parameters.time_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(parameter_axis, parameters.parameter_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // Data transfer could be reduced by using scatter instead
  MPI_Bcast(parameter_density, parameters.parameter_points*parameters.time_points,
            MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Construct rate function since it is reused without further changes
  RateBeta rate(parameters.rate_function_shape,
                parameters.rate_function_center,
                parameters.rate_function_width,
                0.0, 1.0);

  // Allocate enough space for results from this mpi process
  bool *result_escaped = new bool[parameters.simulations_per_time_point*parameters.time_points/world_size];
  double *result_time = new double[parameters.simulations_per_time_point*parameters.time_points/world_size];

#pragma omp parallel for num_threads(parameters.cores_per_node)
  for (size_t i = 0; i < parameters.time_points/world_size; ++i) {
    std::cout << i << ' ' << world_rank << std::endl;
    for (size_t j = 0; j < parameters.simulations_per_time_point; ++j) {
      DAP<RateBeta> dap(rate, 2701); // TODO seed processes properly!
      dap.set_death_rate(0.2); // TODO make into parameter
      dap.add_cell(0.0); // TODO sample starting cell from distribution
      auto result = dap.simulate(parameters.max_population_size);
      result_escaped[i*parameters.simulations_per_time_point + j] = result.first;
      result_time[i*parameters.simulations_per_time_point + j] = result.second;
    }
  }

  // Free dynamic memory
  delete time_axis;
  delete parameter_axis;
  delete parameter_density;
  delete result_escaped;
  delete result_time;

  MPI_Finalize();

  // RateBeta rate(20, 0.8, 3, 0.0, 1.0);
  // DAP<RateBeta> dap(rate, 2701);
  // dap.set_death_rate(0.2);
  // dap.set_noise_sigma(0.1);
  // dap.add_cell(0.0);
  // auto result = dap.simulate(10000);
  // std::cout << result.first << '\t' << result.second << std::endl;

}
