#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <tclap/CmdLine.h>
#include <H5Cpp.h>

#include "cpptoml.h" // TODO Create a conda package, for now link to source dir

#include "dual_adaptation_process.h"


std::string VERSION = "0.0.1";


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
};


template <typename T>
std::vector<T> read_vector(H5::DataSet &ds, H5::PredType pt) {
  // figure out size of data to be read
  H5::DataSpace sp = ds.getSpace();
  hsize_t dims;
  sp.getSimpleExtentDims(&dims);
  // construct vector with enough room for data
  std::vector<T> v(dims);
  // read data
  hsize_t start = 0;
  sp.selectHyperslab(H5S_SELECT_SET, &dims, &start);
  H5::DataSpace sp_mem(1, &dims);
  ds.read(v.data(), pt, sp_mem, sp);
  return v;
}


template <typename T>
std::vector<std::vector<T>> read_vector_2d(H5::DataSet &ds, H5::PredType pt, size_t row_dim) {
  // find size of data to be read
  H5::DataSpace sp = ds.getSpace();
  hsize_t dims[2];
  sp.getSimpleExtentDims(dims);
  size_t column_dim = (row_dim + 1) % 2;
  // allocate vector
  std::vector<std::vector<T>> v;
  v.reserve(dims[row_dim]);
  // copy row by row
  for (size_t i = 0; i < dims[row_dim]; ++i) {
    v.push_back(std::vector<T>(dims[column_dim]));
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
    ds.read(v[i].data(), pt, sp_mem, sp);
  }
  return v;
}


int main(int argc, char** argv) {
  // RateBeta rate(20, 0.8, 3, 0.0, 1.0);
  // DAP<RateBeta> dap(rate, 2701);
  // dap.set_death_rate(0.2);
  // dap.set_noise_sigma(0.1);
  // dap.add_cell(0.0);
  // auto result = dap.simulate(10000);
  // std::cout << result.first << '\t' << result.second << std::endl;

  Arguments arguments;

  std::vector<double> time_axis;
  std::vector<double> parameter_axis;
  std::vector<std::vector<double>> parameter_density;

  try {
    TCLAP::CmdLine cmd("MPI dual adaptation process simulator", ' ', VERSION);
		TCLAP::ValueArg<std::string> a_input_file("i", "input-file", "Path to input file", true, "", "*.hdf5", cmd);
		TCLAP::ValueArg<std::string> a_output_file("o", "output-file", "Path to output file", true, "", "*.hdf5", cmd);

    cmd.parse(argc, argv);

    arguments.infile = a_input_file.getValue();
    arguments.outfile = a_output_file.getValue();

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
      time_axis = read_vector<double>(ds_time_axis, H5::PredType::NATIVE_DOUBLE);

      std::cout << "Loading parameter axis" << std::endl;
      H5::DataSet ds_parameter_axis = gp_parameter_density.openDataSet("parameter_axis");
      parameter_axis = read_vector<double>(ds_parameter_axis, H5::PredType::NATIVE_DOUBLE);

      std::cout << "Loading parameter density" << std::endl;
      H5::DataSet ds_parameter_density = gp_parameter_density.openDataSet("parameter_density");
      parameter_density = read_vector_2d<double>(ds_parameter_density, H5::PredType::NATIVE_DOUBLE, 1);

    } catch (H5::Exception &e) {
      std::cerr << "HDF5 Error:\n\t";
      e.printErrorStack();
      return 1;
    }

    // load simulation parameters from file
  }

  MPI_Finalize();

}
