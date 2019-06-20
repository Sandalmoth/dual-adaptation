#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include <mpi.h>
#include <tclap/CmdLine.h>
#include <H5Cpp.h>

#include "dual_adaptation_process.h"


std::string VERSION = "0.0.0";


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


int main(int argc, char** argv) {
  // RateBeta rate(20, 0.8, 3, 0.0, 1.0);
  // DAP<RateBeta> dap(rate, 2701);
  // dap.set_death_rate(0.2);
  // dap.set_noise_sigma(0.1);
  // dap.add_cell(0.0);
  // auto result = dap.simulate(10000);
  // std::cout << result.first << '\t' << result.second << std::endl;

  Arguments arguments;

  try {
    TCLAP::CmdLine cmd("Mandelbrot drawer", ' ', VERSION);
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

  if (world_rank == 0) {
    try {

      H5::H5File infile(arguments.infile, H5F_ACC_RDONLY);

    } catch (H5::Exception &e) {
      std::cerr << "HDF5 Error:\n\t";
      e.printErrorStack();
      return 1;
    }
    // load simulation parameters from file
  }

  MPI_Finalize();

}
