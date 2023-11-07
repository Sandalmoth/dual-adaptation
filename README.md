# dual-adaptation
Simulation code associated with [...]

## Dependencies
environment.yml contains a conda environment definition with the required packages for building on linux. You also need [Cpptoml](https://github.com/skystrife/cpptoml). 

## Build instructions
The stochastic simulator is written in C++ and needs to be compiled. It's set up to use the CMAKE build system.
```bash
cd code
mkdir bin && cd bin
cmake ..
make
```

## Usage
The simulation workflow is automated with snakemake. For instance
```bash
mkdir results/figures
./switch-config.sh 3
snakemake results/figures/gradual-up-3.default-down-3.abc-fit.pdf
```
will use abc to fit parameters to the data/raw_internal/gradual-up-3.csv and data/raw_internal/default-down-3.csv, and produce a report on how the fitting went. Intermediate files are kept such that running subsequent commands that depend on the abc fitting won't have to redo it. The switch-config program switches between config files that need to be set up to match the input.

Available outputs that snakemake can create are:
```
results/figures/up.down.verify.logistic.pdf
results/figures/up.down.verify.static.pdf
results/figures/up.down.abc-diagnostic.pdf
results/figures/up.down.abc-fit.pdf
results/figures/up.down.mpi.pdf
results/figures/up.down.holiday-processed.pdf
results/figures/up.down.holiday-input.pdf
results/figures/up.down.holiday.pdf
```

It is likely necessary to manually run some of the commands relating to mpi.pdf and holiday.pdf, as the computations are very heavy and intended to run on a cluster. Examine the output of snakemake -n -r [output] for the commands, and do that part manually on a cluster.

A more extensive overview of how to use the code is provided in `notebooks/example.ipynb`
