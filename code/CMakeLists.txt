cmake_minimum_required(VERSION 3.0.0)

add_executable(dap-mpi main_dap_mpi.cpp)
add_executable(statics-mpi main_statics_mpi.cpp)

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(dap-mpi ${MPI_C_LIBRARIES})
target_link_libraries(statics-mpi ${MPI_C_LIBRARIES})

find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
target_link_libraries(dap-mpi ${HDF5_LIBRARIES})
target_link_libraries(dap-mpi ${HDF5_CXX_LIBRARIES} "${CMAKE_ROOT}/../../lib/libhdf5_cpp.a" "${CMAKE_ROOT}/../../lib/libhdf5.a")
target_link_libraries(statics-mpi ${HDF5_LIBRARIES})
target_link_libraries(statics-mpi ${HDF5_CXX_LIBRARIES} "${CMAKE_ROOT}/../../lib/libhdf5_cpp.a" "${CMAKE_ROOT}/../../lib/libhdf5.a")

# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -std=c++17")

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -ggdb") # debug compilation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Werror")
endif()
