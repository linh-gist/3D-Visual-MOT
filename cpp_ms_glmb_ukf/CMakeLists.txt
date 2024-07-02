cmake_minimum_required (VERSION 3.3)
project (cppmsglmb)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
SET( EIGEN3_INCLUDE_DIR "/media/ubuntu/2715608D71CBF6FC/eigen-3.4.0" )
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

INCLUDE_DIRECTORIES ( "./EigenRand" )
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(pybind11)
pybind11_add_module(cppmsglmb src/main.cpp)
