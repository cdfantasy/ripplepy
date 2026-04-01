cmake_minimum_required(VERSION 3.18)
project(ripplepy VERSION 0.1.0 LANGUAGES Fortran C)

# Keep all generated artifacts in the build tree.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_Fortran_STANDARD 90)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# NumPy include path is needed by f2py compilation.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _numpy_include_status
)
if(NOT _numpy_include_status EQUAL 0)
    message(FATAL_ERROR "Failed to query NumPy include directory")
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -Wall -Wextra -cpp")
    set(F77_FLAGS_RELAX "-O3 -Wall -Wextra -cpp -std=legacy -Wno-argument-mismatch")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3")
    set(F77_FLAGS_RELAX "-O3")
endif()

add_subdirectory(fortran)
add_subdirectory(python)