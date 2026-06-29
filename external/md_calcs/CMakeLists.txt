cmake_minimum_required(VERSION 3.14)
project(md_calcs)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# Build the extension directly from the cpp at this level.
pybind11_add_module(md_calcs_par md_calcs_parallel.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(md_calcs_par PRIVATE OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP support not found.")
endif()

# Drop the compiled .so next to the source so `from external.md_calcs import md_calcs_par` works.
set_target_properties(md_calcs_par PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
