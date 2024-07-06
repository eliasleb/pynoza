cmake_minimum_required(VERSION 3.27)
project(pynoza C)

set(CMAKE_C_STANDARD 11)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
add_library(integration_comparison MODULE tests/integration_comparison.c)
target_compile_options(integration_comparison PRIVATE -Wall -Wextra)
target_include_directories(integration_comparison PRIVATE ${Python3_INCLUDE_DIRS})

target_link_libraries(integration_comparison PRIVATE ${Python3_LIBRARIES})

target_compile_definitions(integration_comparison PRIVATE PYBIND11_MODULE_NAME=integration_comparison)
target_link_libraries(integration_comparison PRIVATE pybind11::module)

set_target_properties(integration_comparison PROPERTIES PREFIX "")
set_target_properties(integration_comparison PROPERTIES OUTPUT_NAME "integration_comparison")
set_target_properties(integration_comparison PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS integration_comparison DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
