cmake_minimum_required(VERSION 3.16)
project(neuro_processor_systemc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(SYSTEMC_INCLUDE_DIR systemc
  HINTS "$ENV{SYSTEMC_HOME}/include" /usr/include /usr/local/include)
find_library(SYSTEMC_LIBRARY systemc
  HINTS "$ENV{SYSTEMC_HOME}/lib-linux64" "$ENV{SYSTEMC_HOME}/lib" /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT SYSTEMC_INCLUDE_DIR OR NOT SYSTEMC_LIBRARY)
  message(FATAL_ERROR "SystemC not found. Set SYSTEMC_HOME or SYSTEMC_INCLUDE_DIR/SYSTEMC_LIBRARY.")
endif()

file(GLOB SOURCES CONFIGURE_DEPENDS "src/*.cpp")
add_executable(neuro_accelerator ${SOURCES})
target_include_directories(neuro_accelerator PRIVATE include "${SYSTEMC_INCLUDE_DIR}")
target_link_libraries(neuro_accelerator PRIVATE "${SYSTEMC_LIBRARY}")
