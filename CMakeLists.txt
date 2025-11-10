cmake_minimum_required(VERSION 3.3)
project(vins_lib)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type: Debug or Release" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-std=c++17)
add_definitions(-DCOMPILEDWITHC17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -Wall -fPIC ${AVX_FLAGS} -Wno-dev -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -fPIC ${AVX_FLAGS} -Wno-dev -g")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -Wall -fPIC ${AVX_FLAGS} -Wno-dev -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -fPIC ${AVX_FLAGS} -Wno-dev -O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    add_compile_options(-w)
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time iostreams serialization)
find_package(Ceres REQUIRED)
find_package(spdlog REQUIRED)
find_package(foxglove_lib REQUIRED)
find_package(ethz_apriltag2 REQUIRED)
find_package(apriltag REQUIRED)
find_package(realsense2 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

list(APPEND thirdparty_libraries
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES}
    spdlog::spdlog
    realsense2
)

add_definitions(-DENABLE_ARUCO_TAGS=1)
add_definitions(-DROS_AVAILABLE=0)

# add_subdirectory(thirdparty)
add_subdirectory(src)
add_subdirectory(apps)