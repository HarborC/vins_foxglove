# 最低 CMake 版本要求
cmake_minimum_required(VERSION 3.3)
project(ov_msckf)

# -------------------------------------------------------------------
# 1. 设置构建类型（默认为 Release）
# -------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type: Debug or Release" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# -------------------------------------------------------------------
# 2. 全局 C++ 标准设置
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 3. 查找依赖库：Eigen, OpenCV, Boost, Ceres, PCL
# -------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time iostreams serialization)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(spdlog REQUIRED)

message(STATUS "OPENCV: ${OpenCV_VERSION}  |  BOOST: ${Boost_VERSION}  |  CERES: ${Ceres_VERSION}")

# -------------------------------------------------------------------
# 4. 可选功能开关
# -------------------------------------------------------------------
option(ENABLE_ROS "Enable building with ROS if found" ON)

option(ENABLE_ARUCO_TAGS "Enable aruco tag tracking (disable if no contrib modules)" ON)
if (NOT ENABLE_ARUCO_TAGS)
    add_definitions(-DENABLE_ARUCO_TAGS=0)
    message(WARNING "DISABLING ARUCO TAG TRACKING!")
else()
    add_definitions(-DENABLE_ARUCO_TAGS=1)
endif()

# -------------------------------------------------------------------
# 6. 按需查找 ROS1/ROS2 并包含对应配置文件
# -------------------------------------------------------------------
find_package(catkin QUIET COMPONENTS roscpp)
find_package(ament_cmake QUIET)

if (catkin_FOUND AND ENABLE_ROS)
    message(STATUS "ROS1 detected, including ROS1.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
elseif (ament_cmake_FOUND AND ENABLE_ROS)
    message(STATUS "ROS2 detected, including ROS2.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS2.cmake)
else()
    message(STATUS "No ROS found or ROS support disabled -> using ROS1.cmake without catkin")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
endif()