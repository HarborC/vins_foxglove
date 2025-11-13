# ==== Serial IMU sensor module (new) ====
file(GLOB_RECURSE OVSENSORS_LIBRARY_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors/**/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors/*.cpp"
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors/)

add_library(ov_sensors SHARED ${OVSENSORS_LIBRARY_SOURCES})
target_link_libraries(ov_sensors ${PROJECT_NAME} ${thirdparty_libraries})

add_executable(app app.cpp)
target_link_libraries(app ${PROJECT_NAME} ${thirdparty_libraries} ov_sensors)
target_compile_definitions(app PRIVATE PROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../")

add_executable(pose_eval pose_eval.cpp)
target_link_libraries(pose_eval ${PROJECT_NAME} ${thirdparty_libraries} ov_sensors)
target_compile_definitions(pose_eval PRIVATE PROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../")