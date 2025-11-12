# ==== Serial IMU sensor module (new) ====
file(GLOB_RECURSE OVSENSORS_LIBRARY_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors/**/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors/*.cpp"
)
list(APPEND APP_SRC ${OVSENSORS_LIBRARY_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors/)

add_executable(app app.cpp ${APP_SRC})
target_link_libraries(app ${PROJECT_NAME} ${thirdparty_libraries})
target_compile_definitions(app PRIVATE PROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../")
    