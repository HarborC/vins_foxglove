file(GLOB_RECURSE OVCORE_LIBRARY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ov_core/**/*.cpp"
)
list(APPEND LIBRARY_SOURCES ${OVCORE_LIBRARY_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ov_core/)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ov_core/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

file(GLOB_RECURSE OVINIT_LIBRARY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ov_init/**/*.cpp"
)
list(APPEND LIBRARY_SOURCES ${OVINIT_LIBRARY_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ov_init/)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ov_init/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

file(GLOB_RECURSE OVMSCKF_LIBRARY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ov_msckf/**/*.cpp"
)
list(APPEND LIBRARY_SOURCES ${OVMSCKF_LIBRARY_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ov_msckf/)

# ==== Serial IMU sensor module (new) ====
file(GLOB_RECURSE OVSENSORS_LIBRARY_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors/**/*.cpp"
)
list(APPEND LIBRARY_SOURCES ${OVSENSORS_LIBRARY_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors/)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors/
                DESTINATION include/${PROJECT_NAME}
                FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ov_msckf/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

add_library(${PROJECT_NAME} SHARED ${LIBRARY_SOURCES})
target_link_libraries(${PROJECT_NAME} ${thirdparty_libraries} foxglove_lib apriltag ethz_apriltag2)
target_include_directories(${PROJECT_NAME} PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ov_core>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ov_init>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ov_msckf>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ov_sensors>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        )

install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}-config.cmake
        DESTINATION share/cmake/${PROJECT_NAME}/
     )