add_executable(run_msckf run_msckf.cpp)
target_link_libraries(run_msckf ${PROJECT_NAME} ${thirdparty_libraries})
target_compile_definitions(run_msckf PRIVATE PROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../")

install(TARGETS run_msckf
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

add_executable(record_data record_data.cpp)
target_link_libraries(record_data ${PROJECT_NAME} ${thirdparty_libraries})
target_compile_definitions(record_data PRIVATE PROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../")

install(TARGETS record_data
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)



# add_executable(run_msckf_rs run_msckf_rs.cpp)
# target_link_libraries(run_msckf_rs ${PROJECT_NAME} ${thirdparty_libraries})
# target_compile_definitions(run_msckf_rs PRIVATE PROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../")

# install(TARGETS run_msckf_rs
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin
# )