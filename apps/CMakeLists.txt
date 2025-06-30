add_executable(run_msckf run_msckf.cpp)
target_link_libraries(run_msckf ${PROJECT_NAME} ${thirdparty_libraries})

install(TARGETS run_msckf
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

add_executable(record_data record_data.cpp)
target_link_libraries(record_data ${PROJECT_NAME} ${thirdparty_libraries})
install(TARGETS record_data
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)