if((NOT MSVC) AND Other_Optimization)
    target_compile_options(${PROJECT_NAME} PRIVATE -Ofast)
    target_link_options(${PROJECT_NAME} PRIVATE -Ofast)
endif()
