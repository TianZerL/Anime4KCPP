include_directories(${VapourSynth_SDK_PATH}/include)

target_link_directories(${PROJECT_NAME} PRIVATE ${VapourSynth_SDK_PATH}/lib64)

target_link_libraries(${PROJECT_NAME} VapourSynth)

include(../cmake/ThirdPartyForCore.cmake)
