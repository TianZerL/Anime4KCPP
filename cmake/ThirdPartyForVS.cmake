include_directories(${VapourSynth_SDK_PATH}/include)

if(OS_64_Bit)
    target_link_directories(${PROJECT_NAME} PRIVATE ${VapourSynth_SDK_PATH}/lib64)
else()
    target_link_directories(${PROJECT_NAME} PRIVATE ${VapourSynth_SDK_PATH}/lib32)
endif()

target_link_libraries(${PROJECT_NAME} vapoursynth)

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)
