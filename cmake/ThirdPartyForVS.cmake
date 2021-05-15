include_directories(${VapourSynth_SDK_PATH}/include)

if(OS_64_Bit)
    find_library(VapourSynth_LIBS 
    NAMES vapoursynth 
    PATHS ${VapourSynth_SDK_PATH}/lib64 
    REQUIRED)
else()
    find_library(VapourSynth_LIBS 
    NAMES vapoursynth 
    PATHS ${VapourSynth_SDK_PATH}/lib32 
    REQUIRED)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${VapourSynth_LIBS})

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)
