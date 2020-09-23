include_directories(${AviSynthPlus_SDK_PATH}/include)

if(OS_64_Bit)
    find_library(AviSynthPlus_LIBS 
    NAMES AviSynth 
    PATHS ${AviSynthPlus_SDK_PATH}/lib/x64 
    NO_DEFAULT_PATH REQUIRED)
else()
    find_library(AviSynthPlus_LIBS 
    NAMES AviSynth 
    PATHS ${AviSynthPlus_SDK_PATH}/lib/x86 
    NO_DEFAULT_PATH REQUIRED)
endif()

target_link_libraries(${PROJECT_NAME} ${AviSynthPlus_LIBS})

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)
