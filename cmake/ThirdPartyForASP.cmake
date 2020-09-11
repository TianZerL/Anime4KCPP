include_directories(${AviSynthPlus_SDK_PATH}/include)

if(OS_64_Bit)
    target_link_directories(${PROJECT_NAME} PRIVATE ${AviSynthPlus_SDK_PATH}/lib/x64)
else()
    target_link_directories(${PROJECT_NAME} PRIVATE ${AviSynthPlus_SDK_PATH}/lib/x86)
endif()

target_link_libraries(${PROJECT_NAME} AviSynth)

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)
