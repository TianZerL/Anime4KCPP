include_directories(${AviSynthPlus_SDK_PATH}/include)

target_link_directories(${PROJECT_NAME} PRIVATE ${AviSynthPlus_SDK_PATH}/lib/x64)

target_link_libraries(${PROJECT_NAME} AviSynth)

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)
