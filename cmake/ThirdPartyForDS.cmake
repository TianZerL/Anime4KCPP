include_directories(${DirectShow_SDK_PATH})

target_link_directories(${PROJECT_NAME} PRIVATE ${DirectShow_SDK_PATH}/x64/Release)

target_link_libraries(${PROJECT_NAME} strmbase winmm)

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)