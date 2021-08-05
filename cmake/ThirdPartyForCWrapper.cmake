find_package(OpenCL REQUIRED)

include_directories(${TOP_DIR}/ThirdParty/include ${OpenCL_INCLUDE_DIRS})

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)
