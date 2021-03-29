project(Anime4KCPPCore LANGUAGES CXX)

aux_source_directory(src SOURCE)

include_directories(include)
file(GLOB INCLUDE include/*.hpp)

if(Built_in_kernel)
    add_definitions(-DBUILT_IN_KERNEL)
endif()

if(Use_Legacy_OpenCL_API)
    add_definitions(-DLEGACY_OPENCL_API)
endif()

if(Use_TBB)
    add_definitions(-DUSE_TBB)
endif()

if(Enable_Fast_Math)
    add_definitions(-DENABLE_FAST_MATH)
endif()

if(Enable_NCNN)
    add_definitions(-DENABLE_NCNN)
endif()

add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)

if(WIN32)
    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )
else()
    install(
        TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
    )
endif()
