project(Anime4KCPPCore LANGUAGES CXX)

aux_source_directory(src SOURCE)

include_directories(include)
file(GLOB INCLUDE include/*.hpp)

if(Built_In_Kernel)
    add_definitions(-DBUILT_IN_KERNEL)
endif()

if(Use_Legacy_OpenCL_API)
    add_definitions(-DLEGACY_OPENCL_API)
endif()

if(Enable_Fast_Math)
    add_definitions(-DENABLE_FAST_MATH)
endif()

if(Enable_NCNN)
    add_definitions(-DENABLE_NCNN)
endif()

if(Enable_OpenCV_DNN)
    add_definitions(-DENABLE_OPENCV_DNN)
endif()

if(Enable_Video)
    add_definitions(-DENABLE_VIDEO)
endif()

if(Enable_Preview_GUI)
    add_definitions(-DENABLE_PREVIEW_GUI)
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
