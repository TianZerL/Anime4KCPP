project(Anime4KCPPCore LANGUAGES CXX)

aux_source_directory(src SOURCE)

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

if(Use_Eigen3)
    add_definitions(-DUSE_EIGEN3)
endif(Use_Eigen3)

if(Enable_Video)
    add_definitions(-DENABLE_VIDEO)
endif()

if(Enable_Preview_GUI)
    add_definitions(-DENABLE_PREVIEW_GUI)
endif()

if(Enable_Image_IO)
    add_definitions(-DENABLE_IMAGE_IO)
endif()

if(Build_Static_Core)
    add_library(${PROJECT_NAME} STATIC ${INCLUDE} ${SOURCE})
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS -DAC_STATIC_DEFINE)
else()
    add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})
endif()

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
    ARCHIVE DESTINATION core/lib
    LIBRARY DESTINATION core/lib
    RUNTIME DESTINATION bin
)

install(
    FILES ${INCLUDE} ${CMAKE_CURRENT_BINARY_DIR}/ac_export.h
    DESTINATION core/include
)

install(EXPORT ${PROJECT_NAME} DESTINATION core/cmake)

configure_file(${TOP_DIR}/cmake/Anime4KCPPCoreConfig.cmake.in Anime4KCPPCoreConfig.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Anime4KCPPCoreConfig.cmake DESTINATION core/cmake)
