project(CUDA_Module)

if(Enable_CUDA)
    find_package(CUDA)
    set(CUDA_NVCC_FLAGS -O3)

    include_directories(${TOP_DIR}/Anime4KCore/include)
    file(GLOB SOURCE  *.cu)
    file(GLOB INCLUDE  *.cuh)
    source_group("cuda_include" FILES ${INCLUDE}) 
    source_group("cuda_source" FILES ${SOURCE}) 
    cuda_add_library(CUDA_Module STATIC ${INCLUDE} ${SOURCE})
endif()