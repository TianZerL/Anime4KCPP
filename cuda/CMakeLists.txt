project(CUDA_Module)

if(Enable_CUDA)
    find_package(CUDA)
    set(CUDA_NVCC_FLAGS -O3)

    include_directories(${TOP_DIR}/Anime4KCore/include include)
    file(GLOB SOURCE  src/*.cu)
    file(GLOB INCLUDE  include/*.cuh)

    source_group("cuda_include" FILES ${INCLUDE}) 
    source_group("cuda_source" FILES ${SOURCE}) 
    
    cuda_add_library(CUDA_Module STATIC ${INCLUDE} ${SOURCE})
endif()
