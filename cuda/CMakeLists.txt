project(CUDA_Module)

if(Enable_CUDA)
    find_package(CUDA REQUIRED)

    set(CUDA_NVCC_FLAGS 
    "-O3"
    "-gencode=arch=compute_35,code=sm_35"
    "-gencode=arch=compute_50,code=sm_50"
    "-gencode=arch=compute_52,code=sm_52"
    "-gencode=arch=compute_60,code=sm_60"
    "-gencode=arch=compute_61,code=sm_61"
    "-gencode=arch=compute_70,code=sm_70"
    "-gencode=arch=compute_75,code=sm_75"
    "-gencode=arch=compute_80,code=sm_80"
    "-gencode=arch=compute_80,code=compute_80"
    )

    if(Enable_Fast_Math)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")
    endif()

    include_directories(${TOP_DIR}/Anime4KCore/include include)
    file(GLOB SOURCE  src/*.cu)
    file(GLOB INCLUDE  include/*.cuh)

    source_group("cuda_include" FILES ${INCLUDE}) 
    source_group("cuda_source" FILES ${SOURCE}) 
    
    cuda_add_library(${PROJECT_NAME} STATIC ${INCLUDE} ${SOURCE})
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
