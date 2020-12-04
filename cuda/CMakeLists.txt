project(CUDA_Module LANGUAGES CXX)

if(Enable_CUDA)
    if(${CMAKE_MINOR_VERSION} LESS 18)
        find_package(CUDA REQUIRED)

        set(CUDA_NVCC_FLAGS 
        "${CUDA_NVCC_FLAGS}
        -O3 -Xcompiler=-fPIC
        -gencode=arch=compute_35,code=sm_35
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_60,code=sm_60
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_70,code=sm_70
        -gencode=arch=compute_75,code=sm_75"
        )

        if(${CUDA_VERSION_MAJOR} LESS 11)
            set(CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS} -gencode=arch=compute_75,code=compute_75")
        else()
            set(CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS} -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80")
        endif()

        if(Enable_Fast_Math)
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")
        endif()

        include_directories(${TOP_DIR}/Anime4KCore/include include)
        file(GLOB SOURCE  src/*.cu)
        file(GLOB INCLUDE  include/*.cuh)

        source_group("cuda_include" FILES ${INCLUDE}) 
        source_group("cuda_source" FILES ${SOURCE}) 
        
        cuda_add_library(${PROJECT_NAME} STATIC ${INCLUDE} ${SOURCE})
    else()
        include(CheckLanguage)
        check_language(CUDA)
 
        if(MSVC)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
        else()
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-O3,-fPIC")
        endif()
        
        if(Enable_Fast_Math)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
        endif()

        aux_source_directory(src SOURCE)
        include_directories(${TOP_DIR}/Anime4KCore/include include)
        file(GLOB INCLUDE include/*.cuh)

        source_group("cuda_include" FILES ${INCLUDE}) 
        source_group("cuda_source" FILES ${SOURCE}) 

        message(STATUS "CMAKE_CUDA_COMPILER:         ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CMAKE_CUDA_COMPILER_ID:      ${CMAKE_CUDA_COMPILER_ID}")
        message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
        message(STATUS "CMAKE_CUDA_FLAGS:            ${CMAKE_CUDA_FLAGS}")
        message(STATUS "CMAKE_CUDA_ARCHITECTURES:    ${CMAKE_CUDA_ARCHITECTURES}")

        add_library(${PROJECT_NAME} STATIC ${INCLUDE} ${SOURCE})
    endif()

    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Static)

endif()
