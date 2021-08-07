if(Enable_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_Minimum_CC}-virtual ${CUDA_CC}-real)

    if(${CMAKE_MINOR_VERSION} LESS 18)
        project(CUDA_Module LANGUAGES CXX)

        find_package(CUDA REQUIRED)

        set(CUDA_NVCC_FLAGS 
        "${CUDA_NVCC_FLAGS}
        -arch=sm_${CUDA_CC}
        -gencode=arch=compute_${CUDA_Minimum_CC},code=\"compute_${CUDA_Minimum_CC},compute_${CUDA_Minimum_CC}\""
        )

        if(MSVC)
            set(CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS} -O3")
        else()
            set(CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS} -Xcompiler=-O3,-fPIC")
        endif()

        if(Enable_Fast_Math)
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")
        endif()

        include_directories(${TOP_DIR}/core/include include)
        file(GLOB SOURCE  src/*.cu)
        file(GLOB INCLUDE  include/*.cuh)

        source_group("cuda_include" FILES ${INCLUDE}) 
        source_group("cuda_source" FILES ${SOURCE}) 
        
        cuda_add_library(${PROJECT_NAME} STATIC ${INCLUDE} ${SOURCE})
    else()
        project(CUDA_Module LANGUAGES CUDA) 
 
        if(MSVC)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
        else()
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-O3,-fPIC")
        endif()
        
        if(Enable_Fast_Math)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
        endif()

        aux_source_directory(src SOURCE)
        include_directories(${TOP_DIR}/core/include include)
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

    install(
        TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
        ARCHIVE DESTINATION cuda_module/lib
        LIBRARY DESTINATION cuda_module/lib
        RUNTIME DESTINATION cuda_module/lib
    )
    
    install(EXPORT ${PROJECT_NAME} DESTINATION cuda_module/cmake)

endif()
