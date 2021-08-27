if(Enable_CUDA)
    if(${CMAKE_MINOR_VERSION} LESS 18)
        project(CUDA_Module LANGUAGES CXX)

        find_package(CUDA REQUIRED)

        set(CUDA_NVCC_FLAGS 
        "${CUDA_NVCC_FLAGS}
        -arch=sm_${CUDA_CC}
        -gencode=arch=compute_${CUDA_Minimum_CC},code=\"compute_${CUDA_Minimum_CC},compute_${CUDA_Minimum_CC}\""
        )

        file(GLOB SOURCE  src/*.cu)
        file(GLOB INCLUDE  include/*.cuh)

        source_group("cuda_include" FILES ${INCLUDE}) 
        source_group("cuda_source" FILES ${SOURCE}) 
        
        cuda_add_library(${PROJECT_NAME} OBJECT ${INCLUDE} ${SOURCE})
    else()
        project(CUDA_Module LANGUAGES CUDA)
        
        if(Enable_Fast_Math)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
        endif()

        aux_source_directory(src SOURCE)
        file(GLOB INCLUDE include/*.cuh)

        source_group("cuda_include" FILES ${INCLUDE}) 
        source_group("cuda_source" FILES ${SOURCE}) 

        message(STATUS "CMAKE_CUDA_COMPILER:         ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CMAKE_CUDA_COMPILER_ID:      ${CMAKE_CUDA_COMPILER_ID}")
        message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

        add_library(${PROJECT_NAME} OBJECT ${INCLUDE} ${SOURCE})
    endif()

    target_include_directories(${PROJECT_NAME} PRIVATE ${TOP_DIR}/core/include include)

    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_CUDA)

    set_target_properties(
        ${PROJECT_NAME} PROPERTIES 
            CUDA_RUNTIME_LIBRARY Static
            CUDA_ARCHITECTURES "${CUDA_Minimum_CC}-virtual;${CUDA_CC}-real"
    )

    if((NOT Build_Static_Core) OR (Build_Static_Core AND Build_Static_Core_PIC))
        set_target_properties(
            ${PROJECT_NAME} PROPERTIES 
                POSITION_INDEPENDENT_CODE True
        )
    endif()

    install(
        TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
        ARCHIVE DESTINATION cuda_module/lib
        LIBRARY DESTINATION cuda_module/lib
        RUNTIME DESTINATION cuda_module/lib
    )
    
    install(EXPORT ${PROJECT_NAME} DESTINATION cuda_module/cmake)

endif()
