if(Enable_CUDA)
    project(CUDA_Module LANGUAGES CUDA)

    if((NOT CUDA_Auto_CC) AND (${CMAKE_MINOR_VERSION} LESS 18))
        set(CMAKE_CUDA_FLAGS 
        "${CMAKE_CUDA_FLAGS}
        -arch=sm_${CUDA_CC}
        -gencode=arch=compute_${CUDA_Minimum_CC},code=\"compute_${CUDA_Minimum_CC},compute_${CUDA_Minimum_CC}\""
        )
    endif()

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

    target_include_directories(${PROJECT_NAME} PRIVATE ${TOP_DIR}/core/include include)

    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_CUDA)

    set_target_properties(
        ${PROJECT_NAME} PROPERTIES 
            CUDA_RUNTIME_LIBRARY Static
            CUDA_STANDARD 17
            CUDA_EXTENSIONS OFF
    )

    if((NOT Build_Static_Core) OR (Build_Static_Core AND Build_Static_Core_PIC))
        set_target_properties(
            ${PROJECT_NAME} PROPERTIES 
                POSITION_INDEPENDENT_CODE True
        )
    endif()

endif()
