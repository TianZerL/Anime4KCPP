project(ac VERSION 1.6.0 LANGUAGES CXX)

if(Build_C_Wrapper)

    aux_source_directory(src SOURCE)

    file(GLOB INCLUDE include/*.h)

    if(Build_Static_C_Wrapper)
        add_library(${PROJECT_NAME} STATIC ${INCLUDE} ${SOURCE})
        set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS -DAC_STATIC_DEFINE)
    else()
        add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})
    endif()

    target_compile_definitions(
        ${PROJECT_NAME} 
        PRIVATE 
            ANIME4KCPP_C_WRAPPER_VERSION="${PROJECT_VERSION}"
    )

    include(${TOP_DIR}/cmake/ThirdPartyForCWrapper.cmake)  

    set_target_properties(${PROJECT_NAME}
        PROPERTIES
        PUBLIC_HEADER "${INCLUDE};${CMAKE_CURRENT_BINARY_DIR}/ac_c_export.h"
    )

    install(
        TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
        ARCHIVE DESTINATION c_api/lib
        LIBRARY DESTINATION c_api/lib
        RUNTIME DESTINATION c_api/bin
        PUBLIC_HEADER DESTINATION c_api/include
    )

    install(
        EXPORT ${PROJECT_NAME}
        NAMESPACE Anime4KCPP::
        DESTINATION c_api/cmake
    )
    
    configure_file(${TOP_DIR}/cmake/CWrapperConfig.cmake.in acConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/acConfig.cmake DESTINATION c_api/cmake)

endif()
