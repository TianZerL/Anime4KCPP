if(AC_BUILD_FILTER_AVISYNTH)
    add_subdirectory(avisynth)
endif()
if(AC_BUILD_FILTER_VAPOURSYNTH)
    add_subdirectory(vapoursynth)
endif()
if(AC_BUILD_FILTER_DIRECTSHOW)
    add_subdirectory(directshow)
endif()

if(AC_BUILD_FILTER_AVISYNTH_VAPOURSYNTH)
    project(ac_filter_avs_vs VERSION 3.1.0.0 LANGUAGES CXX)

    set(FILTER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(FILTER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

    include(${DEPENDENCY_DIR}/avisynth.cmake)
    include(${DEPENDENCY_DIR}/vapoursynth.cmake)

    add_library(ac_filter_avs_vs MODULE
        ${FILTER_SOURCE_DIR}/avisynth/src/Filter.cpp
        ${FILTER_SOURCE_DIR}/vapoursynth/src/Filter.cpp
    )

    target_link_libraries(ac_filter_avs_vs PRIVATE ac dep::avisynth dep::vapoursynth)

    ac_check_enable_static_crt(ac_filter_avs_vs)

    install(
        TARGETS ac_filter_avs_vs
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
