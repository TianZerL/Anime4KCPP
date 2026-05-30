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
    project(ac_filter_avs_vs LANGUAGES CXX)

    set(FILTER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(FILTER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

    include(${DEPENDENCY_DIR}/avisynth.cmake)
    include(${DEPENDENCY_DIR}/vapoursynth.cmake)

    add_library(ac_filter_avs_vs MODULE
        ${FILTER_SOURCE_DIR}/avisynth/src/Filter.cpp
        ${FILTER_SOURCE_DIR}/vapoursynth/src/Filter.cpp
    )

    target_link_libraries(ac_filter_avs_vs PRIVATE ac ac_util_misc dep::avisynth dep::vapoursynth)

    target_compile_definitions(ac_filter_avs_vs PRIVATE
        AC_VERSION_MAJOR=${AC_VERSION_MAJOR}
        AC_VERSION_MINOR=${AC_VERSION_MINOR}
    )

    ac_check_enable_static_crt(ac_filter_avs_vs)

    install(
        TARGETS ac_filter_avs_vs
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
