if(AC_BUILD_FILTER_VAPOURSYNTH)
    add_subdirectory(vapoursynth)
endif()
if(AC_BUILD_FILTER_AVISYNTH)
    add_subdirectory(avisynth)
endif()

if(AC_BUILD_FILTER_AVISYNTH_VAPOURSYNTH)
    project(ac_filter_avs_vs VERSION 3.0.0.0 LANGUAGES CXX)

    set(FILTER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(FILTER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

    include(${DEPENDENCY_DIR}/avisynth.cmake)
    include(${DEPENDENCY_DIR}/vapoursynth.cmake)

    add_library(ac_filter_avs_vs SHARED
        ${FILTER_SOURCE_DIR}/avisynth/src/Filter.cpp
        ${FILTER_SOURCE_DIR}/vapoursynth/src/Filter.cpp
    )

    target_link_libraries(ac_filter_avs_vs PRIVATE ac dep::avisynth dep::vapoursynth)

    install(
        TARGETS ac_filter_avs_vs
        RUNTIME DESTINATION filter
    )
endif()
