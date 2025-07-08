if(NOT TARGET dep::directshow)
    add_library(dep_directshow INTERFACE IMPORTED)
    if(AC_PATH_DIRECTSHOW_BASECLASSES)
        set(dep_directshow_PATH ${AC_PATH_DIRECTSHOW_BASECLASSES})
        if(BUILD_ARCH_32BIT)
            set(dep_directshow_ARCH_DIR "")
        else()
            set(dep_directshow_ARCH_DIR "x64")
        endif()

        find_path(dep_directshow_INCLUDE
            NAMES streams.h
            PATHS ${dep_directshow_PATH}
        )
        find_library(dep_directshow_STRMBASE_RELEASE
            NAMES strmbase
            PATHS
                ${dep_directshow_PATH}/${dep_directshow_ARCH_DIR}/Release_MBCS
                ${dep_directshow_PATH}/${dep_directshow_ARCH_DIR}/Release
        )
        find_library(dep_directshow_STRMBASE_DEBUG
            NAMES strmbasd
            PATHS
                ${dep_directshow_PATH}/${dep_directshow_ARCH_DIR}/Debug_MBCS
                ${dep_directshow_PATH}/${dep_directshow_ARCH_DIR}/Debug
        )

        if (dep_directshow_INCLUDE AND (dep_directshow_STRMBASE_RELEASE OR dep_directshow_STRMBASE_DEBUG))
            target_include_directories(dep_directshow INTERFACE $<BUILD_INTERFACE:${dep_directshow_INCLUDE}>)
            target_link_libraries(dep_directshow INTERFACE $<IF:$<CONFIG:Debug>,${dep_directshow_STRMBASE_DEBUG},${dep_directshow_STRMBASE_RELEASE}> winmm)
        elseif(dep_directshow_INCLUDE)
            set(directshow_baseclasses_SOURCE_DIR ${dep_directshow_INCLUDE})
            set(dep_directshow_baseclasses_BUILD TRUE)
        else()
            set(dep_directshow_baseclasses_FETCH TRUE)
        endif()
    else()
        set(dep_directshow_baseclasses_FETCH TRUE)
    endif()

    if(dep_directshow_baseclasses_FETCH)
        message(STATUS "dep: directshow baseclasses not found, will be fetched online.")
        include(FetchContent)
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            FetchContent_Declare(
                directshow
                GIT_REPOSITORY https://github.com/microsoft/Windows-classic-samples.git
                GIT_TAG main
            )
            FetchContent_MakeAvailable(directshow)
            set(directshow_baseclasses_SOURCE_DIR ${directshow_SOURCE_DIR}/Samples/Win7Samples/multimedia/directshow/baseclasses)
            set(dep_directshow_baseclasses_BUILD TRUE)
        else() # For non MSVC compilers
            FetchContent_Declare(
                directshow
                GIT_REPOSITORY https://github.com/TianZerL/DirectShow-BaseClasses-MultiCompiler.git
                GIT_TAG main
            )
            FetchContent_MakeAvailable(directshow)
            target_link_libraries(dep_directshow INTERFACE strmbase)
        endif()
        unset(dep_directshow_baseclasses_FETCH)
    endif()

    if(dep_directshow_baseclasses_BUILD)
        file(GLOB dep_directshow_baseclasses_src "${directshow_baseclasses_SOURCE_DIR}/*.cpp")
        add_library(dep_directshow_baseclasses STATIC ${dep_directshow_baseclasses_src})
        target_include_directories(dep_directshow_baseclasses PUBLIC $<BUILD_INTERFACE:${directshow_baseclasses_SOURCE_DIR}>)
        target_link_libraries(dep_directshow_baseclasses PRIVATE strmiids winmm)
        ac_check_enable_static_crt(dep_directshow_baseclasses)
        target_link_libraries(dep_directshow INTERFACE dep_directshow_baseclasses)
        unset(directshow_baseclasses_SOURCE_DIR)
        unset(dep_directshow_baseclasses_BUILD)
    endif()

    add_library(dep::directshow ALIAS dep_directshow)
endif()
