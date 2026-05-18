if(NOT TARGET dep::doctest)
    add_library(dep_doctest INTERFACE IMPORTED)
    if(AC_PATH_DOCTEST)
        set(dep_doctest_PATH ${AC_PATH_DOCTEST})
        find_path(dep_doctest_INCLUDE
            NAMES doctest/doctest.h
            PATHS ${dep_doctest_PATH}
        )
    endif()
    if(dep_doctest_INCLUDE)
        target_include_directories(dep_doctest INTERFACE $<BUILD_INTERFACE:${dep_doctest_INCLUDE}>)
    else()
        find_package(doctest QUIET)
        if(NOT doctest_FOUND)
            message(STATUS "dep: doctest not found, will be fetched online.")
            include(FetchContent)
            FetchContent_Declare(
                doctest
                GIT_REPOSITORY https://github.com/doctest/doctest.git
                GIT_TAG master
                SOURCE_SUBDIR do_not_find_cmake # To make sure CMakeLists.txt won't run
            )
            FetchContent_MakeAvailable(doctest)
            target_include_directories(dep_doctest INTERFACE $<BUILD_INTERFACE:${doctest_SOURCE_DIR}>)
        else()
            target_link_libraries(dep_doctest INTERFACE doctest::doctest)
        endif()
    endif()
    add_library(dep::doctest ALIAS dep_doctest)
endif()
