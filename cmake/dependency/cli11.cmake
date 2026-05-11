if(NOT TARGET dep::cli11)
    add_library(dep_cli11 INTERFACE IMPORTED)
    if(AC_PATH_CLI11)
        set(dep_cli11_PATH ${AC_PATH_CLI11})
        find_path(dep_cli11_INCLUDE
            NAMES CLI11.hpp
            PATHS ${dep_cli11_PATH}
        )
    endif()
    if(dep_cli11_INCLUDE)
        target_include_directories(dep_cli11 INTERFACE $<BUILD_INTERFACE:${dep_cli11_INCLUDE}>)
    else()
        find_package(CLI11 2.4.0 QUIET)
        if(NOT CLI11_FOUND)
            message(STATUS "dep: cli11 not found or too old, will be fetched online.")
            include(FetchContent)
            FetchContent_Declare(
                cli11
                GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
                GIT_TAG main
            )
            FetchContent_MakeAvailable(cli11)
        endif()
        target_link_libraries(dep_cli11 INTERFACE CLI11::CLI11)
    endif()
    add_library(dep::cli11 ALIAS dep_cli11)
endif()
