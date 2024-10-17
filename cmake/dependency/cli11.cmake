if(NOT TARGET dep::cli11)
    add_library(dep_cli11 INTERFACE IMPORTED)
    find_package(CLI11 QUIET)
    if(NOT CLI11_FOUND)
        message(STATUS "dep: cli11 not found, will be fetched online.")
        include(FetchContent)
        FetchContent_Declare(
            cli11
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(cli11)
    endif()
    target_link_libraries(dep_cli11 INTERFACE CLI11::CLI11)
    add_library(dep::cli11 ALIAS dep_cli11)
endif()
