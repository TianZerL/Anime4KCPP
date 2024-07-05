if(NOT TARGET pybind11::module)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "dep: pybind11 not found, will be fetched online.")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG master
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            TEST_COMMAND ""
            INSTALL_COMMAND ""
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()
