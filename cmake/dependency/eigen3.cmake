if(NOT TARGET dep::eigen3)
    find_package(Eigen3 QUIET)

    if (NOT EIGEN3_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG master
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            TEST_COMMAND ""
            INSTALL_COMMAND ""
        )
        FetchContent_MakeAvailable(Eigen3)
    endif()

    add_library(dep_eigen3 INTERFACE IMPORTED)
    target_link_libraries(dep_eigen3 INTERFACE Eigen3::Eigen)
    add_library(dep::eigen3 ALIAS dep_eigen3)
endif()
