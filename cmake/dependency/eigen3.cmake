if(NOT TARGET dep::eigen3)
    add_library(dep_eigen3 INTERFACE IMPORTED)
    find_package(Eigen3 QUIET)
    if (NOT EIGEN3_FOUND)
        message(STATUS "dep: eigen3 not found, will be fetched online.")
        include(FetchContent)
        if (CMAKE_VERSION VERSION_LESS 3.28)
            FetchContent_Declare(
                eigen3
                GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG master
            )
            FetchContent_GetProperties(eigen3)
            if(NOT eigen3_POPULATED)
                FetchContent_Populate(eigen3)
                add_subdirectory(${eigen3_SOURCE_DIR} ${eigen3_BINARY_DIR} EXCLUDE_FROM_ALL)
            endif()
        else()
            FetchContent_Declare(
                eigen3
                GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG master
                EXCLUDE_FROM_ALL
            )
            FetchContent_MakeAvailable(eigen3)
        endif()
    endif()
    target_link_libraries(dep_eigen3 INTERFACE Eigen3::Eigen)
    add_library(dep::eigen3 ALIAS dep_eigen3)
endif()
