enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 61)

if(NOT TARGET dep::cuda)
    find_package(CUDAToolkit REQUIRED)

    add_library(dep_cuda INTERFACE)
    target_link_libraries(dep_cuda INTERFACE CUDA::cudart_static)
    if(MSVC) #suppress warning
        target_link_options(dep_cuda INTERFACE /NODEFAULTLIB:LIBCMT)     
    endif()
    add_library(dep::cuda ALIAS dep_cuda)
endif()