enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    set(AC_CUDA_ARCH_LIST
        50-real # GTX 750Ti
        52-real # Maxwell
        60-real # Tesla P100
        61-real # Pascal
        70-real # NVIDIA TITAN V
        75      # Turing
        CACHE STRING "CUDA architectures" FORCE
    )
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
    set(AC_CUDA_ARCH_LIST
        50-real # GTX 750Ti
        52-real # Maxwell
        60-real # Tesla P100
        61-real # Pascal
        70-real # NVIDIA TITAN V
        75-real # Turing
        80      # NVIDIA A100
        CACHE STRING "CUDA architectures" FORCE
    )
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    set(AC_CUDA_ARCH_LIST
        50-real # GTX 750Ti
        52-real # Maxwell
        60-real # Tesla P100
        61-real # Pascal
        70-real # NVIDIA TITAN V
        75-real # Turing
        80-real # NVIDIA A100
        86-real # Ampere
        89      # Ada Lovelace
        CACHE STRING "CUDA architectures" FORCE
    )
else()
    set(AC_CUDA_ARCH_LIST
        75-real  # Turing
        80-real  # NVIDIA A100
        86-real  # Ampere
        89-real  # Ada Lovelace
        90-real  # NVIDIA H100
        100-real # NVIDIA B200
        103-real # NVIDIA B300
        120      # Blackwell
        CACHE STRING "CUDA architectures" FORCE
    )
endif()

set(CMAKE_CUDA_ARCHITECTURES ${AC_CUDA_ARCH_LIST})

if(NOT TARGET dep::cuda)
    find_package(CUDAToolkit REQUIRED)
    add_library(dep_cuda INTERFACE)
    target_link_libraries(dep_cuda INTERFACE CUDA::cudart_static)
    if((CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")) AND NOT AC_ENABLE_STATIC_CRT) #suppress warning
        target_link_options(dep_cuda INTERFACE /NODEFAULTLIB:LIBCMT)
    endif()
    add_library(dep::cuda ALIAS dep_cuda)
endif()
