if(NOT TARGET dep::vapoursynth)
    if(WIN32)
        add_library(dep_vapoursynth INTERFACE IMPORTED)
        if(AC_PATH_VAPOURSYNTH_SDK)
            target_link_directories(dep_vapoursynth INTERFACE ${AC_PATH_VAPOURSYNTH_SDK}/$<IF:$<BOOL:${AC_COMPILER_32BIT}>,lib32,lib64>)
            target_include_directories(dep_vapoursynth INTERFACE ${AC_PATH_VAPOURSYNTH_SDK}/include)
        else()
            #TODO
        endif()
    elseif(UNIX)
        #TODO
    endif()
    add_library(dep::vapoursynth ALIAS dep_vapoursynth)
endif()
