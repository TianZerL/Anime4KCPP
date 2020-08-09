set(URL https://github.com/TianZerL/cmdline/raw/master/cmdline.h)
set(SHA1_CMDLINE "689c7e001966d529bc3b6000a3714d9aa26e3996")

if(EXISTS ${TOP_DIR}/ThirdParty/include/cmdline.h)
    file(SHA1 ${TOP_DIR}/ThirdParty/include/cmdline.h LOCAL_SHA1_CMDLINE)

    if(NOT ${LOCAL_SHA1_CMDLINE} STREQUAL ${SHA1_CMDLINE})
        message("Warning:")
        message("   Local SHA1 for comline.h:   ${LOCAL_SHA1_CMDLINE}")
        message("   Expected SHA1:              ${SHA1_CMDLINE}")
        message("   Mismatch SHA1 for cmdline.h, trying to download it...")

        file(
            DOWNLOAD ${URL} ${TOP_DIR}/ThirdParty/include/cmdline.h 
            SHOW_PROGRESS 
            EXPECTED_HASH SHA1=${SHA1_CMDLINE}
        )

    endif()
else()
    file(
        DOWNLOAD ${URL} ${TOP_DIR}/ThirdParty/include/cmdline.h 
        SHOW_PROGRESS 
        EXPECTED_HASH SHA1=${SHA1_CMDLINE}
    )
endif()

find_package(OpenCL REQUIRED)

include_directories(${TOP_DIR}/ThirdParty/include ${OpenCL_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} Anime4KCPPCore)

# Just for G++-8 to enable filesystem
if(CMAKE_CXX_COMPILE_ID MATCHES "GNU")
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()
