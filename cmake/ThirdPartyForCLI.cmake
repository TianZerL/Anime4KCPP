set(URL https://github.com/TianZerL/cmdline/raw/master/cmdline.h)
set(SHA1_CMDLINE "da2d7322f5992f832a62a6c95cddfe6565b766ac")

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
