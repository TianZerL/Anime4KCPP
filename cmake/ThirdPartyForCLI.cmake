set(CMDLINE_H_URL https://github.com/TianZerL/cmdline/raw/master/cmdline.h)
set(SHA1_CMDLINE "383044e4fbc6066249d4102a39431d67ed3657c7")

if(EXISTS ${TOP_DIR}/ThirdParty/include/cmdline/cmdline.h)
    file(SHA1 ${TOP_DIR}/ThirdParty/include/cmdline/cmdline.h LOCAL_SHA1_CMDLINE)

    if(NOT ${LOCAL_SHA1_CMDLINE} STREQUAL ${SHA1_CMDLINE})
        message("Warning:")
        message("   Local SHA1 for comline.h:   ${LOCAL_SHA1_CMDLINE}")
        message("   Expected SHA1:              ${SHA1_CMDLINE}")
        message("   Mismatch SHA1 for cmdline.h, trying to download it...")

        file(
            DOWNLOAD ${CMDLINE_H_URL} ${TOP_DIR}/ThirdParty/include/cmdline/cmdline.h 
            SHOW_PROGRESS 
            EXPECTED_HASH SHA1=${SHA1_CMDLINE}
        )

    endif()
else()
    file(
        DOWNLOAD ${CMDLINE_H_URL} ${TOP_DIR}/ThirdParty/include/cmdline/cmdline.h 
        SHOW_PROGRESS 
        EXPECTED_HASH SHA1=${SHA1_CMDLINE}
    )
endif()

find_package(CURL)
if(${CURL_FOUND})
    message(STATUS "CLI: libcurl found, enable web image download support.")
    target_link_libraries(${PROJECT_NAME} PRIVATE CURL::libcurl)
    add_definitions(-DENABLE_LIBCURL)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${TOP_DIR}/ThirdParty/include/cmdline>)

target_link_libraries(${PROJECT_NAME} PRIVATE Anime4KCPPCore)

if(Use_Boost_filesystem)
    find_package(Boost COMPONENTS filesystem REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE Boost::filesystem)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0) # Just for G++-8 to enable filesystem
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
endif()
