if(Enable_HIP)
	project(HIP_Module LANGUAGES CXX HIP)
	
	file(GLOB SOURCES src/*.hip)
	file(GLOB INCLUDE include/*.hip)
	
	source_group("hip_include" FILES ${INCLUDE}) 
	source_group("hip_source" FILES ${SOURCE}) 
	
	message(STATUS "CMAKE_HIP_COMPILER:         ${CMAKE_HIP_COMPILER}")
	message(STATUS "CMAKE_HIP_COMPILER_ID:      ${CMAKE_HIP_COMPILER_ID}")
	message(STATUS "CMAKE_HIP_COMPILER_VERSION: ${CMAKE_HIP_COMPILER_VERSION}")
	message(STATUS "HIP SOURCES:                ${SOURCES}")
	message(STATUS "HIP INCLUDE:                ${INCLUDE}")
	
	add_library(${PROJECT_NAME} OBJECT ${SOURCES})
	
	target_include_directories(${PROJECT_NAME} PRIVATE ${TOP_DIR}/core/include include)
	
	target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_CUDA ENABLE_HIP)
	
	set_target_properties(
	    ${PROJECT_NAME} PROPERTIES 
	        HIP_STANDARD 17
	)

    if((NOT Build_Static_Core) OR (Build_Static_Core AND Build_Static_Core_PIC))
        set_target_properties(
            ${PROJECT_NAME} PROPERTIES 
                POSITION_INDEPENDENT_CODE True
        )
    endif()
endif()
