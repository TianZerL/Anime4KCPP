project(ac_binding_python VERSION 3.1.0.0 LANGUAGES CXX)

set(BINDING_PYTHON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BINDING_PYTHON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/pybind11.cmake)

pybind11_add_module(pyac MODULE
    ${BINDING_PYTHON_SOURCE_DIR}/src/Binding.cpp
)

target_link_libraries(pyac PRIVATE ac)

ac_check_enable_static_crt(pyac)

install(
    TARGETS pyac
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
