function(minifier_opencl_kernel SRC_VAR DST_VAR)
    string(REGEX REPLACE "//[^\n]*" "" TMP "${${SRC_VAR}}")
    string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" "" TMP "${TMP}")
    string(REGEX REPLACE "[ \t]+" " " TMP "${TMP}")
    string(REGEX REPLACE "\n[ \t]+" "\n" TMP "${TMP}")
    string(REGEX REPLACE "[ \t]+\n" "\n" TMP "${TMP}")
    string(REGEX REPLACE "\n+" "\n" TMP "${TMP}")
    string(REGEX REPLACE "\\(\n" "(" TMP "${TMP}")
    string(REGEX REPLACE "\n\\)" ")" TMP "${TMP}")
    string(REGEX REPLACE "{\n" "{" TMP "${TMP}")
    string(REGEX REPLACE "\\)\n{" "){" TMP "${TMP}")
    string(REGEX REPLACE "[ \t]*,[ \t\r\n]+" "," TMP "${TMP}")
    string(REGEX REPLACE "[ \t]+\\+[ \t\r\n]+" "+" TMP "${TMP}")
    string(REGEX REPLACE "[ \t]+-[ \t\r\n]+" "-" TMP "${TMP}")
    string(REGEX REPLACE "[ \t]+=[ \t\r\n]+" "=" TMP "${TMP}")

    set(${DST_VAR} "${TMP}" PARENT_SCOPE)
endfunction()

function(generate_opencl_kernel_header INPUT_FILE_PATH_VAR OUTPUT_FILE_PATH_VAR)
    file(WRITE "${CORE_BINARY_DIR}/filegen/include/AC/Core/OpenCL/Kernel.cl.hpp.in" [[
#ifndef AC_CORE_OPENCL_KERNEL_HPP
#define AC_CORE_OPENCL_KERNEL_HPP
namespace ac::core::opencl
{
    constexpr const char* KernelString = R"(@OPENCL_KERNEL_CONTENT_MINIFIED@)";
}
#endif]])

    file(READ "${INPUT_FILE_PATH_VAR}" OPENCL_KERNEL_CONTENT)
    minifier_opencl_kernel(OPENCL_KERNEL_CONTENT OPENCL_KERNEL_CONTENT_MINIFIED)

    configure_file(
        ${CORE_BINARY_DIR}/filegen/include/AC/Core/OpenCL/Kernel.cl.hpp.in
        ${OUTPUT_FILE_PATH_VAR}
        @ONLY
    )
endfunction()
