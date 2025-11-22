function(generate_image_resize_simd SIMD_VAR OUTPUT_FILE_PATH_VAR)
    string(TOUPPER ${SIMD_VAR} SIMD_UPPER)
    string(TOLOWER ${SIMD_VAR} SIMD_LOWER)

    file(WRITE "${CORE_BINARY_DIR}/filegen/src/simd/ImageResize${SIMD_UPPER}.cpp.in" [[
#define STBIR_@SIMD_UPPER@

#ifdef STBIR_AVX2
    #define STBIR_USE_FMA
#endif

#ifdef STBIR_USE_FMA
    #define STBIR_AVX
#endif

#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include <stb_image_resize2.h>

extern const auto stbir_resize_extended_@SIMD_LOWER@ = stbir_resize_extended;
]])
    configure_file(
        ${CORE_BINARY_DIR}/filegen/src/simd/ImageResize${SIMD_UPPER}.cpp.in
        ${OUTPUT_FILE_PATH_VAR}
        @ONLY
    )
endfunction()
