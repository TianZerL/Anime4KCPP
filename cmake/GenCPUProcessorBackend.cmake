function(generate_cpu_processor_backend BACKEND_NAME_VAR LAYER_SUFFIX_VAR HEADER_PATH_VAR OUTPUT_FILE_PATH_VAR)
    file(WRITE "${CORE_BINARY_DIR}/filegen/src/simd/ImageResize${SIMD_UPPER}.cpp.in" [[
#define BACKEND_NAME @BACKEND_NAME_VAR@
#define LAYER_SUFFIX @LAYER_SUFFIX_VAR@

#include "AC/Core/Internal/Processor/CPU/@HEADER_PATH_VAR@"
#include "AC/Core/Internal/Processor/CPU/Backend.hpp"
]])
    configure_file(
        ${CORE_BINARY_DIR}/filegen/src/simd/ImageResize${SIMD_UPPER}.cpp.in
        ${OUTPUT_FILE_PATH_VAR}
        @ONLY
    )
endfunction()
