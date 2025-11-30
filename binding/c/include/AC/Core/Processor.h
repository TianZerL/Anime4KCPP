#ifndef AC_BINDING_C_CORE_PROCESSOR_H
#define AC_BINDING_C_CORE_PROCESSOR_H

#include <stdint.h>

#include "AC/Core/Image.h"

enum ac_processor_type
{
    AC_PROCESSOR_CPU    = 0,
    AC_PROCESSOR_OPENCL = 1,
    AC_PROCESSOR_CUDA   = 2
};

typedef struct ac_processor
{
    int device;
    const char* type;
    const char* model;
    struct ac_processor_handle* hptr;
} ac_processor;

AC_C_API int ac_processor_ref(const ac_processor* src, ac_processor* dst);
AC_C_API void ac_processor_unref(ac_processor* processor);
AC_C_API int ac_processor_create(ac_processor* processor);
AC_C_API int ac_processor_process(ac_processor* processor, const ac_image* src, ac_image* dst, double factor);
AC_C_API int ac_processor_ok(const ac_processor* processor);
AC_C_API const char* ac_processor_error(const ac_processor* processor);
AC_C_API const char* ac_processor_name(const ac_processor* processor);
AC_C_API int ac_processor_type(const ac_processor* processor);
AC_C_API const char* ac_processor_type_name(const ac_processor* processor);

AC_C_API const char* ac_processor_info(int processor_type);
AC_C_API const char* ac_processor_list_info();

#endif
