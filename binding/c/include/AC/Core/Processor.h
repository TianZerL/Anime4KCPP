#ifndef AC_BINDING_C_CORE_PROCESSOR_H
#define AC_BINDING_C_CORE_PROCESSOR_H

#include <stdint.h>

#include "AC/Core/Image.h"

enum ACProcessorType
{
    AC_PROCESSOR_CPU    = 0,
    AC_PROCESSOR_OPENCL = 1,
    AC_PROCESSOR_CUDA   = 2
};

typedef struct ACProcessor
{
    int device;
    const char* type;
    const char* model;
    struct ACProcessorHandle* hptr;
} ACProcessor;

AC_C_API ACProcessor* ac_processor_alloc(void);
AC_C_API void ac_processor_free(ACProcessor** processor);
AC_C_API int ac_processor_ref(const ACProcessor* src, ACProcessor* dst);
AC_C_API void ac_processor_unref(ACProcessor* processor);
AC_C_API int ac_processor_create(ACProcessor* processor);
AC_C_API int ac_processor_process(ACProcessor* processor, const ACImage* src, ACImage* dst, double factor);
AC_C_API int ac_processor_ok(const ACProcessor* processor);
AC_C_API const char* ac_processor_error(const ACProcessor* processor);
AC_C_API const char* ac_processor_name(const ACProcessor* processor);
AC_C_API int ac_processor_type(const ACProcessor* processor);
AC_C_API const char* ac_processor_type_name(const ACProcessor* processor);

AC_C_API const char* ac_processor_info(int processor_type);
AC_C_API const char* ac_processor_list_info(void);

#endif
