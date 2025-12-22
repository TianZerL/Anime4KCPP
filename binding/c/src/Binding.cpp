#include <cstdlib>
#include <cstring>
#include <memory>

#include "AC/Core.hpp"

#include "AC/Core/Image.h"
#include "AC/Core/Processor.h"
#include "AC/Error.h"

struct ac_image_handle
{
    ac::core::Image object{};
};
struct ac_processor_handle
{
    std::shared_ptr<ac::core::Processor> object{};
};

namespace detail
{
    static inline void copyProp(const ac::core::Image& src, ac_image* dst)
    {
        dst->width = src.width();
        dst->height = src.height();
        dst->channels = src.channels();
        dst->stride = src.stride();
        dst->element_type = src.type();
        dst->ptr = src.ptr();
    }
}

ac_image* ac_image_alloc()
{
    auto ptr = std::malloc(sizeof(ac_image));
    if (ptr) std::memset(ptr, 0, sizeof(ac_image));
    return static_cast<ac_image*>(ptr);
}
void ac_image_free(ac_image** const image)
{
    if (!(image && *image)) return;
    ac_image_unref(*image);
    std::free(*image);
    *image = nullptr;
}
int ac_image_ref(const ac_image* const src, ac_image* const dst)
{
    if (!(src && dst && src->hptr)) return AC_ERROR(EINVAL);
    if (dst->hptr) dst->hptr->object = src->hptr->object;
    else dst->hptr = new ac_image_handle{ src->hptr->object };
    detail::copyProp(dst->hptr->object, dst);
    return AC_SUCCESS;
}
void ac_image_unref(ac_image* const image)
{
    if (!(image && image->hptr)) return;
    delete image->hptr;
    image->hptr = nullptr;
    image->ptr = nullptr;
}
int ac_image_create(ac_image* image)
{
    if (!image) return AC_ERROR(EINVAL);
    if (!image->hptr) image->hptr = new ac_image_handle{};
    image->hptr->object.create(image->width, image->height, image->channels, image->element_type, image->stride);
    detail::copyProp(image->hptr->object, image);
    return AC_SUCCESS;
}
int ac_image_map(ac_image* image)
{
    if (!image) return AC_ERROR(EINVAL);
    if (!image->hptr) image->hptr = new ac_image_handle{};
    image->hptr->object.map(image->width, image->height, image->channels, image->element_type, image->ptr, image->stride);
    detail::copyProp(image->hptr->object, image);
    return AC_SUCCESS;
}
int ac_image_from(ac_image* image, const void* data)
{
    if (!image) return AC_ERROR(EINVAL);
    if (!image->hptr) image->hptr = new ac_image_handle{};
    image->hptr->object.from(image->width, image->height, image->channels, image->element_type, data, image->stride);
    detail::copyProp(image->hptr->object, image);
    return AC_SUCCESS;
}
int ac_image_view(const ac_image* const src, ac_image* const dst, const int x, const int y, const int w, const int h)
{
    if (!(src && dst && src->hptr)) return AC_ERROR(EINVAL);
    if (dst->hptr) dst->hptr->object = src->hptr->object.view(x, y, w, h);
    else dst->hptr = new ac_image_handle{ src->hptr->object.view(x, y, w, h) };
    detail::copyProp(dst->hptr->object, dst);
    return AC_SUCCESS;
}
int ac_image_clone(const ac_image* const src, ac_image* const dst)
{
    if (!(src && dst && src->hptr)) return AC_ERROR(EINVAL);
    if (dst->hptr) dst->hptr->object = src->hptr->object.clone();
    else dst->hptr = new ac_image_handle{ src->hptr->object.clone() };
    detail::copyProp(dst->hptr->object, dst);
    return AC_SUCCESS;
}
int ac_image_to(const ac_image* const image, void* const data, const int stride)
{
    if (!(image && image->hptr && data)) return AC_ERROR(EINVAL);
    image->hptr->object.to(data, stride);
    return AC_SUCCESS;
}

#ifndef AC_CORE_DISABLE_IMAGE_IO
int ac_imread(const char* const filename, const int mode, ac_image* const image)
{
    if (!(image && image->hptr)) return AC_ERROR(EINVAL);
    auto object = ac::core::imread(filename, mode);
    if (object.empty()) return AC_ERROR(EIO);
    image->hptr->object = object;
    detail::copyProp(image->hptr->object, image);
    return AC_SUCCESS;
}
int ac_imwrite(const char* const filename, const ac_image* const image)
{
    if (!(image && image->hptr)) return AC_ERROR(EINVAL);
    return ac::core::imwrite(filename, image->hptr->object) ? AC_SUCCESS : AC_ERROR(EIO);
}
#endif

int ac_resize(const ac_image* const src, ac_image* const dst, const double fx, const double fy, const int mode)
{
    if (!(src && src->hptr && dst && dst->hptr)) return AC_ERROR(EINVAL);
    ac::core::resize(src->hptr->object, dst->hptr->object, fx, fy, mode);
    detail::copyProp(dst->hptr->object, dst);
    return AC_SUCCESS;
}
int ac_rgb2yuv(const ac_image* const rgb, ac_image* const yuv)
{
    if (!(rgb && rgb->hptr && yuv && yuv->hptr)) return AC_ERROR(EINVAL);
    ac::core::rgb2yuv(rgb->hptr->object, yuv->hptr->object);
    detail::copyProp(yuv->hptr->object, yuv);
    return AC_SUCCESS;
}
int ac_rgba2yuva(const ac_image* const rgba, ac_image* const yuva)
{
    if (!(rgba && rgba->hptr && yuva && yuva->hptr)) return AC_ERROR(EINVAL);
    ac::core::rgba2yuva(rgba->hptr->object, yuva->hptr->object);
    detail::copyProp(yuva->hptr->object, yuva);
    return AC_SUCCESS;
}
int ac_yuv2rgb(const ac_image* const yuv, ac_image* const rgb)
{
    if (!(yuv && yuv->hptr && rgb && rgb->hptr)) return AC_ERROR(EINVAL);
    ac::core::yuv2rgb(yuv->hptr->object, rgb->hptr->object);
    detail::copyProp(rgb->hptr->object, rgb);
    return AC_SUCCESS;
}
int ac_yuva2rgba(const ac_image* const yuva, ac_image* const rgba)
{
    if (!(yuva && yuva->hptr && rgba && rgba->hptr)) return AC_ERROR(EINVAL);
    ac::core::yuva2rgba(yuva->hptr->object, rgba->hptr->object);
    detail::copyProp(rgba->hptr->object, rgba);
    return AC_SUCCESS;
}

int ac_processor_ref(const ac_processor* const src, ac_processor* const dst)
{
    if (!(src && dst && src->hptr)) return AC_ERROR(EINVAL);
    if (dst->hptr) dst->hptr->object = src->hptr->object;
    else dst->hptr = new ac_processor_handle{ src->hptr->object };
    dst->type = src->type;
    dst->device = src->device;
    dst->model = src->model;
    return AC_SUCCESS;
}
void ac_processor_unref(ac_processor* const processor)
{
    if (!(processor && processor->hptr)) return;
    delete processor->hptr;
    processor->hptr = nullptr;
}
int ac_processor_create(ac_processor* const processor)
{
    if (!processor) return AC_ERROR(EINVAL);
    if (!processor->hptr) processor->hptr = new ac_processor_handle{};
    processor->hptr->object = ac::core::Processor::create(processor->type, processor->device, processor->model);
    return ac_processor_ok(processor);
}
int ac_processor_process(ac_processor* const processor, const ac_image* const src, ac_image* const dst, const double factor)
{
    if (!(processor && processor->hptr && src && src->hptr && dst && dst->hptr)) return AC_ERROR(EINVAL);
    processor->hptr->object->process(src->hptr->object, dst->hptr->object, factor);
    detail::copyProp(dst->hptr->object, dst);
    return ac_processor_ok(processor);
}
int ac_processor_ok(const ac_processor* const processor)
{
    if (!(processor && processor->hptr)) return AC_ERROR(EINVAL);
    return processor->hptr->object->ok() ? AC_SUCCESS : AC_ERROR(EINTR);
}
const char* ac_processor_error(const ac_processor* const processor)
{
    if (!(processor && processor->hptr)) return nullptr;
    return processor->hptr->object->error();
}
const char* ac_processor_name(const ac_processor* const processor)
{
    if (!(processor && processor->hptr)) return nullptr;
    return processor->hptr->object->name();
}
int ac_processor_type(const ac_processor* const processor)
{
    if (!(processor && processor->hptr)) return AC_ERROR(EINVAL);
    return processor->hptr->object->type();
}
const char* ac_processor_type_name(const ac_processor* const processor)
{
    if (!(processor && processor->hptr)) return nullptr;
    return processor->hptr->object->typeName();
}

const char* ac_processor_info(const int processor_type)
{
    switch (processor_type)
    {
    case ac::core::Processor::CPU: return ac::core::Processor::info<ac::core::Processor::CPU>();
#ifdef AC_CORE_WITH_OPENCL
    case ac::core::Processor::OpenCL: return ac::core::Processor::info<ac::core::Processor::OpenCL>();
#endif
#ifdef AC_CORE_WITH_CUDA
    case ac::core::Processor::CUDA: return ac::core::Processor::info<ac::core::Processor::CUDA>();
#endif
    default: return "unsupported processor";
    }
}
const char* ac_processor_list_info()
{
    return ac::core::Processor::listInfo();
}
