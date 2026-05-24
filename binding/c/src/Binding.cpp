#include <cstdlib>
#include <cstring>
#include <memory>

#include "AC/Core.hpp"

#include "AC/Core/Image.h"
#include "AC/Core/Processor.h"
#include "AC/Error.h"

struct ACImageHandle
{
    ac::core::Image object{};
};
struct ACProcessorHandle
{
    std::shared_ptr<ac::core::Processor> object{};
};

namespace detail
{
    static inline void copyProp(const ac::core::Image& src, ACImage* const dst)
    {
        dst->width = src.width();
        dst->height = src.height();
        dst->channels = src.channels();
        dst->stride = src.stride();
        dst->element_type = src.type();
        dst->ptr = src.ptr();
    }
}

ACImage* ac_image_alloc()
{
    auto ptr = std::malloc(sizeof(ACImage));
    if (ptr) std::memset(ptr, 0, sizeof(ACImage));
    return static_cast<ACImage*>(ptr);
}
void ac_image_free(ACImage** const image)
{
    if (!(image && *image)) return;
    ac_image_unref(*image);
    std::free(*image);
    *image = nullptr;
}
int ac_image_ref(const ACImage* const src, ACImage* const dst)
{
    if (!(src && dst && src->hptr)) return AC_ERROR(AC_EINVAL);
    if (dst->hptr) dst->hptr->object = src->hptr->object;
    else dst->hptr = new ACImageHandle{ src->hptr->object };
    detail::copyProp(dst->hptr->object, dst);
    return AC_SUCCESS;
}
void ac_image_unref(ACImage* const image)
{
    if (!(image && image->hptr)) return;
    delete image->hptr;
    std::memset(image, 0, sizeof(ACImage));
}
int ac_image_create(ACImage* const image)
{
    if (!image) return AC_ERROR(AC_EINVAL);
    if (!image->hptr) image->hptr = new ACImageHandle{};
    image->hptr->object.create(image->width, image->height, image->channels, image->element_type, image->stride);
    detail::copyProp(image->hptr->object, image);
    return AC_SUCCESS;
}
int ac_image_map(ACImage* const image)
{
    if (!image) return AC_ERROR(AC_EINVAL);
    if (!image->hptr) image->hptr = new ACImageHandle{};
    image->hptr->object.map(image->width, image->height, image->channels, image->element_type, image->ptr, image->stride);
    detail::copyProp(image->hptr->object, image);
    return AC_SUCCESS;
}
int ac_image_from(ACImage* const image, const void* const data)
{
    if (!image) return AC_ERROR(AC_EINVAL);
    if (!image->hptr) image->hptr = new ACImageHandle{};
    image->hptr->object.from(image->width, image->height, image->channels, image->element_type, data, image->stride);
    detail::copyProp(image->hptr->object, image);
    return AC_SUCCESS;
}
int ac_image_view(const ACImage* const src, ACImage* const dst, const int x, const int y, const int w, const int h)
{
    if (!(src && dst && src->hptr)) return AC_ERROR(AC_EINVAL);
    if (dst->hptr) dst->hptr->object = src->hptr->object.view(x, y, w, h);
    else dst->hptr = new ACImageHandle{ src->hptr->object.view(x, y, w, h) };
    detail::copyProp(dst->hptr->object, dst);
    return AC_SUCCESS;
}
int ac_image_clone(const ACImage* const src, ACImage* const dst)
{
    if (!(src && dst && src->hptr)) return AC_ERROR(AC_EINVAL);
    if (dst->hptr) dst->hptr->object = src->hptr->object.clone();
    else dst->hptr = new ACImageHandle{ src->hptr->object.clone() };
    detail::copyProp(dst->hptr->object, dst);
    return AC_SUCCESS;
}
int ac_image_to(const ACImage* const image, void* const data, const int stride)
{
    if (!(image && image->hptr && data)) return AC_ERROR(AC_EINVAL);
    image->hptr->object.to(data, stride);
    return AC_SUCCESS;
}

#ifndef AC_CORE_DISABLE_IMAGE_IO
int ac_imread(const char* const filename, const int mode, ACImage* const image)
{
    if (!image) return AC_ERROR(AC_EINVAL);
    if (!image->hptr) image->hptr = new ACImageHandle{};
    auto object = ac::core::imread(filename, mode);
    if (object.empty()) return AC_ERROR(AC_EIO);
    image->hptr->object = object;
    detail::copyProp(image->hptr->object, image);
    return AC_SUCCESS;
}
int ac_imwrite(const char* const filename, const ACImage* const image)
{
    if (!(image && image->hptr)) return AC_ERROR(AC_EINVAL);
    return ac::core::imwrite(filename, image->hptr->object) ? AC_SUCCESS : AC_ERROR(AC_EIO);
}
#endif

int ac_resize(const ACImage* const src, ACImage* const dst, const double fx, const double fy, const int mode)
{
    if (!(src && src->hptr && dst && dst->hptr)) return AC_ERROR(AC_EINVAL);
    ac::core::resize(src->hptr->object, dst->hptr->object, fx, fy, mode);
    detail::copyProp(dst->hptr->object, dst);
    return AC_SUCCESS;
}
int ac_rgb2yuv(const ACImage* const rgb, ACImage* const yuv)
{
    if (!(rgb && rgb->hptr && yuv && yuv->hptr)) return AC_ERROR(AC_EINVAL);
    ac::core::rgb2yuv(rgb->hptr->object, yuv->hptr->object);
    detail::copyProp(yuv->hptr->object, yuv);
    return AC_SUCCESS;
}
int ac_rgba2yuva(const ACImage* const rgba, ACImage* const yuva)
{
    if (!(rgba && rgba->hptr && yuva && yuva->hptr)) return AC_ERROR(AC_EINVAL);
    ac::core::rgba2yuva(rgba->hptr->object, yuva->hptr->object);
    detail::copyProp(yuva->hptr->object, yuva);
    return AC_SUCCESS;
}
int ac_yuv2rgb(const ACImage* const yuv, ACImage* const rgb)
{
    if (!(yuv && yuv->hptr && rgb && rgb->hptr)) return AC_ERROR(AC_EINVAL);
    ac::core::yuv2rgb(yuv->hptr->object, rgb->hptr->object);
    detail::copyProp(rgb->hptr->object, rgb);
    return AC_SUCCESS;
}
int ac_yuva2rgba(const ACImage* const yuva, ACImage* const rgba)
{
    if (!(yuva && yuva->hptr && rgba && rgba->hptr)) return AC_ERROR(AC_EINVAL);
    ac::core::yuva2rgba(yuva->hptr->object, rgba->hptr->object);
    detail::copyProp(rgba->hptr->object, rgba);
    return AC_SUCCESS;
}

ACProcessor* ac_processor_alloc()
{
    auto ptr = std::malloc(sizeof(ACProcessor));
    if (ptr) std::memset(ptr, 0, sizeof(ACProcessor));
    return static_cast<ACProcessor*>(ptr);
}
void ac_processor_free(ACProcessor** const processor)
{
    if (!(processor && *processor)) return;
    ac_processor_unref(*processor);
    std::free(*processor);
    *processor = nullptr;
}
int ac_processor_ref(const ACProcessor* const src, ACProcessor* const dst)
{
    if (!(src && dst && src->hptr)) return AC_ERROR(AC_EINVAL);
    if (dst->hptr) dst->hptr->object = src->hptr->object;
    else dst->hptr = new ACProcessorHandle{ src->hptr->object };
    dst->type = src->type;
    dst->device = src->device;
    dst->model = src->model;
    return AC_SUCCESS;
}
void ac_processor_unref(ACProcessor* const processor)
{
    if (!(processor && processor->hptr)) return;
    delete processor->hptr;
    std::memset(processor, 0, sizeof(ACProcessor));
}
int ac_processor_create(ACProcessor* const processor)
{
    if (!processor) return AC_ERROR(AC_EINVAL);
    if (!processor->hptr) processor->hptr = new ACProcessorHandle{};
    processor->hptr->object = ac::core::Processor::create(processor->type, processor->device, processor->model);
    return ac_processor_ok(processor);
}
int ac_processor_process(ACProcessor* const processor, const ACImage* const src, ACImage* const dst, const double factor)
{
    if (!(processor && processor->hptr && src && src->hptr && dst && dst->hptr)) return AC_ERROR(AC_EINVAL);
    processor->hptr->object->process(src->hptr->object, dst->hptr->object, factor);
    detail::copyProp(dst->hptr->object, dst);
    return ac_processor_ok(processor);
}
int ac_processor_ok(const ACProcessor* const processor)
{
    if (!(processor && processor->hptr)) return AC_ERROR(AC_EINVAL);
    return processor->hptr->object->ok() ? AC_SUCCESS : AC_ERROR(AC_EPROCESSOR);
}
const char* ac_processor_error(const ACProcessor* const processor)
{
    if (!(processor && processor->hptr)) return nullptr;
    return processor->hptr->object->error();
}
const char* ac_processor_name(const ACProcessor* const processor)
{
    if (!(processor && processor->hptr)) return nullptr;
    return processor->hptr->object->name();
}
int ac_processor_type(const ACProcessor* const processor)
{
    if (!(processor && processor->hptr)) return AC_ERROR(AC_EINVAL);
    return processor->hptr->object->type();
}
const char* ac_processor_type_name(const ACProcessor* const processor)
{
    if (!(processor && processor->hptr)) return nullptr;
    return processor->hptr->object->typeName();
}

const char* ac_processor_info(const int processor_type)
{
    switch (processor_type)
    {
    case AC_PROCESSOR_CPU: return ac::core::Processor::info<ac::core::Processor::CPU>();
#ifdef AC_CORE_WITH_OPENCL
    case AC_PROCESSOR_OPENCL: return ac::core::Processor::info<ac::core::Processor::OpenCL>();
#endif
#ifdef AC_CORE_WITH_CUDA
    case AC_PROCESSOR_CUDA: return ac::core::Processor::info<ac::core::Processor::CUDA>();
#endif
    default: return "unsupported processor";
    }
}
const char* ac_processor_list_info()
{
    return ac::core::Processor::listInfo();
}
