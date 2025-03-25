#include <memory>

#include "AC/Core.hpp"

#include "AC/Core.h"

struct ac_image
{
	ac::core::Image object{};
};
struct ac_processor
{
	std::shared_ptr<ac::core::Processor> object{};
};

ac_image* ac_image_create(const int w, const int h, const int c, const int element_type, void* const data, const int stride)
{
	return new ac_image{ { w, h, c, element_type, data, stride } };
}
ac_image* ac_image_create_empty(void)
{
	return new ac_image{};
}
void ac_image_destroy(ac_image* const image)
{
	delete image;
}
int ac_image_width(const ac_image* const image)
{
	return image->object.width();
}
int ac_image_height(const ac_image* const image)
{
	return image->object.height();
}
int ac_image_channels(const ac_image* const image)
{
	return image->object.channels();
}
int ac_image_stride(const ac_image* const image)
{
	return image->object.stride();
}
int ac_image_size(const ac_image* const image)
{
	return image->object.size();
}
int ac_image_element_size(const ac_image* const image)
{
	return image->object.elementSize();
}
int ac_image_channel_size(const ac_image* const image)
{
	return image->object.channelSize();
}
int ac_image_type(const ac_image* const image)
{
	return image->object.type();
}
uint8_t* ac_image_data(const ac_image* const image)
{
	return image->object.data();
}
uint8_t* ac_image_line(const ac_image* const image, const int idx)
{
	return image->object.line(idx);
}
uint8_t* ac_image_pixel(const ac_image* const image, const int x, const int y)
{
	return image->object.pixel(x, y);
}
void* ac_image_ptr(const ac_image* const image)
{
	return image->object.ptr();
}
void* ac_image_line_ptr(const ac_image* const image, const int idx)
{
	return image->object.ptr(idx);
}
void* ac_image_pixel_ptr(const ac_image* const image, const int x, const int y)
{
	return image->object.ptr(x, y);
}
int ac_image_empty(const ac_image* const image)
{
	return image->object.empty();
}
int ac_image_is_uint(const ac_image* const image)
{
	return image->object.isUint();
}
int ac_image_is_int(const ac_image* const image)
{
	return image->object.isInt();
}
int ac_image_is_float(const ac_image* const image)
{
	return image->object.isFloat();
}
int ac_image_same(const ac_image* const a, const ac_image* const b)
{
	return a->object == b->object;
}

ac_processor* ac_processor_create(const int processor_type, const int device, const char* const model)
{
	return new ac_processor{ ac::core::Processor::create(processor_type, device, model) };
}
void ac_processor_destroy(ac_processor* const processor)
{
	delete processor;
}
void ac_processor_process(ac_processor* const processor, const ac_image* const src, ac_image* const dst, const double factor)
{
	processor->object->process(src->object, dst->object, factor);
}
int ac_processor_ok(const ac_processor* const processor)
{
	return processor->object->ok();
}
const char* ac_processor_error(const ac_processor* const processor)
{
	return processor->object->error();
}
const char* ac_processor_name(const ac_processor* const processor)
{
	return processor->object->name();
}
const char* ac_processor_info(const int processor_type)
{
	switch (processor_type)
	{
	case ac::core::Processor::CPU: return ac::core::Processor::info<ac::core::Processor::CPU>();
#   ifdef AC_CORE_WITH_OPENCL
	case ac::core::Processor::OpenCL: return ac::core::Processor::info<ac::core::Processor::OpenCL>();
#   endif
#   ifdef AC_CORE_WITH_CUDA
	case ac::core::Processor::CUDA: return ac::core::Processor::info<ac::core::Processor::CUDA>();
#   endif
	default: return "unsupported processor";
	}
}
int ac_processor_type(const char* const processor_type_string)
{
	return ac::core::Processor::type(processor_type_string);
}
const char* ac_processor_type_string(const int processor_type)
{
	return ac::core::Processor::type(processor_type);
}

#ifdef AC_CORE_ENABLE_IMAGE_IO
void ac_imread(const char* const filename, const int flag, ac_image* const image)
{
	image->object = ac::core::imread(filename, flag);
}
int ac_imwrite(const char* const filename, const ac_image* const image)
{
	return ac::core::imwrite(filename, image->object);
}
#endif

void ac_resize(const ac_image* const src, ac_image* const dst, const double fx, const double fy)
{
	ac::core::resize(src->object, dst->object, fx, fy);
}
void ac_rgb2yuv(const ac_image* const rgb, ac_image* const yuv)
{
	ac::core::rgb2yuv(rgb->object, yuv->object);
}
void ac_rgba2yuva(const ac_image* const rgba, ac_image* const yuva)
{
	ac::core::rgba2yuva(rgba->object, yuva->object);
}
void ac_yuv2rgb(const ac_image* const yuv, ac_image* const rgb)
{
	ac::core::yuv2rgb(yuv->object, rgb->object);
}
void ac_yuva2rgba(const ac_image* const yuva, ac_image* const rgba)
{
	ac::core::yuva2rgba(yuva->object, rgba->object);
}
