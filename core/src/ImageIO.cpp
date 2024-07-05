#include <cstdint>
#include <cstring>

#ifndef AC_CORE_ENABLE_IMAGE_IO
#   define STBI_NO_STDIO
#   define STBI_WRITE_NO_STDIO
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "AC/Core/Image.hpp"

ac::core::Image ac::core::imdecode(const void* const buffer, const int size, const int flag) noexcept
{
    Image image{};
    int w = 0, h = 0, c = 0;
    std::uint8_t* data = stbi_load_from_memory(static_cast<const std::uint8_t*>(buffer), size, &w, &h, &c, flag);
    if (flag != IMREAD_UNCHANGED) c = flag;
    if (data)
    {
        image.create(w, h, c, ac::core::Image::UInt8);
        std::memcpy(image.ptr(), data, image.size());
        stbi_image_free(data);
    }
    return image;
}

#ifdef AC_CORE_ENABLE_IMAGE_IO
ac::core::Image ac::core::imread(const char* const filename, const int flag) noexcept
{
    Image image{};
    int w = 0, h = 0, c = 0;
    std::uint8_t* data = stbi_load(filename, &w, &h, &c, flag);
    if (flag != IMREAD_UNCHANGED) c = flag;
    if (data)
    {
        image.create(w, h, c, ac::core::Image::UInt8);
        std::memcpy(image.ptr(), data, image.size());
        stbi_image_free(data);
    }
    return image;
}
bool ac::core::imwrite(const char* const filename, const Image& image) noexcept
{
    int idx = -1;
    int count = 0;
    for (auto p = filename; *p != '\0'; p++)
    {
        if (*p == '.') idx = count;
        count++;
    }

    if (idx != -1)
    {
        Image out = image;
        if (!std::strcmp(filename + idx + 1, "png")) return stbi_write_png(filename, out.width(), out.height(), out.channels(), out.ptr(), out.stride());
        unpadding(out, out);
        if (!std::strcmp(filename + idx + 1, "jpg")) return stbi_write_jpg(filename, out.width(), out.height(), out.channels(), out.ptr(), 95);
        if (!std::strcmp(filename + idx + 1, "bmp")) return stbi_write_bmp(filename, out.width(), out.height(), out.channels(), out.ptr());
        if (!std::strcmp(filename + idx + 1, "tga")) return stbi_write_tga(filename, out.width(), out.height(), out.channels(), out.ptr());
    }
    else return stbi_write_jpg(filename, image.width(), image.height(), image.channels(), image.ptr(), 95);

    return false;
}
#endif
