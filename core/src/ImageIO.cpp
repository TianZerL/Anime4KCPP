#include <cctype>
#include <cstdint>
#include <cstring>

#ifdef AC_CORE_DISABLE_IMAGE_IO
#   define STBI_NO_STDIO
#else
#   define STB_IMAGE_WRITE_IMPLEMENTATION
#   include "stb_image_write.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "AC/Core/Image.hpp"

ac::core::Image ac::core::imdecode(const void* const buffer, const int size, const int mode) noexcept
{
    Image image{};
    int w = 0, h = 0, c = 0;
    std::uint8_t* data = stbi_load_from_memory(static_cast<const std::uint8_t*>(buffer), size, &w, &h, &c, mode);
    if (mode > 0 && mode <= 4) c = mode;
    if (data)
    {
        image.from(w, h, c, ac::core::Image::UInt8, data);
        stbi_image_free(data);
    }
    return image;
}

#ifndef AC_CORE_DISABLE_IMAGE_IO
ac::core::Image ac::core::imread(const char* const filename, const int mode) noexcept
{
    Image image{};
    int w = 0, h = 0, c = 0;
    std::uint8_t* data = stbi_load(filename, &w, &h, &c, mode);
    if (mode > 0 && mode <= 4) c = mode;
    if (data)
    {
        image.from(w, h, c, ac::core::Image::UInt8, data);
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

    if ((idx != -1) && (idx + 1 < count))
    {
        char ext[5] = "";
        for (int i = 0; i < 4 && idx + 1 + i < count; i++) ext[i] = static_cast<char>(std::tolower(static_cast<unsigned char>((filename + idx + 1)[i])));

        if (!std::strcmp(ext, "png"))
        {
            Image out = image;
            return stbi_write_png(filename, out.width(), out.height(), out.channels(), out.ptr(), out.stride());
        }
        if (!std::strcmp(ext, "jpg") || !std::strcmp(ext, "jpeg"))
        {
            Image out = unpadding(image);
            return stbi_write_jpg(filename, out.width(), out.height(), out.channels(), out.ptr(), 95);
        }
        if (!std::strcmp(ext, "bmp"))
        {
            Image out = unpadding(image);
            return stbi_write_bmp(filename, out.width(), out.height(), out.channels(), out.ptr());
        }
        if (!std::strcmp(ext, "tga"))
        {
            Image out = unpadding(image);
            return stbi_write_tga(filename, out.width(), out.height(), out.channels(), out.ptr());
        }
    }
    else return stbi_write_png(filename, image.width(), image.height(), image.channels(), image.ptr(), image.stride());

    return false;
}
#endif
