#include <cctype>
#include <cstdint>
#include <cstring>

#ifdef AC_CORE_DISABLE_IMAGE_IO
#   define STBI_NO_STDIO
#else
#   define STB_IMAGE_WRITE_IMPLEMENTATION
#   include <stb_image_write.h>
#   ifdef AC_CORE_WITH_FPNG
#       include <fpng.h>
#   endif
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

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
    auto point = std::strrchr(filename, '.');

    if (point && *(++point))
    {
        char ext[5] = "";
        for (int i = 0; point[i] && i < 4; i++) ext[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(point[i])));

        if (!std::strcmp(ext, "png"))
        {
#       ifdef AC_CORE_WITH_FPNG
            [[maybe_unused]] static const bool fpngInitFlag = (fpng::fpng_init(), true);
            Image out = unpadding(image);
            return fpng::fpng_encode_image_to_file(filename, out.ptr(), out.width(), out.height(), out.channels());
#       else
            Image out = image;
            return stbi_write_png(filename, out.width(), out.height(), out.channels(), out.ptr(), out.stride());
#       endif
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

    return false;
}
#endif
