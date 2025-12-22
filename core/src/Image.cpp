#include <algorithm>
#include <cstring>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

#ifndef AC_CORE_STRIDE_ALIGN
#   define AC_CORE_STRIDE_ALIGN 32
#endif

struct ac::core::Image::ImageData
{
    void* data;

    ImageData(const int size) noexcept : data(fastMalloc(size)) {}
    ~ImageData() noexcept
    {
        fastFree(data);
        data = nullptr;
    }
};

ac::core::Image::Image() noexcept : Image(0, 0, 0, UInt8, nullptr, 0) {}
ac::core::Image::Image(const int w, const int h, const int c, const ElementType elementType, const int stride) : Image(w, h, c, elementType, nullptr, stride) {}
ac::core::Image::Image(const int w, const int h, const int c, const ElementType elementType, void* const data, const int stride) :
    w(0), h(0), c(0), elementType(UInt8), pitch(0), pixels(nullptr), dptr(nullptr)
{
    if (data) map(w, h, c, elementType, data, stride);
    else create(w, h, c, elementType, stride);
}
ac::core::Image::Image(const Image&) noexcept = default;
ac::core::Image::Image(Image&&) noexcept = default;
ac::core::Image::~Image() noexcept = default;
ac::core::Image& ac::core::Image::operator=(const Image&) noexcept = default;
ac::core::Image& ac::core::Image::operator=(Image&&) noexcept = default;

void ac::core::Image::create(const int w, const int h, const int c, const ElementType elementType, const int stride)
{
    int lineSize = w * c * (elementType & 0xff);
    if (!(h > 0 && lineSize > 0)) return;
    int pitch = (stride >= lineSize) ? stride : align(lineSize, AC_CORE_STRIDE_ALIGN);
    int size = h * pitch;
    this->w = w;
    this->h = h;
    this->c = c;
    this->elementType = elementType;
    this->pitch = pitch;
    this->dptr = std::make_shared<ImageData>(size);
    this->pixels = this->dptr->data;
}
void ac::core::Image::map(const int w, const int h, const int c, const ElementType elementType, void* const data, const int stride) noexcept
{
    int lineSize = w * c * (elementType & 0xff);
    if (!(h > 0 && lineSize > 0) || !data) return;
    int pitch = (stride >= lineSize) ? stride : lineSize;
    this->w = w;
    this->h = h;
    this->c = c;
    this->elementType = elementType;
    this->pitch = pitch;
    this->dptr = nullptr;
    this->pixels = data;
}
void ac::core::Image::from(const int w, const int h, const int c, const ElementType elementType, const void* const data, const int stride)
{
    int lineSize = w * c * (elementType & 0xff);
    if (!(h > 0 && lineSize > 0) || !data) return;
    int pitch = (stride >= lineSize) ? stride : lineSize;
    create(w, h, c, elementType);
    auto src = static_cast<const std::uint8_t*>(data);
    for (int i = 0; i < h; i++) std::memcpy(line(i), src + i * pitch, lineSize);
}
void ac::core::Image::to(void* const data, const int stride) const noexcept
{
    int lineSize = width() * pixelSize();
    if (!(height() > 0 && lineSize > 0) || !data) return;
    int pitch = (stride >= lineSize) ? stride : lineSize;
    auto dst = static_cast<std::uint8_t*>(data);
    for (int i = 0; i < height(); i++) std::memcpy(dst + i * pitch, line(i), lineSize);
}
ac::core::Image ac::core::Image::view(const int x, const int y, const int w, const int h) const noexcept
{
    Image dst{};

    int intersectX = std::max(x, 0);
    int intersectY = std::max(y, 0);
    int intersectW = std::min(width(), x + w) - intersectX;
    int intersectH = std::min(height(), y + h) - intersectY;

    if (intersectW > 0 && intersectH > 0)
    {
        dst.w = intersectW;
        dst.h = intersectH;
        dst.c = channels();
        dst.elementType = type();
        dst.pitch = stride();
        dst.pixels = ptr(intersectX, intersectY);
        dst.dptr = this->dptr;
    }

    return dst;
}
ac::core::Image ac::core::Image::clone() const
{
    Image dst{};
    dst.from(width(), height(), channels(), type(), ptr(), stride());
    return dst;
}
