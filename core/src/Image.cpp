#include <cstddef>
#include <cstdlib>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

#ifndef AC_MALLOC_ALIGN
#define AC_MALLOC_ALIGN 32
#endif

namespace ac::core::detail
{
    template <typename T>
    static inline constexpr T* alignPtr(T* const ptr, const int n) noexcept
    {
        auto srcIntptr = reinterpret_cast<std::uintptr_t>(ptr);
        auto dstIntptr = align(srcIntptr, n);
        return ptr + (dstIntptr - srcIntptr);
    }

    static inline void* alignedAlloc(std::size_t size) noexcept
    {
        auto buffer = static_cast<std::uint8_t**>(std::malloc(size + sizeof(void*) + AC_MALLOC_ALIGN));
        if (!buffer) return nullptr;
        std::uint8_t** ptr = alignPtr(buffer + 1, AC_MALLOC_ALIGN);
        ptr[-1] = reinterpret_cast<std::uint8_t*>(buffer);
        return ptr;
    }
    static inline void alignedFree(void* ptr) noexcept
    {
        if (ptr) std::free(static_cast<std::uint8_t**>(ptr)[-1]);
    }
}

struct ac::core::Image::ImageData
{
    void* data;

    ImageData(int size) noexcept : data(detail::alignedAlloc(size)) {}
    ~ImageData() noexcept
    {
        detail::alignedFree(data);
    }
};

ac::core::Image::Image() noexcept : Image(0, 0, 0, UInt8, nullptr, 0) {}
ac::core::Image::Image(const int w, const int h, const int c, const ElementType elementType, const int stride) : Image(w, h, c, elementType, nullptr, stride) {}
ac::core::Image::Image(const int w, const int h, const int c, const ElementType elementType, void* const data, const int stride) :
    w(w), h(h), c(c), elementType(elementType),
    pitch(stride > 0 ? stride : w * c * (elementType & 0xff)),
    pixels(nullptr), dptr(nullptr)
{
    int size = this->h * this->pitch;
    if (!(size > 0)) return;
    if (data) this->pixels = data;
    else
    {
        this->dptr = std::make_shared<ImageData>(size);
        this->pixels = this->dptr->data;
    }
}
ac::core::Image::Image(const Image&) noexcept = default;
ac::core::Image::Image(Image&&) noexcept = default;
ac::core::Image::~Image() noexcept = default;
ac::core::Image& ac::core::Image::operator=(const Image&) noexcept = default;
ac::core::Image& ac::core::Image::operator=(Image&&) noexcept = default;

void ac::core::Image::create(const int w, const int h, const int c, const ElementType elementType, const int stride)
{
    int pitch = stride > 0 ? stride : w * c * (elementType & 0xff);
    int size = h * pitch;
    if (!(size > 0)) return;
    this->w = w;
    this->h = h;
    this->c = c;
    this->elementType = elementType;
    this->pitch = pitch;
    this->dptr = std::make_shared<ImageData>(size);
    this->pixels = this->dptr->data;
}
