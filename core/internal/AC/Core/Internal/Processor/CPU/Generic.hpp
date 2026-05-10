#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_GENERIC_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_GENERIC_HPP

#include <array>
#include <cstddef>
#include <utility>

#include "AC/Util/Macro.hpp"

namespace ac::core::cpu
{
    struct OpImplGeneric
    {
    public:
        static constexpr int alignment = sizeof(float);

    private:
        template <std::size_t... i>
        static AC_FORCE_INLINE auto vector_dot_impl(const float* const v1, const float* const v2, std::index_sequence<i...>) noexcept
        {
            return (... + (v1[i] * v2[i]));
        }

        template <int vsize>
        static AC_FORCE_INLINE auto vector_dot(const float* const v1, const float* const v2) noexcept
        {
            return vector_dot_impl(v1, v2, std::make_index_sequence<vsize>{});
        }

        template <std::size_t... p>
        static AC_FORCE_INLINE auto block_sum_impl(const float* const* const rptr, const float* const* const kptr, const int c, std::index_sequence<p...>) noexcept
        {
            return (... + (rptr[p][c] * kptr[p][c]));
        }

        template <int cpos, typename K>
        static AC_FORCE_INLINE auto block_sum(const float* const* const rptr, K&& kptr, const int c) noexcept
        {
            return block_sum_impl(rptr, kptr.data(), c, std::make_index_sequence<cpos>{});
        }

        template <int cin, std::size_t... p>
        static AC_FORCE_INLINE auto make_kptr_impl(const float* const kernels, const int n, std::index_sequence<p...>) noexcept
        {
            return std::array<const float*, sizeof...(p)>{ (kernels + n * cin * sizeof...(p) + cin * p)... };
        }

        template <int cin, int cpos>
        static AC_FORCE_INLINE auto make_kptr(const float* const kernels, const int n) noexcept
        {
            return make_kptr_impl<cin>(kernels, n, std::make_index_sequence<cpos>{});
        }

    public:
        template <int vsize>
        static AC_FORCE_INLINE float dot(const float* const v1, const float* const v2) noexcept
        {
            return vector_dot<vsize>(v1, v2);
        }

        template <int cout, int cpos>
        static AC_FORCE_INLINE void conv_cin1(const float* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cpos;
                out[n] = vector_dot<cpos>(rptr, kptr) + biases[n];
            }
        }

        template <int cin, int cout, int cpos>
        static AC_FORCE_INLINE void conv(const float* const* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            for (int n = 0; n < cout; n++)
            {
                auto kptr = make_kptr<cin, cpos>(kernels, n);
                float sum = 0.0f;
                for (int c = 0; c < cin; c++) sum += block_sum<cpos>(rptr, kptr, c);
                out[n] = sum + biases[n];
            }
        }
    };
}

#endif
