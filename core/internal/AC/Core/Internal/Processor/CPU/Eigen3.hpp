#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_EIGEN3_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_EIGEN3_HPP

#include <Eigen/Core>

#include "AC/Util/Macro.hpp"

namespace ac::core::cpu
{
    struct OpImplEigen3
    {
    public:
        static constexpr int alignment = 4 * sizeof(float);

    public:
        template <int vsize>
        static AC_FORCE_INLINE float dot(const float* const v1, const float* const v2) noexcept
        {
            Eigen::Map<const Eigen::Vector<float, vsize>> r1{ v1 };
            Eigen::Map<const Eigen::Vector<float, vsize>> r2{ v2 };

            return r1.dot(r2);
        }

        template <int cout, int cpos>
        static AC_FORCE_INLINE void conv_cin1(const float* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            Eigen::Map<const Eigen::Vector<float, cpos>> r{ rptr };

            for (int n = 0; n < cout; n++)
            {
                Eigen::Map<const Eigen::Vector<float, cpos>> k{ kernels + n * cpos };
                out[n] = r.dot(k) + biases[n];
            }
        }

        template <int cin, int cout, int cpos>
        static AC_FORCE_INLINE void conv(const float* const* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            Eigen::Matrix<float, cin * cpos, 1> r;
            for (int p = 0; p < cpos; p++) r.template segment<cin>(p * cin) = Eigen::Map<const Eigen::Matrix<float, cin, 1>>{ rptr[p] };

            Eigen::Map<const Eigen::Matrix<float, cout, cin * cpos, Eigen::RowMajor>> k{ kernels };
            Eigen::Map<const Eigen::Matrix<float, cout, 1>> b{ biases };
            Eigen::Map<Eigen::Matrix<float, cout, 1>> s{ out };

            s.noalias() = k * r + b;
        }
    };
}

#endif
