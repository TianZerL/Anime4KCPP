#ifndef AC_CORE_INTERNAL_MODEL_PARAM_HPP
#define AC_CORE_INTERNAL_MODEL_PARAM_HPP

#ifndef AC_CORE_PARAM_ALIGN
#   define AC_CORE_PARAM_ALIGN 64
#endif

namespace ac::core::model::param
{
#include "AC/Core/Internal/Model/Param/ACNet.p"
#include "AC/Core/Internal/Model/Param/ARNet.p"
#include "AC/Core/Internal/Model/Param/ArtCNN.p"
#include "AC/Core/Internal/Model/Param/FSRCNNX.p"
}

#endif
