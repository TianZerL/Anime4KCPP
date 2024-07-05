#ifndef AC_VIDEO_FILTER_HPP
#define AC_VIDEO_FILTER_HPP

#include "AC/Video/Pipeline.hpp"

namespace ac::video
{
    enum
    {
        FILTER_AUTO     = 0,
        FILTER_PARALLEL = 1,
        FILTER_SERIAL   = 2
    };

    void filter(Pipeline& pipeline, bool (*callback)(Frame& /*src*/, Frame& /*dst*/, void* /*userdata*/), void* userdata, int flag);
}

#endif
