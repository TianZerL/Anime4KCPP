#ifndef AC_VIDEO_FILTER_HPP
#define AC_VIDEO_FILTER_HPP

#include "AC/Video/Pipeline.hpp"

#define AC_VIDEO_FILTER_MODE_AUTO     0x0
#define AC_VIDEO_FILTER_MODE_PARALLEL 0x1
#define AC_VIDEO_FILTER_MODE_SERIAL   0x2

#define AC_VIDEO_FILTER_MODE_MASK 0xf

#define AC_VIDEO_FILTER_MODE_PARALLEL_WORKERS_SHIFT 4
#define AC_VIDEO_FILTER_MODE_PARALLEL_WORKERS_MASK  (0xffff << AC_VIDEO_FILTER_MODE_PARALLEL_WORKERS_SHIFT)

#define AC_VIDEO_FILTER_MODE_PARALLEL_WITH_WORKERS(n) (AC_VIDEO_FILTER_MODE_PARALLEL | ((n) << AC_VIDEO_FILTER_MODE_PARALLEL_WORKERS_SHIFT))

namespace ac::video
{
    void filter(Pipeline& pipeline, bool (*callback)(Frame& /*src*/, Frame& /*dst*/, void* /*userdata*/), void* userdata, int flag);
}

#endif
