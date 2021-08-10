#pragma once

#define GET_ACNET_TYPE_INDEX(HDN, HDNLevel) (HDN) ? (((HDNLevel) > 3 || (HDNLevel) < 1) ? 1 : HDNLevel) : 0

namespace Anime4KCPP
{
    enum ACNetType
    {
        HDNL0 = 0,
        HDNL1,
        HDNL2,
        HDNL3,
        TotalTypeCount
    };
}
