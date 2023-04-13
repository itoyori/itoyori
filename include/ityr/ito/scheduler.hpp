#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/options.hpp"
#include "ityr/ito/sched/serial.hpp"
#include "ityr/ito/sched/randws.hpp"
#include "ityr/ito/sched/adws.hpp"

namespace ityr::ito {

using scheduler = ITYR_CONCAT(scheduler_, ITYR_ITO_SCHEDULER);

}
