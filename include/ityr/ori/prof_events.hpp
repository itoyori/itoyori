#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"

namespace ityr::ori {

struct prof_event_get : public common::profiler::event {
  using event::event;
  std::string str() const override { return "core_get"; }
};

struct prof_event_put : public common::profiler::event {
  using event::event;
  std::string str() const override { return "core_put"; }
};

struct prof_event_checkout : public common::profiler::event {
  using event::event;
  std::string str() const override { return "core_checkout"; }
};

struct prof_event_checkin : public common::profiler::event {
  using event::event;
  std::string str() const override { return "core_checkin"; }
};

struct prof_event_release : public common::profiler::event {
  using event::event;
  std::string str() const override { return "cache_release"; }
};

struct prof_event_release_lazy : public common::profiler::event {
  using event::event;
  std::string str() const override { return "cache_release_lazy"; }
};

struct prof_event_acquire : public common::profiler::event {
  using event::event;
  std::string str() const override { return "cache_acquire"; }
};

struct prof_event_acquire_wait : public common::profiler::event {
  using event::event;
  std::string str() const override { return "cache_acquire_wait"; }
};

class prof_events {
public:
  prof_events() {}

private:
  common::profiler::event_initializer<prof_event_get>          ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_put>          ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_checkout>     ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_checkin>      ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_release>      ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_acquire>      ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_release_lazy> ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_acquire_wait> ITYR_ANON_VAR;
};

}
