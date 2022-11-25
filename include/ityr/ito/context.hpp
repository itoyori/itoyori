#pragma once

#include "ityr/common/util.hpp"

#if defined(__x86_64__)

#include "ityr/ito/arch/x86_64.hpp"
namespace ityr::ito {
using context_frame = context_frame_x86_64;
using context = context_x86_64;
}

#elif defined(__aarch64__)

#include "ityr/ito/arch/aarch64.hpp"
namespace ityr::ito {
using context_frame = context_frame_aarch64;
using context = context_aarch64;
}

#else
#error "This architecture is not supported"
#endif

namespace ityr::ito {

ITYR_TEST_CASE("[ityr::ito::context] save_context_with_call()") {
  int x = 3;
  int y = 7;
  void* random_addr = (void*)0x123456;

  auto fn = [](context_frame* cf, void* xp, void* yp) {
    ITYR_CHECK(*((int*)xp) == 3);
    ITYR_CHECK(*((int*)yp) == 7);
    ITYR_CHECK(cf->parent_frame == (void*)0x123456);
  };

  context::save_context_with_call((context_frame*)random_addr, fn, &x, &y);
}

ITYR_TEST_CASE("[ityr::ito::context] save_context_with_call() and resume()") {
  bool ok = false;
  // save context frame 1 (cf1)
  context::save_context_with_call(nullptr, [](context_frame* cf1, void* ok_, void*) {

    // save context frame 2 (cf2)
    context::save_context_with_call(cf1, [](context_frame* cf2, void* cf1_, void*) {
      context_frame* cf1 = (context_frame*)cf1_;
      ITYR_CHECK(cf2->parent_frame == cf1);

      // save context frame 3 (cf3)
      context::save_context_with_call(cf2, [](context_frame* cf3, void* cf1_, void* cf2_) {
        context_frame* cf1 = (context_frame*)cf1_;
        context_frame* cf2 = (context_frame*)cf2_;
        ITYR_CHECK(cf2->parent_frame == cf1);
        ITYR_CHECK(cf3->parent_frame == cf2);

        // resume cf2 by discarding cf3 and the current execution context
        context::resume(cf2);

        // should not reach here
        ITYR_CHECK(false);
      }, (void*)cf1, (void*)cf2);

      // should not reach here
      ITYR_CHECK(false);
    }, (void*)cf1, nullptr);

    // should reach here (check by setting a flag ok)
    *((bool*)ok_) = true;
  }, &ok, nullptr);

  ITYR_CHECK(ok);
}

ITYR_TEST_CASE("[ityr::ito::context] call_on_stack()") {
  int x = 3;
  int y = 7;

  size_t stack_size = 128 * 1024;
  void* stack_buf = std::malloc(stack_size);

  context::call_on_stack(stack_buf, stack_size,
                         [](void* xp, void* yp, void* stack_buf, void* stack_size_p) {
    ITYR_CHECK(*((int*)xp) == 3);
    ITYR_CHECK(*((int*)yp) == 7);
    size_t stack_size = *((size_t*)stack_size_p);

    int a = 3;

    // Check if the local variable address is within the requested stack
    ITYR_CHECK((uintptr_t)stack_buf <= (uintptr_t)&a);
    ITYR_CHECK((uintptr_t)&a < (uintptr_t)stack_buf + stack_size);
  }, &x, &y, stack_buf, &stack_size);

  std::free(stack_buf);
}

ITYR_TEST_CASE("[ityr::ito::context] jump_to_stack()") {
  size_t stack_size = 128 * 1024;
  void* stack_buf = std::malloc(stack_size);

  context::save_context_with_call(nullptr, [](context_frame* cf, void* stack_buf, void* stack_size_p) {
    size_t stack_size = *((size_t*)stack_size_p);

    // Set a canary to the stack bottom
    uint8_t canary = 55;
    *((uint8_t*)stack_buf + stack_size - 1) = canary;

    // jump to the stack just above the canary location
    context::jump_to_stack((uint8_t*)stack_buf + stack_size - 2,
                           [](void* cf_, void* stack_buf, void* stack_size_p, void* canary_p) {
      context_frame* cf = (context_frame*)cf_;
      size_t stack_size = *((size_t*)stack_size_p);
      uint8_t canary = *((size_t*)canary_p);

      int a = 3;

      // Check if the local variable address is within the requested stack
      ITYR_CHECK((uintptr_t)stack_buf <= (uintptr_t)&a);
      ITYR_CHECK((uintptr_t)&a < (uintptr_t)stack_buf + stack_size);

      // Check canary is not changed
      ITYR_CHECK(*((uint8_t*)stack_buf + stack_size - 1) == canary);

      // Check if the local variable address is within the requested stack
      ITYR_CHECK((uintptr_t)stack_buf <= (uintptr_t)&a);
      ITYR_CHECK((uintptr_t)&a < (uintptr_t)stack_buf + stack_size);

      // resume the main context
      context::resume(cf);
    }, cf, stack_buf, &stack_size, &canary);

    // should not reach here
    ITYR_CHECK(false);
  }, stack_buf, &stack_size);

  std::free(stack_buf);
}

}
