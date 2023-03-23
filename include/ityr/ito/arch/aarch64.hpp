#pragma once

#include "ityr/common/util.hpp"

namespace ityr::ito {

#if defined(__ARM_FEATURE_SVE)
#define ITYR_AARCH64_FLOAT_CLOBBERS \
    "p0" , "p1" , "p2" , "p3" , "p4" , "p5" , "p6" , "p7" , \
    "p8" , "p9" , "p10", "p11", "p12", "p13", "p14", "p15", \
    "z0" , "z1" , "z2" , "z3" , "z4" , "z5" , "z6" , "z7" , \
    "z8" , "z9" , "z10", "z11", "z12", "z13", "z14", "z15", \
    "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", \
    "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
#else
#define ITYR_AARCH64_FLOAT_CLOBBERS \
    "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" , \
    "v8" , "v9" , "v10", "v11", "v12", "v13", "v14", "v15", \
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
#endif

#if defined(__FUJITSU)
// Fujitsu compiler (trad mode of v1.2.31) generates the illegal instruction
//     ldp     x19, x19, [x29, #-16]
// if we specify x19 in the clobbered list.
// As a workaround, we save x19 in the stack explicitly
#define ITYR_AARCH64_SAVE_R19    "stp x0, x19, [sp, #-16]!\n\t"
#define ITYR_AARCH64_RESTORE_R19 "ldp x0, x19, [sp], #16\n\t"
#define ITYR_AARCH64_CLOBBER_R19
#else
#define ITYR_AARCH64_SAVE_R19
#define ITYR_AARCH64_RESTORE_R19
#define ITYR_AARCH64_CLOBBER_R19 "x19",
#endif

// must be a callee-saved register
#define ITYR_AARCH64_ORIG_SP_REG "x20"

struct context_frame_aarch64 {
  void*                  fp;
  void*                  lr;
  void*                  top;
  context_frame_aarch64* parent_frame;
};

class context_aarch64 {
  using context_frame = context_frame_aarch64;
  using save_context_fn_t = void (*)(context_frame*, void*, void*);
  using call_on_stack_fn_t = void (*)(void*, void*, void*, void*);
  using jump_to_stack_fn_t = void (*)(void*, void*, void*, void*);

public:
  static void save_context_with_call(context_frame*    parent_cf,
                                     save_context_fn_t fn,
                                     void*             arg0,
                                     void*             arg1) {
    register void* parent_cf_x9 asm("x9")  = reinterpret_cast<void*>(parent_cf);
    register void* fn_x10       asm("x10") = reinterpret_cast<void*>(fn);
    register void* arg0_x1      asm("x1")  = arg0;
    register void* arg1_x2      asm("x2")  = arg1;
    asm volatile (
        ITYR_AARCH64_SAVE_R19
        /* stack top and parent field of context */
        "sub x0, sp, #32\n\t"
        "stp x0, %0, [sp, #-16]!\n\t"
        /* save FP (r29) and LR (r30) */
        "adr x30, 1f\n\t"
        "stp x29, x30, [sp, #-16]!\n\t"
        /* call function */
        "mov x0, sp\n\t"
        "blr %1\n\t"
        /* skip saved FP and LR when normally returned */
        "add sp, sp, #16\n\t"

        "1:\n\t"
        /* skip parent field */
        "add sp, sp, #16\n\t"
        ITYR_AARCH64_RESTORE_R19
      : "+r"(parent_cf_x9), "+r"(fn_x10), "+r"(arg0_x1), "+r"(arg1_x2)
      :
      : "x0", "x3", "x4", "x5", "x6", "x7",
        "x8", "x11", "x12", "x13", "x14", "x15",
        "x16", "x17", "x18", ITYR_AARCH64_CLOBBER_R19 "x20", "x21", "x22", "x23",
        "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        ITYR_AARCH64_FLOAT_CLOBBERS,
        "cc", "memory"
    );
  }

  static void resume(context_frame* cf) {
    asm volatile (
        "mov sp, %0\n\t"
        "ldp x29, x30, [sp], #16\n\t"
        "ret\n\t"
      :
      : "r"(cf)
      :
    );
    // discard the current context
  }

  static void call_on_stack(void*              stack_buf,
                            size_t             stack_size,
                            call_on_stack_fn_t fn,
                            void*              arg0,
                            void*              arg1,
                            void*              arg2,
                            void*              arg3) {
    uintptr_t sp = reinterpret_cast<uintptr_t>(stack_buf) + stack_size - 1;
    sp &= 0xFFFFFFFFFFFFFFF0;

    register void* sp_x9   asm("x9")  = reinterpret_cast<void*>(sp);
    register void* fn_x10  asm("x10") = reinterpret_cast<void*>(fn);
    register void* arg0_x0 asm("x0")  = arg0;
    register void* arg1_x1 asm("x1")  = arg1;
    register void* arg2_x2 asm("x2")  = arg2;
    register void* arg3_x3 asm("x3")  = arg3;
    asm volatile (
        "mov " ITYR_AARCH64_ORIG_SP_REG ", sp\n\t"
        "mov sp, %0\n\t"
        "blr %1\n\t"
        "mov sp, " ITYR_AARCH64_ORIG_SP_REG "\n\t"
      : "+r"(sp_x9), "+r"(fn_x10),
        "+r"(arg0_x0), "+r"(arg1_x1), "+r"(arg2_x2), "+r"(arg3_x3)
      :
      : "x4", "x5", "x6", "x7",
        "x8", "x11", "x12", "x13", "x14", "x15",
        "x16", "x17", "x18", ITYR_AARCH64_ORIG_SP_REG,
        /* callee-saved registers are saved */
        ITYR_AARCH64_FLOAT_CLOBBERS,
        "cc", "memory"
    );
  }

  static void jump_to_stack(void*              stack_ptr,
                            jump_to_stack_fn_t fn,
                            void*              arg0,
                            void*              arg1,
                            void*              arg2,
                            void*              arg3) {
    uintptr_t sp = reinterpret_cast<uintptr_t>(stack_ptr) & 0xFFFFFFFFFFFFFFF0;

    register void* arg0_x0 asm("x0") = arg0;
    register void* arg1_x1 asm("x1") = arg1;
    register void* arg2_x2 asm("x2") = arg2;
    register void* arg3_x3 asm("x3") = arg3;
    asm volatile (
        "mov sp, %0\n\t"
        "blr %1\n\t"
      :
      : "r"(sp), "r"(fn),
        "r"(arg0_x0), "r"(arg1_x1), "r"(arg2_x2), "r"(arg3_x3)
      :
    );
    // discard the current context
  }

  static void clear_parent_frame(context_frame* cf) {
    // Workaround for generating backtracing.
    // Backtracing in libunwind often causes segfault because of the stack management.
    // That is, because stacks are moved across different nodes, their parent stack may not exist
    // in the current node. Thus, we clear the frame pointer and instruction pointer outside the
    // current stack (which should be in the parent stack area), so that backtracing does not
    // go further than that.
    if (cf->parent_frame) {
      cf->parent_frame->fp = nullptr;
      cf->parent_frame->lr = nullptr;
    }
  }

};

}

