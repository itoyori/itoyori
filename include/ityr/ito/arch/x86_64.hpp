#pragma once

#include "ityr/common/util.hpp"

namespace ityr::ito {

#if defined(__AVX512F__)
#define ITYR_X86_64_FLOAT_REGS \
    "%xmm0" , "%xmm1" , "%xmm2" , "%xmm3" , "%xmm4" , "%xmm5" , "%xmm6" , "%xmm7" , \
    "%xmm8" , "%xmm9" , "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", \
    "%xmm16", "%xmm17", "%xmm18", "%xmm19", "%xmm20", "%xmm21", "%xmm22", "%xmm23", \
    "%xmm24", "%xmm25", "%xmm26", "%xmm27", "%xmm28", "%xmm29", "%xmm30", "%xmm31"
#else
#define ITYR_X86_64_FLOAT_REGS \
    "%xmm0" , "%xmm1" , "%xmm2" , "%xmm3" , "%xmm4" , "%xmm5" , "%xmm6" , "%xmm7" , \
    "%xmm8" , "%xmm9" , "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
#endif

struct context_frame_x86_64 {
  void*                 rip;
  void*                 rsp;
  void*                 rbp;
  context_frame_x86_64* parent_frame;
};

class context_x86_64 {
  using context_frame = context_frame_x86_64;
  using save_context_fn_t = void (*)(context_frame*, void*, void*);
  using call_on_stack_fn_t = void (*)(void*, void*, void*, void*);
  using jump_to_stack_fn_t = void (*)(void*, void*, void*, void*);

public:
  static void save_context_with_call(context_frame*    parent_cf,
                                     save_context_fn_t fn,
                                     void*             arg0,
                                     void*             arg1) {
    register void* parent_cf_r8 asm("r8") = reinterpret_cast<void*>(parent_cf);
    register void* fn_r9        asm("r9") = reinterpret_cast<void*>(fn);
    asm volatile (
        /* save red zone */
        "sub  $128, %%rsp\n\t"
        /* 16-byte sp alignment for SIMD registers */
        "mov  %%rsp, %%rax\n\t"
        "and  $0xFFFFFFFFFFFFFFF0, %%rsp\n\t"
        "push %%rax\n\t"
        /* alignment */
        "sub  $0x8, %%rsp\n\t"
        /* parent field of context frame */
        "push %0\n\t"
        /* push rbp */
        "push %%rbp\n\t"
        /* sp */
        "lea  -16(%%rsp), %%rax\n\t"
        "push %%rax\n\t"
        /* ip */
        "lea  1f(%%rip), %%rax\n\t"
        "push %%rax\n\t"
        /* call function */
        "mov  %%rsp, %%rdi\n\t"
        "call *%1\n\t"
        /* pop ip from stack */
        "add  $8, %%rsp\n\t"

        "1:\n\t" /* ip is popped with ret operation at resume */
        /* pop sp */
        "add  $8, %%rsp\n\t"
        /* pop rbp */
        "pop  %%rbp\n\t"
        /* parent field of context frame and align */
        "add  $16, %%rsp\n\t"
        /* revert sp alignmment */
        "pop  %%rsp\n\t"
        /* restore red zone */
        "add  $128, %%rsp\n\t"
      : "+r"(parent_cf_r8), "+r"(fn_r9),
        "+S"(arg0), "+d"(arg1)
      :
      : "%rax", "%rbx", "%rcx", "%rdi",
        "%r10", "%r11", "%r12", "%r13", "%r14", "%r15",
        ITYR_X86_64_FLOAT_REGS,
        "cc", "memory"
    );
  }

  static void resume(context_frame* cf) {
    asm volatile (
        "mov  %0, %%rsp\n\t"
        "ret\n\t"
      :
      : "g"(cf)
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

    register void* sp_r8 asm("r8") = reinterpret_cast<void*>(sp);
    register void* fn_r9 asm("r9") = reinterpret_cast<void*>(fn);
    asm volatile (
        "mov  %%rsp, %%rax\n\t"
        "mov  %0, %%rsp\n\t"
        /* alignment for SIMD register accesses */
        "sub  $0x8, %%rsp\n\t"
        "push %%rax\n\t"
        "call *%1\n\t"
        "pop  %%rsp\n\t"
      : "+r"(sp_r8), "+r"(fn_r9),
        "+D"(arg0), "+S"(arg1), "+d"(arg2), "+c"(arg3)
      :
      : "%rax", "%rbx",
        "%r10", "%r11", "%r12", "%r13", "%r14", "%r15",
        ITYR_X86_64_FLOAT_REGS,
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

    asm volatile (
        "mov  %0, %%rsp\n\t"
        "call *%1\n\t"
      :
      : "g"(sp), "r"(fn),
        "D"(arg0), "S"(arg1), "d"(arg2), "c"(arg3)
      :
    );
    // discard the current context
  }

};

}
