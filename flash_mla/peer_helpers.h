#pragma once


// 一次传输 8 个 bf16 (128 bits)，这是最高效的方式
__device__ __forceinline__ void st_async_peer_bf16_x8(
    unsigned short v0, unsigned short v1, unsigned short v2, unsigned short v3,
    unsigned short v4, unsigned short v5, unsigned short v6, unsigned short v7,
    void* ptr,
    void* barrier_ptr
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint32_t peer_addr = smem_addr ^ 0x1000000;
    
    uint32_t bar_smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
    uint32_t peer_bar_addr = bar_smem_addr ^ 0x1000000;

    // Pack 8 bf16 into 4 registers (b32)
    uint32_t r0, r1, r2, r3;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(r0) : "h"(v0), "h"(v1));
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(r1) : "h"(v2), "h"(v3));
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(r2) : "h"(v4), "h"(v5));
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(r3) : "h"(v6), "h"(v7));

    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b128 [%0], {%1, %2, %3, %4}, [%5];"
        : 
        : "r"(peer_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(peer_bar_addr)
        : "memory"
    );
}

// 设置 Barrier 期望的 Transaction 字节数
__device__ __forceinline__ void expect_peer_tx(void* barrier_ptr, int bytes) {
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
    asm volatile(
        // expect_tx表示这次操作不是为了增加arrived thread count，
        // 而是expected transaction count
        // mbarrier.arrive.expect_tx.shared.b64 state, [barrier_addr], tx_count
        // 这条指令会原子地将tx_count加到[barrier_addr]指向的mbarrier对象的pending transaction counter上 
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        : 
        : "r"(bar_addr), "r"(bytes)
        : "memory"
    );
}