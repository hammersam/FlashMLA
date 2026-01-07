# ruff: noqa
import torch
import tilelang
from tilelang import language as T
import argparse

@tilelang.jit(
    out_idx=[-2, -1],
    compile_flags=[
        "-I/./",
        "-O3",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG"
    ],
    include_files=["peer_helpers.h"],
)
def decode_splitkv_mla_fp8(
    batch,
    heads_q,
    heads_k,
    dim,
    topk,
    block_size=64,
    num_splits=1,
    threads=384,
    max_num_blocks=65536, # Max physical blocks in KV cache
):
    """
    Mimics splitkv_mla.cu:
    1. Split-KV: Grid z-dim represents splits.
    2. 3-Warpgroup Pipeline: 2 Consumers, 1 Producer.
    3. FP8 Dequantization: Producer loads int8, dequants to bf16.
    4. Crossover: Structure allows for peer SMEM access (simulated here).
    5. Paged Attention: Supports loading KV from non-contiguous physical blocks via token indices.
    """
    
    # Shapes
    # Q: [batch, 1, heads_q, dim]
    # KV is sparse/paged. We assume a flattened view or block table lookup.
    # For simplicity in this kernel, we assume KV is [TotalBlocks, BlockSize, HeadsK, Dim]
    # Indices: [batch, heads_q, topk // block_size]
    
    # FP8 is stored as int8
    dtype_q = "bfloat16"
    dtype_kv_storage = "int8" 
    dtype_comp = "float"
    
    # Dimensions
    H_Q = heads_q
    H_K = heads_k
    D = dim
    BI = block_size
    
    # Grid: [Batch * HeadsQ, NumSplits]
    # We map bx to (Batch, HeadQ)
    
    @T.prim_func
    def main(
        Q: T.Tensor([batch, 1, H_Q, D], dtype_q),
        KV_FP8: T.Tensor([max_num_blocks, BI, H_K, D], dtype_kv_storage), # Physical KV pool
        KV_Scales: T.Tensor([max_num_blocks, BI, H_K, 1], "float32"), # Per-token scales
        Indices: T.Tensor([batch, H_Q, topk], "int32"), # Token indices (TopK)
        Output: T.Tensor([batch, num_splits, H_Q, D], "float32"), # Partial Output
        LSE: T.Tensor([batch, num_splits, H_Q], "float32"), # Partial LSE
    ):
        with T.Kernel(
            batch * H_Q,
            num_splits,
            threads=threads
        ) as (bx, by):
            
            # Parse Grid
            batch_idx = bx // H_Q
            head_idx = bx % H_Q
            split_idx = by
            
            # Cluster Logic (Hypothetical)
            cluster_size = 2
            rank_in_cluster = bx % cluster_size
            peer_rank = (rank_in_cluster + 1) % cluster_size
            
            # Shared Memory Allocations
            # Q: [1, D] (One query vector per block)
            Q_shared = T.alloc_shared([1, D], dtype_q)
            
            # KV Buffers (Double Buffering)
            # K, V: [BI, D]
            # We allocate 2 buffers for pipelining
            K_shared = T.alloc_shared([2, BI, D], dtype_q)
            V_shared = T.alloc_shared([2, BI, D], dtype_q)
            
            # Accumulators
            acc_o = T.alloc_fragment([1, D], dtype_comp)
            acc_m = T.alloc_fragment([1], dtype_comp)
            acc_l = T.alloc_fragment([1], dtype_comp)
            
            # Pipeline Barriers
            # Producer -> Consumer
            bar_kv_ready = T.alloc_barrier(arrive_count=128) # Producer (128 threads) arrives
            # Consumer -> Producer
            bar_kv_free = T.alloc_barrier(arrive_count=256) # Consumers (256 threads) arrive
            # Remote Producer -> Consumer (Crossover)
            # 一个memory barrier，专门用于跟踪asynchronous memory transactions，
            # 在crossover中用于确保peer block写到local block's shared memory中的数据已经完全到达
            #
            # Local Block (Consumer)                     Peer Block (Producer)
            # ----------------------                     ---------------------
            # 1. expect_peer_tx(bar, 128 bytes)
            # (Barrier计数 += 128)
            #         |
            #         |                                2. st_async(data, peer_smem, peer_bar)
            #         |                                   (写入数据 -> 硬件传输 -> 写入完成)
            #         |                                            |
            #         |<-------------------------------------------+
            #         | (硬件自动操作: Barrier计数 -= 128)
            #         |
            # 3. barrier_wait(bar)
            # (如果计数==0，继续执行)
            # (开始计算 Attention)
            #
            bar_kv_remote_ready = T.alloc_barrier(arrive_count=1)
            
            # Init Accumulators
            T.fill(acc_o, 0)
            T.fill(acc_m, -5e4) # -inf
            T.fill(acc_l, 0)
            
            # Load Q
            tx = T.get_thread_binding()
            if tx < 32: # Use one warp to load Q
                for i in T.vectorized(D // 32):
                    # Simple coalesced load
                    Q_shared[0, tx * (D // 32) + i] = Q[batch_idx, 0, head_idx, tx * (D // 32) + i]
            
            T.barrier_arrive(bar_kv_free[0]) # Signal buffer 0 is free
            T.barrier_arrive(bar_kv_free[1]) # Signal buffer 1 is free
            
            # Calculate Split Range
            total_blocks = topk // BI
            blocks_per_split = T.ceildiv(total_blocks, num_splits)
            start_block = split_idx * blocks_per_split
            end_block = T.min(start_block + blocks_per_split, total_blocks)
            num_iters = end_block - start_block
            
            # --- Main Loop ---
            for i in T.Pipelined(num_iters, num_stages=2):
                # Pipeline Stage: Load & Dequant (Producer) vs Compute (Consumer)
                
                # === Producer Logic (WG2: Threads 256-383) ===
                if tx >= 256:
                    buf_idx = i % 2
                    
                    # Wait for buffer to be free
                    T.barrier_wait(bar_kv_free[buf_idx], (i // 2) & 1)
                    
                    
                    # 3. Load FP8 KV, Dequantize, and Store to Shared
                    # We have 128 threads to load BI*D elements.
                    # BI=64, D=512 (example) -> 32768 elements.
                    # Each thread loads 256 elements.
                    
                    # Vectorized Load & Dequant Loop
                    # Crossover: Split work between blocks in cluster
                    rows_per_block = BI // cluster_size
                    
                    curr_sparse_block_idx = start_block + i

                    # Set expectation for remote data
                    # Each block receives 'rows_per_block' rows from peer
                    # Each row has D elements (bf16) + V part (bf16)
                    # Total bytes = rows_per_block * D * 2 (bytes/bf16) * 2 (K and V)
                    if tx == 256: # Only one thread needs to set expectation
                        expected_bytes = rows_per_block * D * 2 * 2
                        # 数据传输开始之前，local block(同时作为数据的接收方和消费方)需要
                        # 告诉barrier预计会有多少字节的数据通过异步写入到达
                        T.call_extern("expect_peer_tx", bar_kv_remote_ready[buf_idx], expected_bytes)
                        # 满足 arrive_count=1 的期望，使得 barrier 能够仅等待事务完成
                        T.barrier_arrive(bar_kv_remote_ready[buf_idx])

                    row_offset = rank_in_cluster * rows_per_block
                    
                    for r_local in T.serial(rows_per_block):
                        r = row_offset + r_local
                        
                        # 1. Load Token Index (Paged Attention Indirection)
                        # Indices shape: [batch, heads, topk]
                        # We access the r-th token in the current sparse block
                        token_idx = Indices[batch_idx, head_idx, curr_sparse_block_idx * BI + r]
                        
                        # 2. Compute Physical Address
                        # Handle invalid tokens (-1) by mapping to 0 and masking later (or relying on scale=0/mask)
                        valid_token = token_idx != -1
                        safe_token_idx = T.max(0, token_idx)
                        
                        phys_block_idx = safe_token_idx // BI
                        phys_offset = safe_token_idx % BI
                        
                        # 3. Load Scale (Per-token)
                        scale = KV_Scales[phys_block_idx, phys_offset, 0, 0]
                        
                        # Vectorize by 8 (128 bits) for efficient st_async
                        for c_outer in T.serial(D // 8):
                            # We process 8 elements at a time
                            # Each thread processes 8 elements? No, we have 128 threads.
                            # Total elements per row = D = 512.
                            # 512 / 8 = 64 vectors.
                            # We have 128 threads.
                            # Let's map threads to vectors.
                            
                            vec_idx = r * (D // 8) + c_outer
                            if vec_idx % 128 == (tx - 256):
                                # Load & Dequant 8 values
                                vals_bf16 = T.alloc_fragment([8], dtype_q)
                                for v in T.serial(8):
                                    c = c_outer * 8 + v
                                    # Load from Physical Pool
                                    val_int8 = KV_FP8[phys_block_idx, phys_offset, 0, c]
                                    val_fp32 = T.cast(val_int8, "float32") * scale
                                    vals_bf16[v] = T.cast(val_fp32, dtype_q)
                                    
                                    if not valid_token:
                                        vals_bf16[v] = T.cast(0.0, dtype_q)

                                    # Store Local
                                    K_shared[buf_idx, r, c] = vals_bf16[v]
                                    V_shared[buf_idx, r, c] = vals_bf16[v] # Dummy V

                                # Store Remote (Peer) - Vectorized 128-bit write
                                T.call_extern("st_async_peer_bf16_x8",
                                              vals_bf16[0], vals_bf16[1], vals_bf16[2], vals_bf16[3],
                                              vals_bf16[4], vals_bf16[5], vals_bf16[6], vals_bf16[7],
                                              K_shared.access_ptr("w", offset=[buf_idx, r, c_outer * 8]),
                                              bar_kv_remote_ready[buf_idx])
                                
                                T.call_extern("st_async_peer_bf16_x8",
                                              vals_bf16[0], vals_bf16[1], vals_bf16[2], vals_bf16[3],
                                              vals_bf16[4], vals_bf16[5], vals_bf16[6], vals_bf16[7],
                                              V_shared.access_ptr("w", offset=[buf_idx, r, c_outer * 8]),
                                              bar_kv_remote_ready[buf_idx])

                    # Signal Data Ready
                    T.cp_async_barrier_noinc(bar_kv_ready[buf_idx])

                # === Consumer Logic (WG0 & WG1: Threads 0-255) ===
                elif tx < 256:
                    buf_idx = i % 2
                    
                    # Wait for Data
                    T.barrier_wait(bar_kv_ready[buf_idx], (i // 2) & 1)
                    T.barrier_wait(bar_kv_remote_ready[buf_idx], (i // 2) & 1)
                    
                    # Compute S = Q @ K.T
                    # Q: [1, D], K: [BI, D] -> S: [1, BI]
                    # We use T.gemm or manual dot product
                    
                    # Local Accumulator for S
                    S_local = T.alloc_fragment([BI], dtype_comp)
                    T.fill(S_local, 0)
                    
                    # GEMM: S = Q * K^T
                    # Simplified dot product loop
                    for k in T.serial(BI):
                        dot = T.alloc_fragment([1], dtype_comp)
                        T.fill(dot, 0)
                        for d in T.serial(D):
                            # In a real kernel, this is a Tensor Core MMA
                            dot[0] += T.cast(Q_shared[0, d], dtype_comp) * T.cast(K_shared[buf_idx, k, d], dtype_comp)
                        S_local[k] = dot[0]
                    
                    # Online Softmax Update
                    # 1. Calc Max
                    m_local = T.alloc_fragment([1], dtype_comp)
                    T.fill(m_local, -5e4)
                    for k in T.serial(BI):
                        m_local[0] = T.max(m_local[0], S_local[k])
                    
                    # Warp Reduce Max (Simplified)
                    # ...
                    
                    # Update Global Max & Rescale O
                    m_prev = acc_m[0]
                    m_new = T.max(m_prev, m_local[0])
                    acc_m[0] = m_new
                    
                    scale_o = T.exp2(m_prev - m_new)
                    acc_l[0] = acc_l[0] * scale_o
                    for d in T.serial(D):
                        acc_o[0, d] = acc_o[0, d] * scale_o
                        
                    # Compute Exp & Sum
                    l_local = T.alloc_fragment([1], dtype_comp)
                    T.fill(l_local, 0)
                    for k in T.serial(BI):
                        S_local[k] = T.exp2(S_local[k] - m_new)
                        l_local[0] += S_local[k]
                        
                    acc_l[0] += l_local[0]
                    
                    # GEMM: O += S * V
                    # S: [1, BI], V: [BI, D] -> O: [1, D]
                    for d in T.serial(D):
                        dot = T.alloc_fragment([1], dtype_comp)
                        T.fill(dot, 0)
                        for k in T.serial(BI):
                            dot[0] += S_local[k] * T.cast(V_shared[buf_idx, k, d], dtype_comp)
                        acc_o[0, d] += dot[0]
                        
                    # Signal Buffer Free
                    T.barrier_arrive(bar_kv_free[buf_idx])
            
            # --- Epilogue ---
            # Store Partial Results
            if tx < D:
                # Simple store by first D threads
                Output[batch_idx, split_idx, head_idx, tx] = acc_o[0, tx]
            if tx == 0:
                LSE[batch_idx, split_idx, head_idx] = T.log2(acc_l[0]) + acc_m[0]

    return main

if __name__ == "__main__":
    # Simple compilation test
    B, H, D, TopK = 1, 32, 512, 2048
    kernel = decode_splitkv_mla_fp8(B, H, 1, D, TopK)
    print("Kernel compiled successfully.")
    print(kernel.get_kernel_source())