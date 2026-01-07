# ruff: noqa
import torch
import tilelang
from tilelang import language as T
import argparse


@tilelang.jit(
    out_idx=[-2, -1],
    compile_flags=[
        "-O3",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ],
)
def sparse_mla_fwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=0,
    threads=384,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, 'non-casual is not supported'
    assert topk % block_I == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            'here we solve the H padding automatically, other wise you '
            'should handle Q copy and Output copy with your mask (when '
            'kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would '
            'be handled automatically)'
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    assert NI % 2 == 0, 'NI should be a multiple of 2'
    D = dim
    D_tail = tail_dim
    KV_stride = kv_stride
    if head_kv > 64:
        assert head_kv % 64 == 0, 'head_kv should be a multiple of 64'
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    # 从32->64能够减少用于读kvcache所花的时间，如果num_query_head = 128,
    # num_kv_head = 1, 原本同样的kvcache需要读4次，改变之后只需要读2次了
    # TODO: 当H_per_block = 64, num_query_head = 128时，有两个
    # thread block会读取相同的kvcache，能不能将两次hbm读取优化成只读
    # 一次，比如说读到L2 cache中然后boardcast到两个thread block的shared
    # memory中，或者说使用thread block cluster功能，相同的kvcache
    # 只需要从hbm中读一次就能够读到cluster中所有thread block的shared mem中,
    # 最重要的是能够将两个hbm读减少至一次，这会极大提升io throughput(tb/s)和tflops/s

    # NOTE: tilelang中有T.call_extern和T.ptx
    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor(1, indices_dtype), # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(
            # 如果CP0为True(i.e., 序列开始部分)，需要跳过前(KV_stride - 1)
            # 无法看到任何KV的query，同时要小心seq_len < kv_stride导致grid size为负
            (max(0, seq_len - kv_stride + 1) if CP0 else seq_len) * REPLICATE_H,
            batch,
            kv_group,
            threads=threads
        ) as (bx, by, bz):
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            
            KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
            
            O_shared_l = Q_shared_l
            O_shared_r = Q_shared_r

            # 对这个query而言，当前BI中的kv是否可见
            # producer会交替写入buf0和buf1的mask，为了避免出现consumer0还在
            # 读取buf0 mask时，producer已经开始写入buf1 mask的情况，选择使用
            # 两个buf mask
            is_kv_valid = T.alloc_shared([2, BI], "bool", scope="shared")

            acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
            acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
            
            # WG0计算S0(BI_2*i)，WG1计算S1(BI_2*i+1)，通过shared mem进行分享

            # Reuse K_tail_shared for S_shared to save memory when dimensions match
            # 必须进行复用，否则h100中sm的shared mem不够用(> 228kb)，属于shared mem bound
            S_shared_0 = K_tail_shared_0
            S_shared_1 = K_tail_shared_1
            
            # WG0和WG1双方交换local max，比较然后计算出global max，并据此rescale自己的O_L或O_R
            row_max_shared_0 = T.alloc_shared([H_per_block], accum_dtype)
            row_max_shared_1 = T.alloc_shared([H_per_block], accum_dtype)

            # 分别用于存储偶数BI和奇数BI的sum of exps，之后需要加起来进行整合
            row_sum_shared_0 = T.alloc_shared([H_per_block], accum_dtype)
            row_sum_shared_1 = T.alloc_shared([H_per_block], accum_dtype)

            # acc_s, sumexp, m_i都需要分别为consumer0和consumer1分配一份
            acc_s_0 = T.alloc_fragment([H_per_block, BI], accum_dtype)
            acc_s_1 = T.alloc_fragment([H_per_block, BI], accum_dtype)
            
            sumexp_0 = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i_0 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_0 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev_0 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_peer_0 = T.alloc_fragment([H_per_block], accum_dtype)
            
            sumexp_1 = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i_1 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_1 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev_1 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_peer_1 = T.alloc_fragment([H_per_block], accum_dtype)
            
            bar_q = T.alloc_barrier(arrive_count=384)
            
            # Producer -> Consumer Barriers
            bar_k_0_ready = T.alloc_barrier(arrive_count=128) # Prod arrives
            bar_k_1_ready = T.alloc_barrier(arrive_count=128) # Prod arrives
            
            # Consumer -> Producer Barriers (Both consumers must arrive)
            bar_k_0_free = T.alloc_barrier(arrive_count=256) 
            bar_k_1_free = T.alloc_barrier(arrive_count=256)
            
            # Inter-Consumer Barriers (Seesaw Sync)
            bar_stats_0_ready = T.alloc_barrier(arrive_count=128) # Cons 0 arrives
            bar_stats_1_ready = T.alloc_barrier(arrive_count=128) # Cons 1 arrives
            
            bar_S_0_ready = T.alloc_barrier(arrive_count=128) # Cons 0 arrives
            bar_S_1_ready = T.alloc_barrier(arrive_count=128) # Cons 1 arrives

            b_i, g_i = by, bz
            # 如果是第一个chunk，直接从第(kv_stride - 1)个token开始计算
            s_i = (bx + (KV_stride - 1 if CP0 else 0)) if REPLICATE_H == 1 else (
                bx // REPLICATE_H + (KV_stride - 1 if CP0 else 0))
            q_i = q_start_index_s[0] + s_i
            # 有时候为了减少kvcache的大小，可能不会为每一个token存放一份KV，而是
            # 每隔KV_stride个token存储一份KV(一般是stride window中的最后一个token)，
            # 所以当前query能够看到的kv范围应该是[0:max_kv_i]
            max_kv_i = (q_i + 1 - KV_stride) // KV_stride

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()

            T.copy(Q[b_i, s_i, H0:H1, 0:D // 2], Q_shared_l)
            T.copy(Q[b_i, s_i, H0:H1, D // 2:D], Q_shared_r)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            # 非阻塞地将barrier内部计数器加一，producer线程可以提前去加载kv
            T.barrier_arrive(bar_q)

            if tx >= 256:

                # producer: prefetch kvcache to shared mem
                T.set_max_nreg(72, 0)
                
                prefetch_indices_0 = T.alloc_fragment([4], indices_dtype)
                prefetch_indices_1 = T.alloc_fragment([4], indices_dtype)
                
                # Prime the Pump! 预读iter_0的索引
                for r in T.serial(4):
                    # 这里读取会产生long scoreboard stall，但是在循环开始之前只发生一次
                    prefetch_indices_0[r] = Indices[b_i, s_i, g_i, r * 16 + (tx - 256) // 8]
                    prefetch_indices_1[r] = Indices[b_i, s_i, g_i, BI + r * 16 + (tx - 256) // 8]

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    # 等待KV_shared_0_l和KV_shared_0_r都被使用完毕

                    T.barrier_wait(bar_k_0_free[0], (i_i & 1))

                    # 一个Block大小`BI`是64，加载过程被分为4次迭代，每次只处理16个indices
                    # producer一共有128个线程，8个连续线程一组协作加载一个index对应的kv
                    for r in T.serial(4):
                        # mitigate long scoreboard stall here
                        index = prefetch_indices_0[r]
                        is_kv_valid[0, r * 16 + (tx - 256) // 8] = index <= max_kv_i
                        if is_kv_valid[0, r * 16 + (tx - 256) // 8]:
                            # 这里假设dim = 512, tail_dim = 64
                            with T.attr("default", "async_scope", 1):
                                # 8个线程协作加载一行KV_dim（长度为512），分为4个iteration进行加载，
                                # 每个iteration中，每个线程分别为KV_shared_0_l和KV_Shared_0_r
                                # 加载连续的8个元素，8个线程总共分别加载了64个元素
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        # (tx - 256) // 8决定了线程负责哪一行，而
                                        # (tx - 256) % 8决定了线程加载一行中的哪一部份
                                        KV_shared_0_l[r * 16 + (tx - 256) // 8,
                                                      64 * u + (tx - 256) % 8 * 8 +
                                                      v] = KV[b_i, index, g_i,
                                                              64 * u + (tx - 256) % 8 * 8 + v]
                                        KV_shared_0_r[r * 16 + (tx - 256) // 8,
                                                      64 * u + (tx - 256) % 8 * 8 +
                                                      v] = KV[b_i, index, g_i, D // 2 +
                                                              64 * u + (tx - 256) % 8 * 8 + v]
                            with T.attr("default", "async_scope", 1):
                                # tail_dim(长度为64)只需8个线程一组协作一个iteration就能够加载完成
                                for v in T.vectorized(8):
                                    K_tail_shared_0[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 +
                                                    v] = KV[b_i, index, g_i,
                                                            D + (tx - 256) % 8 * 8 + v]
                    T.cp_async_barrier_noinc(bar_k_0_ready[0])

                    if i_i + 1 < T.ceildiv(NI, 2):
                        # 为下一轮kv数据加载异步预取需要的索引，能够和本轮的kv数据加载实现重叠隐藏延迟
                        for r in T.serial(4):
                            prefetch_indices_0[r] = Indices[b_i, s_i, g_i, ((i_i + 1) * 2) * BI + r * 16 + (tx - 256) // 8]

                    # Buffer 1
                    T.barrier_wait(bar_k_1_free[0], (i_i & 1))

                    for r in T.serial(4):
                        index = prefetch_indices_1[r]
                        is_kv_valid[1, r * 16 + (tx - 256) // 8] = index <= max_kv_i
                        if is_kv_valid[1, r * 16 + (tx - 256) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        KV_shared_1_l[r * 16 + (tx - 256) // 8,
                                                      64 * u + (tx - 256) % 8 * 8 +
                                                      v] = KV[b_i, index, g_i,
                                                              64 * u + (tx - 256) % 8 * 8 + v]
                                        KV_shared_1_r[r * 16 + (tx - 256) // 8,
                                                      64 * u + (tx - 256) % 8 * 8 +
                                                      v] = KV[b_i, index, g_i, D // 2 +
                                                              64 * u + (tx - 256) % 8 * 8 + v]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_1[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 +
                                                    v] = KV[b_i, index, g_i,
                                                            D + (tx - 256) % 8 * 8 + v]
                    T.cp_async_barrier_noinc(bar_k_1_ready[0])

                    if i_i + 1 < T.ceildiv(NI, 2):
                        for r in T.serial(4):
                            prefetch_indices_1[r] = Indices[b_i, s_i, g_i, ((i_i + 1) * 2 + 1) * BI + r * 16 + (tx - 256) // 8]

            elif tx < 128:
                # 检查是否已经有384个线程arrive过bar_q(phase0已经完成)，如果
                # 没有的话继续等待，否则直接通过
                T.barrier_wait(bar_q, 0)

                # pre-arrive free barriers to indicate buffers are initially free
                # 在最开始phase0用于告诉producer可以往两个buffer中加载数据
                T.barrier_arrive(bar_k_0_free[0])
                T.barrier_arrive(bar_k_1_free[0])
                
                # Consumer 0 (WG0): Responsible for Even Blocks and O_L (Left Half)
                T.set_max_nreg(216, 1)
                T.fill(sumexp_0, 0)
                for h_i in T.Parallel(H_per_block):
                    m_i_0[h_i] = -5e4
                T.fill(acc_o_l, 0)

                # 每个iteration两个consumer合作计算两个BI
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # --- Step 1: Compute S0 = Q @ K0^T (Even Block) ---
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                    T.fill(acc_s_0, 0)
                    T.gemm(Q_shared_l, KV_shared_0_l, acc_s_0, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_0_r, acc_s_0, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_0, acc_s_0, transpose_B=True, wg_wait=-1)

                    T.copy(m_i_0, m_i_prev_0)
                    T.wait_wgmma(0)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        if not is_kv_valid[0, bi_i]:
                            acc_s_0[h_i, bi_i] = -5e4
                    T.reduce_max(acc_s_0, m_i_0, dim=1, clear=False)

                    # --- Step 2: Local Softmax Stats & Exchange ---
                    T.copy(m_i_0, row_max_shared_0)
                    T.barrier_arrive(bar_stats_0_ready)
                    # 如果consumer0在iter_i等待到了consumer1传递过来的
                    # local max值，这也意味着consumer1在iter_i-1已经使用
                    # 完了consumer0传递的S_0，所以下面可以不用阻塞直接往其中写入
                    T.barrier_wait(bar_stats_1_ready, (i_i & 1))
                    T.copy(row_max_shared_1, m_i_peer_0)
                    
                    # Update global max and scale O
                    for h_i in T.Parallel(H_per_block):
                        m_i_0[h_i] = T.max(m_i_0[h_i], m_i_peer_0[h_i])
                        
                    # Scale O_L
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= T.exp2((m_i_prev_0[h_i] - m_i_0[h_i]) * sm_scale)
                    
                    # Scale SumExp
                    for h_i in T.Parallel(H_per_block):
                        sumexp_0[h_i] *= T.exp2((m_i_prev_0[h_i] - m_i_0[h_i]) * sm_scale)
                        
                    # Compute P0 = exp(S0 - m_new)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s_0[h_i, bi_i] = T.exp2(acc_s_0[h_i, bi_i] * sm_scale - m_i_0[h_i] * sm_scale)
                    
                    # Update SumExp with P0
                    T.reduce_sum(acc_s_0, sumexp_i_0, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp_0[h_i] += sumexp_i_0[h_i]
                        
                    # --- Step 3: O_L += P0 @ V0_L (Self-Attention) ---
                    # Wait for S0 buffer to be free (consumed by peer in prev iter)
                    # T.barrier_wait(bar_S_0_free, (i_i & 1))
                    T.copy(acc_s_0, S_shared_0)
                    T.barrier_arrive(bar_S_0_ready)
                    
                    T.gemm(S_shared_0, KV_shared_0_l, acc_o_l, transpose_B=False, wg_wait=-1)
                    
                    # --- Step 4: O_L += P1 @ V1_L (Cross-Attention) ---
                    # Wait for P1 (S1) from peer
                    T.barrier_wait(bar_S_1_ready, (i_i & 1))
                    
                    T.gemm(S_shared_1, KV_shared_1_l, acc_o_l, transpose_B=False, wg_wait=-1)
                    
                    # NOTE: 但是k_0和k_1除了要被consumer0使用之外，也会被consumer1使用，所以这里并不会带来太大的性能提升
                    # 除了最近的异步gemm(i.e., S_shared_1 @ KV_shared_1_k)之外，其他都需要等待结束
                    T.wait_wgmma(1)
                    T.barrier_arrive(bar_k_0_free[0])
                    # 等待所有异步gemm结束
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_k_1_free[0])
                    
                T.copy(sumexp_0, row_sum_shared_0)
                T.barrier_arrive(bar_stats_0_ready) # Reuse barrier
                T.barrier_wait(bar_stats_1_ready, T.ceildiv(NI, 2) & 1)
                T.copy(row_sum_shared_1, sumexp_i_0) # Reuse sumexp_i buffer
                
                for h_i in T.Parallel(H_per_block):
                    sumexp_0[h_i] += sumexp_i_0[h_i]
                    
                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_l[h_i, d_i] /= sumexp_0[h_i]
                
                for h_i in T.Parallel(H_per_block):
                    sumexp_0[h_i] = T.log2(sumexp_0[h_i]) + m_i_0[h_i] * sm_scale
                    
                T.copy(acc_o_l, O_shared_l)
                T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0:D // 2])
                T.copy(sumexp_0, Lse[b_i, s_i, H0:H1]) # Write LSE

            elif tx >= 128 and tx < 256:
                T.barrier_wait(bar_q, 0)

                # pre-arrive free barriers to indicate buffers are initially free
                # 在最开始phase0用于告诉producer可以往两个buffer中加载数据
                T.barrier_arrive(bar_k_0_free[0])
                T.barrier_arrive(bar_k_1_free[0])

                # Consumer 1 (WG1): Responsible for Odd Blocks and O_R (Right Half)
                # NOTE: 256 * 216 + 128 * 72 = 64,512 < 65536(H100 SM RegFile Limit)，
                # 设置的寄存器数量再多就会发生hang了，都必须是8的倍数
                T.set_max_nreg(216, 1)
                T.fill(sumexp_1, 0)
                for h_i in T.Parallel(H_per_block):
                    m_i_1[h_i] = -5e4
                T.fill(acc_o_r, 0)
                
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # --- Step 1: Compute S1 = Q @ K1^T (Odd Block) ---
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                    T.fill(acc_s_1, 0)
                    T.gemm(Q_shared_l, KV_shared_1_l, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_1_r, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_1, acc_s_1, transpose_B=True, wg_wait=-1)

                    # --- Step 2: Local Softmax Stats & Exchange ---
                    T.copy(m_i_1, m_i_prev_1)
                    T.wait_wgmma(0)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        if not is_kv_valid[1, bi_i]:
                            acc_s_1[h_i, bi_i] = -5e4
                    
                    T.reduce_max(acc_s_1, m_i_1, dim=1, clear=False)
                    T.copy(m_i_1, row_max_shared_1)
                    T.barrier_arrive(bar_stats_1_ready)
                    T.barrier_wait(bar_stats_0_ready, (i_i & 1))
                    T.copy(row_max_shared_0, m_i_peer_1)
                    
                    for h_i in T.Parallel(H_per_block):
                        m_i_1[h_i] = T.max(m_i_1[h_i], m_i_peer_1[h_i])
                        
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= T.exp2((m_i_prev_1[h_i] - m_i_1[h_i]) * sm_scale)
                    
                    for h_i in T.Parallel(H_per_block):
                        sumexp_1[h_i] *= T.exp2((m_i_prev_1[h_i] - m_i_1[h_i]) * sm_scale)
                        
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s_1[h_i, bi_i] = T.exp2(acc_s_1[h_i, bi_i] * sm_scale - m_i_1[h_i] * sm_scale)
                    
                    T.reduce_sum(acc_s_1, sumexp_i_1, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp_1[h_i] += sumexp_i_1[h_i]

                    # --- Step 3: O_R += P1 @ V1_R (Self-Attention) ---
                    T.copy(acc_s_1, S_shared_1)

                    T.barrier_arrive(bar_S_1_ready)
                    
                    T.gemm(S_shared_1, KV_shared_1_r, acc_o_r, transpose_B=False, wg_wait=-1)
                    
                    # --- Step 4: O_R += P0 @ V0_R (Cross-Attention) ---
                    T.barrier_wait(bar_S_0_ready, (i_i & 1))
                    
                    T.gemm(S_shared_0, KV_shared_0_r, acc_o_r, transpose_B=False, wg_wait=-1)
                    
                    
                    T.wait_wgmma(1)
                    T.barrier_arrive(bar_k_1_free[0])
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_k_0_free[0])

                T.copy(sumexp_1, row_sum_shared_1)
                T.barrier_arrive(bar_stats_1_ready)
                T.barrier_wait(bar_stats_0_ready, T.ceildiv(NI, 2) & 1)
                T.copy(row_sum_shared_0, sumexp_i_1)
                
                for h_i in T.Parallel(H_per_block):
                    sumexp_1[h_i] += sumexp_i_1[h_i]

                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_r[h_i, d_i] /= sumexp_1[h_i]
                    
                T.copy(acc_o_r, O_shared_r)
                T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2:D])
                
    return main


def sparse_mla_fwd_interface(q,
                             kv,
                             indices,
                             q_start_index_s,
                             kv_stride,
                             sm_scale=None,
                             is_casual=True,
                             return_kernel=False,
                             print_kernel=False):
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, 'you should assign dim otherwise'
    dim = 512

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    if q_start_index_s != 0:
        assert q_start_index_s > kv_stride, "If it is because each cp has too short length, you should fix the logic involving CP0 (cp_rank == 0), to make sure q with pos < KV_Stride - 1 is masked (or you may just ignore how this is handled if nan in these q's Out would not effect others, which is reported to be likely to happen by wangding)"
    CP0 = q_start_index_s == 0

    # 对kernel进行编译
    kernel = sparse_mla_fwd(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride,
                            kv_group, sm_scale, is_casual, CP0)

    if print_kernel:
        print(kernel.get_kernel_source())

    if return_kernel:
        return kernel

    out, lse, = kernel(q, kv, indices, torch.tensor([q_start_index_s], 
                       dtype=torch.int32, device="cuda"))
    if q_start_index_s == 0 and kv_stride > 1:
        # 将前(kv_stride - 1)个位置的输出置为0，因为无法看到kv没有进行计算
        out[:, :kv_stride - 1, :, :] = 0
    return out, lse


def ref_sparse_mla_fwd_interface(q,
                                 kv,
                                 indices,
                                 q_start_index_s,
                                 kv_stride=1,
                                 sm_scale=None,
                                 is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape
    if q_start_index_s is None:
        q_start_index_s = sk * kv_stride - sq

    assert kv.shape[-1] == 576, 'you should assign dim otherwise'
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    num_kv_per_index = 1
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        q_start_index_s, sq + q_start_index_s, dtype=torch.int32,
        device="cuda").view(-1, 1) >= torch.arange(
            kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32, device="cuda").view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def test_sparse_mla_fwd_pipelined(B=1,
                                  S=4096,
                                  SKV=8192,
                                  H=128,
                                  HKV=1,
                                  DQK=576,
                                  DV=512,
                                  topk=2048,
                                  dtype=torch.bfloat16,
                                  # query在全局序列位置中(或者说相对于kv)的偏移量
                                  q_start_s_index=2048,
                                  check_correctness=True,
                                  profile=False):
    KV_stride = 1

    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device='cuda').requires_grad_(True) / 10
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device='cuda').requires_grad_(True) / 10
    q_start_s_index_t = torch.tensor([q_start_s_index], dtype=torch.int32, device="cuda")

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device='cuda')
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                # 加上偏移量q_start_s_index转换为全局序列位置
                i_i = torch.randperm(min(max(1, ((t + q_start_s_index) // KV_stride)), SKV))[:topk]
                indices[b, t, h, :len(i_i)] = i_i

    print("index generation finished")

    kernel = sparse_mla_fwd_interface(
        q, kv, indices, q_start_s_index, KV_stride, return_kernel=True, print_kernel=True)

    def fn():
        return kernel(q, kv, indices, q_start_s_index_t)

    if check_correctness:
        tl_out, tl_lse = fn()
        assert KV_stride == 1, "KV_stride > 1 not supported"
        # if q_start_s_index == 0 and KV_stride > 1:
        #     tl_out[:, :KV_stride - 1, :, :] = 0
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, q_start_s_index, KV_stride)
        print(f"tl_out: {tl_out}")
        print(f"ref_out: {ref_out}")
        torch.testing.assert_close(tl_out, ref_out, rtol=1e-3, atol=1e-3)

    if profile:
        print("Profiling mode: running minimal iterations (1 warmup + 1 run)...")
        fn()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        return

    from tilelang.profiler import do_bench
    ms = do_bench(
        fn,
        rep=20,
        warmup=10,
    )
    print(f"Average time: {ms:.3f} ms")
    print(f'fwd io bandwidth = ', (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    tflops = (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12
    print(f'fwd tflops = {tflops:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.test_correctness:
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 1, 1024, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
        test_sparse_mla_fwd_pipelined(
            B, S, SKV, H, HKV, DQK, DV, topk, dtype, check_correctness=True, profile=args.profile)
    else:
        # Prefill Benchmark: long context
        print(" --- Prefill Benchmark --- ")
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 2, 4096, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
        test_sparse_mla_fwd_pipelined(
            B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start_s_index=4096, check_correctness=False, profile=args.profile)

        # Decode Benchmark: large batch size, high throughput generation
        print("\n --- Decode Benchmark --- ")
        # Increase batch size to saturate h100 for decode
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 128 * 16, 2, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
        test_sparse_mla_fwd_pipelined(
            B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start_s_index=2048 + 4096, check_correctness=False, profile=args.profile)
