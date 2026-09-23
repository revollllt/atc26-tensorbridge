#pragma once

#include <type_traits>

#include <tensorbridge/common/tile.cuh>
#include <tensorbridge/dequant/fpma.cuh>
#include <tensorbridge/memory/g2s.cuh>
#include <tensorbridge/memory/r2s.cuh>
#include <tensorbridge/memory/s2g.cuh>
#include <tensorbridge/memory/s2r.cuh>
#include <tensorbridge/mma/wgmma.cuh>
#include <tensorbridge/scheduler/gemm.cuh>

namespace tensorbridge {

// NVFP4-weight x FP8-activation GEMM for SM90, `D = A @ B.T`:
//
//     A    [M, K]        FP8 E4M3 activations
//     SFA  [M]           FP32 per-token activation scales
//     B    [N, K / 2]    packed NVFP4 weights, repacked offline into thread order
//     SFB  [K / 16, N]   prefolded E4M3 weight scales
//     D    [M, N]        BF16 output
//
// `Config` (tile, pipeline depth, backend) is chosen per shape by
// `csrc/heuristics/sm90.hpp` and emitted by the JIT together with the WGMMA
// instruction of the tile.
//
// Each CTA runs one producer warp group and eight consumer warps. The producer
// streams K blocks from global memory into a ring of shared-memory stages
// (`g2s`); the consumers pull weights into registers (`s2r`), dequantize them
// there (`fpma`), multiply by the activations in shared memory (`wgmma`), and
// finally write the output tile back (`r2s`, `s2g`).
template <class Config>
__global__ __launch_bounds__(Tile::kNumThreads, 1) void sm90_fp8_nvfp4_gemm(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b,
    const __grid_constant__ std::conditional_t<Config::kUseTmaStore, CUtensorMap const, void *const> output,
    const float *gmem_sfa,
    const uint8_t *gmem_sfb,
    const float *gmem_global_scale,
    int32_t *locks,
    uint32_t shape_m) {
  constexpr uint32_t kNumStages = Config::kNumStages;
  constexpr uint32_t kKIters = Tile::kKIters;
  constexpr bool kFourBuffers = has_four_register_buffers(Config::kBlockM);
  constexpr uint32_t kNumBuffers = WeightRegisters<Config>::kNumBuffers;
  constexpr bool kStsmEpilogue = has_stsm_epilogue(Config::kBlockM) && Config::kUseTmaStore;

  static_assert(Config::kBlockM % 8 == 0 && Config::kBlockM <= 256, "Invalid token tile");
  static_assert(Config::kShapeN % Tile::kBlockN == 0 && Config::kShapeK % Tile::kBlockK == 0, "N and K must be 128-aligned");
  static_assert(kNumStages >= 3, "Too few stages to overlap loads with the mainloop");
  static_assert(Config::kMulticastA * Config::kMulticastB <= 2, "Clusters hold at most two CTAs");
  static_assert(!Config::kPrebroadcastScale || kFourBuffers, "Prebroadcast scales need the four-buffer tiles");
  static_assert(!Config::kUseStreamK || Config::kUseTmaStore, "Stream-K slices accumulate through TMA");
  // Known issue, inherited from the original implementation: with Stream-K, the
  // members of a multicast cluster sporadically store stale shared memory in the
  // tail of their output tile. Until that is understood, clusters are data
  // parallel only; `tests/` pins both multicast directions in that mode.
  static_assert(!Config::kUseStreamK || Config::kMulticastA * Config::kMulticastB == 1,
                "Stream-K runs on independent CTAs only");

  // The warp groups split the register file; the producer needs few registers,
  // and what it gives up goes to the consumers' accumulators and buffers.
  constexpr uint32_t kProducerRegs = Config::kNumProducerRegs;
  constexpr uint32_t kConsumerRegs = (256 - kProducerRegs / 2) / 8 * 8;
  static_assert(kProducerRegs >= 24 && kProducerRegs <= 128 && kProducerRegs % 8 == 0, "Invalid producer registers");

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage<Config> *>(shared_memory);
  auto scheduler = Scheduler<Config>(shape_m);

  if (threadIdx.x >= Tile::kNumMathThreads) {
    // ---- Producer warp group: global -> shared -------------------------------
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kProducerRegs));

    auto producer = Producer<Config>(smem, &tensor_map_a, &tensor_map_b, gmem_sfa, gmem_sfb);
    producer.init_barriers();
    __syncthreads();

    while (scheduler.get_next_block()) {
      uint32_t k_blocks_left = scheduler.slice_num_k_blocks;
      producer.seek(scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, shape_m);

      // Fill all but one slot; from then on, every slot the consumers release
      // is answered by loading the slot behind it.
      producer.wait_tile_stored();
      producer.template load_stage<true>(0);
      TB_UNROLL
      for (uint32_t stage_id = 1; stage_id < kNumStages - 1; stage_id++) {
        producer.load_stage(stage_id, stage_id < k_blocks_left);
      }

      while (k_blocks_left) {
        TB_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          if (k_blocks_left == 1) producer.load_token_scales();
          producer.wait_stage_consumed(stage_id);
          producer.load_stage(stage_id + kNumStages - 1, k_blocks_left >= kNumStages);
          k_blocks_left--;
          if (!k_blocks_left) break;
        }
      }
    }
  } else {
    // ---- Consumer warps: shared -> register -> WGMMA -> shared -> global -----
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(kConsumerRegs));

    WeightRegisters<Config> regs;
    float accum[kNumAccumulators<Config>];
    float token_scales[Config::kBlockM / 8];

    auto consumer = Consumer<Config>(smem);
    auto tile_writer = OutputTileWriter<Config>(smem.d);
    // A TMA descriptor is passed by value (grid constant) and used by address.
    auto output_ptr = [&]() -> const void * {
      if constexpr (Config::kUseTmaStore) return &output; else return output;
    };
    auto tile_store = OutputTileStore<Config>(smem.d, output_ptr(), locks);

    auto s2r = [&](uint32_t stage_id, uint32_t k_iter) {
      consumer.load_weights(stage_id, k_iter, regs.packed[k_iter % kNumBuffers]);
      consumer.load_scales(stage_id, k_iter, regs.scales[k_iter % kNumBuffers]);
    };
    // Two k32 calls per 16-byte weight read (dual-MMA preinterleaved layout).
    auto s2r_pair = [&](uint32_t stage_id, uint32_t k_iter) {
      consumer.load_weights_pair(stage_id, k_iter, regs.packed[k_iter], regs.packed[k_iter + 1]);
      consumer.load_scales(stage_id, k_iter, regs.scales[k_iter]);
      consumer.load_scales(stage_id, k_iter + 1, regs.scales[k_iter + 1]);
    };
    auto dequant = [&](uint32_t buffer_id) {
      fpma::dequant<Config::kPrebroadcastScale>(regs.packed[buffer_id], regs.scales[buffer_id], regs.fragment[buffer_id]);
    };
    auto issue = [&](uint32_t stage_id, uint32_t k_iter) {
      issue_wgmma<Config>(smem.a[stage_id], k_iter, regs.fragment[k_iter % kNumBuffers], accum);
    };

    __syncthreads();
    consumer.release(SharedStorage<Config>::kEpilogueSlot);

    while (scheduler.get_next_block()) {
      uint32_t k_blocks_left = scheduler.slice_num_k_blocks;
      const bool is_full_depth = scheduler.is_full_depth();

      TB_UNROLL
      for (uint32_t i = 0; i < kNumAccumulators<Config>; i++) accum[i] = 0;
      tile_store.seek(scheduler.m_block_id, scheduler.n_block_id, shape_m);
      const float global_scale = gmem_global_scale[0];

      consumer.template wait_stage<true>(0);
      if constexpr (!kFourBuffers) {
        s2r(0, 0);
        dequant(0);
      } else if (is_full_depth) {
        s2r_pair(0, 0);
        dequant(0);
      }

      while (k_blocks_left) {
        TB_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          const uint32_t next_stage_id = (stage_id + 1) % kNumStages;
          const bool has_next = k_blocks_left > 1;

          if constexpr (!kFourBuffers) {
            // Two buffers: load and dequantize call `i + 1` around the issue of
            // call `i`, then drain, since the buffer of call `i` is needed next.
            TB_UNROLL
            for (uint32_t k_iter = 0; k_iter < kKIters; k_iter++) {
              s2r((stage_id + (k_iter + 1) / kKIters) % kNumStages, (k_iter + 1) % kKIters);
              issue(stage_id, k_iter);
              if (k_iter == kKIters - 2) {
                consumer.release(stage_id);
                if (has_next) consumer.wait_stage(next_stage_id);
              }
              dequant((k_iter + 1) % kNumBuffers);
              ptx::wgmma_wait_group<0>();
            }
          } else if (is_full_depth) {
            // Lagged schedule, one call always in flight:
            //
            //     issue(i) -> wait<3> -> s2r(i + 2, i + 3) -> dequant(i + 1)
            //
            // `wait<3>` only blocks once all four buffers are busy, so loading
            // and the FPMA bridge run under the WGMMAs. The stage seam (release,
            // wait, first load of the next slot) sits behind the last issue for
            // the same reason. Kept a real loop: unrolled, the four bodies
            // outgrow the register file.
            uint32_t next_stage_landed = 1;
            for (uint32_t k_iter = 0; k_iter < kKIters; k_iter++) {
              issue(stage_id, k_iter);
              ptx::wgmma_wait_group<3>();

              if (k_iter == 0 && has_next) next_stage_landed = consumer.probe_stage(next_stage_id);

              if (k_iter == kKIters - 1) {
                consumer.release(stage_id);
                if (has_next) {
                  consumer.wait_stage(next_stage_id, next_stage_landed);
                  s2r_pair(next_stage_id, 0);
                  dequant(0);
                }
              } else {
                if (k_iter == 0) s2r_pair(stage_id, 2);
                dequant(k_iter + 1);
              }
            }
          } else {
            // Stream-K slice: the conservative schedule. Refill each buffer once
            // its previous call has retired, then issue the four calls together.
            ptx::wgmma_wait_group<3>();
            s2r(stage_id, 0);
            dequant(0);
            ptx::wgmma_wait_group<2>();
            s2r(stage_id, 1);
            dequant(1);
            ptx::wgmma_wait_group<1>();
            s2r(stage_id, 2);
            dequant(2);
            ptx::wgmma_wait_group<0>();
            s2r(stage_id, 3);
            dequant(3);

            TB_UNROLL
            for (uint32_t k_iter = 0; k_iter < kKIters; k_iter++) issue(stage_id, k_iter);

            consumer.release(stage_id);
            if (has_next) consumer.wait_stage(next_stage_id);
          }

          k_blocks_left--;
          if (!k_blocks_left) break;
        }
      }

      // ---- Epilogue: scale, round to BF16, transpose, store ------------------
      consumer.wait_token_scales();
      consumer.load_token_scales(token_scales);
      if constexpr (kFourBuffers) ptx::wgmma_wait_group<0>();

      tile_store.sync_math_threads();
      if (!is_full_depth) tile_store.lock_slice(scheduler.lock_idx, scheduler.slice_id);

      TB_UNROLL
      for (uint32_t i = 0; i < Config::kBlockM / 8; i++) token_scales[i] *= global_scale;
      if constexpr (kStsmEpilogue) {
        // A slice's partial sums take the scalar writer.
        if (is_full_depth) {
          tile_writer.write_stsm(accum, token_scales);
        } else {
          tile_writer.write_scalar(accum, token_scales);
        }
      } else {
        tile_writer.write_scalar(accum, token_scales);
      }
      tile_store.sync_math_threads();

      if constexpr (Config::kUseTmaStore) {
        ptx::tma_fence_shared_writes();
        tile_store.store_tma(scheduler.slice_id, scheduler.slice_count);
      } else {
        tile_store.store_direct();
      }
      tile_store.sync_math_threads();
      if (!is_full_depth) tile_store.unlock_slice(scheduler.lock_idx, scheduler.slice_id, scheduler.slice_count);

      if constexpr (Config::kUseTmaStore) ptx::tma_wait_store_group<0, true>();
      consumer.release(SharedStorage<Config>::kEpilogueSlot);
    }
  }

  __syncthreads();
  if constexpr (Config::kMulticastA * Config::kMulticastB > 1) {
    asm volatile("barrier.cluster.arrive;\n");
    asm volatile("barrier.cluster.wait;\n");
  }
}

}  // namespace tensorbridge
