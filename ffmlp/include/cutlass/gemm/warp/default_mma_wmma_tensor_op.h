/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Default warp-level GEMM operators selected by data type, size, and layouts of operands.
*/

#pragma once

#include "cutlass/arch/wmma.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_tensor_op_wmma.h"

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    ///< Size of the Gemm problem (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements
    typename ElementA_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA_,
    /// Data type of B elements
    typename ElementB_,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB_,
    /// Element type of C matrix
    typename ElementC_,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC_,
    /// Operator describing the tensor operation
    typename Operator_ = arch::OpMultiplyAdd,
    /// Number of partitions along K dimension
    int PartitionsK = 1
>
struct DefaultMmaTensorOpWmma;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for m-by-n-by-kgroup
template <
    ///< Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements
    typename ElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Data type of B elements
    typename ElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Element type of C matrix
    typename ElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Operator describing the tensor operation
    typename Operator_,
    /// Number of partitions along K dimension
    int PartitionsK>
struct DefaultMmaTensorOpWmma {
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Wmma<
          InstructionShape_, 
          ElementA,
          LayoutA, 
          ElementB,
          LayoutB, 
          ElementC,
          LayoutC, 
          Operator_>,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOpWmma<
        WarpShape_,
        ElementA, 
        LayoutA, 
        ElementB, 
        LayoutB,
        ElementC, 
        LayoutC, 
        Policy, 
        PartitionsK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

#endif
