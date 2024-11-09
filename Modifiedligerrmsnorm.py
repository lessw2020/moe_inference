import math
import torch
import triton
import triton.language as tl
from triton.language.math import rsqrt

@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Llama-style RMS normalization: y_i = (x_i / RMS) * (offset + wi)
    where RMS = sqrt(sum(x_i^2) / N) is computed in fp32
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # Cast to fp32 for RMS computation
    X_row_fp32 = X_row.to(tl.float32)
    mean_square = tl.sum(X_row_fp32 * X_row_fp32, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)
    
    # Cache rstd for backward pass
    tl.store(RSTD_ptr, rstd)
    
    # Normalize in fp32, then cast back to original dtype for weight multiplication
    X_row = (X_row_fp32 * rstd).to(X_row_dtype)
    Y_row = X_row * (offset + W_row)
    
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)

@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    dY_ptr += row_start * dY_row_stride
    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start

    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    W_row = W_row + offset

    for _ in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(RSTD_ptr)

        X_row = X_row.to(tl.float32)
        m = (dY_row * W_row).to(tl.float32)
        
        dX_row = rstd_row * m
        dX_row += (rstd_row) * (
            -(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row
        )
        
        # Calculate gradient of W using original dtype
        dW_row += dY_row * (X_row * rstd_row).to(X_dtype)

        tl.store(dY_ptr + col_offsets, dX_row.to(X_dtype), mask=mask)

        dY_ptr += dY_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += RSTD_row_stride

    tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)

def calculate_settings(n_cols):
    """Calculate BLOCK_SIZE and num_warps based on n_cols."""
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    return BLOCK_SIZE, num_warps

def rms_norm_forward(X, W, eps, offset=0.0):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty_like(X)
    # RSTD is always computed/stored in fp32 for Llama
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    assert X.shape[1] == W.shape[0], "Incompatible dimensions"

    _rms_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        offset,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps

def rms_norm_backward(dY, X, W, RSTD, offset, BLOCK_SIZE, num_warps):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("Feature dimension must be < 64KB")
        
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)
    
    _rms_norm_backward_kernel[grid](
        dY,
        dY.stride(0),
        X,
        X.stride(0),
        X.dtype,
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        _dW,
        _dW.stride(0),
        n_rows,
        n_cols,
        offset,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    dX = dY.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype)
    return dX, dW

class LigerRMSNormFunction(torch.autograd.Function):
    """
    Performs RMSNorm (Root Mean Square Normalization) using Llama-style casting,
    where only the inverse RMS computation is done in fp32.
    """
    @staticmethod
    def forward(ctx, X, W, eps, offset=0.0):
        """
        Args:
            X: Input tensor of shape (B, T, H) or (BxT, H)
            W: Weight tensor of shape (H,)
            eps: Small constant for numerical stability
            offset: Offset added to weights (default: 0.0)
        """
        Y, X, RSTD, BLOCK_SIZE, num_warps = rms_norm_forward(X, W, eps, offset)
        ctx.offset = offset
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, RSTD = ctx.saved_tensors
        dX, dW = rms_norm_backward(
            dY, X, W, RSTD, ctx.offset, ctx.BLOCK_SIZE, ctx.num_warps
        )
        return dX, dW, None, None
