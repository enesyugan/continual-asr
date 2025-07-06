import torch
from time import time


def spmm_python(csr_input, mat, size):
    """
    Pythonic SPMM (sparse * dense) using scatter_add.
    Args:
        row: 1D LongTensor of row indices of nonzeros (nnz,)
        col: 1D LongTensor of col indices of nonzeros (nnz,)
        value: 1D Tensor of nonzero values (nnz,) or None (assume 1)
        mat: 2D Tensor of shape (N, K), the dense matrix on the right
        size: tuple (M, N), the shape of the sparse matrix
    Returns:
        Tensor of shape (M, K)
    """

    coo = csr_input.to_sparse()
    row, col = coo.indices()
    value = coo.values()

    M, N = size
    K = mat.size(1)
    if value is None:
        value = torch.ones_like(row, dtype=mat.dtype)

    # sparse matrix multiply: sum over N
    # output[i] += value[nz] * mat[col[nz]]
    output = torch.zeros((M, K), dtype=mat.dtype, device=mat.device)
    updates = value.unsqueeze(1) * mat[col]  # shape: (nnz, K)
    output.index_add_(0, row, updates)
    return output


class SpspmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices_A, values_A, shape_A,
                indices_B, values_B, shape_B):
        # Save for backward

        ctx.shapes = (shape_A, shape_B)
        ctx.indices = (indices_A, indices_B)

        M = shape_A[0]
        K = shape_A[1]
        assert K == shape_B[0]
        N = shape_B[1]

        sparse_A = torch.sparse_coo_tensor(indices_A, values_A, size=shape_A, device=values_A.device).to_sparse_csr()
        sparse_B = torch.sparse_coo_tensor(indices_B, values_B, size=shape_B, device=values_B.device).to_sparse_csr()

        ctx.save_for_backward(sparse_A, sparse_B)

        sparse_C = torch.sparse.mm(sparse_A, sparse_B)  # M x N

        return sparse_C.to_dense()

    @staticmethod
    def backward(ctx, dense_grad_C):
        sparse_A, sparse_B = ctx.saved_tensors
        shape_A, shape_B = ctx.shapes
        indices_A, indices_B = ctx.indices

        # M x N \times N x K -> M x K
        # dense_grad_A = torch.sparse.mm(dense_grad_C, sparse_B.t()).t()
        #
        # dense_grad_A = spmm_sum(indices_B[0], sparse_B.crow_indices(), sparse_B.col_indices(),
        #                         sparse_B.values(), None, None, dense_grad_C.t())
        dense_grad_A = spmm_python(sparse_B, dense_grad_C.t(), shape_B)
        grad_values_A = dense_grad_A.t()[[indices_A[0], indices_A[1]]]
        # K x M \times M x N -> K x N

        del dense_grad_A  # <-- explicitly free memory
        dense_grad_B = torch.sparse.mm(sparse_A.t(), dense_grad_C)
        grad_values_B = dense_grad_B[[indices_B[0], indices_B[1]]]

        return None, grad_values_A, None, None, grad_values_B, None


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.amp.autocast_mode._cast(args, 'cuda', torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.amp.autocast_mode._cast(args, 'cuda', torch.half)


def spspmm_autograd(values_A, indices_A, shape_A,
                    values_B, indices_B, shape_B):
    args = _cast_if_autocast_enabled(values_A, indices_A, shape_A,
                                     values_B, indices_B, shape_B)
    with torch.amp.autocast('cuda', enabled=False):
        return SpspmmFunction.apply(*args)
