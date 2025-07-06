import torch
from time import time
# import torch_sparse

from loras.sparse_linear import spspmm_autograd, spmm_python
import spmm_cuda
from spmm_cuda import spmm_sum



# Dimensions
M, K, N = 1024, 32768, 1024
nnz = 8192
B = 32


class SparseLORA(torch.nn.Module):

    def __init__(self, M, N, K, nnz):

        super(SparseLORA, self).__init__()

        self.M = M
        self.N = N
        self.K = K  # K is bottleneck size
        self.nnz = nnz

        rows_A = torch.randint(0, M, (nnz,))
        cols_A = torch.randint(0, K, (nnz,))

        self.indices_A = torch.stack([rows_A, cols_A])
        self.values_A = torch.nn.Parameter(torch.randn(nnz, requires_grad=True))
        self.shape_A = (M, K)

        rows_B = torch.randint(0, K, (nnz,))
        cols_B = torch.randint(0, N, (nnz,))
        self.indices_B = torch.stack([rows_B, cols_B])
        self.values_B = torch.nn.Parameter(torch.randn(nnz, requires_grad=True))
        self.shape_B = (K, N)

    def forward(self, x):

        # x should be [B x M]

        AB = spspmm_autograd(self.indices_A, self.values_A, self.shape_A,

                             self.indices_B, self.values_B, self.shape_B)

        # print(AB)
        output = torch.nn.functional.linear(x, AB.t())

        return output

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
dtype = torch.bfloat16
dtype = torch.float16

model = SparseLORA(M, N, K, nnz)
model = model.to(dtype)
model = model.cuda()

optimizer = torch.optim.AdamW(model.parameters())


model.zero_grad()

x = torch.randn((B, M), dtype=dtype)
x = x.cuda()

y = model(x)

loss = y.sum() * 1000

loss.backward()

print(model.values_A.grad)
print(model.values_B.grad)

optimizer.step()

# SPEED TEST
num_iters = 10000

test_dense = torch.randn((M, N), dtype=dtype).cuda()
test_sparse = torch.sparse_coo_tensor(model.indices_B, model.values_B,
                                      size=model.shape_B, device=model.values_B.device).to_sparse_csr()

test_sparse = test_sparse.cuda()

with torch.no_grad():
    torch.cuda.profiler.start()
    torch.cuda.synchronize()
    start_time = time()
    for _ in range(num_iters):
        dense_out = spmm_sum(model.indices_B[0], test_sparse.crow_indices(), test_sparse.col_indices(),
                             test_sparse.values(), None, None, test_dense.t())

    torch.cuda.synchronize()
    stop_time = time()
    print(F"\ncustom spmm cuda bf16 {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

    torch.cuda.profiler.start()
    torch.cuda.synchronize()
    start_time = time()
    for _ in range(num_iters):
        dense_out = torch.sparse.mm(test_dense, test_sparse.t()).t()

    torch.cuda.synchronize()
    stop_time = time()
    print(F"\ntorch.sparse spmm bf16 {(stop_time - start_time) * 1000. / num_iters:.4f} ms")


    torch.cuda.profiler.start()
    torch.cuda.synchronize()
    start_time = time()
    for _ in range(num_iters):
        dense_out = spmm_python(test_sparse, test_dense.t(), model.shape_B)

    torch.cuda.synchronize()
    stop_time = time()
    print(F"\nPytorch spmm bf16 {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

