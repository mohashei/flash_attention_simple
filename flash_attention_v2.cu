#include <torch/torch.h>
#include <torch/types.h>
#include <vector>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template<int MAX_D>
__global__ void flash_attention_v2(
    float* O,
    float* L,
    const float* Q,
    const float* K,
    const float* V,
    const int M,
    const int N,
    const int d,
    const int Bc,
    const float softmax_scale
) {
    const int tx = threadIdx.x;
    // B, nh, ceil(N / M)
    const int bx = blockIdx.x;  // bx = batch
    const int by = blockIdx.y;  // by = head
    const int bz = blockIdx.z;  // bz = seqlen index

    // Offset into Q,K,V,O,L - different for each batch and head and part of the sequence
    // gridDim.y = nh, gridDim.z = ceil(N / M), we will process M elements of the sequence at a time
    // offset for QKV
    const int q_offset = (bx * gridDim.y * N * d) + (by * N * d) + (bz * M * d);
    const int kv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int valid_rows = min(M, N - bz * M);
    if (valid_rows <= 0) return;
    // offset for L
    const int lse_offset = (bx * gridDim.y * N) + (by * N) + bz * M;

    // Define SRAM for Q,K,V,S
    const int tile_size = Bc * d;  // size of Qi, Kj, Vj
    extern __shared__ float sram[];
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[2 * tile_size];
    float* S = &sram[3 * tile_size];
    float acc_o[MAX_D] = {0.0f};
    float l;
    float m;

    const int Tr = CEIL_DIV(M, Bc);
    const int Tc = CEIL_DIV(N, Bc);

    // Load Q tile (outer loop)
    for (int i = 0; i < Tr; i++) {

        // Load Qi ∈ R^(Bc x d) to SRAM
        // Make sure the threads are accessing contiguous elements.
        // No check because Bc is the number of threads.
        for (int x = 0; x < tile_size; x+=Bc) {
            Qi[x + tx] = Q[q_offset + (tile_size * i) + x + tx];
        }

        __syncthreads();

        l = 0.f;
        m = -INFINITY;
        for (int x = 0; x < d; x++) {
            acc_o[x] = 0.0f;
        }
        // Inner loop over K/V blocks
        for (int j = 0; j < Tc; j++) {
            // Load Kj, Vj ∈ R^(Bc x d) to SRAM
            // Make sure the threads are accessing contiguous elements.
            // No check because Bc is the number of threads.
            for (int x = 0; x < tile_size; x+=Bc) {
                Kj[x + tx] = K[kv_offset + (tile_size * j) + x + tx];
                Vj[x + tx] = V[kv_offset + (tile_size * j) + x + tx];
            }

            __syncthreads();

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // Compute new m
            float row_m_new = max(m, row_m);

            // P = exp(S - row_m_new), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m_new);
                row_l += S[(Bc * tx) + y];
            }

            float row_m_exp = __expf(m - row_m_new);
            float row_l_new = row_l + l * row_m_exp;

            // Write O, L to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                acc_o[x] = row_m_exp * acc_o[x] + pv;
            }
            m = row_m_new;
            l = row_l_new;
        }
        for (int x = 0; x < d; x++) {
            O[q_offset + (tile_size * i) + (tx * d) + x] = acc_o[x] / l;
        }
        L[lse_offset + tx] = __logf(l) + m;
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

void launch_flash_attention(
    at::Tensor& O,
    at::Tensor& L,
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    const int M,
    const int Bc,
    const float scale
) {
    const int N_dim = Q.size(2);
    const int d_dim = Q.size(3);
    constexpr int MAX_D = 128;
    if (MAX_D < d_dim) {
        std::cerr << "Head hidden dimension too large, should be at most " << MAX_D;
    }
    
    const dim3 grid(Q.size(0), Q.size(1), (N_dim + M - 1)/M);
    const dim3 block(Bc, 1, 1);
    const size_t shared_mem = (3 * Bc * d_dim + Bc * Bc) * sizeof(float);

    flash_attention_v2<MAX_D><<<grid, block, shared_mem>>>(
        O.data_ptr<float>(),
        L.data_ptr<float>(),
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        M,
        N_dim,
        d_dim,
        Bc,
        scale
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Post-kernel sync error: " 
                << cudaGetErrorString(err) << "\n";
    }
}