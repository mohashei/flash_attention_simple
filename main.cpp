#include <stdio.h>
#include <ATen/ATen.h>
#include <vector>
#include <iostream>

at::Tensor compute_reference(at::Tensor Q, at::Tensor K, at::Tensor V, float scale) {
    return at::scaled_dot_product_attention(
        Q, K, V, {}, 0.0, false, static_cast<double>(scale)
    );
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
);

int main(int argc, char **argv) {
    // Default values
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 256;
    int d = 64;
    int M = 256;
    int Bc = 16;
    float scale = 1.0f;
    int inspect_batch = -1;
    int inspect_heads = -1;
    int inspect_seq = -1;
    int inspect_d = -1;

    // Parse command line arguments
    std::vector<std::string> args(argv + 1, argv + argc);
    
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--batch-size") {
            if (i + 1 < args.size()) {
                batch_size = std::stoi(args[++i]);
            }
        } else if (args[i] == "--num-heads") {
            if (i + 1 < args.size()) {
                num_heads = std::stoi(args[++i]);
            }
        } else if (args[i] == "--seq-len") {
            if (i + 1 < args.size()) {
                seq_len = std::stoi(args[++i]);
            }
        } else if (args[i] == "--d") {
            if (i + 1 < args.size()) {
                d = std::stoi(args[++i]);
            }
        } else if (args[i] == "--M") {
            if (i + 1 < args.size()) {
                M = std::stoi(args[++i]);
            }
        } else if (args[i] == "--Bc") {
            if (i + 1 < args.size()) {
                Bc = std::stoi(args[++i]);
            }
        } else if (args[i] == "--scale") {
            if (i + 1 < args.size()) {
                scale = std::stof(args[++i]);
            }
        } else if (args[i] == "--inspect") {
            if (i + 4 < args.size()) {
                inspect_batch = std::stoi(args[++i]);
                inspect_heads = std::stoi(args[++i]);
                inspect_seq = std::stoi(args[++i]);
                inspect_d = std::stoi(args[++i]);
            }
        } else if (args[i] == "--help") {
            std::cout << "Usage: program [options]\n"
                      << "Options:\n"
                      << "  --batch-size N     Set batch size\n"
                      << "  --num-heads N      Set number of heads\n"
                      << "  --seq-len N        Set sequence length\n"
                      << "  --d N              Set dimension size\n"
                      << "  --M N              Set M value\n"
                      << "  --Bc N             Set Bc value\n"
                      << "  --scale F          Set scale factor\n"
                      << "  --inspect bhnd     If set, prints bhnd element of output tensor.";
            return 0;
        }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    
    auto Q = at::randn({batch_size, num_heads, seq_len, d}, options);
    auto K = at::randn_like(Q);
    auto V = at::randn_like(Q);
    auto O_kernel = at::zeros_like(Q);
    auto L_kernel = at::zeros({batch_size, num_heads, seq_len}, options);

    auto O_ref = compute_reference(Q, K, V, scale);

    try {
        // launch_flash_attention(O_kernel, L_kernel, Q, K, V, M, Bc, scale);
        launch_flash_attention(O_kernel, L_kernel, Q, K, V, M, Bc, scale);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    const float rtol = 1e-3f;
    const float atol = 1e-5f;
    bool all_close = at::allclose(O_kernel.cpu(), O_ref.cpu(), rtol, atol);

    if (!all_close) {
        // Calculate absolute differences between tensors
        auto diff = (O_kernel.cpu() - O_ref.cpu()).abs();
        
        // Find maximum and minimum differences
        auto max_diff = diff.max();
        auto min_diff = diff.min();
        
        // Get indices of maximum and minimum differences
        auto max_indices = at::argmax(diff);
        auto min_indices = at::argmin(diff);
        
        // Print debug information
        std::cout << "Test failed! Debug information:\n";
        std::cout << "Maximum difference: " << max_diff.item<float>() << "\n";
        std::cout << "Minimum difference: " << min_diff.item<float>() << "\n";
        std::cout << "Location of maximum difference: " << max_indices.item<float>() << "\n";
        std::cout << "Location of maximum difference: " << min_indices.item<float>() << "\n";
    }

    if (inspect_batch > -1) {
        
        // Validate indices
        if (inspect_batch >= 0 && inspect_batch < batch_size &&
            inspect_heads >= 0 && inspect_heads < num_heads &&
            inspect_seq >= 0 && inspect_seq < seq_len &&
            inspect_d >= 0 && inspect_d < d) {
            // Get the values at the specified indices
            auto kernel_val = O_kernel.cpu()[inspect_batch][inspect_heads][inspect_seq][inspect_d].item<float>();
            auto ref_val = O_ref.cpu()[inspect_batch][inspect_heads][inspect_seq][inspect_d].item<float>();
            auto diff = std::abs(kernel_val - ref_val);
            
            std::cout << "\nInspecting element at indices [b=" << inspect_batch
                      << ", h=" << inspect_heads << ", n=" << inspect_seq << ", d=" << inspect_d << "]:\n";
            std::cout << "Kernel value: " << kernel_val << "\n";
            std::cout << "Reference value: " << ref_val << "\n";
            std::cout << "Absolute difference: " << diff << "\n";
        } else {
            std::cout << "Invalid indices provided!\n";
        }
    }
    
    std::cout << (all_close ? "✅ Success" : "❌ Failure") << "\n";
    return all_close ? 0 : 1;
}