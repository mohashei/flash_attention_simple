# flash_attention_simple
A simple non-optimal flash attention implementation. Relies on C++ and CUDA without python.

Based on this [implementation](https://github.com/tspeterkim/flash-attention-minimal) but with some differences in the details.

In particular
1) Better access patterns to global memory.
2) Better caching of local results.
3) Ability to split a single sequence computation into multiple blocks.
4) Fewer non-MMA arithmetic operations (based on [flash attention v2](https://arxiv.org/pdf/2307.08691))

Usage:

CMake to build (assuming you cloned):
```
cd flash_attention_v2
cmake -B build -S .
cmake --build build
```

Running instructions:
```
build/flash_attention_v2 \
--batch-size batch_size \
--num-heads num_heads \
--M M \
--seq-len seq_length \
--inspect b h n d
```

Here `inpsect` allows you to inspect a given element with respect to the reference implementation.
