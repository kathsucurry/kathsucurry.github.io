---
layout: post
title:  "Reduction (Sum): part 0 | Introduction"
date:   2025-10-14
categories: cuda
---

In this series, we will explore how to implement parallel sum reduction using CUDA. We'll begin with an introductory post covering the following key topics:

1. [What is sum reduction, and how can we parallelize it?](#1-what-is-sum-reduction-and-how-can-we-parallelize-it)
2. [Choosing peak memory bandwidth as the metric](#2-choosing-peak-memory-bandwidth-as-the-metric)
3. [Setting up the implementation code](#3-setting-up-the-implementation-code)

**Note: this post assumes that the readers are already familiar with threads and blocks in GPU programming.** The Github link to the implementation code can be found [here](https://github.com/kathsucurry/cuda_reduction_sum).

<br />


# 1. What is sum reduction, and how can we parallelize it?


Given a list of elements, reduction applies an operation on each of the elements and accumulates them into one single value--hence, the term "reduce". In the case of sum reduction, the operation sums all the elements together, outputting the final sum value. For instance, with a list of length 8 containing the elements `[1, 2, 3, 4, 5, 6, 7, 8]`, the sum reduction output would be `1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36`. Other reduction operations include, but are not limited to, `max`, `min`, and `mean`. 

To implement sum reduction, the most naive way is to add the elements sequentially:

```c++
int array[] = {1, 2, 3, 4, 5, 6, 7, 8};
int N       = 8;
int output  = 0; // For storing the sum reduction output; initialize to 0.

for (size_t i{0}; i < N; ++i) {
    output += array[i];
}
```

The above solution requires 8 time steps to complete, and this `O(N)` approach does not scale well with large inputs. So to improve performance and better utilize available resources (e.g., the GPU), we can restructure it as a tree-based approach:

![image Tree-based approach example](/assets/images/2025-10-14-reduction_sum/fig1_tree_approach.png)
<p style="text-align: center;"><i>One tree-based approach to parallelize our problem example.</i></p>

Compared to the 8 time steps in the sequential approach, it now only requires 3 time steps: 4 sum operations in the first step, 2 sum operations in the second step, and one final sum operation in the third step.

Note that here, we assume sufficient resources to perform all 4 sum operations simultaneously. In practice, however, often the data is so large that we need to break the computations into multiple batches, with each batch handled by a separate thread block. Once we obtain the partial sum from each batch, we then invoke the reduction kernel one more time to compute the final sum. See the figure below for an example.

![image Multiple kernel invocations](/assets/images/2025-10-14-reduction_sum/fig2_multiple_kernel_invocations.png)
<p style="text-align: center;"><i>We may need to perform multiple kernel invocations to handle large datasets. Figure taken from Mark Harris' deck.</i></p>


# 2. Choosing memory bandwidth as the metric

Reduction is an example of memory-bound kernels: for each element loaded, only one floating-point operation is performed. In such cases, optimizing memory access efficiency becomes the priority where we strive to *reach* the GPU peak bandwidth. In contrast, compute-bound kernels--such as matrix multiplication--are limited by arithmetic intensity rather than memory access. We will look into matrix multiplication in a future post.

### What is my GPU peak memory bandwidth?

There are two ways to obtain the GPU peak memory bandwidth. One is to look up the specifications. For example, I'm using the RTX 5070 Ti, and according to the [specification document](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), the memory bandwidth is 896 GB/s.

Another way is to calculate the peak bandwidth by multiplying the memory bus width and memory clock speed obtained from `cudaGetDeviceProperties`:

```c++
// Code taken and slightly modified from Lei Mao's reduction article: https://leimao.github.io/blog/CUDA-Reduction/.
int device_id{0};
cudaGetDevice(&device_id);
cudaDeviceProp device_prop; // For storing the device properties.
cudaGetDeviceProperties(&device_prop, device_id);
    
// Calculate peak bandwidth:
// 1) Obtain memory clock rate in kHz and convert to Hz.
// Note that memoryClockRate is deprecated in CUDA 12.x and removed in CUDA 13.
double const memory_clock_hz{device_prop.memoryClockRate * 1000.0};

// 2) Obtain memory bus width in bits and convert to bytes.
double const memory_bus_width_bytes{device_prop.memoryBusWidth / 8.0};

// 3) Factor of 2.0 for Double Data Rate (DDR), then divide by 1.0e9 to convert from bytes/s to GB/s.
// The result (for RTX 5070 Ti) should roughly be 896 GB/s.
float const peak_bandwidth{static_cast<float>(2.0f * memory_clock_hz * memory_bus_width_bytes / 1.0e9)};
```

**Memory clock rate** determines the rate to which data is transferred to and from the GPU in kHz. **Memory bus width** determines the number of bits that can be simultaneously transferred to and from the GPU. We then incorporate *Double Data Rate (DDR)*, where data is transferred *twice* per clock cycle since it's transferred on both rising and falling edges of the clock signal.

In brief, we aim to *reach* for the GPU peak memory bandwidth (896 GB/s on RTX 5070 Ti) in our optimization efforts. Note the use of word "reach" rather than "achieve" here; the peak memory bandwidth is a theoretical maximum and typically cannot be fully realized in practice due to various overheads.


# 3. Setting up the implementation code

I used, and slightly modified, [Lei Mao's reduction code setting](https://leimao.github.io/blog/CUDA-Reduction/) as follows.

1. **Error handling, output verification, and performance measurement**:  I include error handling (`CHECK_CUDA_ERROR()` and `CHECK_LAST_CUDA_ERROR()`), kernel output verification (within `profile_batched_kernel()`), and performance measurement (`measure_performance_in_ms`), with minor renaming based on personal preferences. All of this can be found in [`utils.cuh`](https://github.com/kathsucurry/cuda_reduction_sum/blob/main/src/utils.cuh).
2. **Improved memory management**: instead of allocating, freeing, and reallocating device memory for each kernel invocation, I now allocate and free memory once in `main()` (see [`run_kernel.cu`](https://github.com/kathsucurry/cuda_reduction_sum/blob/main/run_kernel.cu)). This reduces overhead and speeds up the overall process--without affecting the kernel performance.
3. **Kernel organization**: all kernel implementations are located in the [`kernels` directory](https://github.com/kathsucurry/cuda_reduction_sum/tree/main/src/kernels). This organization was inspired by [Simon's matmul kernel optimization repo](https://github.com/siboehm/SGEMM_CUDA).



### Computing the effective bandwidth

Recall in [section 1](#1-what-is-sum-reduction-and-how-can-we-parallelize-it) that we may need to perform multiple kernel invocations to obtain the final sum. In this post, we primarily want to measure the effective memory bandwidth of only the *first* kernel invocation to align with both [Lei Mao's](https://leimao.github.io/blog/CUDA-Reduction/) and [Mark Harris'](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) implementations. This means that we would 1) load all the elements from the input list from the global memory and 2) store the partial sums of each batch at the end of the kernel execution back to the global memory.

The calculation for getting the effective memory bandwidth can be found below.

```c++
// Assume num_elements and batch_size have been defined (see additional notes).
// Assume latency has been computed (see additional notes).

// Total bytes transferred = bytes loaded (i.e., all the elements from the input list) + 
//                           bytes stored (i.e., number of batches/thread blocks, when partial sums are stored).
size_t num_bytes{num_elements * sizeof(float) + batch_size * sizeof(float)};

// Compute effective bandwidth in GB/s.
float const effective_bandwidth{(num_bytes * 1e-6f) / latency};
```

**Additional notes**

`num_elements` denotes the total number of elements in our input list, while `batch_size` specifies the number of thread blocks.

`latency` is measured using the following steps:

1. **Warmup the kernel.** Run the kernel at least once to clear out any initial overheads and bring the system to a steady, optimal state. The warmup run is typically slower, so we don't include it in the final measurement.
2. **Execute the kernel multiple times.** This helps smooth out variability in runtime and provides more consistent results.
3. **Compute the average latency.** Averaging the runtimes over several runs leads to more reliable estimate than measuring a single execution.

<br />

---

<br />

With the basics covered, we are now ready to optimize the sum reduction algorithm! Click [here]({% link _posts/2025-10-14-reduction_sum_part1.markdown %}) to continue to the next part of the series.