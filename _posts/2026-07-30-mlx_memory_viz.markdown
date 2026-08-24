---
layout: post
title:  "Building a Memory Profiler for MLX"
date:   2026-07-30
categories: mlx
---

The main goal of this write-up is mostly to help myself understand what I'm actually doing, as much as I can. If you have any input or feedback, I'd love to hear it! 

The post is organized into the following sections:

- [How to use](#how-to-use)
- [Motivation](#motivation)
- [The plan](#the-plan)
- [Plan 1: Understand PyTorch's side](#plan-1-understand-pytorchs-side)
   - [What happens behind the scene?](#what-happens-behind-the-scene)
      - [1. `torch.cuda.memory._record_memory_history(): turning on/off memory recording](#1-torchcudamemory_record_memory_history-turning-onoff-memory-recording)
      - [2. How trace entries are recorded](#2-how-trace-entries-are-recorded)
      - [3. `torch.cuda.memory._dump_snapshot("out.pickle")`: taking a snapshot](#3-torchcudamemory_dump_snapshotoutpickle-taking-a-snapshot)
      - [4. Capturing the context](#4-capturing-the-context)
- [Plan 2: Understand MLX's side](#plan-2-understand-mlxs-side)
  - [The allocator](#the-allocator)
  - [Computation graph construction](#computation-graph-construction)
  - [When `eval` is triggered](#when-eval-is-triggered)
- [Plan 3: Bridge the two](#plan-3-bridge-the-two)
  - [1. Enable recording memory events.](#1-enable-recording-memory-events)
  - [2. Capture the primitive name associated with each event.](#2-capture-the-primitive-name-associated-with-each-event)
  - [3. Implement traceback capture.](#3-implement-traceback-capture)
  - [4. Format the output to match PyTorch's snapshot format.](#4-format-the-output-to-match-pytorchs-snapshot-format)
- [Summary](#summary)

My implementation can be found [here](https://github.com/kathsucurry/mlx/tree/v0.32.1-memory-viz).

<hr class="hr-top" /><hr />

# How to use

Use the MLX version [here](https://github.com/kathsucurry/mlx/tree/v0.32.1-memory-viz).

```python
import mlx.core as mx
from mlx._memory_viz import dump_snapshot

# 1. Turn on the recording.
mx.record_memory_events(enabled=True, max_entries=100000)

# 2. Add things to record.
# ...

# 3. Get the memory events, either:
# 3.1. Get a list of event dictionaries to play around with, or
events = mx.get_memory_events()

# 3.2. Dump the events into a pickle file compatible with PyTorch's memory viz.
dump_snapshot(mx.get_memory_events(), "output.pickle")

# 4. Turn off the recording.
mx.record_memory_events(enabled=False)
```

<hr class="hr-top" /><hr />

# Motivation

I ran into a small problem while learning [MLX](https://github.com/ml-explore/mlx): I couldn't find a memory profiler I liked... Maybe I just missed some obvious existing tools out there, but here are two tools I found.

<div style="padding-left: 20px;">

<h4> 1. MLX's built-in memory API </h4>

<pre><code>import mlx.core as mx

mx.get_active_memory() # The number of bytes currently allocated (exclues the buffer cache)
mx.get_cache_memory()  # The number of bytes held in the allocator's reuse cache
mx.get_peak_memory()   # THe peak memory mark since program start</code></pre>

It's simple and dependency-free, but it only gives me numbers. I'd like to know what was contributing to these numbers.

<br /><br />

<h4> 2. Xcode's Metal Debugger </h4>

<pre><code># run with: MTL_CAPTURE_ENABLED=1 python script.py
mx.metal.start_capture("out.gputrace")   # Path must not already exist
...
mx.metal.stop_capture()</code></pre>

Then open the <code>.gputrace</code> in Xcode. This is a powerful tool, but to my understanding, it's aimed at GPU work (e.g., kernel dispatches) and not at attributing allocations to the lines of code that caused them. It also has a much steeper learning curve.

<br /><br />

</div>

I really like PyTorch's [memory snapshot visualization](https://pytorch.org/blog/understanding-gpu-memory-1/) where I can record allocation history, dump a pickle, and drop it into [docs.pytorch.org/memory_viz](https://docs.pytorch.org/memory_viz), and see how each allocation is tied back to the code line that made it. So I decided to build one for MLX. Specifically, **to enable MLX output a pickle in a format compatible with PyTorch's snapshot format** so that the existing viewer works out of the box. No new frontend required.

<hr class="hr-top" /><hr />

# The plan

1. **Understand PyTorch's side.** How are the memory events recorded? What ends up in the output pickle file?
2. **Understand MLX's side.** Where does MLX allocate? How does lazy evaluation influence the implementation?
3. **Bridge the two.** Output MLX allocation events in PyTorch's snapshot schema.

I'm new to allocator internals and still learning C++, so I did get assistance from Claude Code, mostly for navigating through unfamiliar codebases, for the C++ I couldn't yet write unassisted, for correctness verification, and for getting suggestions for relevant readings.

<hr class="hr-top" /><hr />

# Plan 1: Understand PyTorch's side

*Note: the PyTorch version described here is `v2.14.0a0` with head commit `68b353e2`. Some details that are deemed irrelevant are omitted.*

<!-- START OF DIV -->
<div class="div-author-note">
<h4>Author's Note</h4>
PyTorch allows capturing memory events for CUDA (NVIDIA GPUs) and XPU (Intel GPUs). Since I've mainly used the profiler for CUDA, I'll focus my descriptions on CUDA functionality that will be relevant to the implementation on MLX.
</div>
<!-- END OF DIV -->

In PyTorch, we capture memory events using the following functions:

```python
import torch

# Start the recording and set the max capacity of 100_000 entries.
# If there are >100_000 entries, only the *last* 100_000 entries will be stored. 
torch.cuda.memory._record_memory_history(max_entries=100_000) 

# The training run we want to capture.
...

# Save the snapshot.
torch.cuda.memory._dump_snapshot("out.pickle")

# Stop the recording.
torch.cuda.memory._record_memory_history(enabled=None)
```

### What happens behind the scene?

I'll divide this section into 4 parts:
1. [`torch.cuda.memory._record_memory_history(): turning on/off memory recording](#1-torchcudamemory_record_memory_history-turning-onoff-memory-recording)
2. [How trace entries are recorded](#2-how-trace-entries-are-recorded)
3. [`torch.cuda.memory._dump_snapshot("out.pickle")`: taking a snapshot](#3-torchcudamemory_dump_snapshotoutpickle-taking-a-snapshot)
4. [Capturing the context](#4-capturing-the-context)

<hr class="hr-single" />

#### 1. `torch.cuda.memory._record_memory_history()`: turning on/off memory recording

![_record_memory_history() function calls](/assets/images/2026-07-30-mlx_memory_viz/plan1_pytorch_code1.png)
<p style="text-align: center;"><i>Figure 1. The call path from <code>torch.cuda.memory._record_memory_history()</code> down to the per-device allocator. Note that some details have been omitted for simplicity.</i></p>

The diagram above may look intimidating, but for our purposes, I'd say `torch.cuda.memory._record_memory_history()` does two main things (marked with `*` in the image above):

1. **Set up the configuration**: which context recording function to use and which events to capture.
2. **Set `record_history` on each `DeviceCachingAllocator`**. `record_history` is a private boolean member of `DeviceCachingAllocator`, which is used as the flag the allocator checks later to decide whether to record an event at all.

So **no recording happens yet** at this point. By default, the `enabled` argument in `torch.cuda.memory._record_memory_history()` in the new implementation is set to `all`, which is then converted into a `true` value by the time it reaches `c10::cuda::CUDACachingAllocator::recordHistory()`.

<hr class="hr-single" />

#### 2. How trace entries are recorded

All trace entries are captured as `TraceEntry` objects, created inside `record_trace()`, a private function on `DeviceCachingAllocator`. If you search for every `record_trace()` call site in `c10/cuda/CUDACachingAllocator.cpp`, you'll notice they're called in **memory allocation and deallocation paths**. The one exception is `snapshot()`, which records an entry purely as a marker of when the user takes a snapshot. **So whenever the user calls a PyTorch function that triggers an allocation or deallocation, `record_trace()` is invoked**.

`record_trace()` then does the following (with some details omitted):
1. Return early if `record_history` is `false` and no trace trackers are registered (I won't cover trace trackers here.)
2. **Create a `TraceEntry` holding the details of the event**: its type (allocation, free, and so on), the address, the size, the time it's created, etc.
3. If `record_history` is `true`, **insert the newly created entry into the ring buffer `alloc_buffer`**.

<hr class="hr-single" />

#### 3. `torch.cuda.memory._dump_snapshot("out.pickle")`: taking a snapshot

![_dump_snapshot() function calls](/assets/images/2026-07-30-mlx_memory_viz/plan1_pytorch_code2.png)
<p style="text-align: center;"><i>Figure 2. The call path from <code>torch.cuda.memory._dump_snapshot()</code> down to the per-device allocator. Note that some details have been omitted for simplicity.</i></p>

As you may have expected, the diagram for `_dump_snapshot()` is very much aligned with `_record_memory_history()`. Once `THCPModule_memorySnapshot()` receives the output, it parses the output into a dictionary and passes it back to the Python side. If `_dump_snapshot()` is called, the dictionary snapshot is then dumped into a pickle file.

Of the two outputs, `device_traces` and `segments`, **we'll focus only on `device_traces` in this post**.

<hr class="hr-single" />

#### 4. Capturing the context

One essential component is stack trace capture. Having the call chain that leads to each allocation/deallocation, particularly the Python frames, provides helpful context for debugging. *How does PyTorch capture this context*? In order to answer this question, I'll trace the code **from the output back to where it's captured**.

**The list of `{filename, line, name}`s comes from `TraceEntry`s' context**

The [docstring of `_snapshot()`](https://github.com/pytorch/pytorch/blob/e4bdcb097b91e69f2cd74789d7cc51703c617ce4/torch/cuda/memory.py#L1045-L1131) in `torch/cuda/memory.py` provides clear details of how the memory state is represented. The component we're interested in is the `Frame` TypedDict, containing `filename`, `line`, and `name` fields (plus optional FX debug fields, which I will omit in my explanation). Both `TraceEntry` and `Block` carry a `frames` field holding a list of these `Frame`s.

In `THCPModule_memorySnapshot()`, tracing how the snapshot's `trace_entries` is used, we see in [`traceEntryToDict()`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/cuda/Module.cpp#L902) that the context comes from **the `context_` field held by `TraceEntry`**. `getCapturedTracebackFromContext(te.context_)` returns a raw `CapturedTraceback*` (I'll talk more about `CapturedTraceback` later), which is appended to `to_gather_frames`. After all entries are processed, the whole batch of frames in `to_gather_frames` is [symbolized at once into filename, line, and name information by `py_symbolize()`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/cuda/Module.cpp#L1034), and the results are written back into each entry's `frames` key.

**How each `TraceEntry`'s context is created and passed to `record_trace()`**

Recall that [`TraceEntry`s are created by `record_trace()`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/c10/cuda/CUDACachingAllocator.cpp#L4432). Tracing where its context argument comes from leads to `maybeGatherContext()` (one example [here](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/c10/cuda/CUDACachingAllocator.cpp#L1724)). This function loads and invokes `context_recorder_`, which is the **context-recording callback function** installed when `recordHistory()` was called (within the `_record_memory_history()` call path mentioned in [point 1](#1-torchcudamemory_record_memory_history-turning-onoff-memory-recording)), or returns nothing if recording is disabled, hence the `maybe`. `maybeGatherContext()` is called at most once per invocation by the allocator methods that initiate traceable events (e.g., `malloc`, `free`, `release_blocks`, `emptyCache`).

So now we know when and where "context"s are created and used to create a `TraceEntry`. But how do we define a "context" to begin with?

**What is a "context"?**

*To be honest, this is probably one of the parts I struggled with the most. I hope I've understood and explained it correctly, and any feedback is always welcome.*

As a member of `TraceEntry`, `context_` is a [shared pointer to `c10::GatheredContext`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/c10/core/CachingDeviceAllocator.h#L175). `GatheredContext` itself is [defined as an *empty* polymorphic base (apart from a virtual destructor) on the **allocator**'s side (`c10/core/Allocator.h`)](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/c10/core/Allocator.h#L347). Its purpose is to be a type the allocator can access **without knowing what's inside**, i.e., an *opaque* handler (which also means it's free from any potentially heavy dependencies).

<!-- START OF DIV -->
<div class="div-author-note">
<h4>Author's Note</h4>
<code>GatheredContext</code> being an opaque handler will be important <a href="#3-implement-traceback-capture">when we implement traceback capture</a> later. Just a heads-up!
</div>
<!-- END OF DIV -->

I mentioned `CapturedTraceback*` very briefly earlier as the type returned by `getCapturedTracebackFromContext()`. `CapturedTraceback` [inherits `GatheredContext`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/profiler/combined_traceback.h#L23) and lives on the **profiler**'s side (`torch/csrc/profiler/combined_traceback.h`). This is where we see the implementations we're looking for. Looking at the header, it holds three separate frame vectors: `frames_` (Python), `cpp_frames_`, and `script_frames_`, and its static `gather(python, script, cpp)` takes one flag per kind.

Looking back at `c10::cuda::_record_memory_history()` in `torch/csrc/cuda/memory_snapshot.cpp` from [point 1](#1-torchcudamemory_record_memory_history-turning-onoff-memory-recording), we see that the context-gathering callback `recorder` is set to either `gather` or `gather_with_cpp`, which [call `CapturedTraceback::gather()` with the appropriate arguments](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/cuda/memory_snapshot.cpp#L117). `recorder` is then passed to `c10::cuda::CUDACachingAllocator::recordHistory()`.

<!-- START OF DIV -->
<div class="div-interested">
<h4>If You're Curious...</h4>
I had so many questions when I looked at the definition of <code>GatheredContext</code> for the first time:
<br /><br />

<pre><code>// used to hold traceback information in allocators
// ...
struct GatheredContext {
  virtual ~GatheredContext() = default;
};</code></pre>

As the comment above the definition mentions, it's empty because it exists only for "hold[ing] traceback information in allocators", which explains the emptiness. But is there any significance behind declaring the destructor virtual? <i>Yes.</i> There are at least two possible reasons:
<br /><br />

<ol>
<li><b>Ensures clean deletion of the derived class objects.</b>
<br />
If an object is deleted through a base class pointer (in our case, a <code>GatheredContext*</code> that points to a <code>CapturedTraceback</code>) and the base class's destructor is non-virtual, the behavior is undefined. In practice, only the base part is destroyed and the derived data members (<code>frames_</code>, <code>cpp_frames_</code>, etc) are leaked. Giving the base class a virtual destructor ensures the entire object is destroyed: the runtime looks up the <i>actual</i> type (<code>CapturedTraceback</code>), runs its destructor first, then the base's.

<br /><br />
<i>Though</i> note that <code>std::shared_ptr</code> type-erases its deleter at construction, so since our context here is held using <code>std::shared_ptr</code>, the right destructor would run even without the <code>virtual</code> keyword. 
<br /><br />
</li>
<li><b>Allows the  <code>dynamic_cast<CapturedTraceback*>(x.get())</code> in <code>getCapturedTracebackFromContext</code>.</b>
<br />
At runtime, <code>dynamic_cast</code> needs to know that <code>x.get()</code> actually points to a <code>CapturedTraceback</code>. But how does it acquire that information?
<br /><br />
The moment a class declares <i>any</i> virtual function, it becomes a polymorphic type. In practice, compilers implement this by adding an invisible field, the vptr ("virtual table pointer"), to every <b>object</b> (not the class!) of that class. The vptr points to the <b>class</b>'s vtable, which is an array of function pointers used to invoke the appropriate function implementations. The table also contains a reference to the class's <b>type information</b>, the field the <code>dynamic_cast</code> uses to determine whether <code>x.get()</code> points to the right type.
<br />
</li>
</ol>

Source:
<ul>
<li>Effective C++ 3rd Edition, Item 7: Declare destructors virtual in polymorphic base classes, Scott Meyers (2005)</li>
<li><a href="https://leimao.github.io/blog/CPP-Virtual-Table/">C++ Virtual Table, Lei Mao (2023)</a></li>
</ul>

</div>
<!-- END OF DIV -->

**How are Python tracebacks captured?**

Let's start with `python_support_`, a linked list of unwinders that is a static atomic pointer to `CapturedTraceback::Python`. At first, it is initialized as a null pointer in [`torch.csrc/profiler/combined_traceback.cpp`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/profiler/combined_traceback.cpp#L6). When `import torch` is invoked, part of the eager startup registration includes invoking `installCapturedTracebackPython()` ([`torch/csrc/profiler/python/init.cpp`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/profiler/python/init.cpp#L788)) and `python_support_` is modified such that its head now points to a new `PythonTraceback` (see the figure below).

![How python_support_ is updated](/assets/images/2026-07-30-mlx_memory_viz/plan1_pytorch_code3.png)
<p style="text-align: center;"><i>Figure 3. How invoking `import torch` leads to updating <code>python_support_</code>.</i></p>

When [`CapturedTraceback::gather()`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/profiler/combined_traceback.cpp#L8) is called with `python = True`, it walks the unwinder linked list until an unwinder can and does gather Python frames. For each, it first checks [`PythonTraceback::canGather()`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/profiler/python/combined_traceback.cpp#L27) (GIL safety on the current thread). If safe, it calls [`PythonTraceback::gather()`](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/profiler/python/combined_traceback.cpp#L44), which gets the current frame via `PyEval_GetFrame()` and walks up the stack (`PyFrame_GetBack()`), storing `(code, lasti)` per frame. Once frames are captured, the loop stops and does not visit remaining unwinders (if any).

More on how Python frames are handled in PyTorch will be discussed [when we implement traceback capture](#3-implement-traceback-capture).

<hr class="hr-top" /><hr />

# Plan 2: Understand MLX's side

*Note: the MLX version described here is `v0.32.1` with head commit `255f953f`. Some details that are deemed irrelevant are omitted.*

One main thing I learned from PyTorch's implementation is this: at each memory allocation/deallocation event in the allocator, capture the stack *at that moment* and attach it to the trace entry as context.

In MLX, we can create a trace entry in the allocation/deallocation path in pretty much the same way, but we *can't capture the traceback there*. MLX is lazy, so arrays are only materialized when needed. By the time the allocator actually runs, the Python code that constructed the array has long since returned, so every traceback would point at the `eval()` or whatever else that triggers materialization rather than at the code responsible for the allocation.

Before diving into how to resolve this difference, let's look at the components of MLX that may be relevant for our solution:

- [The allocator](#the-allocator)
- [Computation graph construction](#computation-graph-construction)
- [When `eval` is triggered](#when-eval-is-triggered)

I'll only cover the Metal-backend implementation here for simplicity.

<hr class="hr-single" />

### The allocator

MLX currently has a much simpler allocator than PyTorch's. It has an abstract base class `Allocator`, which backend-specific allocators derive from (see the figure below).

![MLX's Allocators](/assets/images/2026-07-30-mlx_memory_viz/plan2_mlx_code1.png)
<p style="text-align: center;"><i>Figure 4. MLX's Allocators.</i></p>

You might be wondering why there seem to be two sets of allocation and deallocation: `malloc`/`free` versus `make_buffer`/`release`. Briefly:
- `malloc` and `free` handle buffers that MLX allocates and owns. `free` prefers to recycle a buffer into the cache when possible, so `malloc` can reuse it later.
- `make_buffer` and `release` handle buffers that wrap external memory or foreign raw pointers (e.g., NumPy arrays) *without copying*. Since MLX doesn't own that memory, `release` tears down the wrapper and never recycles it.

<hr class="hr-single" />

### Computation graph construction

A computation graph is a directed acyclic graph describing how a result gets computed. Each node is an array, and an edge from one array to another means the second was computed from the first (as in, the first is the input to the operation that produces the second array). Below is a computation graph example (produced using `mx.export_to_dot()` **before** running any `mx.eval()` implemented in `mlx/graph_utils.cpp`) along with the computation code.

![A computation graph example](/assets/images/2026-07-30-mlx_memory_viz/plan2_mlx_code2.png)
<p style="text-align: center;"><i>Figure 5. A computation graph example.</i></p>

How is the graph constructed? What happens when we invoke each of the line in the example code above? To answer the questions, we'll start by delving into the [class `array`](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/array.h#L26).

#### The `array` class (and its nested `ArrayDesc` struct)

As mentioned in the docstring, "an array is really a node in the graph". Looking at its private member, it actually only holds [a shared pointer to `ArrayDesc` object, `array_desc_`](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/array.h#L533). An `array` is therefore just one pointer wide, which is what makes passing it around by value cheap. Most of its public functions are thin accessors that forward to **the data contained within `array_desc_`**.

Looking at the [`ArrayDesc` struct definition](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/array.h#L478), we see the following information:
- **The metadata**: shape, strides, size, dtype, offset, data_size (how many elements of the buffer it actually accesses), flags (contiguity information).
- **The operation**: a `std::shared_ptr<Primitive> primitive`, which knows how to compute the array's data from its inputs. For leaf arrays, the primitive would be null.
- **The status**, which is one of:
  - `unscheduled`: the computation producing the output array has not been scheduled yet.
  - `evaluated`: the array's evaluation has been run, but the computation is not necessarily complete. Its memory has been allocated, and if the array is not a tracer, it has been detached from the graph (with its primitive and inputs dropped) so the upstream graph can be freed.
  - `available`: if the array is the output of a computation, then the computation is complete and the data is safe to read. 
- **The event**: a handle to the completion signal for the operation that produces this array's data. It is what promotes an array fom `evaluated` to `available`.
- **The tracer flag** (`is_tracer`): marks an array that is being used inside a graph transform such as `grad` or `compile`, and so must not be detached from the graph at eval time.
- **The data**: a `std::shared_ptr<Data>`, where [`Data`](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/array.h#L231-L246) is the buffer along with its deleter.
- **The inputs**: the inputs to the operation held by the array
- **The siblings**: the co-outputs of a multi-output operation, along with `position`, this array's index in that output list.

There are multiple possible levels of sharing here:
1. Multiple `array` handles sharing one `ArrayDesc` are **the same array**. Also notice that `id()` of the array returns the address of that shared `ArrayDesc` object, so these handles all report the same `id()`.
2. Multiple `ArrayDesc`s sharing one `Data` are different **views** onto the same memory, each with its own shape, strides, and offset.
3. Multiple `ArrayDesc`s sharing one `Primitive` indicate the co-outputs of the same multi-output operation

I will show examples of each in the section below.

#### What happens when we run each line in the Fig. 5 example?

Now that we know what `array` is, we can look into what the lines in the example code actually do. Let's start with the following lines.

```python
a = mx.array([1, 2, 3, 4], dtype=mx.float32)
b = mx.array([5, 6, 7, 8], dtype=mx.float32)
```

Each line above [calls `create_array()`](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/python/src/array.cpp#L310), which is defined in `python/src/convert.cpp`. Tracing it through a few more layers eventually reaches [`return mx::array(vals.begin(), shape, dtype);`](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/python/src/convert.cpp#L613). It calls [one of `array`'s constructor](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/array.h#L542-L549), which initializes `array_desc_` via `ArrayDesc`'s [constructor](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/array.cpp#L259-L262) and its `init()` function. Notice that these *leaf* arrays are immediately `available`, i.e., **the data is materialized right away and no evaluation is needed**.

`c = a + b`, on the other hand, invokes `a.__add__(b)`, which [calls `mx::add(a, b)`](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/python/src/array.cpp#L547-L556) that is defined in [`mlx/ops.cpp`](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/ops.cpp#L2947-L2954). It in turn calls [an `array`'s constructor](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/array.cpp#L19-L40), [an `ArrayDesc`'s constructor](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/array.cpp#L264-L275), and invokes ArrayDesc's `init()`.

`.reshape()` is an interesting one since it can behave in several ways. `mx.reshape(a, (4, 1))` (producing node `K` in the figure) and `mx.reshape(c, (1, 4))` (node `L`) follow paths similar to the operations above, ending in the `array` and `ArrayDesc` constructors. Note that creating a new `array` and `array_desc_` doesn't necessarily mean memory gets allocated at `eval` time. During `eval`, MLX first [checks the input's layout](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/backend/common/common.cpp#L148-L180). A row-contiguous input always yields a view, otherwise MLX still tries to express the new shape as strides over the existing buffer. It only copies as the last resort. When a view is produced, the new `ArrayDesc` points to the same data instead of allocating a new buffer (recall the third level of sharing mentioned above).

`out2 = d.reshape((4, 4))`, on the other hand, behaves differently. You might have noticed `d` is missing in the figure! What happened to `d`? By this point, `d`'s shape is already `(4, 4)`, so [`reshape()` function](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/ops.cpp#L413-L423) simply returns `d` by value, copy-constructing a new handle. `out2` and `d` are therefore two different `array` handles sharing one `ArrayDesc` (the first level of sharing above).

One other operation I'd like to highlight is `split()`, which produces multiple outputs (i.e., generating new `array` objects with their own `ArrayDesc`s) that share a single `split` primitive (the second level of sharing above). Each output holds references to the others as its `siblings`.

In summary, **`array`-related Python operations that may end up with allocations all pass through `ArrayDesc`'s `init()`**. It runs while Python frame that created the array is still on the stack.

<hr class="hr-single" />

### When `eval` is triggered

When we run `mx.eval(out1, out2, out3)` at the end, it passes through [these lines](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/python/src/transforms.cpp#L1187-L1192) before reaching the core function [`eval()`](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/transforms.cpp#L336-L351). As long as there's any unscheduled graph work, it calls `eval_impl()` and waits for it to finish.

[`eval_impl()`](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/transforms.cpp#L80-L320) starts by wrapping the outputs the user asks for in a synthetic `Synchronizer` node (`sync` below), giving the graph a single root to traverse from. It then does three things:

1. [Compute the out-degree](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/transforms.cpp#L116-L178) (i.e., the number of consumer of each node in the graph) using depth-first search (DFS). For the example above:

    ```
    | Node     | Out-degree | Consumer nodes     | Primitive |
    |----------|------------|--------------------|-----------|
    | a        | 3          | c, K, M            | leaf      |
    | d (out2) | 3          | split*, out3, sync | Matmul    |
    | split1-4*| 2          | out1               | Split     |
    | K        | 1          | d (out2)           | Reshape   |
    | L        | 1          | d (out2)           | Reshape   |
    | M        | 1          | out3               | Broadcast |
    | b        | 1          | c                  | leaf      |
    | c        | 1          | L                  | Add       |
    | out1     | 1          | sync               | Add       |
    | out3     | 1          | sync               | Add       |


    *siblings share a single count. out1 has split1 and split3 as its inputs.
    ```

2. [Build the tape](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/transforms.cpp#L184-L225), a deque holding the order in which operations run, so that every producer runs before its consumers and each operation runs exactly once. The tape is built with breadth-first search (BFS). The example above generates:

    ```
    tape: [sync, out1, out3, split3, M, d (out2), K, L, c]
    execution: c --> L --> K --> d (out2) --> M --> split3 --> out3 --> out1 --> sync
    ```

3. [Run the operations in the tape](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/transforms.cpp#L228-L307). Each primitive is invoked in [these lines](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/transforms.cpp#L264-L268), and **that is where output buffers are allocated**.

<hr class="hr-top" /><hr />

# Plan 3: Bridge the two

Recall that in PyTorch, whenever recording is enabled and it's GIL-safe (i.e., when the thread legally holds the GIL), the Python stack is gathered *synchronously during the alloc/free call* and passed to the trace-recording function placed in the allocator.

In MLX, however, while it's still straightforward to place the recording function in the allocator, its lazy computation design means the Python context has to be captured separately. This is because by the time the allocation happens, the Python frame that created the array is long gone. One idea is to capture the context *during* graph construction instead. As discussed earlier, `array`-related Python operations that may end up with allocations all pass through `ArrayDesc`'s `init()`, so we can record the context there and **carry it on the array descriptor `array_desc_`**, then to be picked up by the memory trace entry when the buffer is eventually allocated.

Here are the implementation steps. My main goal at this point is a working prototype, so I'm not worrying much about optimization, and I might overlook some details. Some naming may also change in the future.

1. [Enable recording memory events.](#1-enable-recording-memory-events)
2. [Capture the primitive name associated with each event.](#2-capture-the-primitive-name-associated-with-each-event)
3. [Implement traceback capture.](#3-implement-traceback-capture)
4. [Format the output to match PyTorch's snapshot format.](#4-format-the-output-to-match-pytorchs-snapshot-format)

<hr class="hr-single" />

### 1. Enable recording memory events.

This step is fairly straightforward and it entails the following substeps:

   - [Backend] Implement a recording function, along with a class representing a memory event, and call it from the allocator's allocation and deallocation paths.
   - [Backend] Ensure the memory event class contains a flag to indicate whether the recording is enabled or not.
   - [Python-facing] Provide a functionality to toggle the recording functionality (similar to PyTorch's `_record_memory_history()`).
   - [Python-facing] Expose the captured events to Python (for testing purposes).

<hr class="hr-single" />

### 2. Capture the primitive name associated with each event.

Since we're not capturing C++ stacks, the primitive names carry most of the signal about where an allocation came from.

To record which primitive an allocation/deallocation belongs to, we need to get the primitive's name from `array` down to the event recording function. Recall from the [`eval` description section](when-eval-is-triggered) that primitives are invoked in [these lines](https://github.com/ml-explore/mlx/blob/255f953f99c3403df19fa4d92462143139c3dfff/mlx/transforms.cpp#L264-L268) where `arr` is passed to either `gpu::eval()` or `cpu::eval()`.

One solution is to thread the name through the call path, passing it alongside `arr` all the way down to where memory is allocated or freed. The problem is that the call path can get tedious, so plumbing an extra argument through every branch that might allocate/deallocate gets messy pretty fast...

Another solution is to create a context object: an object that stores information at one point in the call stack so it can be read further down without being passed explicitly. We Implement two structs in `mlx/op_context.h`:

```c++
struct OpInfo {
  std::string_view primitive_name;
};

inline thread_local OpInfo current_op;

struct OpContext {
  explicit OpContext(std::string_view primitive_name) {
    current_op = {.primitive_name = primitive_name};
  }
  ~OpContext() {
    current_op = {};
  }

  OpContext(const OpContext&) = delete;
  OpContext& operator=(const OpContext&) = delete;
};
```

We can then set the primitive name right before `gpu::eval(arr)` or `cpu::eval(arr)` and read it back inside the memory event recording function. That way, the allocator is able to pick it up no matter how deep the call path goes.

<hr class="hr-single" />

### 3. Implement traceback capture.

Possibly the most challenging part of the implementation for me. I relied heavily on Claude Code for this one due to my lack of experience.

#### Challenges

In both PyTorch and MLX, there is a Python layer and a C++ layer. The allocator runs in pure C++ and handles allocations, deallocations, and trace recording. Meanwhile, we want to capture Python stacks and attach the captured `PyObject`s to the recorded trace entries. Here is the catch: *the allocator cannot include Python headers* (i.e., cannot use `PyObject`). PyTorch solves this with the three layers illustrated in Figure 6 below.

![The layering design in PyTorch](/assets/images/2026-07-30-mlx_memory_viz/plan3_pytorch.png)
<p style="text-align: center;"><i>Figure 6. The layering design in PyTorch.</i></p>

1. **Layer 1**: The allocator sees only an *opaque* `GatheredContext` and invokes `context_recorder_` (which we discussed earlier) to produce one.
2. **Layer 2**: torch C++ sees the `PyFrame` type, which stores the frame's code object as a `void*` rather than a `PyObject*`. A `void*` can be stored, moved, and compared, but never dereferenced, meaning, this layer cannot INCREF, DECREF, or read a filename. Anything that needs to *interpret* the pointer goes through a virtual interface that Layer 3 registers at import time.
3. **Layer 3**: The Python binding file, which implements all the Python-specific operations, from walking the frame stack, resolving frames to filenames and line numbers, to releasing references.

<!-- START OF DIV -->
<div class="div-interested">
<h4>If You're Curious...</h4>

<code>Py_INCREF()</code> and <code>Py_DECREF()</code> are Python C APIs for managing an object's reference count. When <code>Py_DECREF()</code> brings the count to zero, it immediately invokes the object's deallocator, which releases the object.
</div>
<!-- END OF DIV -->

Here is where the problem lies. Layer 2 *owns* strong references to code objects but cannot safely release them. Releasing means `Py_DECREF`, which may only be called when holding the GIL, and `~CapturedTraceback` can run on a thread that is currently holding the allocator's device lock. Acquiring the GIL there can deadlock (explained in the comments [here](https://github.com/pytorch/pytorch/blob/68b353e2d8b7879b819209055f5524d0e7a1e9d1/torch/csrc/profiler/python/combined_traceback.cpp#L10-L21)): one thread holds the GIL and waits for the device lock, while another holds the device lock and waits for the GIL. So instead of freeing immediately, the destructor defers. It pushes the doomed frame pointers onto a global `to_free_frames` vector. The actual `Py_DECREF`s happen at the top of the next `gather()` call, which by construction holds the GIL and runs outside the device lock.

#### Solution

To mitigate this problem, instead of passing the `PyObject*` (in the form of `void*`) to the allocator, we can intern each captured stack in a table on the Python side and pass an integer ID into the memory event recording instead. The table owns the only references to the code objects and never releases them during a session, so destroying a memory event just destroys an integer (no `Py_DECREF` and no GIL).

This entails the following steps:

1. In `mlx/traceback.h|cpp`, implement the hook mechanism: `set_traceback_tracking()` to enable tracking, `set_traceback_capture_func()` to install the capture hook (which is invoked by `install_traceback_capture()`), and `capture_traceback()` to invoke the hook. `capture_traceback()` returns a `TracebackId`, and the core only transports it.

2. Add a new `ArrayDesc` field `traceback` for storing the traceback ID.

3. Capture the traceback when the `ArrayDesc` is created (we discussed earlier how the `init()` function can be a reliable capture point): 
  ```c++
void array::ArrayDesc::init() {
  ...
  traceback = detail::capture_traceback();
}
  ```

4. Add the traceback ID to `OpInfo`, so the thread-local `current_op` can carry it from the eval loop to the allocator.

5. Pass the array's traceback ID when constructing `op_context` in `transforms.cpp`.

6. Add a traceback ID field in `MemoryEvent`.

7. Call `set_traceback_tracking(enabled)` in `record_memory_event()` (next to the existing `set_op_tracking(enabled)`), and attach the traceback in `maybe_record_events()`:
  ```c++
if (MemoryEvent::is_alloc(action)) {
  event.primitive_name = detail::current_op.primitive_name;
  // Add the lines below.
  event.traceback = detail::current_op.traceback != detail::no_traceback
      ? detail::current_op.traceback
      : detail::capture_traceback();
}
  ```
  The fallback in the last line exists for leaf arrays. Leaf arrays allocate eagerly at construction without `eval()`, so no `OpContext` ever runs and `current_op` carries no ID. Consequently, the ID sitting in `ArrayDesc` is unreachable from the allocator, whose interface is just `malloc(size)`. But leaf construction happens on the Python thread with the GIL held, so the allocator can simply capture directly. This is safe because the capture function *never attempts* to acquire GIL; it only checks if GIL is held. In other use cases (the eval path), the check fails and the fallback harmlessly returns `no_traceback` (though `current_op` already has the ID anyway).

8. Implement the capture function itself in `python/src/traceback.h|cpp`, along with the interning table. For deduplication, the table hashes each captured stack. I picked the Fowler–Noll–Vo hash for its simplicity and for how easily it folds frame by frame into a running hash.

9. Also in `python/src/traceback.h|cpp`, implement `resolve_traceback(TracebackId id)`, which turns an ID back into a readable stack: a list of `(filename, function name, line number)` tuples, ordered outermost frame first to match Python's own traceback convention. This is called from the `get_memory_events()` binding.

<hr class="hr-single" />

### 4. Format the output to match PyTorch's snapshot format.

PyTorch's snapshot dictionary [contains the following fields](https://github.com/pytorch/pytorch/blob/e4bdcb097b91e69f2cd74789d7cc51703c617ce4/torch/csrc/cuda/Module.cpp#L1027-L1032): `segments`, `device_traces`, `allocator_settings`, `external_annotations`, `host_segments`, and `host_traces`. For our first prototype, we'll only include `device_traces` and `segments`. Specifically, we'll implement `device_traces` and leave `segments` as an empty list.

Since we already have `mx.get_memory_events()`, we just need a function that reformats its output into a form the memory viz web app can load. Note that we want to handle two different timelines:

1. **Active memory**: the memory arrays are actually using right now.
  - Allocation actions: `{"AllocNew", "AllocReuse", "AllocMakeBuffer"}`, to be replaced with `alloc`
  - Free actions: `{"FreeActiveToCache", "FreeActiveToOS", "Release"}`, to be replaced with `free_completed`
2. **Reserved memory**: the memory taken from the OS that hasn't been released yet, i.e., active memory + what's retained in the cache
  - Relevant allocation actions: `{"AllocNew", "AllocMakeBuffer"}`, to be replaced with `segment_alloc`
  - Relevant free actions: `{"FreeActiveToOS", "Release", "FreeCacheToOS"}`, to be replaced with `segment_free`

Since I don't currently capture C++ stacks, the primitive name carries most of the useful signal. In my current implementation, primitives are used as event categories, which the viewer displays as graph colors and legend entries. The problem is that there's only a limited number of colors, so distinct primitives end up sharing one. To work around this, I also added the primitive name to the user metadata, where it shows up as text (shown in the figure below).

![User metadata in memory viz](/assets/images/2026-07-30-mlx_memory_viz/plan3_user_metadata.png)
<p style="text-align: center;"><i>Figure 7. User metadata displayed on memory viz.</i></p>

<hr class="hr-top" /><hr />

# Summary

To reiterate, the goal of this post was to help myself work through the designs behind PyTorch and MLX, and my reasoning for how I implemented memory viz for MLX. I'm still learning, and the implementation is only at the prototype stage. I feel my explanations could use more of the big picture, and I'd welcome any corrections.

Overall this has been a great learning experience. Everything I picked up about C++ along the way also led me to discover some really helpful CppCon talks! I don't know where this project will go, but I hope it turns out to be useful to others too.