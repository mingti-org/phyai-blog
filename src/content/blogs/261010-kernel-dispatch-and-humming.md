---
title: "Kernel dispatch and Humming quantization in PhyAI"
description: "How PhyAI selects kernels for each workload, and how Humming brings configurable low-precision inference to the same execution path."
published: 2026-10-10
author: "PhyAI Research"
readTime: "7 min read"
---

A kernel that works well for a large batch can be a poor choice for a robot waiting on one action chunk. Change the GPU, attention layout, or weight precision, and the choice changes again. Keeping those decisions inside every model soon becomes a maintenance problem: each new backend brings another set of conditions to check and another execution path to validate.

PhyAI's kernel dispatcher handles those decisions across models. Layers describe the computation they need, and the dispatcher checks which implementations can execute it before applying a selection policy. Humming quantization makes this useful in practice: a model can load selected projections in INT4 and execute them with Humming while its other layers keep their existing precision and kernels.

## A kernel choice belongs to a workload

The dispatcher covers matrix multiplication, attention, normalization, activation, convolution, embeddings, rotary position encoding, and modulation. These operations have different constraints. A GEMM implementation may require a particular weight layout; an attention backend may depend on a paged KV cache or a specific head dimension. A library being installed is only one part of eligibility.

PhyAI records these requirements with each implementation. At a call site, it combines them with facts about the GPU, tensor dtypes and shapes, quantization format, model role, and available libraries. It also checks the execution mode: a kernel used inside a CUDA Graph must declare that it supports capture.

This lets model code ask for an operation through a common interface while the dispatcher handles backend selection. FlashInfer, PhyAI's own kernels, Humming, and reference implementations can coexist in the catalog. Adding an implementation means declaring where it is valid and how to prepare it, so other models can reuse it when their calls meet those requirements.

<figure class="architecture-figure systems-figure" data-wide-content>
  <picture>
    <source media="(max-width: 620px)" srcset="../../blog/261010-kernel-dispatch-mobile.svg" />
    <img src="../../blog/261010-kernel-dispatch.svg" alt="A model operation and its hardware, dtype, shape, and quantization facts enter the dispatcher. Capability checks and policy select a backend, then prepared choices are cached for execution." loading="lazy" decoding="async" />
  </picture>
  <figcaption>The dispatcher separates a model's operation from the implementation used to execute it. Each backend remains subject to its own compatibility requirements.</figcaption>
</figure>

## Put deployment choices in a policy

Kernel policies are YAML files. A rule can prefer particular implementations, restrict a call to a compatible family, or pin one implementation for a measured workload. Rules can match the operation, model role, hardware, precision, and input dimensions. PhyAI resolves ordinary kernel rules by priority and reports ambiguous matches instead of silently depending on their order.

For example, this rule selects FlashInfer's FA2 implementation for the π0.5 action expert's paged attention on an SM90 GPU:

```yaml
schema: phyai.kernel/v1
profile: static

rules:
  - id: pi05-expert-fa2
    priority: 10
    match:
      op: attention_paged
      role: expert
      model.family: pi05
      device.arch: sm90
    restrict_to: flashinfer.attention_paged.fa2
```

The rule records a choice that a team can measure in the π0.5 execution path and reproduce across runs. On another GPU, the team can change its policy and keep the model implementation.

The dispatcher can explain its decision, including which rules matched and why it rejected a candidate. A missing dependency, unsupported layout, or capture restriction becomes a concrete diagnosis. Policies also control reference fallback; a deployment can require an error when no selected implementation is usable.

## Keep selection work out of repeated inference

Policy evaluation is useful during setup, but a control loop should not keep paying for the same decision. PhyAI caches prepared selections and memoizes them at call sites. Once the relevant configuration has been warmed up, it can freeze the selected choices for repeated execution. Changes to the selector or policy invalidate the corresponding cached decisions.

There is also an optional `autotune` profile. For supported operations such as ordinary GEMM and normalization, it measures eligible implementations and stores the fastest measured choice. The built-in benchmark can synthesize inputs for these tensor operations from their shapes and dtypes.

Attention needs a live runner and KV state, so it does not use that same synthetic benchmark. Teams can measure attention in the real model path and encode their result in a policy, as in the example above. This distinction keeps the tuning decision tied to an execution setup that the benchmark can actually represent.

## Humming makes the separation useful

Quantization changes the representation of a layer's weights. Executing those weights requires a kernel that understands the resulting format. PhyAI puts both decisions in one policy file: the `quantization` section chooses the layers and precision; the top-level kernel rules choose their execution implementations.

With Humming, PhyAI can perform post-training quantization when it loads an original FP16, BF16, or FP32 checkpoint. The current loading path uses round-to-nearest quantization. The user supplies the unquantized weights and a policy; PhyAI prepares the selected layers for low-precision execution.

The supplied π0.5 W4A16 policy is a concrete example. It quantizes attention and MLP projections in the language model and action expert to INT4 with a group size of 128. Activations remain BF16 or FP16. The vision tower, embeddings, normalization layers, and action/time heads retain their configured precision. An optional value head also retains its precision.

<figure class="architecture-figure systems-figure" data-wide-content>
  <picture>
    <source media="(max-width: 620px)" srcset="../../blog/261010-humming-policy-mobile.svg" />
    <img src="../../blog/261010-humming-policy.svg" alt="The PI0.5 W4A16 preset loads original weights, keeps vision and other excluded layers at their configured precision, quantizes language and action-expert projections to INT4, and routes those projections to Humming with BF16 or FP16 activations." loading="lazy" decoding="async" />
  </picture>
  <figcaption>The supplied π0.5 W4A16 preset applies quantization selectively. INT4 describes the chosen projection weights; it does not describe the precision of the entire model.</figcaption>
</figure>

With PhyAI installed and the Python environment activated, run the preset from the repository root:

```bash
PHYAI_KERNEL_CONFIG=examples/pi05/kernel_policies/humming_w4a16.yaml \
python examples/pi05/run_pi05.py --checkpoint /path/to/pi05_base
```

Replace the checkpoint path with the original model directory. Humming is included in the standard PhyAI installation. This BF16 example requires NVIDIA SM80 or newer, and the local weight dimensions must satisfy the preset's group-size constraints.

Inside the preset, quantized GEMMs carry the Humming layout. The dispatch rule then selects the matching implementation:

```yaml
rules:
  - id: humming-quantized-linear
    match:
      op: gemm
      quant.layout: humming
    restrict_to: humming.gemm.w4a16
```

That rule belongs alongside the preset's quantization section. Selecting a Humming kernel alone does not convert the weights. Likewise, a normal BF16 GEMM cannot execute a packed Humming weight simply because both operations are called GEMM.

## Choose precision by layer and hardware

The Humming integration supports INT4, INT8, FP8, MXFP4, and NVFP4 weight schemes, with several weight/activation combinations. PhyAI includes presets for π0.5 and Cosmos3 so users can start from a concrete layer-selection policy and adjust it for their checkpoint.

Hardware and dtype requirements differ by scheme. FP8 A8 combinations require SM89 or newer; the documented BF16 combinations otherwise start at SM80, subject to their format and shape constraints. Changing the quantization policy requires creating an engine and loading the weights again. The [quantization guide](https://phyai.mintlify.app/quantization/overview) lists the supported combinations and explains how to match layers.

Lower precision reduces the storage needed for the selected weights, but the effect on model latency depends on the workload and kernel. Action quality also needs evaluation with the intended checkpoint and tasks. PhyAI's policy makes those experiments easier to specify: keep sensitive layers at their original precision, change the candidate projections, and compare the resulting execution and outputs.

Kernel dispatch entered `main` in September through [PR #57](https://github.com/mingti-org/phyai/pull/57), followed by [Humming PTQ support in October](https://github.com/mingti-org/phyai/pull/69). The [π0.5 presets](https://github.com/mingti-org/phyai/tree/main/examples/pi05/kernel_policies) are a starting point for trying the two together: choose a supported scheme, load the original checkpoint, and evaluate the resulting actions and inference time.

For the rest of this release, read the [October update](../261010-october-update/). The companion post, [Data-parallel inference in PhyAI](../261010-unified-serving-and-gateway/), explains how to run complete model replicas across GPUs and hosts.
