---
title: "PhyAI Day 0 Support for MiniCPM-RobotManip"
description: "How PhyAI brought MiniCPM-RobotManip to optimized inference on release day, reaching 36.77 Hz on NVIDIA H20 and 47.85 Hz on RTX 5090."
published: 2026-07-19
author: "PhyAI Research"
readTime: "7 min read"
---

PhyAI supports **MiniCPM-RobotManip from Day 0**. Through the same compact engine API used by other Physical AI models, the original checkpoint can be loaded directly and run through a model-specific execution path without rewriting the surrounding inference stack. Day 0 support here means more than making the model run: the first release includes its preprocessing contract, weight mapping, complete vision-language-action path, CUDA Graph execution, selectable action precision, and dedicated Triton kernels.

## MiniCPM-RobotManip

MiniCPM-RobotManip is a 1.5B-parameter open-source general-purpose VLA built on MiniCPM-V 4.6. It combines a SigLIP vision tower, a Qwen3.5 hybrid language backbone, and a DiT action head. The language backbone interleaves Gated DeltaNet linear-attention layers with full-attention layers, while the 16-layer DiT alternates action self-attention and cross-attention to the VLM representation. The policy predicts a 30-step action chunk in a unified 80D action space and refines it through four clean-action steps.

Three properties make the model particularly useful for real robots:

- **Generality.** MiniCPM-RobotManip is trained across embodiments, scenes, and tasks, while retaining task-level capability comparable to specialized models.
- **Visual efficiency.** Its MiniCPM-V vision path compresses each frame from 256 visual tokens to 64, a 4x reduction that lowers the cost of continuous multi-camera input while preserving useful visual information.
- **Native visual context.** Visual history is placed directly in the model context. Streaming context management, relevant-history retention, and context-specific post-training let the policy use prior observations without treating every frame as an isolated image.

This combination matters in deployment: the model remains compact, spends fewer tokens on each camera frame, and is designed to carry temporal evidence across a robot's observation stream.

### Model-side efficiency

The MiniCPM team also reports a model-side latency comparison before any PhyAI runtime optimization. On an NVIDIA H100 with three camera views and BF16 execution, MiniCPM-RobotManip completes end-to-end action prediction in **120 ms**, compared with **234 ms** for π0.5 under the same setup. That is approximately **49% lower latency** in the benchmark reported with the model release.

<figure class="benchmark-figure analysis-figure" data-wide-content>
	<img class="benchmark-chart" src="../../blog/minicpm-robotmanip-model-latency.svg" alt="Bar chart showing the MiniCPM team's H100 three-view BF16 benchmark, with MiniCPM-RobotManip at 120 milliseconds and pi 0.5 at 234 milliseconds." />
	<figcaption>Model-side benchmark reported by the MiniCPM team: NVIDIA H100, three-view input, BF16, end-to-end action prediction latency. This result is separate from the PhyAI runtime benchmarks below.</figcaption>
</figure>

### MiniCPM-RobotManip in action

The release demonstration shows MiniCPM-RobotManip controlling a dual-arm setup for a real-world sandwich-making task. It provides a useful complement to the latency numbers: visual compression and native context are being evaluated in a continuous, multi-camera manipulation setting rather than on isolated images.

<figure class="minicpm-demo-figure" data-wide-content>
	<img src="../../blog/minicpm-robotmanip-sandwich-demo.gif" alt="MiniCPM-RobotManip real-world dual-arm sandwich-making demonstration shown from multiple camera views." loading="lazy" decoding="async" />
	<figcaption>Real-world sandwich-making demonstration from the MiniCPM release deck, shown at 3x speed in the original presentation.</figcaption>
</figure>

## How PhyAI supports it

The integration begins at the model boundary. A MiniCPM-V 4.6 processor converts camera images and the task instruction into the exact token, pixel, and vision-grid tensors expected by the checkpoint; robot state is normalized into the policy's 80D input contract. The resulting request is passed to the PhyAI engine through the `minicpm_gr00t` plugin. PhyAI also remaps the training checkpoint names to its inference modules, so the original PTH or Safetensors checkpoint can be loaded directly.

The runtime then separates the work into a VLM stage and an action stage. Vision layout metadata, including position IDs, 2x2 window indices, inverse indices, and cumulative sequence lengths, is built once for each input shape. The VLM result conditions a fixed four-step action loop, which produces the complete action chunk.

### CUDA Graphs for the complete repeated path

PhyAI captures the fixed-shape VLM path and the complete four-step action loop in separate, shape-keyed CUDA Graphs. New inputs are copied into static buffers and replayed without rebuilding the Python launch sequence on every control step. Capturing the whole action loop, rather than graphing each DiT call independently, also removes host launch overhead between the four refinement steps.

On NVIDIA H20 in BF16, CUDA Graph execution raises throughput from **10.12 Hz to 33.28 Hz**, a **3.29x** speedup over the native path.

### Triton kernels for MiniCPM's hybrid backbone

Two recurring sequences in the Qwen3.5 hybrid backbone received model-specific Triton kernels:

- **RMSNorm + SiLU gate.** The fused kernel computes the RMS statistics in FP32, applies the learned weight, evaluates the SiLU gate, and writes the gated result once. This replaces separate normalization, activation, and multiplication launches and avoids materializing their intermediate tensors.
- **Depthwise causal Conv1d + SiLU + Q/K/V split.** Gated DeltaNet layers project a mixed QKV tensor before a short depthwise causal convolution. The fused kernel performs the convolution, applies SiLU, and writes Q, K, and V directly into three contiguous outputs. It removes the transpose-heavy Conv1d path and avoids writing and rereading a combined activated QKV tensor before splitting it.

With these fusions added to CUDA Graph execution, H20 BF16 throughput reaches **36.77 Hz**: **3.63x** the native path and a further **10.5%** over CUDA Graph alone. The kernels have numerical tests against the corresponding PyTorch operations for both FP32 and BF16 execution.

<figure class="benchmark-figure" data-wide-content>
	<div class="benchmark-grid">
		<img class="benchmark-chart" src="../../blog/minicpm-day0-h20-optimization.svg" alt="Bar chart showing MiniCPM-RobotManip BF16 throughput on NVIDIA H20 increasing from 10.12 Hz natively to 33.28 Hz with CUDA Graph and 36.77 Hz with CUDA Graph plus Triton fusion." />
		<img class="benchmark-chart" src="../../blog/minicpm-day0-three-view-throughput.svg" alt="Grouped bar chart showing MiniCPM-RobotManip three-view throughput on RTX 5090, H20, and H100 for PhyAI FP32 and BF16 action execution, with an EmbodyEvalKit baseline on H100." />
	</div>
	<figcaption>MiniCPM-RobotManip throughput; higher is better. Left: the H20 optimization sequence uses BF16. Right: the three-view workload uses 448x448 images and 64 text tokens. H20 FP32 is reported as approximately 32 Hz in the source table; the precise final H20 BF16 result is 36.77 Hz. Libero two-view results are excluded.</figcaption>
</figure>

## Three-view benchmark

The three-view benchmark reports steady-state model inference after warm-up and excludes one-time preprocessing. Its current command-line interface exposes `--action-dtype float32` and `--action-dtype bfloat16`, allowing the action head's accuracy and throughput trade-off to be measured explicitly.

On RTX 5090, PhyAI reaches **36.23 Hz with FP32 actions** and **47.85 Hz with BF16 actions**. This path runs directly with `uv run python`; it does not require the `PHYAI_FORCE_LINEAR_KERNEL=torch` override. On H100, the same three-view workload reaches **40.20 Hz in FP32** and **62.35 Hz in BF16**, compared with **9.44 Hz** for the EmbodyEvalKit path. The H20 BF16 result is **36.77 Hz**.

The dtype labels apply to the action parameters and activations; the MiniCPM-V vision-language backbone runs in BF16 in both modes. As with all throughput measurements, comparisons should keep the checkpoint, input shape, warm-up, timing scope, and software environment fixed.

## Building Day 0 support in the open

MiniCPM-RobotManip shows the role we want PhyAI to play: a new Physical AI model should be able to move from release to a faithful, optimized inference path without every team rebuilding the same runtime machinery. PhyAI is an open-source community project, and we welcome developers and researchers to help extend model coverage, validate numerical accuracy, improve kernels, and carry these systems from research experiments into large-scale deployment.
