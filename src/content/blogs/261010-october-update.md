---
title: "2026 Oct Update"
description: "Configurable kernel dispatch with Humming quantization, data-parallel serving across GPUs and hosts, PI0.5 PPO rollouts, and support for MolmoAct2 and Qwen-Image 2.1."
published: 2026-10-10
author: "PhyAI Research"
readTime: "7 min read"
featured: true
---

## TL;DR

- Configure kernel selection and Humming quantization through one YAML policy, with INT4, INT8, FP8, MXFP4, and NVFP4 options.
- Scale independent requests across complete model replicas, using the Engine for local DP and a gateway for multiple hosts and LeRobot clients.
- Run π0.5 PPO rollouts with actor weight updates. The PhyAI runtime is merged; RLinf integration is under review.
- Run MolmoAct2 continuous-action inference and Qwen-Image 2.1 image generation and editing.

This update covers changes merged through October 10, 2026. Two companion articles explain the systems work in more detail: [kernel dispatch and Humming quantization](../261010-kernel-dispatch-and-humming/), and [data-parallel inference across GPUs and hosts](../261010-unified-serving-and-gateway/).

## 1. Kernel dispatch and Humming quantization

Choosing a kernel depends on the workload. The GPU matters, but so do batch size, tensor layout, precision, and the particular stage of a model. We moved those decisions into a shared dispatcher so each model can reuse the same selection machinery.

Layers describe the operation they need, and the dispatcher checks candidate implementations against their capabilities. A YAML policy can then prefer a backend, restrict a workload to a measured implementation, or define a fallback. The system covers GEMM, attention, normalization, convolution, and other recurring operations. It caches prepared choices for repeated execution and supports optional autotuning for operations that its built-in benchmark can measure.

<figure class="architecture-figure systems-figure" data-wide-content>
  <picture>
    <source media="(max-width: 620px)" srcset="../../blog/261010-kernel-dispatch-mobile.svg" />
    <img src="../../blog/261010-kernel-dispatch.svg" alt="PhyAI's kernel dispatcher combines an operation's requirements with a policy to choose and cache a compatible execution backend." loading="lazy" decoding="async" />
  </picture>
  <figcaption>Model layers share the selection machinery while retaining the operation shapes and execution paths their architectures need.</figcaption>
</figure>

Humming quantization is a practical use of this design. One policy chooses which layers to quantize and which kernels execute the resulting weights. PhyAI can load an original checkpoint and quantize selected layers to INT4, INT8, FP8, MXFP4, or NVFP4, subject to each scheme's hardware and shape requirements.

The supplied π0.5 W4A16 preset, for example, uses INT4 weights for the language model and action expert's attention/MLP projections while keeping BF16 or FP16 activations. The vision tower and input/output heads retain their configured precision. Those quantized projections use compatible Humming kernels; other operations keep their own selected implementations. We also provide Cosmos3 presets.

Teams can keep sensitive layers at their original precision and evaluate lower-precision projections on their own checkpoint and tasks. The [kernel dispatcher and Humming article](../261010-kernel-dispatch-and-humming/) explains how to specify those choices and includes a runnable preset example.

## 2. Data-parallel inference and multi-host serving

Many robots or environments can request actions from the same policy at once. Data-parallel (DP) inference handles that independent work by running complete model replicas on multiple GPUs. Each replica processes its own input through the full model path.

PhyAI's unified Engine now manages this replica pool. A deployment configuration sets the number of copies and their GPU placement. Applications submit concurrent requests through `Engine.submit()`, and the local dispatcher routes each whole request to a healthy replica with the fewest requests in flight. Each submission has its own future; the caller can collect results in the required order.

<figure class="architecture-figure systems-figure" data-wide-content>
  <picture>
    <source media="(max-width: 620px)" srcset="../../blog/261010-dp-replicas-mobile.svg" />
    <img src="../../blog/261010-dp-replicas.svg" alt="Four complete PI0.5 policy replicas process independent requests concurrently, with one Engine routing requests and returning their individual results." loading="lazy" decoding="async" />
  </picture>
  <figcaption>DP scales the capacity to process independent requests. Each request runs within one complete replica, which retains the model's own execution path and batch contract.</figcaption>
</figure>

This keeps DP outside the model implementation. π0.5 can run as several independent single-GPU replicas. A model such as Cosmos3 can use a supported TP/CFG group for each replica and then replicate that group. Applications that want to distribute a large batch prepare separate requests and assemble the results themselves.

The gateway extends the pool across hosts. For π0.5, each network replica runs in its own model server process, and the gateway routes requests among healthy servers registered under the same model name. This gives separate replicas the concurrency to serve different clients. The LeRobot adapter connects its asynchronous clients to that pool through a shared address.

The first benefit is more capacity for simultaneous inference, with less queueing when there is enough independent work to use the replicas. The [Data-parallel inference in PhyAI article](../261010-unified-serving-and-gateway/) explains the local and multi-host paths, provides a concurrent submission example, and covers the current server-adapter constraints.


## 3. π0.5 rollouts for reinforcement learning

PPO needs more from inference than a final action. The training system needs the sampled trajectory and its probabilities, and the rollout model needs to receive updated actor weights as training progresses. PhyAI now provides those pieces for π0.5.

The rollout runtime returns actions, the denoising chain, per-transition log probabilities, and optional values. An external controller supplies the sampling schedule and owns the training algorithm. PhyAI executes the model and returns the data needed for training. It also accepts actor weight updates without rebuilding the engine, then refreshes weight-dependent state and recaptures the affected CUDA Graphs.

The [RLinf integration PR](https://github.com/RLinf/RLinf/pull/1587) reports measured π0.5 PPO results on H800 GPUs. In its controlled four-GPU profile, PhyAI more than halved prediction time on the critical path. The improvement in the complete RL step is smaller because environment execution, actor updates, and other work still take time.

<div class="release-table-wrap" role="region" aria-label="Four-H800 PI0.5 PPO profile results" tabindex="0">
  <table class="release-table">
    <caption>π0.5 PPO profile on 4×H800; lower time is better.</caption>
    <thead><tr><th scope="col">Measured scope</th><th scope="col">HF baseline</th><th scope="col">PhyAI</th><th scope="col">Result</th></tr></thead>
    <tbody>
      <tr><th scope="row">Critical-path prediction</th><td>142.57 s</td><td>68.74 s</td><td>2.07× faster</td></tr>
      <tr><th scope="row">Mean complete RL step</th><td>772.45 s</td><td>657.79 s</td><td>14.8% less time</td></tr>
    </tbody>
  </table>
</div>

<figure class="benchmark-figure rl-pr-figure" id="pi05-rollout-profile" data-wide-content>
  <a href="../../blog/261010-rlinf-pi05-rollout-profile.png" target="_blank" rel="noopener" aria-label="Open the full-size PI0.5 rollout profile">
    <img class="benchmark-chart" src="../../blog/261010-rlinf-pi05-rollout-profile.png" width="4650" height="3120" alt="Original RLinf PR timeline of one stable PI0.5 PPO step on four H800 GPUs: 769.8 seconds for HF and 663.0 seconds for PhyAI, with actor, environment, and rollout activity shown separately." loading="lazy" decoding="async" />
  </a>
  <figcaption>One stable π0.5 PPO step on four H800 GPUs. Hatched bars summarize prediction time on the rank with the largest total (144.0 s for HF and 68.7 s for PhyAI); their position does not indicate additional work. The table above reports aggregate profile timings. Original figure from <a href="https://github.com/RLinf/RLinf/pull/1587">RLinf PR #1587</a>. <a href="../../blog/261010-rlinf-pi05-rollout-profile.png" target="_blank" rel="noopener">Open full-size figure</a>.</figcaption>
</figure>

A separate run on eight H800 GPUs completed 500 PPO steps from a fresh start on LIBERO-Spatial. The PR reports mean `success_once` of 0.8797 for PhyAI and 0.8684 for its stitched HF reference aligned by global step. This is a training-run comparison; the timing results above come from the separate four-GPU profile.

<figure class="benchmark-figure rl-pr-figure" id="pi05-rl-convergence" data-wide-content>
  <a href="../../blog/261010-rlinf-pi05-convergence.png" target="_blank" rel="noopener" aria-label="Open the full-size PI0.5 training convergence figure">
    <img class="benchmark-chart" src="../../blog/261010-rlinf-pi05-convergence.png" width="2544" height="1344" alt="Original RLinf PR chart of LIBERO-Spatial training success_once over PPO steps 0 to 499 on eight H800 GPUs, comparing the PhyAI fresh-start run with the HF reference aligned by global step." loading="lazy" decoding="async" />
  </a>
  <figcaption>LIBERO-Spatial training on eight H800 GPUs. PhyAI's fresh-start run covers 500 PPO steps and is compared with a stitched HF reference aligned by global step. The bold red curve is PhyAI's 20-step rolling mean. Original figure from <a href="https://github.com/RLinf/RLinf/pull/1587">RLinf PR #1587</a>. <a href="../../blog/261010-rlinf-pi05-convergence.png" target="_blank" rel="noopener">Open full-size figure</a>.</figcaption>
</figure>

These measurements describe the integration versions used for those experiments and predate the final cleanup commits. The PR documents the software stack, training configuration, and separate focused validation. PhyAI's rollout and weight-update support is merged; the RLinf integration remains under upstream review on October 10. The current gateway's bundled π0.5 server is a separate serving path and needs a matching adapter for RLinf's SGLang-style payloads.

For teams building a rollout controller, the [π0.5 integration guide](https://phyai.mintlify.app/models/pi05/ws1#training-rollout-integration) describes the request fields and weight-update lifecycle. The backend work gives training systems access to PhyAI's inference runtime while keeping sampling and PPO decisions in the training framework.

## 4. New model support

### MolmoAct2

[MolmoAct2 support](https://phyai.mintlify.app/models/molmoact2/inference) landed on October 10. The model uses camera images, a task instruction, and robot state to predict a sequence of continuous actions. PhyAI runs its vision-language backbone and action expert in BF16 and supports CUDA Graph execution for the repeated action-velocity computation.

The integration includes checkpoint loading, observation processing, action postprocessing, and an example using the `franka_droid` configuration. That example takes three RGB camera views and an eight-dimensional state, then returns 15 actions with eight values each. The base checkpoint is intended for task adaptation; the example validates the inference path rather than establishing a robot task success rate. Depth reasoning and the action-expert depth gate are outside the currently supported configuration.

### Qwen-Image 2.1

[Qwen-Image 2.1](https://phyai.mintlify.app/models/qwen-image/generation) adds text-to-image generation and image editing to PhyAI. Editing accepts up to ten reference images. The execution path includes conditioning, diffusion sampling, and VAE decoding, with examples that save the generated result as a PNG.

The model supports tensor parallelism and optional parallel CFG branches through the shared Engine deployment layer. A user can start with one GPU or configure more devices for the supported topology without replacing the model entry point. The implementation entered `main` on October 5.

August also brought the GR00T N1.7 integration with mixed-length batching, CUDA Graph profiles, and LIBERO evaluation tools. π0 gained additional LeRobot alignment and benchmarking tools, and the [MiniCPM-Robot Day-0 work](../260719-day-0-minicpm-robotic/) merged into the main branch. The two newly added model families in this October update are MolmoAct2 and Qwen-Image 2.1.

All four areas are available to inspect in the [PhyAI repository](https://github.com/mingti-org/phyai). Start with the [installation guide](https://phyai.mintlify.app/) for source and nightly packages, then use the model and deployment guides for the workload you want to run.

## Next

1. We are tuning kernels for NVIDIA Thor (SM110) using a KDA-like approach. Early results look promising, and we plan to merge these optimizations into PhyAI in a future update.
2. Support for Lingbot-VLA and Xiaomi VLA is also in progress.
