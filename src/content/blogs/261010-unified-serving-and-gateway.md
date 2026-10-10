---
title: "Data-parallel inference in PhyAI"
description: "Run complete policy replicas on multiple GPUs, distribute independent requests, and extend the replica pool across hosts."
published: 2026-10-10
author: "PhyAI Research"
readTime: "7 min read"
---

A policy can fit on one GPU while the application needs to serve many robots or environments at once. Each observation starts another inference request. Running several complete copies of the policy lets different GPUs process those requests concurrently.

This is the data-parallel (DP) inference problem behind PhyAI's serving update. PhyAI places and manages the replicas, then distributes requests through a shared Engine interface. The gateway extends request distribution to independently running model servers across hosts. LeRobot is one way to connect clients to that pool.

## A complete policy on each GPU

Consider four GPUs running π0.5. Each GPU holds a complete policy replica, loaded from the same checkpoint. Four independent requests can execute on four replicas at the same time. Each replica runs the policy's full vision-language-action path and produces the result for its own input.

The unit of routing is a request. It can contain a single observation or a batch that the application has already prepared. PhyAI sends it whole to one replica, so the model scheduler keeps its existing input contract and per-replica batch behavior.

<figure class="architecture-figure systems-figure" data-wide-content>
  <picture>
    <source media="(max-width: 620px)" srcset="../../blog/261010-dp-replicas-mobile.svg" />
    <img src="../../blog/261010-dp-replicas.svg" alt="Independent requests enter one PhyAI Engine and are routed to four complete PI0.5 replicas on GPU 0 through GPU 3. Each replica processes its own request and returns a separate result." loading="lazy" decoding="async" />
  </picture>
  <figcaption>Inference DP with four complete π0.5 replicas. Each request stays within one replica. The diagram shows concurrent execution capacity, not a measured fourfold speedup.</figcaption>
</figure>

Adding replicas gives the application more capacity for independent work. The compute path of an individual request stays within its selected replica. Under concurrent load, extra replicas can also reduce time spent waiting for an available GPU. Actual throughput depends on request sizes, preprocessing, data transfers, and how much work the application keeps in flight.

## Configure the pool, then submit work

`DeploymentConfig.replica_count` sets the number of complete model copies. For single-GPU π0.5 replicas, assigning four devices and setting `replica_count=4` creates a four-replica pool. PhyAI starts the workers, loads each replica, and waits for them to become ready.

The application then submits independent requests through one Engine. The local dispatcher chooses the healthy replica with the fewest requests in flight and rotates among ties. Each submission returns a future associated with that request, so the caller can collect results in the order it needs.

The helper below accepts prepared `PI05Request` objects, each with batch size 1 and the model's default three-camera input contract. Call it from your application's `if __name__ == "__main__":` entry point because the managed workers use Python's `spawn` method.

```python
import torch

from phyai import DeploymentConfig, Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig
from phyai.models.pi05.main_pi05 import PI05Args


def infer_many(requests, checkpoint_dir):
    devices = (0, 1, 2, 3)
    args = EngineArgs(
        plugin="pi05",
        plugin_args=PI05Args(
            checkpoint_dir=checkpoint_dir,
            max_batch_size=1,
        ),
        config=EngineConfig(
            device=DeviceConfig(
                target="cuda", params_dtype=torch.bfloat16
            ),
        ),
    )
    with Engine(
        args,
        deployment=DeploymentConfig.local(
            devices=devices, replica_count=len(devices)
        ),
    ) as engine:
        futures = [engine.submit(request) for request in requests]
        return [future.result().cpu() for future in futures]
```

Submitting the requests before waiting is what exposes concurrency to the pool. A loop that calls `step()` and waits for each result leaves only one request outstanding. In a long-running application, keep the Engine alive and continue feeding it work; the helper includes construction and cleanup to show the complete lifecycle.

Here, `max_batch_size` applies to each replica. The device indices refer to GPUs visible to the launching process. Results are copied to CPU before the engine closes because CUDA results returned through IPC belong to the worker's memory. The [π0.5 guide](https://phyai.mintlify.app/models/pi05/ws1) describes how to prepare the input tensors and configure other camera or batch shapes.

For a large batch of independent observations, the caller can split it into valid per-replica requests, submit those requests, and concatenate their results. PhyAI preserves those request boundaries; batch splitting and result assembly remain explicit application steps.

## Keep DP outside the model implementation

The model plugin continues to implement the execution of one replica. It owns the model's preprocessing contract, forward path, and scheduler. The shared deployment layer owns the replica pool. This lets existing model integrations gain multi-replica serving through the same request interface.

Model parallelism fits inside that arrangement. A model that needs several GPUs for one replica can use a supported TP or CFG topology, then replicate that complete group. Four GPUs could hold four single-GPU replicas, or two replicas with two ranks each. `ParallelConfig` describes the ranks inside a replica; `DeploymentConfig` describes how many copies to run and where to put them.

The available combinations depend on the model. π0.5 currently uses single-GPU replicas. Cosmos3 supports TP and CFG inside a replica. Communication groups stay within each replica, and each group receives its own request. The [parallel-serving guide](https://phyai.mintlify.app/deployment/parallel-serving) describes the topology and placement rules.

## Extend DP across model servers

Across hosts, the gateway distributes independent requests among model servers registered under the same model name. Each server owns its checkpoint and Engine. The gateway chooses a healthy server using its in-flight request count and forwards the complete request there.

For π0.5, the concrete serving layout is one independent model server per GPU replica. Two hosts with two servers each provide four separately addressable copies of the policy. Clients use one gateway address while those servers execute different requests concurrently.

<figure class="architecture-figure systems-figure" data-wide-content>
  <picture>
    <source media="(max-width: 620px)" srcset="../../blog/261010-dp-multi-host-mobile.svg" />
    <img src="../../blog/261010-dp-multi-host.svg" alt="A gateway routes independent client requests among four PI0.5 model servers grouped across two hosts. Each server has its own Engine and one complete GPU replica." loading="lazy" decoding="async" />
  </picture>
  <figcaption>DP across hosts with one model server per π0.5 replica. The gateway routes at the server level; each selected server executes the request on its own replica.</figcaption>
</figure>

Start from the [π0.5 server YAML](https://github.com/mingti-org/phyai/blob/main/examples/pi05/server.yaml). On each host, prepare two configurations with ports `50063` and `50064`, the same `model_name: pi05`, and the intended checkpoint and robot settings. Bind each server process to its own GPU with `CUDA_VISIBLE_DEVICES`, then register all four endpoints:

```bash
phyai-gateway \
  --backend pi05 10.0.0.11:50063 \
  --backend pi05 10.0.0.11:50064 \
  --backend pi05 10.0.0.12:50063 \
  --backend pi05 10.0.0.12:50064
```

Replace the addresses with your model hosts. The operator keeps the registered replicas on the intended model version and matches their input/output contracts. The gateway checks health and supports adding or removing servers while it runs, so the pool can change without changing every client's address.

There is a practical reason to use separate π0.5 server processes here. Its current server adapter serializes preprocessing, inference, and postprocessing to protect request-specific processor state. Configuring several Engine replicas behind that single adapter does not expose concurrent serving through it. Direct `Engine.submit()` and a gateway over independent servers are the two DP paths described in this post.

LeRobot's asynchronous client can connect through the gateway's adapter using its existing request and response formats. Match the registered model name, camera and action settings, and control rate, then point the client at the gateway. The [LeRobot setup guide](https://phyai.mintlify.app/deployment/server-gateway#connect-a-lerobot-client) covers those details. The client adapter provides access to the replica pool; the independent server processes provide its concurrency.

## Operate the replicas as a pool

Request routing is stateless, so successive observations from one client can reach different servers. Applications that keep session history inside a model server need to account for that state before distributing requests. The local dispatcher and gateway count in-flight requests; requests with different costs can still produce uneven GPU utilization.

Local managed replicas also share a supervisor. A worker failure can make that local pool unavailable, and the application or process manager is responsible for rebuilding it. At the gateway level, health checks remove unavailable model servers from new request selection. Requests that have already started are not automatically retried.

The pool shown here handles ordinary inference requests. The π0.5 PPO `rollout_step()` and actor weight-update APIs currently operate on an inline or single external core. An RL framework manages those rollout workers and their weight synchronization separately, as described in the [October update](../261010-october-update/).

The DP execution layer landed in [PR #58](https://github.com/mingti-org/phyai/pull/58), followed by the [gateway in PR #67](https://github.com/mingti-org/phyai/pull/67). Start with enough replicas for the independent requests your application can supply, then measure completed requests, queueing time, and GPU utilization under the workload you expect. The companion article on [kernel dispatch and Humming quantization](../261010-kernel-dispatch-and-humming/) explains the execution choices within each replica.
