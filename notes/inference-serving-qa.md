# LLM Serving Layer — Follow-up Q&A

Deeper dives on the serving taxonomy (see `inference-serving-taxonomy.md`).

---

## 1. Are these taxonomies finite?

**No — they're living.** The top-level buckets (memory, scheduling,
parallelism, compute, routing) have been stable for ~2 years, but new
sub-techniques show up every few months. Disaggregation didn't exist as a
category before 2023. Speculative decoding variants (Medusa, EAGLE,
lookahead) are still proliferating. Think of the taxonomy as a working
frame, not a closed ontology.

---

## 2. PagedAttention with long requests

**Yes — one request spans many pages.** Pages are typically 16 tokens. A
10k-token context = ~625 pages per layer. The engine keeps a **page table**
(exactly like an OS virtual memory page table) mapping "logical position in
this request's KV" → "physical page in GPU memory." The pages don't need to
be contiguous, which is the whole point — it eliminates fragmentation.

---

## 3. Is prefix caching what LLM providers call "prompt caching"?

**Conceptually yes, but different implementations.** Anthropic's prompt
caching, OpenAI's automatic prompt caching, and Gemini's context caching
are all forms of prefix-KV reuse. Differences from in-engine RadixAttention:

- **Explicit markers** (Anthropic uses `cache_control`; OpenAI and Gemini
  auto-detect).
- **Longer-lived** — providers persist prefix KV to fast storage (not just
  GPU RAM) so it survives across requests and minutes of idle time.
- **Billing-visible** — you see cache hit/miss tokens in usage.

In-engine prefix caching (vLLM, SGLang) is the mechanism; provider prompt
caching is the productized API surface on top.

---

## 4. Does KV cache quantization use LoRA?

**No — unrelated.** LoRA is a *fine-tuning* technique: low-rank adapter
matrices added to frozen weights. KV cache quantization is a *numerical
representation* change: store KV tensors in int8/fp8/int4 instead of fp16,
purely a memory/compute optimization. They compose (you can serve a
LoRA-tuned model with a quantized KV cache), but they're orthogonal.

---

## 5. How does the system know when to evict/compress KV?

Every method has a different heuristic. The main families:

- **Attention-score based** (H2O) — track how much attention each past
  token receives across heads. Tokens with consistently tiny scores rarely
  influence output → safe to drop.
- **Window + sink** (StreamingLLM) — keep the first 4 "attention sink"
  tokens and a sliding window of recent tokens. Drop the middle. Works
  because attention concentrates on those two regions.
- **Position-based heuristics** (SnapKV, pyramid KV) — drop less-
  informative positions, often layer-dependent.
- **Memory-pressure-triggered** — at the engine level, evict whole requests
  (not tokens) when the KV pool is full. vLLM preempts and re-prefills the
  request when space frees up.

There's no free lunch — all token-level eviction trades some quality for
memory.

---

## 6. Is continuous batching like a websocket?

**No — different layers of the stack.** A websocket is a *network
transport*: a persistent bidirectional connection between client and server.
Continuous batching is a *server-side GPU scheduling strategy*.

What's actually going on:

- The HTTP/SSE/websocket connection delivers a request and streams tokens
  back.
- Inside the server, the GPU processes many requests in one "batch step"
  each forward pass. **Continuous** = the batch composition changes every
  step: finished sequences leave, new ones join, without waiting for the
  whole batch to complete.
- You could do continuous batching with zero websockets (plain HTTP SSE is
  more common). And you could have websockets with no batching at all.

---

## 7. How does chunked prefill work?

**Setup:** prefill (processing the prompt) is compute-heavy; decode
(generating one token) is memory-bandwidth heavy. If a 32k-token prompt
arrives, a single prefill step takes seconds — during which no other
request can generate tokens. Everyone stalls.

**Chunked prefill:** split the prefill into, say, 512-token chunks. Each
scheduler step does one chunk *plus* decode steps for everyone else already
generating. Roughly:

```
step 1: prefill chunk 1 of long request + decode tokens for requests A,B,C
step 2: prefill chunk 2 of long request + decode tokens for A,B,C,D
...
step N: last chunk → long request starts decoding too
```

Tradeoff: TTFT (time to first token) for the long request goes up slightly,
but ongoing requests don't stall. Net throughput and p99 latency improve.

---

## 8. How is load balancing configured?

Depends on the layer you're working at.

- **L4 (transport)** — round-robin or least-connections across replicas.
  Cloud LBs (ALB, GCP LB), HAProxy.
- **L7 (HTTP-aware)** — Envoy, nginx, Traefik. Can inspect headers, path,
  body. LLM gateways (LiteLLM, Portkey) sit here.
- **LLM-specific** — needs prefix-aware or session-aware routing. Hash on
  system prompt or conversation ID → consistently route to a replica whose
  KV cache already has the prefix. Tools: vLLM's router, SGLang's router,
  Envoy with a custom Lua/WASM filter, or your own gateway.

Key insight: naïve round-robin is bad for LLMs because it destroys KV
cache locality. Prefix-aware routing can 2–5× throughput for chat workloads.

---

## 9. Kubernetes for a Docker user

You already know the hard part (containers). K8s is the orchestration
layer on top.

**Mental model:**

| Docker concept        | K8s concept                           |
|-----------------------|----------------------------------------|
| Container             | Container (inside a Pod)               |
| `docker run`          | `kubectl apply -f deployment.yaml`     |
| `docker-compose.yml`  | `Deployment` + `Service` manifests     |
| One host              | A cluster of many hosts (nodes)        |
| Restart policy        | Controller reconciles desired state    |

**Core objects you'll actually use:**

- **Pod** — one or more containers that share a network and run together.
  Usually one pod = one container.
- **Deployment** — declarative spec: "I want 3 replicas of this pod,
  always." K8s reconciles reality to match.
- **Service** — stable virtual IP + DNS name pointing at a set of pods.
  Pods are ephemeral; services give you a stable endpoint.
- **Ingress / Gateway** — exposes services to the outside world with a
  hostname and TLS.
- **Node** — a machine (VM) in the cluster. GPU nodes have GPUs.
- **Namespace** — a logical folder for isolation (dev/staging/prod).
- **ConfigMap / Secret** — config and credentials mounted into pods.
- **HorizontalPodAutoscaler (HPA)** — "scale replicas when CPU > 70% (or
  a custom metric like queue depth)."

**What an LLM serving stack on K8s looks like:**

```
Ingress (exposes vllm.mycompany.com + TLS)
   │
   ▼
Service (vllm-svc, load-balances across pods)
   │
   ▼
Deployment (3 replicas of vLLM pod, each needs 1 GPU)
   │
   ▼
Pods scheduled onto GPU Nodes
```

You'd write a `Deployment` YAML specifying the vLLM container image, GPU
request (`nvidia.com/gpu: 1`), env vars, and model mount. K8s finds a GPU
node with capacity and starts the pod. If the pod crashes, K8s restarts
it. If you bump replicas to 10, K8s spins up 7 more.

**What K8s adds over Docker:**

- Self-healing (pods die → reschedule automatically)
- Rolling updates (deploy new version with no downtime)
- Declarative config (Git-ops friendly)
- Autoscaling (pods and nodes)
- Service discovery (pods find each other by DNS)
- Secrets/config management
- Resource scheduling (match GPU pods to GPU nodes)

**If you want to learn by doing:** install `minikube` or `kind` locally
(single-node K8s on your Mac), and `kubectl apply` a simple nginx
Deployment + Service. That'll click the mental model in an hour.
