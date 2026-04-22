# Linux → Docker → Kubernetes Ramp

Prerequisite ladder for build-project 14 (K8s LeaderWorkerSet deployment)
in `buildable-projects.md`. Aimed at a Python backend engineer comfortable
with Docker but new to Linux ops and K8s.

Total scope: **~4–5 weeks** if starting from scratch; **~2–3 weeks** if
Linux/bash is already fluent.

---

## Part 1 — Linux / bash primer (~1 week)

K8s runs on Linux; every pod is a Linux process in a namespace/cgroup, and
debugging means `kubectl exec` into a container and running Linux commands.
Fluency here pays off daily.

### 1.1 Filesystem & navigation
- Hierarchy: `/etc` (config), `/var` (state/logs), `/opt` (third-party
  apps), `/usr` (binaries), `/tmp` (scratch), `/proc` (kernel info),
  `/dev` (devices).
- Commands: `ls -la`, `cd`, `pwd`, `find`, `tree`, `du -sh`, `df -h`.
- Paths: absolute vs relative, `~`, `.`, `..`, globbing (`*`, `?`, `[]`).

### 1.2 Processes & signals
- `ps aux`, `top`, `htop`, `pgrep`, `pkill`.
- Signals: `SIGTERM` (15), `SIGKILL` (9), `SIGHUP` (1). K8s sends SIGTERM
  on pod shutdown — your app must handle it gracefully.
- `&`, `jobs`, `fg`, `bg`, `nohup`, `disown`, `Ctrl+C`, `Ctrl+Z`.

### 1.3 Permissions & users
- `chmod`, `chown`, `chgrp`, octal vs symbolic mode (`755`, `u+x`).
- Users/groups: `id`, `whoami`, `sudo`, `/etc/passwd`, `/etc/group`.
- Matters in K8s because `securityContext.runAsUser` and filesystem
  permissions on mounted volumes bite frequently.

### 1.4 Networking basics
- `curl -v`, `wget`, `ping`, `dig`, `nslookup`, `host`.
- `ss -tulpn` (modern `netstat`) — see which process owns which port.
- `/etc/hosts`, `/etc/resolv.conf` — DNS basics (K8s DNS mimics these).
- Ports: privileged (<1024), ephemeral range, TCP vs UDP, localhost vs
  0.0.0.0.

### 1.5 Streams, pipes, redirection
- `stdin` (0), `stdout` (1), `stderr` (2). `|`, `>`, `>>`, `2>&1`, `/dev/null`.
- `tee`, `xargs`, `cat`, `less`, `head`, `tail -f`.
- K8s logs are just container stdout/stderr streamed; `kubectl logs` =
  `tail -f` over the network.

### 1.6 Text tools
- `grep -r`, `sed`, `awk`, `cut`, `sort`, `uniq`, `wc`, `tr`.
- `jq` for JSON (kubectl output is JSON/YAML constantly).
- `yq` for YAML.

### 1.7 Environment & shell config
- `export`, `env`, `printenv`, `unset`.
- `~/.bashrc`, `~/.zshrc`, `~/.profile`, `PATH`, shell aliases.
- `$?`, `$$`, `$@`, `$#` in scripts; `&&`, `||`, `;` chaining.

### 1.8 Package management & systemd (know of, not master)
- `apt` (Debian/Ubuntu), `yum`/`dnf` (RHEL/Fedora), `apk` (Alpine — most
  container base images).
- `systemctl status/start/stop/restart`, `journalctl -u <svc> -f`.

### 1.9 SSH & file transfer
- `ssh user@host`, key pairs (`~/.ssh/id_ed25519`, `authorized_keys`).
- `scp`, `rsync -avz`, `ssh -L` for port forwarding (K8s uses the same
  concept with `kubectl port-forward`).

### 1.10 Bash scripting minimum
- Shebang (`#!/usr/bin/env bash`), `set -euo pipefail`.
- Variables, `if/then/fi`, `for/while`, functions, `$(...)`.
- Enough to read K8s entrypoint scripts and write debug one-liners.

**Resources:**
- *The Linux Command Line* — William Shotts (free PDF at
  linuxcommand.org/tlcl.php). The canonical beginner book.
- `missing.csail.mit.edu` — MIT's "Missing Semester" course. Short,
  excellent.
- `man <command>` and `tldr <command>` (install `tldr` — curated examples).

---

## Part 2 — Docker primer (~3–5 days, gap-filling)

Assuming the basics (`docker run`, `docker ps`, `docker-compose up`) are
already known. The gaps that matter for K8s:

### 2.1 Images, layers, and the build cache
- `Dockerfile` instructions: `FROM`, `RUN`, `COPY`, `ADD`, `CMD`,
  `ENTRYPOINT`, `ENV`, `ARG`, `WORKDIR`, `EXPOSE`, `USER`, `HEALTHCHECK`.
- Each instruction = a layer. Order from least-changing to
  most-changing (deps before app code) so the cache isn't busted.
- `.dockerignore` — mirror `.gitignore` logic; avoid shipping `.git/` or
  `__pycache__/` into images.

### 2.2 CMD vs ENTRYPOINT
- `ENTRYPOINT` = the executable; `CMD` = default args. K8s `command:` and
  `args:` map 1:1 to these. Getting this wrong is a common "my pod
  CrashLoopBackOff" cause.
- Prefer exec form (`["python", "app.py"]`) over shell form
  (`python app.py`) — exec form receives SIGTERM correctly, shell form
  masks it via `/bin/sh -c`.

### 2.3 Multi-stage builds
- `FROM ... AS builder` → `FROM ... AS runtime` with `COPY --from=builder`.
- Ship only runtime deps, not build toolchains. Matters a lot for
  Python + CUDA images — base on `nvidia/cuda:...-runtime` not `-devel`.

### 2.4 Image size & security
- Prefer `python:3.12-slim` or `distroless` over full Ubuntu.
- Run as non-root: `RUN useradd -m app && USER app`.
- Scan with `docker scout`, `trivy`, or `grype`.

### 2.5 Volumes, bind mounts, tmpfs
- Bind mount = host path → container path (dev loop).
- Named volume = Docker-managed storage.
- K8s equivalents: `hostPath` (bind mount, avoid in prod),
  `emptyDir` (scratch), `PersistentVolumeClaim` (named volume, the real
  production path).

### 2.6 Networking
- Bridge (default), host, none, user-defined networks.
- Port publishing: `-p host:container`. K8s handles this via `Service` —
  the container always binds the same port; the Service virtualizes it.

### 2.7 Registries
- `docker login`, `docker tag`, `docker push`.
- Public: Docker Hub, GHCR, Quay.io. Private: ECR, GCR/Artifact Registry,
  ACR. K8s pulls images from these via `imagePullSecrets`.
- Tags vs digests: always pin by digest (`@sha256:...`) in production
  manifests.

### 2.8 Runtime internals (know of)
- Docker Engine sits on top of `containerd` and `runc`. K8s talks to
  `containerd` directly (Docker-shim was removed in K8s 1.24).
- **OCI image format** — the standard Docker images follow; K8s runs any
  OCI image, Docker-built or not.
- Linux namespaces (pid, net, mnt, user, ipc, uts) + cgroups (CPU, memory
  limits) are what make a "container." K8s pods = a set of containers
  sharing network and IPC namespaces.

### 2.9 docker-compose → K8s manifest mental map

| docker-compose concept        | K8s concept                         |
|-------------------------------|-------------------------------------|
| `service:`                    | `Deployment` + `Service`            |
| `image:`                      | `spec.template.spec.containers[].image` |
| `ports: "8080:80"`            | `Service.spec.ports`                |
| `environment:`                | `env:` or `envFrom: configMapRef`   |
| `volumes: ./data:/data`       | `PersistentVolumeClaim` + `volumeMounts` |
| `depends_on:`                 | readiness probes + init containers  |
| `networks:`                   | Services + NetworkPolicies          |
| `restart: always`             | default (managed by the controller) |

**Resources:**
- *Docker Deep Dive* — Nigel Poulton. Short, approachable.
- Docker's own docs at docs.docker.com — the "Best practices for
  Dockerfiles" page is essential.

---

## Part 3 — Kubernetes ramp (~2–3 weeks)

Now the content from before.

### Week 1 — Fundamentals (no GPU)
1. Install `kind` or `minikube` on Mac — single-node K8s locally.
2. Work through **Kubernetes Basics** on kubernetes.io (interactive,
   ~2 hours).
3. Book: *Kubernetes in Action*, Marko Lukša (2nd ed., 2023) —
   chapters 1–6 are the canonical Docker→K8s on-ramp.
4. Hands-on: deploy nginx as Deployment + Service + Ingress. Then do the
   same for the FastAPI project-9 server (CPU-only, no model).
5. Core objects to master: Pods, Deployments, ReplicaSets, Services,
   ConfigMaps, Secrets, Namespaces, `kubectl` basics.

### Week 2 — Stateful + autoscaling
1. Add persistent volumes (PV, PVC) — mount a local model file.
2. StatefulSets, Jobs, CronJobs.
3. Horizontal Pod Autoscaler on CPU → then on a **custom metric**
   (queue depth via Prometheus adapter).
4. Helm basics — package the server as a chart.
5. Free video: **TechWorld with Nana** full K8s course (YouTube, ~4h).

### Week 3 — GPU + inference-specific
1. Move to a real cluster with a GPU (GKE/EKS/runpod/Lambda Labs —
   ~$1–3/hr for a single T4 or L4). Docker Desktop on Mac can't expose
   GPUs to K8s.
2. Install the **NVIDIA GPU Operator** — standard way to expose GPUs to
   pods (`resources.limits: nvidia.com/gpu: 1`).
3. Deploy vLLM as a single-GPU Deployment, expose via Service, hit with
   `curl`.
4. Add a **Gateway API** resource (successor to Ingress) with host
   routing.
5. Read **LeaderWorkerSet** docs
   (github.com/kubernetes-sigs/lws) — what enables multi-host tensor
   parallelism.
6. *Now* build-project 14 is tractable: LWS deployment, HPA on queue
   depth, prefix-affinity routing via llm-d's gateway plugin.

---

## Anchor resources

**Linux/bash**
- *The Linux Command Line* — William Shotts (free)
- MIT Missing Semester — missing.csail.mit.edu

**Docker**
- *Docker Deep Dive* — Nigel Poulton
- docs.docker.com → "Best practices for Dockerfiles"

**Kubernetes**
- *Kubernetes in Action* (2nd ed.) — Marko Lukša, Manning 2023
- *Kubernetes Up & Running* (3rd ed.) — Hightower/Burns/Beda,
  O'Reilly 2022
- kubernetes.io official tutorials
- **TechWorld with Nana** — YouTube full course
- KubeCon talks, "AI on K8s" track (2024–2025)

**GPU / inference on K8s**
- NVIDIA GPU Operator docs
- LeaderWorkerSet — github.com/kubernetes-sigs/lws
- llm-d quickstart — github.com/llm-d/llm-d
- vllm-project/production-stack
