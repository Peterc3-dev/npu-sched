# npu-sched

Job scheduler daemon and CLI for AMD XDNA 2 NPU workloads.

## Features

- Priority queue with configurable concurrency (0-9 priority levels, higher = sooner)
- REST API via axum — submit, list, cancel, and inspect jobs over HTTP
- Live NPU hardware status: module, device, firmware, column count, driver version
- Per-job timeout with automatic kill and partial output capture
- Output truncation at 1 MB to prevent memory blowout
- History eviction keeps memory bounded (configurable `--max-history`)
- Built-in CLI client — no curl required for basic operations

## Install

```
cargo build --release
```

Binary lands at `target/release/npu-sched`.

## Usage

Start the daemon:

```
npu-sched serve --port 7890 --concurrency 1
```

Submit a job:

```
npu-sched submit --name "inference" --cmd "python3 run_model.py" --priority 8 --timeout 120
```

Check NPU and queue status:

```
npu-sched status
```

List all jobs:

```
npu-sched jobs
```

Cancel a pending job:

```
npu-sched cancel <job-uuid>
```

## API Endpoints

| Method   | Path         | Description              |
|----------|--------------|--------------------------|
| `GET`    | `/health`    | Health check             |
| `GET`    | `/status`    | NPU status + queue stats |
| `POST`   | `/jobs`      | Submit a job             |
| `GET`    | `/jobs`      | List all jobs            |
| `GET`    | `/jobs/:id`  | Get job details          |
| `DELETE` | `/jobs/:id`  | Cancel a pending job     |

---

Built with Rust + axum + tokio.
