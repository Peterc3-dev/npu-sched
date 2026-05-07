use std::collections::BinaryHeap;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Json, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::Router;
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::sync::{Mutex, Notify};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// NPU Status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NpuStatus {
    module_loaded: bool,
    device_exists: bool,
    device_accessible: bool,
    firmware_version: Option<String>,
    columns: Option<u32>,
    driver_version: Option<String>,
}

fn check_npu_status() -> NpuStatus {
    // Check if amdxdna module is loaded
    let module_loaded = std::fs::read_to_string("/proc/modules")
        .map(|s| s.lines().any(|l| l.starts_with("amdxdna ")))
        .unwrap_or(false);

    let dev_path = Path::new("/dev/accel/accel0");
    let device_exists = dev_path.exists();

    // Check if accessible (readable)
    let device_accessible = if device_exists {
        std::fs::metadata(dev_path)
            .map(|_| true)
            .unwrap_or(false)
    } else {
        false
    };

    // Try to read firmware version from sysfs
    let firmware_version = read_sysfs_firmware_version();

    // Try to read columns info
    let columns = read_sysfs_columns();

    // Try to read driver version
    let driver_version = read_driver_version();

    NpuStatus {
        module_loaded,
        device_exists,
        device_accessible,
        firmware_version,
        columns,
        driver_version,
    }
}

fn read_sysfs_firmware_version() -> Option<String> {
    // Try common sysfs paths for amdxdna firmware version
    let paths = [
        "/sys/class/accel/accel0/device/fw_version",
        "/sys/module/amdxdna/parameters/firmware_version",
    ];
    for p in &paths {
        if let Ok(v) = std::fs::read_to_string(p) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    // Fallback: parse from dmesg-style info or xrt-smi if we had it
    None
}

fn read_sysfs_columns() -> Option<u32> {
    let paths = [
        "/sys/class/accel/accel0/device/npu_columns",
        "/sys/module/amdxdna/parameters/npu_columns",
    ];
    for p in &paths {
        if let Ok(v) = std::fs::read_to_string(p) {
            if let Ok(n) = v.trim().parse::<u32>() {
                return Some(n);
            }
        }
    }
    None
}

fn read_driver_version() -> Option<String> {
    if let Ok(v) = std::fs::read_to_string("/sys/module/amdxdna/version") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return Some(v);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Job model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Job {
    id: Uuid,
    name: String,
    command: String,
    priority: u8,
    status: JobStatus,
    exit_code: Option<i32>,
    stdout: Option<String>,
    stderr: Option<String>,
    submitted_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
    timeout_secs: u64,
}

// Wrapper for priority queue ordering: higher priority first, then earlier submission
#[derive(Debug, Clone)]
struct PendingJob {
    id: Uuid,
    priority: u8,
    submitted_at: DateTime<Utc>,
}

impl PartialEq for PendingJob {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for PendingJob {}

impl PartialOrd for PendingJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first
        self.priority
            .cmp(&other.priority)
            // Earlier submission first (reverse because BinaryHeap is max-heap)
            .then_with(|| other.submitted_at.cmp(&self.submitted_at))
    }
}

// ---------------------------------------------------------------------------
// Scheduler state
// ---------------------------------------------------------------------------

struct SchedulerInner {
    jobs: Vec<Job>,
    queue: BinaryHeap<PendingJob>,
    running_count: usize,
    concurrency_limit: usize,
    max_history: usize,
}

struct Scheduler {
    inner: Mutex<SchedulerInner>,
    notify: Notify,
}

type AppState = Arc<Scheduler>;

const MAX_OUTPUT_BYTES: usize = 1_048_576; // 1 MB

impl Scheduler {
    fn new(concurrency_limit: usize, max_history: usize) -> Self {
        Self {
            inner: Mutex::new(SchedulerInner {
                jobs: Vec::new(),
                queue: BinaryHeap::new(),
                running_count: 0,
                concurrency_limit,
                max_history,
            }),
            notify: Notify::new(),
        }
    }
}

/// Truncate a string to at most `max_bytes` bytes (on a char boundary).
/// Returns the (possibly truncated) string with a note appended if truncated.
fn truncate_output(s: String, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s;
    }
    // Find a valid char boundary at or before max_bytes
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut truncated = s[..end].to_string();
    truncated.push_str("\n... [OUTPUT TRUNCATED — exceeded 1 MB limit]");
    truncated
}

/// Remove the oldest completed/failed/cancelled/timed-out jobs when total exceeds max_history.
fn evict_old_jobs(inner: &mut SchedulerInner) {
    if inner.max_history == 0 || inner.jobs.len() <= inner.max_history {
        return;
    }

    // Count how many we need to remove
    let excess = inner.jobs.len() - inner.max_history;

    // Collect indices of completed (non-pending, non-running) jobs, oldest first
    let mut removable_indices: Vec<usize> = inner
        .jobs
        .iter()
        .enumerate()
        .filter(|(_, j)| matches!(j.status, JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled | JobStatus::TimedOut))
        .map(|(i, _)| i)
        .collect();

    // Sort by completed_at (oldest first), falling back to submitted_at
    removable_indices.sort_by(|&a, &b| {
        let time_a = inner.jobs[a].completed_at.unwrap_or(inner.jobs[a].submitted_at);
        let time_b = inner.jobs[b].completed_at.unwrap_or(inner.jobs[b].submitted_at);
        time_a.cmp(&time_b)
    });

    // Remove up to `excess` oldest completed jobs (remove from highest index first to avoid shifting)
    let to_remove: usize = excess.min(removable_indices.len());
    let mut indices_to_remove: Vec<usize> = removable_indices[..to_remove].to_vec();
    indices_to_remove.sort_unstable_by(|a, b| b.cmp(a)); // descending so removal doesn't shift
    for idx in indices_to_remove {
        inner.jobs.remove(idx);
    }
}

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SubmitRequest {
    name: String,
    #[serde(rename = "cmd")]
    command: String,
    #[serde(default = "default_priority")]
    priority: u8,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
}

fn default_priority() -> u8 {
    5
}

fn default_timeout() -> u64 {
    300
}

#[derive(Serialize, Deserialize)]
struct SubmitResponse {
    id: Uuid,
    message: String,
    timeout_secs: u64,
}

#[derive(Serialize, Deserialize)]
struct StatusResponse {
    npu: NpuStatus,
    queue: QueueStats,
}

#[derive(Serialize, Deserialize)]
struct QueueStats {
    pending: usize,
    running: usize,
    completed: usize,
    failed: usize,
    cancelled: usize,
    timed_out: usize,
    total: usize,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    uptime_secs: u64,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ---------------------------------------------------------------------------
// API handlers
// ---------------------------------------------------------------------------

async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    let inner = state.inner.lock().await;
    let _ = inner; // just prove we can acquire the lock
    Json(HealthResponse {
        status: "ok".to_string(),
        uptime_secs: 0, // could track with a start time, but not critical
    })
}

async fn status_handler(State(state): State<AppState>) -> impl IntoResponse {
    let npu = check_npu_status();
    let inner = state.inner.lock().await;

    let mut pending = 0usize;
    let mut running = 0usize;
    let mut completed = 0usize;
    let mut failed = 0usize;
    let mut cancelled = 0usize;
    let mut timed_out = 0usize;

    for job in &inner.jobs {
        match job.status {
            JobStatus::Pending => pending += 1,
            JobStatus::Running => running += 1,
            JobStatus::Completed => completed += 1,
            JobStatus::Failed => failed += 1,
            JobStatus::Cancelled => cancelled += 1,
            JobStatus::TimedOut => timed_out += 1,
        }
    }

    Json(StatusResponse {
        npu,
        queue: QueueStats {
            pending,
            running,
            completed,
            failed,
            cancelled,
            timed_out,
            total: inner.jobs.len(),
        },
    })
}

async fn submit_handler(
    State(state): State<AppState>,
    Json(req): Json<SubmitRequest>,
) -> impl IntoResponse {
    let priority = req.priority.min(9);
    let now = Utc::now();
    let id = Uuid::new_v4();

    // BUG 2 fix: validate timeout_secs — 0 defaults to 300, clamp to 1..=3600
    let timeout_secs = if req.timeout_secs == 0 {
        300
    } else {
        req.timeout_secs.clamp(1, 3600)
    };

    let job = Job {
        id,
        name: req.name,
        command: req.command,
        priority,
        status: JobStatus::Pending,
        exit_code: None,
        stdout: None,
        stderr: None,
        submitted_at: now,
        started_at: None,
        completed_at: None,
        timeout_secs,
    };

    {
        let mut inner = state.inner.lock().await;
        inner.jobs.push(job);
        inner.queue.push(PendingJob {
            id,
            priority,
            submitted_at: now,
        });
    }

    // Wake the executor loop
    state.notify.notify_one();

    (
        StatusCode::CREATED,
        Json(SubmitResponse {
            id,
            message: "Job submitted".to_string(),
            timeout_secs,
        }),
    )
}

async fn list_jobs_handler(State(state): State<AppState>) -> impl IntoResponse {
    let inner = state.inner.lock().await;
    Json(inner.jobs.clone())
}

async fn get_job_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let inner = state.inner.lock().await;
    match inner.jobs.iter().find(|j| j.id == id) {
        Some(job) => Ok(Json(job.clone())),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Job {} not found", id),
            }),
        )),
    }
}

async fn cancel_job_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let mut inner = state.inner.lock().await;
    match inner.jobs.iter_mut().find(|j| j.id == id) {
        Some(job) => {
            if job.status == JobStatus::Pending {
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(Utc::now());
                // Remove from priority queue
                let old_queue: Vec<PendingJob> = inner.queue.drain().collect();
                for pj in old_queue {
                    if pj.id != id {
                        inner.queue.push(pj);
                    }
                }
                Ok(Json(serde_json::json!({"message": "Job cancelled", "id": id})))
            } else {
                Err((
                    StatusCode::CONFLICT,
                    Json(ErrorResponse {
                        error: format!("Job {} is {:?}, can only cancel pending jobs", id, job.status),
                    }),
                ))
            }
        }
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Job {} not found", id),
            }),
        )),
    }
}

// ---------------------------------------------------------------------------
// Job executor loop
// ---------------------------------------------------------------------------

async fn executor_loop(state: AppState) {
    loop {
        // Wait for notification or check periodically
        tokio::select! {
            _ = state.notify.notified() => {},
            _ = tokio::time::sleep(Duration::from_secs(1)) => {},
        }

        // Try to dequeue and run jobs up to concurrency limit
        loop {
            let job_to_run = {
                let mut inner = state.inner.lock().await;
                if inner.running_count >= inner.concurrency_limit {
                    break;
                }
                // Pop from priority queue, skipping cancelled jobs
                loop {
                    match inner.queue.pop() {
                        Some(pj) => {
                            // Check if still pending (might have been cancelled)
                            let still_pending = inner
                                .jobs
                                .iter()
                                .any(|j| j.id == pj.id && j.status == JobStatus::Pending);
                            if still_pending {
                                break Some(pj.id);
                            }
                            // else skip and continue popping
                        }
                        None => break None,
                    }
                }
            };

            match job_to_run {
                Some(job_id) => {
                    // Mark as running
                    {
                        let mut inner = state.inner.lock().await;
                        if let Some(job) = inner.jobs.iter_mut().find(|j| j.id == job_id) {
                            job.status = JobStatus::Running;
                            job.started_at = Some(Utc::now());
                        }
                        inner.running_count += 1;
                    }

                    // Spawn the execution
                    let state_clone = Arc::clone(&state);
                    tokio::spawn(async move {
                        run_job(state_clone, job_id).await;
                    });
                }
                None => break,
            }
        }
    }
}

/// Read all bytes from an optional child pipe handle, returning a String.
async fn read_pipe(pipe: Option<tokio::process::ChildStdout>) -> String {
    use tokio::io::AsyncReadExt;
    match pipe {
        Some(mut p) => {
            let mut buf = Vec::new();
            let _ = p.read_to_end(&mut buf).await;
            String::from_utf8_lossy(&buf).to_string()
        }
        None => String::new(),
    }
}

/// Read all bytes from an optional child stderr pipe handle, returning a String.
async fn read_pipe_stderr(pipe: Option<tokio::process::ChildStderr>) -> String {
    use tokio::io::AsyncReadExt;
    match pipe {
        Some(mut p) => {
            let mut buf = Vec::new();
            let _ = p.read_to_end(&mut buf).await;
            String::from_utf8_lossy(&buf).to_string()
        }
        None => String::new(),
    }
}

async fn run_job(state: AppState, job_id: Uuid) {
    let (command, timeout_secs) = {
        let inner = state.inner.lock().await;
        match inner.jobs.iter().find(|j| j.id == job_id) {
            Some(j) => (j.command.clone(), j.timeout_secs),
            None => return,
        }
    };

    // BUG 1 fix: spawn child separately so we can kill it on timeout.
    // Take stdout/stderr handles before waiting so we retain the Child for kill().
    let spawn_result = Command::new("sh")
        .arg("-c")
        .arg(&command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    let result = match spawn_result {
        Ok(mut child) => {
            // Take the pipe handles so we can read them after wait/kill
            let child_stdout = child.stdout.take();
            let child_stderr = child.stderr.take();

            match tokio::time::timeout(
                Duration::from_secs(timeout_secs),
                child.wait(),
            )
            .await
            {
                Ok(Ok(status)) => {
                    let exit_code = status.code().unwrap_or(-1);
                    let stdout = read_pipe(child_stdout).await;
                    let stderr = read_pipe_stderr(child_stderr).await;
                    Ok(Ok((exit_code, stdout, stderr)))
                }
                Ok(Err(e)) => Ok(Err(format!("Failed to wait: {}", e))),
                Err(_timeout) => {
                    // Timeout fired — kill the child process
                    if let Err(e) = child.kill().await {
                        tracing::warn!(
                            "Failed to kill timed-out child for job {}: {}",
                            job_id,
                            e
                        );
                    }
                    // Reap the child to avoid zombie
                    let _ = child.wait().await;
                    // Collect any partial output after kill
                    let partial_stderr = read_pipe_stderr(child_stderr).await;
                    let stderr_msg = if partial_stderr.is_empty() {
                        format!("Job timed out after {} seconds", timeout_secs)
                    } else {
                        format!(
                            "Job timed out after {} seconds. Partial stderr:\n{}",
                            timeout_secs, partial_stderr
                        )
                    };
                    Err(stderr_msg)
                }
            }
        }
        Err(e) => Ok(Err(format!("Failed to spawn: {}", e))),
    };

    let mut inner = state.inner.lock().await;
    inner.running_count = inner.running_count.saturating_sub(1);

    if let Some(job) = inner.jobs.iter_mut().find(|j| j.id == job_id) {
        job.completed_at = Some(Utc::now());

        match result {
            Ok(Ok((exit_code, stdout, stderr))) => {
                job.exit_code = Some(exit_code);
                // BUG 3 fix: truncate stdout/stderr to 1 MB
                job.stdout = Some(truncate_output(stdout, MAX_OUTPUT_BYTES));
                job.stderr = Some(truncate_output(stderr, MAX_OUTPUT_BYTES));
                if exit_code == 0 {
                    job.status = JobStatus::Completed;
                } else {
                    job.status = JobStatus::Failed;
                }
            }
            Ok(Err(e)) => {
                job.status = JobStatus::Failed;
                job.stderr = Some(truncate_output(
                    format!("Execution error: {}", e),
                    MAX_OUTPUT_BYTES,
                ));
            }
            Err(timeout_msg) => {
                job.status = JobStatus::TimedOut;
                job.stderr = Some(truncate_output(timeout_msg, MAX_OUTPUT_BYTES));
            }
        }
    }

    // BUG 3 fix: evict oldest completed jobs if we exceed max_history
    evict_old_jobs(&mut inner);

    // Notify executor there might be room for more jobs
    drop(inner);
    state.notify.notify_one();
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "npu-sched", about = "NPU job scheduler for AMD XDNA 2")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the scheduler daemon
    Serve {
        /// Port to listen on
        #[arg(long, default_value = "7890")]
        port: u16,
        /// Max concurrent NPU jobs
        #[arg(long, default_value = "1")]
        concurrency: usize,
        /// Max completed jobs to keep in history (oldest evicted first)
        #[arg(long, default_value = "1000")]
        max_history: usize,
    },
    /// Show NPU status and queue stats
    Status {
        /// Daemon URL
        #[arg(long, default_value = "http://127.0.0.1:7890")]
        url: String,
    },
    /// Submit a job
    Submit {
        /// Job name
        #[arg(long)]
        name: String,
        /// Shell command to execute
        #[arg(long)]
        cmd: String,
        /// Priority (0-9, higher = sooner)
        #[arg(long, default_value = "5")]
        priority: u8,
        /// Timeout in seconds
        #[arg(long, default_value = "300")]
        timeout: u64,
        /// Daemon URL
        #[arg(long, default_value = "http://127.0.0.1:7890")]
        url: String,
    },
    /// List all jobs
    Jobs {
        /// Daemon URL
        #[arg(long, default_value = "http://127.0.0.1:7890")]
        url: String,
    },
    /// Cancel a pending job
    Cancel {
        /// Job ID to cancel
        id: Uuid,
        /// Daemon URL
        #[arg(long, default_value = "http://127.0.0.1:7890")]
        url: String,
    },
}

// ---------------------------------------------------------------------------
// CLI HTTP helpers (minimal, no extra deps — just use TCP + hand-rolled HTTP)
// ---------------------------------------------------------------------------

async fn cli_get(url: &str) -> Result<String, String> {
    let url_parsed: url::Url = url
        .parse()
        .map_err(|e| format!("Bad URL: {}", e))
        // fallback for no url crate
        .or_else(|_| Err("Invalid URL".to_string()))?;

    let host = url_parsed
        .host_str()
        .ok_or("No host")?
        .to_string();
    let port = url_parsed.port().unwrap_or(80);
    let path = if url_parsed.path().is_empty() {
        "/"
    } else {
        url_parsed.path()
    };

    let addr = format!("{}:{}", host, port);
    let mut stream = tokio::net::TcpStream::connect(&addr)
        .await
        .map_err(|e| format!("Connection failed (is the daemon running?): {}", e))?;

    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        path, host
    );

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|e| format!("Write failed: {}", e))?;

    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("Read failed: {}", e))?;

    let response = String::from_utf8_lossy(&buf).to_string();
    // Extract body after \r\n\r\n
    match response.find("\r\n\r\n") {
        Some(pos) => Ok(response[pos + 4..].to_string()),
        None => Ok(response),
    }
}

async fn cli_post(url: &str, body: &str) -> Result<String, String> {
    let url_parsed: url::Url = url
        .parse()
        .map_err(|e| format!("Bad URL: {}", e))
        .or_else(|_| Err("Invalid URL".to_string()))?;

    let host = url_parsed
        .host_str()
        .ok_or("No host")?
        .to_string();
    let port = url_parsed.port().unwrap_or(80);
    let path = if url_parsed.path().is_empty() {
        "/"
    } else {
        url_parsed.path()
    };

    let addr = format!("{}:{}", host, port);
    let mut stream = tokio::net::TcpStream::connect(&addr)
        .await
        .map_err(|e| format!("Connection failed (is the daemon running?): {}", e))?;

    let request = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path, host, body.len(), body
    );

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|e| format!("Write failed: {}", e))?;

    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("Read failed: {}", e))?;

    let response = String::from_utf8_lossy(&buf).to_string();
    match response.find("\r\n\r\n") {
        Some(pos) => Ok(response[pos + 4..].to_string()),
        None => Ok(response),
    }
}

async fn cli_delete(url: &str) -> Result<String, String> {
    let url_parsed: url::Url = url
        .parse()
        .map_err(|e| format!("Bad URL: {}", e))
        .or_else(|_| Err("Invalid URL".to_string()))?;

    let host = url_parsed
        .host_str()
        .ok_or("No host")?
        .to_string();
    let port = url_parsed.port().unwrap_or(80);
    let path = if url_parsed.path().is_empty() {
        "/"
    } else {
        url_parsed.path()
    };

    let addr = format!("{}:{}", host, port);
    let mut stream = tokio::net::TcpStream::connect(&addr)
        .await
        .map_err(|e| format!("Connection failed (is the daemon running?): {}", e))?;

    let request = format!(
        "DELETE {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        path, host
    );

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|e| format!("Write failed: {}", e))?;

    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("Read failed: {}", e))?;

    let response = String::from_utf8_lossy(&buf).to_string();
    match response.find("\r\n\r\n") {
        Some(pos) => Ok(response[pos + 4..].to_string()),
        None => Ok(response),
    }
}

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, concurrency, max_history } => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "info".parse().unwrap()),
                )
                .init();

            let state = Arc::new(Scheduler::new(concurrency, max_history));

            // Print startup NPU status
            let npu = check_npu_status();
            tracing::info!("NPU module loaded: {}", npu.module_loaded);
            tracing::info!("NPU device exists: {}", npu.device_exists);
            tracing::info!("NPU device accessible: {}", npu.device_accessible);
            if let Some(ref fw) = npu.firmware_version {
                tracing::info!("NPU firmware: {}", fw);
            }
            if let Some(cols) = npu.columns {
                tracing::info!("NPU columns: {}", cols);
            }

            // Start executor loop
            let executor_state = Arc::clone(&state);
            tokio::spawn(async move {
                executor_loop(executor_state).await;
            });

            let app = Router::new()
                .route("/health", get(health_handler))
                .route("/status", get(status_handler))
                .route("/jobs", post(submit_handler))
                .route("/jobs", get(list_jobs_handler))
                .route("/jobs/:id", get(get_job_handler))
                .route("/jobs/:id", delete(cancel_job_handler))
                .with_state(state);

            let addr = format!("127.0.0.1:{}", port);
            tracing::info!("npu-sched daemon listening on {}", addr);

            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .expect("Failed to bind");

            axum::serve(listener, app)
                .await
                .expect("Server error");
        }

        Commands::Status { url } => {
            let endpoint = format!("{}/status", url);
            match cli_get(&endpoint).await {
                Ok(body) => {
                    match serde_json::from_str::<StatusResponse>(&body) {
                        Ok(status) => {
                            println!("=== NPU Status ===");
                            println!(
                                "  Module loaded:    {}",
                                if status.npu.module_loaded { "yes" } else { "no" }
                            );
                            println!(
                                "  Device exists:    {}",
                                if status.npu.device_exists { "yes" } else { "no" }
                            );
                            println!(
                                "  Device accessible:{}",
                                if status.npu.device_accessible {
                                    " yes"
                                } else {
                                    " no"
                                }
                            );
                            if let Some(ref fw) = status.npu.firmware_version {
                                println!("  Firmware:         {}", fw);
                            }
                            if let Some(cols) = status.npu.columns {
                                println!("  Columns:          {}", cols);
                            }
                            if let Some(ref dv) = status.npu.driver_version {
                                println!("  Driver version:   {}", dv);
                            }
                            println!();
                            println!("=== Queue Stats ===");
                            println!("  Pending:    {}", status.queue.pending);
                            println!("  Running:    {}", status.queue.running);
                            println!("  Completed:  {}", status.queue.completed);
                            println!("  Failed:     {}", status.queue.failed);
                            println!("  Cancelled:  {}", status.queue.cancelled);
                            println!("  Timed out:  {}", status.queue.timed_out);
                            println!("  Total:      {}", status.queue.total);
                        }
                        Err(_) => {
                            // Just print raw JSON
                            println!("{}", body);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Submit {
            name,
            cmd,
            priority,
            timeout,
            url,
        } => {
            let endpoint = format!("{}/jobs", url);
            let body = serde_json::json!({
                "name": name,
                "cmd": cmd,
                "priority": priority.min(9),
                "timeout_secs": timeout,
            });
            match cli_post(&endpoint, &body.to_string()).await {
                Ok(resp) => {
                    match serde_json::from_str::<SubmitResponse>(&resp) {
                        Ok(sr) => {
                            println!("Job submitted: {}", sr.id);
                        }
                        Err(_) => {
                            println!("{}", resp);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Jobs { url } => {
            let endpoint = format!("{}/jobs", url);
            match cli_get(&endpoint).await {
                Ok(body) => {
                    match serde_json::from_str::<Vec<Job>>(&body) {
                        Ok(jobs) => {
                            if jobs.is_empty() {
                                println!("No jobs.");
                                return;
                            }
                            println!(
                                "{:<38} {:<20} {:<4} {:<12} {:<20}",
                                "ID", "NAME", "PRI", "STATUS", "SUBMITTED"
                            );
                            println!("{}", "-".repeat(96));
                            for job in &jobs {
                                let status_str = format!("{:?}", job.status).to_lowercase();
                                println!(
                                    "{:<38} {:<20} {:<4} {:<12} {:<20}",
                                    job.id,
                                    truncate(&job.name, 20),
                                    job.priority,
                                    status_str,
                                    job.submitted_at.format("%Y-%m-%d %H:%M:%S"),
                                );
                            }
                        }
                        Err(_) => {
                            println!("{}", body);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Cancel { id, url } => {
            let endpoint = format!("{}/jobs/{}", url, id);
            match cli_delete(&endpoint).await {
                Ok(resp) => {
                    println!("{}", resp);
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

// Minimal URL parser to avoid adding a `url` crate dependency
mod url {
    #[allow(dead_code)]
    pub struct Url {
        pub scheme: String,
        pub host: String,
        pub port: Option<u16>,
        pub path: String,
    }

    impl Url {
        pub fn host_str(&self) -> Option<&str> {
            Some(&self.host)
        }

        pub fn port(&self) -> Option<u16> {
            self.port
        }

        pub fn path(&self) -> &str {
            &self.path
        }
    }

    impl std::str::FromStr for Url {
        type Err = String;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            // Parse: scheme://host[:port][/path]
            let rest = if let Some(r) = s.strip_prefix("http://") {
                r
            } else if let Some(r) = s.strip_prefix("https://") {
                r
            } else {
                return Err("Unsupported scheme".to_string());
            };

            let scheme = if s.starts_with("https") {
                "https"
            } else {
                "http"
            }
            .to_string();

            let (authority, path) = match rest.find('/') {
                Some(pos) => (&rest[..pos], &rest[pos..]),
                None => (rest, "/"),
            };

            let (host, port) = match authority.rfind(':') {
                Some(pos) => {
                    let port_str = &authority[pos + 1..];
                    match port_str.parse::<u16>() {
                        Ok(p) => (authority[..pos].to_string(), Some(p)),
                        Err(_) => (authority.to_string(), None),
                    }
                }
                None => (authority.to_string(), None),
            };

            Ok(Url {
                scheme,
                host,
                port,
                path: path.to_string(),
            })
        }
    }
}
