//! Regression test for the lost-spawn race in the blocking pool that was
//! introduced by #7757 (and showed up as a hang in omicron's
//! `artifact_store::test::issue_7796` — oxidecomputer/omicron#10277).
//!
//! The bug: `Spawner::spawn_task` read `num_idle_threads` with a Relaxed
//! atomic load before deciding whether to spawn a new blocking worker or
//! notify an idle one. The counter was only decremented by a worker
//! *after* `wait_for_task` returned a task, so a spawner could read a
//! stale positive value for a worker that had already claimed its
//! notification, take the notify-only path, and have the notification
//! delivered to a worker that was no longer idle. If the worker's
//! task was long-running or blocking, the newly-pushed task would be
//! stranded in the queue with no one free to run it.
//!
//! Strategy: each iteration spins up a fresh current-thread runtime
//! (so the blocking pool starts empty and grows workers as spawns
//! arrive — this is where the race fires most reliably), walks through
//! a sequence that mirrors omicron's
//! `ArtifactStore::writer()` / `write_stream()` pattern (transient
//! writers whose senders are dropped immediately, interleaved with
//! persistent writers whose blocking tasks park in `blocking_recv`
//! for the iteration's lifetime), and then drains the persistent
//! writers. If the bug strands any of the persistent writer's
//! blocking tasks, `JoinSet::join_next` never returns. A tight
//! per-iteration timeout turns that hang into a test failure.
//!
//! On the pre-#7757 code the test passes trivially because the
//! spawner made the idle/spawn decision under the single
//! blocking-pool mutex. On the post-#7757 code without this fix the
//! test hangs on iteration 0–11 every run observed (100% over 10
//! attempts). With the fix it completes all iterations in about a
//! second.

#![warn(rust_2018_idioms)]
#![cfg(all(feature = "full", not(target_os = "wasi"), not(miri)))]

use std::time::Duration;

use tokio::runtime::Builder;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

/// Number of iterations to run the scenario. Each iteration spins up
/// a fresh current-thread runtime, exercises a handful of
/// `spawn_blocking` calls, and tears the runtime down. Experimentally
/// the buggy code strands a task in the first iteration on both
/// x86_64 Linux and macOS; the loop exists to make the signal
/// overwhelming rather than marginal. On the fixed code this whole
/// suite runs in a few seconds.
const ITERATIONS: usize = 500;

/// Per-iteration timeout. On the fixed code, iterations consistently
/// finish in well under a millisecond. On the buggy code, a strand is
/// permanent, so any reasonable timeout suffices. Keep this tighter
/// than the pool's 10-second `keep_alive` so a timed-out worker can't
/// accidentally rescue a strand and mask a real failure.
const ITER_TIMEOUT: Duration = Duration::from_secs(1);

/// Blocking tasks per simulated writer. Matches omicron's
/// `TestBackend::new(2)` setup.
const TASKS_PER_WRITER: usize = 2;

#[test]
fn blocking_pool_does_not_strand_tasks() {
    for iter in 0..ITERATIONS {
        // Fresh runtime per iteration, matching `#[tokio::test]`
        // semantics in omicron. On the buggy code, a freshly-created
        // blocking pool growing its first few workers is apparently
        // where the stale-read race fires most reliably.
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let completed = rt.block_on(async {
            tokio::time::timeout(ITER_TIMEOUT, scenario()).await.is_ok()
        });

        assert!(
            completed,
            "iteration {iter}: blocking task stranded in pool queue (hang)",
        );
    }
}

async fn scenario() {
    // The omicron pattern starts each `store.writer()` call with a
    // couple of awaited quick `spawn_blocking` operations (via
    // `tokio::fs::create_dir`). Those awaits return the async
    // worker to the pool briefly, letting blocking-pool workers
    // transition between the idle and running states — which is
    // exactly when the stale `num_idle_threads` read can fire.
    let quick = || async {
        tokio::task::spawn_blocking(|| {}).await.unwrap();
    };

    // Equivalent to omicron's `first_writer`:
    quick().await;
    quick().await;
    let (persistent_senders, persistent_handles) = make_writer();

    // Equivalent to the two `let _ = store.writer(...).await;`
    // calls. Each creates and immediately drops a writer.
    for _ in 0..2 {
        quick().await;
        quick().await;
        let (senders, mut handles) = make_writer();
        drop(senders);
        while let Some(res) = handles.join_next().await {
            res.unwrap();
        }
    }

    // Equivalent to omicron's `fourth_writer`. Creating this while
    // the first persistent writer's blocking tasks are still parked
    // in `blocking_recv` is the spawn that hits the stale-read
    // window in practice.
    quick().await;
    quick().await;
    let (second_senders, second_handles) = make_writer();

    // Drive both persistent writers to completion. If any blocking
    // task was stranded in the queue, `JoinSet::join_next` blocks
    // forever waiting for a task that will never run; the outer
    // `tokio::time::timeout` converts that hang into a test
    // failure.
    drive_writer(persistent_senders, persistent_handles).await;
    drive_writer(second_senders, second_handles).await;
}

type Senders = Vec<mpsc::Sender<()>>;
type Handles = JoinSet<()>;

fn make_writer() -> (Senders, Handles) {
    let mut senders = Vec::with_capacity(TASKS_PER_WRITER);
    let mut handles = JoinSet::new();
    for _ in 0..TASKS_PER_WRITER {
        let (tx, mut rx) = mpsc::channel::<()>(1);
        senders.push(tx);
        handles.spawn_blocking(move || {
            while rx.blocking_recv().is_some() {}
        });
    }
    (senders, handles)
}

async fn drive_writer(senders: Senders, mut handles: Handles) {
    for tx in &senders {
        tx.send(()).await.unwrap();
    }
    drop(senders);
    while let Some(res) = handles.join_next().await {
        res.unwrap();
    }
}
