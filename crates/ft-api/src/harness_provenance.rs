//! Provenance the campaign's evidence contract requires from a head-to-head
//! harness — specifically, the incumbent's **version** (`frankentorch-wnku0`).
//!
//! # Why the version is provenance, not trivia
//!
//! A vs-PyTorch ratio is only as good as the arm beside it, and that arm has a
//! version. The `lane-sweep-reps16` re-bank measured one unchanged ELF against
//! two PyTorch builds on the same host:
//!
//! | lane | vs torch 2.12.1 | vs torch 2.13.0 | our arm |
//! |---|---|---|---|
//! | `max_pool1d` | 2.43x slower | **1.29x slower** | moved <3% |
//! | `avg_pool2d` | 6.87x slower | **4.21x slower** | moved <3% |
//!
//! PyTorch 2.13.0 is ~1.9x and ~1.8x *slower* than 2.12.1 on those two ops.
//! Upgrading the oracle venv would therefore have "improved" two lanes by ~1.9x
//! with **zero FrankenTorch change** — a free win available to anyone who
//! re-runs after a `pip install -U torch` and quotes the delta. Nothing in the
//! previous provenance block could have caught that.
//!
//! So: the version is emitted **on the same block a quoter copies**, and it is
//! self-reported by the Python child *in the same invocation* rather than
//! remembered, configured, or looked up — the same standard the executing-ELF
//! digest is held to.
//!
//! # The rule this module exists to make unavoidable
//!
//! [`INCUMBENT_MOVED_RULE`] states it: **a delta whose incumbent arm moved is
//! not a win.** If the incumbent's version, build, or measured time changed
//! between two runs, the difference between those runs is not attributable to
//! our code, and quoting it as a speedup is proof-class inflation.

/// The line the Python arm must print for its version to be captured.
///
/// Chosen to be unmistakable in a mixed stdout stream and trivially greppable
/// out of an archived run log.
pub const VERSION_MARKER: &str = "PT_TORCH_VERSION ";

/// The standing rule that accompanies a version in any quoted provenance block.
pub const INCUMBENT_MOVED_RULE: &str =
    "a delta whose incumbent arm moved (version, build, or measured time) is NOT a win";

/// Python that every live-torch harness must include so its arm self-reports.
///
/// Emitted before any timing so that a harness which dies mid-measurement still
/// leaves its provenance behind.
pub const VERSION_PROBE_PY: &str = "print('PT_TORCH_VERSION %s' % torch.__version__, flush=True)";

/// Extract the torch version the Python child reported about itself.
///
/// Returns [`None`] when the marker is absent or carries no value — callers
/// must treat that as a hard failure, not a blank field. See
/// [`require_reported_version`].
#[must_use]
pub fn parse_reported_version(stdout: &str) -> Option<&str> {
    stdout.lines().find_map(|line| {
        let value = line.trim().strip_prefix(VERSION_MARKER)?;
        let value = value.trim();
        // A marker with an empty payload is a missing version, not a version.
        if value.is_empty() { None } else { Some(value) }
    })
}

/// Same as [`parse_reported_version`] but fails the run when the version is absent.
///
/// This is the enforcement point: a harness calls it with `?` while assembling
/// its provenance block, so **a run cannot emit ratios without also emitting the
/// version they were measured against**. Silently printing "unknown" would let
/// exactly the free-win substitution above through.
///
/// Returns an error rather than panicking so the failure travels through the
/// harness's own `Result` and this stays usable from library code.
///
/// # Errors
///
/// Errors when the child's stdout carries no usable [`VERSION_MARKER`] line,
/// which means either the probe was omitted from the Python source or the child
/// died before reaching it. Either way the ratios that follow are unquotable.
pub fn require_reported_version(stdout: &str) -> Result<&str, MissingIncumbentVersion> {
    parse_reported_version(stdout).ok_or(MissingIncumbentVersion)
}

/// The arm's actual payload — its first non-empty line that is not the version
/// marker.
///
/// Adding a provenance line to a script that previously printed **only** its
/// result silently breaks any caller doing `stdout.trim().parse()`. This is the
/// paired accessor: provenance comes off the top, the payload is what remains.
#[must_use]
pub fn payload_line(stdout: &str) -> Option<&str> {
    stdout
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with(VERSION_MARKER))
}

/// Check that a newly-observed arm agrees with the first arm's version.
///
/// A harness that launches one process per lane (the gauntlet bench launches
/// nine) has no structural guarantee they were the same interpreter: a
/// `PYTORCH_PYTHON` change mid-run, a stale venv on one path, or a rebuilt
/// environment between lanes all produce a table whose rows are measured against
/// **different incumbents** while each row still looks internally consistent.
///
/// Returns `None` when they agree, or the message to fail with when they do not.
#[must_use]
pub fn version_disagreement(first: (&str, &str), observed: (&str, &str)) -> Option<String> {
    let (first_version, first_label) = first;
    let (version, label) = observed;
    if first_version == version {
        return None;
    }
    Some(format!(
        "PyTorch arms disagree on their version: `{first_label}` reported {first_version} but \
         `{label}` reported {version}. Every row in this table would be measured against a \
         different incumbent, so none of them is quotable."
    ))
}

/// The PyTorch arm ran but never said which PyTorch it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingIncumbentVersion;

impl std::fmt::Display for MissingIncumbentVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the PyTorch arm did not self-report its version: no `{VERSION_MARKER}` line in its \
             stdout. A ratio without its incumbent's version is not quotable \
             ({INCUMBENT_MOVED_RULE}). Add `{VERSION_PROBE_PY}` to the harness's Python source."
        )
    }
}

impl std::error::Error for MissingIncumbentVersion {}

/// SHA-256 of the binary that is executing right now (`frankentorch-fl87u`).
///
/// # Why a harness reports the digest of its own ELF
///
/// Because a build can report success and still leave the OLD binary in place,
/// and nothing else in the pipeline notices. Two rch modes do exactly that:
/// `rch exec -- sh -c "... cargo build ..."` compiles remotely, prints
/// `Finished`, exits 0, and never syncs the artifact back; and a stale per-worker
/// sync manifest can make a build fail or serve old bytes for a file that exists
/// locally. Retrieval can also land the artifact somewhere other than the path
/// you then run — a worker-scoped `CARGO_TARGET_DIR` rewrite leaves
/// `target/release/...` untouched while the build genuinely succeeded elsewhere.
///
/// The failure is silent and it is worst exactly where it matters: a self-A/B
/// that rebuilds one arm and re-runs it measures the SAME binary twice and
/// prints a perfectly plausible "no change" row. The executing-ELF digest is what
/// catches that, which is why it is printed beside every ratio this repo quotes.
///
/// # Panics
///
/// Panics if the current executable path or `sha256sum` is unavailable — a
/// harness that cannot identify its own binary must not go on to print ratios.
#[must_use]
pub fn executing_elf_sha256() -> String {
    let executable = std::env::current_exe().expect("current executable must be available");
    let output = std::process::Command::new("sha256sum")
        .arg(executable)
        .output()
        .expect("sha256sum must be available");
    assert!(output.status.success(), "sha256sum failed");
    String::from_utf8(output.stdout)
        .expect("sha256sum output must be UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum must print a digest")
        .to_owned()
}

/// Catch an A/B whose two arms are the same binary.
///
/// This is the `frankentorch-fl87u` trap in one check. If a rebuild silently
/// failed to reach the path being run, both arms execute identical code, the
/// deltas come out near zero (or wherever the noise lands), and the row reads as
/// a clean result rather than a broken experiment. Comparing the two arms'
/// executing-ELF digests is the cheapest way to know, and it is only useful if
/// something actually compares them — hence a function rather than a convention.
///
/// Returns `None` when the arms genuinely differ, or the message to fail with
/// when they do not.
#[must_use]
pub fn identical_arm_digests(before: &str, after: &str) -> Option<String> {
    if before.trim() != after.trim() || before.trim().is_empty() {
        return None;
    }
    Some(format!(
        "both A/B arms executed the SAME binary (executing_elf_sha256={}). The rebuild did not \
         reach the path being run, so this measured one binary twice and any delta is noise. See \
         frankentorch-fl87u: use a BARE `rch exec -- cargo build ...` with no `sh -c` and no \
         pipes, check where a worker-scoped CARGO_TARGET_DIR retrieved the artifact, and \
         `rch sync --worker <id> --force` if a stale manifest is suspected.",
        before.trim()
    ))
}

/// Render the provenance rows that carry the incumbent's identity.
///
/// Kept as one function so the three live-torch harnesses cannot drift into
/// three different provenance formats.
#[must_use]
pub fn incumbent_provenance_block(version: &str, threads: usize) -> String {
    format!(
        "incumbent=PyTorch {version} (self-reported by the arm, same invocation), threads={threads}\n\
         incumbent_rule={INCUMBENT_MOVED_RULE}"
    )
}

/// Read the first line of a `/proc` or `/sys` file, or `"unknown"`.
fn first_line_of(path: &str) -> String {
    std::fs::read_to_string(path).map_or_else(
        |_| "unknown".to_owned(),
        |text| text.lines().next().unwrap_or("unknown").trim().to_owned(),
    )
}

/// Live announcements in `dir`, excluding `me`, reclaiming every slot whose holder is gone.
///
/// The liveness predicate is a parameter so the dead / recycled / live branches can all be driven
/// by a test — the reclaim path in particular, which is the one that decides whether a killed run
/// strands every future run behind a phantom overlap.
fn scan_measurement_slots(
    dir: &std::path::Path,
    me: u32,
    is_live: &dyn Fn(u32) -> bool,
) -> Vec<String> {
    let mut others = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return others;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(pid) = stem.parse::<u32>() else {
            continue;
        };
        if pid == me {
            continue;
        }
        if is_live(pid) {
            others.push(
                std::fs::read_to_string(&path)
                    .unwrap_or_default()
                    .trim()
                    .to_owned(),
            );
        } else {
            let _ = std::fs::remove_file(&path);
        }
    }
    others
}

/// An announcement that this process is MEASURING, removed when it drops.
///
/// `frankentorch-hi9r6`, item 195. Item 194 recorded two FrankenTorch h2h harnesses sampling
/// simultaneously, inverting every conv2d ratio by up to 4x, with neither agent able to know: this
/// host has a BUILD slot and no MEASUREMENT slot. `concurrent_measurement_block` can see the
/// overlap in `/proc` afterwards; this lets a run see it BEFORE it spends ten minutes, and lets
/// the other agent's run be named rather than merely counted.
///
/// Advisory, never blocking. It does not wait for the slot to free — a measurement that waits is a
/// measurement that hangs, and agents share this host by design. It reports, the row carries the
/// report, and a reader three weeks later can tell an overlapped row from a clean one.
#[derive(Debug)]
pub struct MeasurementSlot {
    path: std::path::PathBuf,
}

impl Drop for MeasurementSlot {
    fn drop(&mut self) {
        // Best effort: a slot left behind by a killed process is reclaimed by the liveness check
        // in `announce_measurement`, so failing here costs a stale line and not correctness.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Announce this measurement and report any other live one.
///
/// Returns the guard — which must be BOUND (`let _slot = ...`, never `let _ = ...`) so the
/// announcement lives as long as the run — and the provenance line to print.
///
/// The slot directory is `FT_H2H_SLOT_DIR` or `/data/tmp/ft-h2h-slots`. Files are ~100 bytes and
/// named by pid. A slot whose pid is gone, or whose pid has been recycled by a process that is not
/// a harness, is deleted on sight: the directory self-heals rather than accumulating tombstones
/// that would make every future run look contended.
#[must_use]
pub fn announce_measurement(label: &str) -> (Option<MeasurementSlot>, String) {
    let dir =
        std::env::var("FT_H2H_SLOT_DIR").unwrap_or_else(|_| "/data/tmp/ft-h2h-slots".to_owned());
    let dir = std::path::PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return (
            None,
            format!(
                "measurement_slot=UNAVAILABLE (cannot create {}); overlap with another harness \
                 cannot be detected in advance, only after the fact by the /proc scan above",
                dir.display()
            ),
        );
    }

    let me = std::process::id();
    let mut others = scan_measurement_slots(&dir, me, &slot_holder_is_live);

    let path = dir.join(format!("{me}.slot"));
    let elf = executing_elf_sha256();
    let record = format!(
        "pid={me} host={} elf={} lanes={label}",
        first_line_of("/proc/sys/kernel/hostname"),
        &elf[..elf.len().min(16)]
    );
    let guard = std::fs::write(&path, &record)
        .ok()
        .map(|()| MeasurementSlot { path });

    let line = if others.is_empty() {
        format!("measurement_slot=held [{record}]; no other live h2h announcement")
    } else {
        others.sort();
        format!(
            "measurement_slot=OVERLAPPED [{record}]; {} other live announcement(s): {} — those \
             runs and this one sample the same cores at the same time. Item 194 measured what that \
             does: every conv2d ratio changed SIGN, and both drift gates passed while it happened. \
             No row from an overlapped window is quotable.",
            others.len(),
            others.join(" | ")
        )
    };
    (guard, line)
}

/// Is `pid` still a live harness? A bare `/proc/<pid>` check would accept a recycled pid belonging
/// to something else entirely, which would strand a slot forever and make every later run report a
/// phantom overlap.
fn slot_holder_is_live(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/cmdline")).is_ok_and(|raw| {
        let cmdline = raw.replace('\0', " ");
        cmdline.contains("_h2h") || cmdline.contains("gauntlet")
    })
}

/// Other processes that were MEASURING when this run started — `frankentorch-hi9r6`, item 193.
///
/// WHY THIS IS PROVENANCE AND NOT A NICETY. Item 193 records a run whose drift gates BOTH passed
/// while every conv2d ratio inverted — `conv2d_masked` read 2.45x SLOWER at loadavg 27 and 1.63x
/// FASTER at loadavg 85, same ELF, same torch build, twenty minutes apart. The drift gate measures
/// STABILITY, and a uniformly overloaded host is perfectly stable; the incumbent's absolute time
/// went 3.1 ms -> 89 ms and the gate had nothing to say about it. What refused those rows was the
/// A/A nulls, which is one layer of defence for a confound that inverts the sign of the answer.
///
/// The specific cause was two h2h harnesses and an unrelated numerical benchmark all sampling at
/// once. That is invisible in loadavg (which cannot say WHAT the load is) and invisible to every
/// gate here, but it is trivially visible in `/proc` — so the run can simply say so, and a reader
/// three weeks later can tell a contended row from a quiet one without having been in the room.
///
/// Deliberately a REPORT and not a refusal: agents share this host by design, and a harness that
/// exits because a peer is busy would be a worse failure than one that says what it saw. It is
/// also deliberately not a wait — blocking until the host is quiet is how a measurement turns into
/// a hang.
///
/// PRECISION, STATED HONESTLY. The classifier matches substrings of `/proc/<pid>/cmdline`, so a
/// SHELL that merely names one of these paths is reported too — verified against a live host,
/// where a `zsh -c` invoking a harness was flagged as a `torch-arm`. That over-reports, and the
/// direction is chosen: a spurious line makes a reader check, while a missed one makes a contended
/// row look clean. What it cannot see at all is a measurement that shares neither a name nor a
/// path with these patterns, so `none` means "none of the shapes we know", never "the host is
/// quiet" — the loadavg and clock lines beside it are what say that.
#[must_use]
pub fn concurrent_measurement_block() -> String {
    let others = concurrent_measurement_processes();
    let (active, idle): (Vec<_>, Vec<_>) = others
        .iter()
        .partition(|(_, _, share)| *share >= CONTENTION_ACTIVE_FLOOR);
    let render = |set: &[&(u32, &'static str, f64)]| {
        let mut listed: Vec<String> = set
            .iter()
            .map(|(pid, kind, share)| format!("{kind}[{pid}] {:.0}%", share * 100.0))
            .collect();
        listed.sort();
        listed.join(" ")
    };
    // Named but not burning CPU: reported, never voiding. These are overwhelmingly shell wrappers
    // and rch clients whose work is on another machine, and treating them as contention voided
    // real runs.
    let idle_note = if idle.is_empty() {
        String::new()
    } else {
        format!(
            "\nconcurrent_measurements_idle={} (matched by name, burning no local CPU — shell \
             wrappers, and rch clients whose bench runs on a remote worker; reported, not \
             voiding): {}",
            idle.len(),
            render(&idle.iter().collect::<Vec<_>>())
        )
    };
    if active.is_empty() {
        return format!(
            "concurrent_measurements=none ACTIVE (scanned /proc for other h2h harnesses, criterion \
             benches and torch processes, then measured each candidate's local CPU over 300ms; \
             this run's own incumbent child is excluded){idle_note}"
        );
    }
    format!(
        "concurrent_measurements={} DETECTED: {} — another process was sampling inside this \
         window AND burning local CPU, so BOTH runs' arms are contended and NEITHER is quotable \
         however its gates read. The drift gate cannot see this: it measures stability, and a \
         uniformly overloaded host is stable.{idle_note}",
        active.len(),
        render(&active.iter().collect::<Vec<_>>())
    )
}

/// `(pid, kind)` for processes that look like they are measuring, excluding this process and its
/// direct children — the incumbent arm is our own child and must not count as contention.
/// `ppid` from `/proc/<pid>/stat`, which is `pid (comm) state ppid ...`. `comm` may contain
/// spaces and parentheses, so the fields are taken from AFTER the last ')' rather than by index.
fn parent_of(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    stat.rsplit(')')
        .next()?
        .split_whitespace()
        .nth(1)
        .and_then(|field| field.parse::<u32>().ok())
}

/// CPU ticks (`utime + stime`) burned by `pid`, or `None` if it is gone.
///
/// `comm` can contain spaces and parentheses, so fields are taken from after the LAST `)`:
/// `utime` is field 14 and `stime` field 15, which land at indices 11 and 12 of that tail.
fn cpu_ticks_of(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let tail = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = tail.split_whitespace().collect();
    let utime = fields.get(11)?.parse::<u64>().ok()?;
    let stime = fields.get(12)?.parse::<u64>().ok()?;
    Some(utime + stime)
}

/// Fraction of one core each candidate is burning, sampled over `SAMPLE`.
///
/// # Why a name match is not enough
///
/// Item 213 removed the detector's self-detection; this removes the other half of the same bug.
/// A run of mine was voided by `concurrent_measurements=3 DETECTED: criterion-bench[90285]
/// criterion-bench[90321] criterion-bench[90322]`, and the three processes were:
///
/// - `rch exec -- cargo bench --profile release-perf -p fnp-python --bench …` — an rch CLIENT,
///   whose bench runs on a REMOTE worker and burns nothing on this host;
/// - two `zsh -c …` wrappers whose command lines merely CONTAIN that text.
///
/// So one logical job, running on another machine, counted three times and voided the run. The
/// same shape voids any run taken while a peer's shell mentions `--bench`, including a peer's
/// `pgrep`, and it is not rare: it is how agents on this box launch everything.
///
/// Naming is not evidence of sampling. **Burning CPU is**, and it is measurable in 300 ms, so the
/// detector measures it rather than inferring it from a string. A process below the floor is still
/// PRINTED — the doc above promises visibility — but it no longer voids the run.
const CONTENTION_SAMPLE: std::time::Duration = std::time::Duration::from_millis(300);

/// Floor for calling a named process an ACTIVE measurement, as a fraction of one core.
///
/// A criterion bench or a peer harness that is actually sampling runs at 100% of a core or far
/// more; a shell wrapper or an rch client sits at nought. Anything in between is reported and left
/// to the reader. 10% over a 300 ms window is three `utime` ticks, which is above the 10 ms tick
/// resolution rather than at it.
const CONTENTION_ACTIVE_FLOOR: f64 = 0.10;

fn concurrent_measurement_processes() -> Vec<(u32, &'static str, f64)> {
    let me = std::process::id();
    // Our ancestor chain, walked once. Bounded because a pid cannot be its own ancestor and pid 1
    // has no parent; the explicit cap is belt-and-braces against a malformed `/proc`.
    let mut ancestors = Vec::new();
    let mut walk = me;
    for _ in 0..64 {
        match parent_of(walk) {
            Some(parent) if parent != 0 && parent != walk => {
                ancestors.push(parent);
                walk = parent;
            }
            _ => break,
        }
    }
    // Candidates matched by NAME. The measured set returned below carries a third field, so this
    // is annotated rather than inferred from the return type.
    let mut found: Vec<(u32, &'static str)> = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        if pid == me {
            continue;
        }
        let dir = entry.path();
        // Skip our own family in BOTH directions — item 213. Children are the incumbent arm we
        // spawned. ANCESTORS are the shell that launched us, and excluding them is not cosmetic:
        // this harness is normally invoked as `PYTORCH_PYTHON=/…/torchvenv-…/bin/python …`, so the
        // invoking shell's command line names the torch venv and matches the `torch-arm` pattern.
        // Every single run therefore self-reported one concurrent measurement, which is the
        // failure mode that makes a detector worthless: an alarm that is always on is not read.
        if ancestors.contains(&pid) {
            continue;
        }
        if parent_of(pid) == Some(me) {
            continue; // our own incumbent arm
        }
        let Ok(raw) = std::fs::read_to_string(dir.join("cmdline")) else {
            continue;
        };
        let cmdline = raw.replace('\0', " ");
        let kind = if cmdline.contains("_h2h") || cmdline.contains("gauntlet") {
            "h2h-harness"
        } else if cmdline.contains("torchvenv") || cmdline.contains("site-packages/torch") {
            "torch-arm"
        } else if cmdline.contains("--bench") || cmdline.contains("/benches/") {
            "criterion-bench"
        } else {
            continue;
        };
        found.push((pid, kind));
    }
    // Second pass: how much local CPU is each of them ACTUALLY burning. Sampled once for the whole
    // candidate set rather than per process, so the wall cost is one 300 ms sleep however many
    // matched.
    let before: Vec<Option<u64>> = found.iter().map(|(pid, _)| cpu_ticks_of(*pid)).collect();
    std::thread::sleep(CONTENTION_SAMPLE);
    let window = CONTENTION_SAMPLE.as_secs_f64();
    found
        .into_iter()
        .zip(before)
        .map(|((pid, kind), start)| {
            // USER_HZ is 100 for /proc regardless of CONFIG_HZ. A process that exited during the
            // sample reads as 0.0, which is the right answer: it is not contending with us now.
            let share = match (start, cpu_ticks_of(pid)) {
                (Some(a), Some(b)) => (b.saturating_sub(a) as f64 / 100.0) / window,
                _ => 0.0,
            };
            (pid, kind, share)
        })
        .collect()
}

/// Render the rows that identify the MACHINE a row was measured on.
///
/// # Why the machine is provenance
///
/// Measured across the rch fleet on 2026-08-15: the same cubic `splu` cell read
/// **1.2693x on one worker and 0.0093x on another** — a 13.6x swing — with BOTH
/// A/A nulls PASSING. A passing null controls within-invocation noise; it says
/// nothing about between-machine differences in CPU model, cache, memory
/// bandwidth or resident contention. Independently, an external-load veto was
/// found not to predict the ratio at all (load varied 4.9x while the ratio
/// spread 6.46%, r = -0.35), so load is not a usable stand-in for machine
/// identity either.
///
/// The consequences are structural, and this block only serves the last one:
///
/// 1. Both arms must be sampled in the SAME invocation on the SAME machine.
/// 2. A row that does not name its machine cannot be compared to any other row.
///
/// The harnesses in this crate satisfy (1) by construction — the incumbent is a
/// co-process interleaved by the balanced square, so there is one machine per
/// invocation by definition. This prints the identity of that machine so a
/// banked row can still be placed afterwards.
#[must_use]
pub fn measurement_host_block(rayon_threads: usize) -> String {
    let host = first_line_of("/proc/sys/kernel/hostname");
    let cpu = std::fs::read_to_string("/proc/cpuinfo").map_or_else(
        |_| "unknown".to_owned(),
        |text| {
            text.lines()
                .find_map(|line| line.strip_prefix("model name"))
                .and_then(|rest| rest.split_once(':'))
                .map_or_else(|| "unknown".to_owned(), |(_, name)| name.trim().to_owned())
        },
    );
    let governor = first_line_of("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    let isa = {
        let mut features = vec![std::env::consts::ARCH];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                features.push("avx512f");
            } else if std::arch::is_x86_feature_detected!("avx2") {
                features.push("avx2");
            }
        }
        features.join("+")
    };
    format!(
        "measurement_host={host} cpu={cpu:?} isa={isa} governor={governor} \
         rayon_threads={rayon_threads} online_cpus={}\n\
         host_rule=both arms are sampled in ONE invocation on THIS machine; a row measured \
         elsewhere is not comparable to this one, A/A PASS or not",
        std::thread::available_parallelism()
            .map_or_else(|_| "unknown".to_owned(), |count| count.get().to_string())
    )
}

/// Instantaneous per-core clocks, sorted ascending, in MHz.
///
/// frankentorch-68pwz: **cores on this machine run at different clocks AT THE SAME
/// INSTANT.** Measured here: 64 cores spanning 1429 MHz to 4018 MHz, a 2.812x spread,
/// with roughly a quarter parked at the floor while the rest sit at 3192-4018. It is
/// bimodal, not a gradient.
///
/// That makes clock a first-class confound for any two-arm ratio. Our arm runs 64 rayon
/// threads and therefore spans every core, so a parallel join waits on whichever core is
/// parked; the incumbent runs 8 threads that may land anywhere. **A ratio whose arms sat
/// at different clocks is partly a frequency ratio.** Reading `scaling_cur_freq` is the
/// only way to notice, because loadavg is blind to it — the spread above was observed at
/// loadavg 28.
pub fn cpu_mhz_sorted() -> Vec<f64> {
    let mut mhz: Vec<f64> = (0..)
        .map_while(|cpu| {
            std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
            ))
            .ok()
        })
        .filter_map(|raw| raw.trim().parse::<f64>().ok().map(|khz| khz / 1000.0))
        .collect();
    if mhz.is_empty() {
        // No cpufreq sysfs (containers, some VMs). /proc/cpuinfo still reports a
        // per-core MHz on x86, and an empty result is reported honestly rather than
        // faked with a constant.
        mhz = std::fs::read_to_string("/proc/cpuinfo")
            .unwrap_or_default()
            .lines()
            .filter(|line| line.starts_with("cpu MHz"))
            .filter_map(|line| line.split(':').nth(1)?.trim().parse::<f64>().ok())
            .collect();
    }
    mhz.sort_by(f64::total_cmp);
    mhz
}

/// `(min, median, max, spread)` of the current per-core clocks, or `None` if the machine
/// does not report them. `spread` is `max / min` — the factor by which two cores can
/// differ AT THE SAME MOMENT, which is the number that decides whether a row's arms were
/// comparable.
#[must_use]
pub fn cpu_mhz_stats() -> Option<(f64, f64, f64, f64)> {
    let mhz = cpu_mhz_sorted();
    if mhz.is_empty() {
        return None;
    }
    let min = mhz[0];
    let max = mhz[mhz.len() - 1];
    let median = mhz[mhz.len() / 2];
    Some((
        min,
        median,
        max,
        if min > 0.0 { max / min } else { f64::NAN },
    ))
}

/// One line of clock provenance for a banked row.
///
/// Says what the clocks were AND how comparability was ensured, because "we measured the
/// clocks" and "the arms ran at the same clock" are different claims and only the second
/// licenses a ratio.
#[must_use]
pub fn cpu_clock_block(pinned: Option<&str>) -> String {
    match cpu_mhz_stats() {
        None => "cpu_mhz=UNAVAILABLE (no cpufreq sysfs and no /proc/cpuinfo MHz); this \
                 host cannot show whether the two arms ran at comparable clocks"
            .to_string(),
        Some((min, median, max, spread)) => {
            let comparability = match pinned {
                Some(set) => format!(
                    "arms pinned to the SAME core set [{set}], so both saw the same clock \
                     domain"
                ),
                None => "arms NOT pinned: our 64 rayon threads span every core including \
                         any parked at the floor, the incumbent's 8 land wherever the \
                         scheduler puts them, so this row's arms are NOT known to have \
                         run at comparable clocks"
                    .to_string(),
            };
            format!(
                "cpu_mhz min={min:.0} median={median:.0} max={max:.0} spread={spread:.3}x; \
                 {comparability}"
            )
        }
    }
}

/// The 1-minute load average, or `None` if it cannot be read.
#[must_use]
pub fn load_average_1m() -> Option<f64> {
    first_line_of("/proc/loadavg")
        .split_whitespace()
        .next()
        .and_then(|field| field.parse::<f64>().ok())
}

/// The widest load drift a run may show and still be quoted.
///
/// Chosen from the runs already banked rather than by taste
/// (`frankentorch-2h8vi`): the certified rows ran at drifts of about 1.07x
/// (`frankentorch-y4nj9`, load 8.63 -> 9.25) and 1.15x, while the pair that
/// certified OPPOSITE directions under different estimators
/// (`NEGATIVE_EVIDENCE` item 18) ran at 1.30x and 1.41x. 1.25 sits between the
/// two populations, so it admits every row this campaign has certified and
/// refuses both members of the contradicting pair.
pub const MAX_LOAD_DRIFT: f64 = 1.25;

/// Whether the host stayed put underneath a measurement.
///
/// This exists because an A/A null cannot see it. A null establishes that an arm
/// was STABLE WITHIN the invocation; a run in which every arm is uniformly 40%
/// slow has immaculate nulls and is still not comparable to anything. Item 18
/// recorded two runs of the same lane, both with all nulls passing, certifying
/// **opposite directions** — the distinguishing variable was the host moving
/// under the measurement, which nothing in the protocol was watching.
///
/// The signal is DRIFT IN EITHER DIRECTION, not level. A steady load of 25 is
/// measurable; 6 -> 30 is not, and 30 -> 6 is equally not, because the arms
/// sampled early and late in the balanced square then saw different machines. A
/// gate keyed on absolute load would have rejected the `y4nj9` certified row
/// (steady 8.63 -> 9.25) for the wrong reason.
#[must_use]
pub fn load_drift_is_quotable(start: Option<f64>, end: Option<f64>) -> bool {
    let (Some(start), Some(end)) = (start, end) else {
        // Unknown drift is not quotable: a missing signal must not read as a
        // passing one.
        return false;
    };
    // A very light host makes the ratio jumpy for reasons that are not the
    // measurement (0.2 -> 0.5 is 2.5x and means nothing), so compare against a
    // floor rather than the raw readings.
    let floor = 1.0_f64;
    let lo = start.max(floor);
    let hi = end.max(floor);
    let drift = if hi >= lo { hi / lo } else { lo / hi };
    drift <= MAX_LOAD_DRIFT
}

/// Whether the host stayed put, judged over a SERIES of load samples rather than
/// an endpoint pair.
///
/// `frankentorch-68pwz`, NEGATIVE_EVIDENCE item 49. [`load_drift_is_quotable`]
/// compares only the first and last reading, and that is a real blind spot: a run
/// measured at `17.78` and `17.72` passes, and those two numbers are equally
/// consistent with a steady host and with one that climbed to 60 and came back.
/// The arms of a balanced square are interleaved THROUGHOUT the run, so a mid-run
/// excursion lands on some samples and not others — precisely the confound the
/// gate exists to catch, and precisely the one an endpoint pair cannot see.
///
/// I hit this myself: item 49's allocator experiment was invalidated by load, and
/// the run I was comparing it against had been certified steady by exactly this
/// two-point check.
///
/// The rule is unchanged — DRIFT IN EITHER DIRECTION, not level — but it is now
/// evaluated as `max(samples) / min(samples)`, which is the endpoint check when
/// the extremes happen to be the endpoints and strictly stricter otherwise. It can
/// only reject runs the old gate accepted; it never accepts one the old gate
/// refused.
///
/// Fewer than two samples cannot show drift, so they are not quotable: a missing
/// signal must not read as a passing one, the same reasoning as the `None` arm of
/// the pairwise gate.
#[must_use]
pub fn load_series_is_quotable(samples: &[f64]) -> bool {
    if samples.len() < 2 {
        return false;
    }
    let floor = 1.0_f64;
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;
    for &s in samples {
        if !s.is_finite() || s < 0.0 {
            // A malformed reading fails closed rather than being skipped, so a
            // partly-unreadable series cannot masquerade as a clean one.
            return false;
        }
        let v = s.max(floor);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    hi / lo <= MAX_LOAD_DRIFT
}

/// The worst drift in a series, for reporting alongside a row.
///
/// Returns `None` for a series too short or malformed to judge, matching
/// [`load_series_is_quotable`] so a row cannot print a reassuring number while the
/// gate refuses it.
#[must_use]
pub fn load_series_drift(samples: &[f64]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let floor = 1.0_f64;
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;
    for &s in samples {
        if !s.is_finite() || s < 0.0 {
            return None;
        }
        let v = s.max(floor);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    Some(hi / lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_drift_admits_the_certified_runs_and_refuses_the_contradicting_pair() {
        // frankentorch-y4nj9's certified row: steady, and it must stay quotable.
        assert!(load_drift_is_quotable(Some(8.63), Some(9.25)));
        // A steady HIGH load is measurable — the signal is drift, not level.
        assert!(load_drift_is_quotable(Some(25.0), Some(26.0)));
        // NEGATIVE_EVIDENCE item 18, the two runs that certified opposite
        // directions with every null passing.
        assert!(!load_drift_is_quotable(Some(21.8), Some(28.3)));
        assert!(!load_drift_is_quotable(Some(21.9), Some(30.8)));
        // Drift DOWNWARD is just as disqualifying: the arms sampled early and
        // late saw different machines either way.
        assert!(!load_drift_is_quotable(Some(30.0), Some(6.0)));
        // A missing reading must not read as a pass.
        assert!(!load_drift_is_quotable(None, Some(9.0)));
        assert!(!load_drift_is_quotable(Some(9.0), None));
        // A nearly idle host must not be failed by ratio jitter at tiny values.
        assert!(load_drift_is_quotable(Some(0.2), Some(0.9)));
    }

    /// The block must name the machine even when every optional source is
    /// missing — a provenance line that silently drops fields is worse than one
    /// that says `unknown`, because a reader cannot tell the two apart.
    #[test]
    fn measurement_host_block_names_every_field() {
        let block = measurement_host_block(8);
        for field in [
            "measurement_host=",
            "cpu=",
            "isa=",
            "governor=",
            "rayon_threads=8",
            "online_cpus=",
            "host_rule=",
        ] {
            assert!(block.contains(field), "missing {field} in:\n{block}");
        }
        assert!(
            !block.contains("cpu=\"\""),
            "cpu must not be blank:\n{block}"
        );
    }

    #[test]
    fn parses_a_reported_version() {
        let stdout = "PT_TORCH_VERSION 2.12.1+cpu\nPT max_pool1d 6.9760 -1.234\n";
        assert_eq!(parse_reported_version(stdout), Some("2.12.1+cpu"));
    }

    #[test]
    fn parses_when_the_marker_is_not_the_first_line() {
        let stdout = "some warning from numpy\nPT_TORCH_VERSION 2.13.0+cpu\n";
        assert_eq!(parse_reported_version(stdout), Some("2.13.0+cpu"));
    }

    #[test]
    fn tolerates_surrounding_whitespace() {
        let stdout = "   PT_TORCH_VERSION   2.12.1+cpu   \n";
        assert_eq!(parse_reported_version(stdout), Some("2.12.1+cpu"));
    }

    /// NEGATIVE CASE: a harness that forgot the probe must not silently produce
    /// a versionless provenance block. A naive `unwrap_or("unknown")` passes the
    /// happy-path tests above and fails this one.
    #[test]
    fn absent_marker_is_none_not_a_blank_version() {
        let stdout = "PT max_pool1d 6.9760 -1.234\nPT conv3d 5.5300 -9.9\n";
        assert_eq!(parse_reported_version(stdout), None);
    }

    /// NEGATIVE CASE: the marker present but empty is a *missing* version. This
    /// is the shape a `print('PT_TORCH_VERSION %s' % '')` bug produces, and it
    /// must not be reported as though a version were captured.
    #[test]
    fn empty_marker_payload_is_none() {
        assert_eq!(parse_reported_version("PT_TORCH_VERSION \n"), None);
        assert_eq!(parse_reported_version("PT_TORCH_VERSION    \n"), None);
    }

    #[test]
    fn require_returns_the_version_when_present() {
        assert_eq!(
            require_reported_version("PT_TORCH_VERSION 2.12.1+cpu\n"),
            Ok("2.12.1+cpu")
        );
    }

    /// NEGATIVE CASE: the enforcement point must actually stop the run. A
    /// harness using `?` on this cannot go on to print ratios.
    #[test]
    fn require_errors_when_the_arm_did_not_report() {
        assert_eq!(
            require_reported_version("PT max_pool1d 6.9760 -1.234\n"),
            Err(MissingIncumbentVersion)
        );
    }

    /// The failure must say what to do about it, or it just looks like a crash.
    #[test]
    fn missing_version_error_names_the_marker_and_the_rule() {
        let message = MissingIncumbentVersion.to_string();
        assert!(message.contains(VERSION_MARKER.trim_end()), "{message}");
        assert!(message.contains("is NOT a win"), "{message}");
        assert!(message.contains("torch.__version__"), "{message}");
    }

    /// The exact stdout shape the gauntlet's Python arms now emit.
    #[test]
    fn payload_line_skips_the_provenance_line() {
        let stdout = "PT_TORCH_VERSION 2.12.1+cpu\n0.008927762014\n";
        assert_eq!(payload_line(stdout), Some("0.008927762014"));
        assert_eq!(
            payload_line(stdout).and_then(|l| l.parse::<f64>().ok()),
            Some(0.008_927_762_014)
        );
    }

    /// NEGATIVE CASE: the bug this accessor exists to prevent. A caller doing
    /// `stdout.trim().parse()` on a script that gained a provenance line gets a
    /// parse failure; `payload_line` must not reproduce that by returning the
    /// marker.
    #[test]
    fn payload_line_never_returns_the_marker_line() {
        assert_eq!(payload_line("PT_TORCH_VERSION 2.13.0+cpu\n"), None);
        assert_eq!(payload_line(""), None);
        assert_eq!(payload_line("\n  \n"), None);
    }

    #[test]
    fn payload_line_tolerates_blank_lines_and_ordering() {
        assert_eq!(
            payload_line("\n\nPT_TORCH_VERSION 2.12.1\n\n1.5\n"),
            Some("1.5")
        );
        assert_eq!(payload_line("1.5\nPT_TORCH_VERSION 2.12.1\n"), Some("1.5"));
    }

    #[test]
    fn matching_arm_versions_are_not_a_disagreement() {
        assert_eq!(
            version_disagreement(
                ("2.12.1+cpu", "PyTorch max_pool1d"),
                ("2.12.1+cpu", "PyTorch conv3d")
            ),
            None
        );
    }

    /// NEGATIVE CASE: the multi-process defect. Two lanes on different
    /// interpreters must be caught, and the message must name BOTH lanes or it
    /// is not actionable — you cannot fix what you cannot locate.
    #[test]
    fn differing_arm_versions_are_caught_and_both_lanes_named() {
        let message = version_disagreement(
            ("2.12.1+cpu", "PyTorch max_pool1d"),
            ("2.13.0+cpu", "PyTorch conv3d"),
        )
        .expect("differing versions must be a disagreement");
        assert!(message.contains("2.12.1+cpu"), "{message}");
        assert!(message.contains("2.13.0+cpu"), "{message}");
        assert!(message.contains("max_pool1d"), "{message}");
        assert!(message.contains("conv3d"), "{message}");
        assert!(message.contains("different incumbent"), "{message}");
        assert!(message.contains("quotable"), "{message}");
    }

    /// The exact pair this session measured. 2.12.1 vs 2.13.0 moved two lanes'
    /// ratios by ~1.9x with no code change, so this is the substitution the
    /// check exists to stop.
    #[test]
    fn the_observed_version_pair_is_a_disagreement() {
        assert!(
            version_disagreement(("2.12.1+cpu", "a"), ("2.13.0+cpu", "b")).is_some(),
            "the two oracles measured this session must not be silently mixed"
        );
    }

    #[test]
    fn provenance_block_names_the_version_and_the_rule() {
        let block = incumbent_provenance_block("2.12.1+cpu", 8);
        assert!(block.contains("2.12.1+cpu"), "version must appear: {block}");
        assert!(
            block.contains("threads=8"),
            "thread count must appear: {block}"
        );
        assert!(
            block.contains("is NOT a win"),
            "the incumbent-moved rule must travel with the block: {block}"
        );
    }

    /// The digest must be a real SHA-256 and stable within a process, or it
    /// cannot identify anything.
    #[test]
    fn executing_elf_digest_is_a_stable_sha256() {
        let digest = executing_elf_sha256();
        assert_eq!(digest.len(), 64, "not a SHA-256: {digest}");
        assert!(
            digest.chars().all(|c| c.is_ascii_hexdigit()),
            "not hex: {digest}"
        );
        assert_eq!(digest, executing_elf_sha256(), "digest must be stable");
    }

    /// **THE `frankentorch-fl87u` TRAP.** Two arms with the same digest means the
    /// rebuild never reached the binary being run, so the experiment measured one
    /// binary twice. That must be caught, and the message must say what to do.
    #[test]
    fn identical_arm_digests_are_caught_and_the_message_is_actionable() {
        let digest = "7286dcfc85bc6c77caff8b434be4429f05a4261e75fd011f1b0dc70d54fb982c";
        let message = identical_arm_digests(digest, digest)
            .expect("an A/B that ran one binary twice must be caught");
        assert!(message.contains(digest), "must name the digest: {message}");
        assert!(message.contains("SAME binary"), "{message}");
        assert!(
            message.contains("fl87u"),
            "must point at the bead: {message}"
        );
        assert!(
            message.contains("sh -c"),
            "must name the known cause: {message}"
        );
    }

    /// Whitespace differences are formatting, not a different binary.
    #[test]
    fn digests_differing_only_in_whitespace_are_still_identical() {
        assert!(identical_arm_digests(" abc123 ", "abc123\n").is_some());
    }

    /// NEGATIVE CASE: a genuine A/B must not be flagged, or the guard is noise
    /// and gets ignored.
    #[test]
    fn genuinely_different_arms_are_not_flagged() {
        assert_eq!(
            identical_arm_digests(
                "7286dcfc85bc6c77caff8b434be4429f05a4261e75fd011f1b0dc70d54fb982c",
                "c96e881e99e0f5b9b560898596a02876573d8868a1f4b29553921ea4496afc33",
            ),
            None
        );
    }

    /// NEGATIVE CASE: two MISSING digests are not evidence that one binary ran
    /// twice — they are evidence of nothing, and must not produce a confident
    /// accusation.
    #[test]
    fn two_empty_digests_are_not_an_identical_arm_claim() {
        assert_eq!(identical_arm_digests("", ""), None);
        assert_eq!(identical_arm_digests("  ", "\n"), None);
    }

    /// The Python probe and the Rust parser must agree on the marker, or the
    /// harness reports nothing and every run hard-fails.
    #[test]
    fn python_probe_emits_the_marker_the_parser_looks_for() {
        assert!(
            VERSION_PROBE_PY.contains(VERSION_MARKER.trim_end()),
            "probe `{VERSION_PROBE_PY}` must print marker `{VERSION_MARKER}`"
        );
    }

    /// THE BLIND SPOT ITEM 49 FOUND, encoded so it cannot come back.
    ///
    /// The real run that motivated this read `17.78` at start and `17.72` at end
    /// and was certified steady. Those same endpoints are equally consistent with
    /// a host that climbed to 60 in the middle — and the balanced square samples
    /// throughout, so a mid-run excursion hits some samples and not others. The
    /// pairwise gate accepts both; the series gate must separate them.
    #[test]
    fn series_gate_catches_the_mid_run_excursion_the_pairwise_gate_cannot_see() {
        let endpoints_only = (Some(17.78_f64), Some(17.72_f64));
        assert!(
            load_drift_is_quotable(endpoints_only.0, endpoints_only.1),
            "the pairwise gate accepted this real run, which is the premise"
        );

        // Same endpoints, steady throughout: both gates agree it is quotable.
        let steady = [17.78, 17.80, 17.69, 17.75, 17.72];
        assert!(load_series_is_quotable(&steady));

        // Same endpoints, 3.4x excursion in the middle: the pairwise gate still
        // says yes and is WRONG; the series gate refuses.
        let excursion = [17.78, 42.0, 60.5, 25.1, 17.72];
        assert!(
            !load_series_is_quotable(&excursion),
            "a 60.5/17.72 = 3.4x mid-run excursion must not be quotable"
        );
        let drift = load_series_drift(&excursion).expect("series is judgeable");
        assert!(
            (drift - 60.5 / 17.72).abs() < 1e-9,
            "reported drift {drift} should be max/min over the whole series"
        );
    }

    /// The series gate must never ACCEPT what the pairwise gate rejected — it is
    /// strictly stricter, not merely different. Checked on the two real voided
    /// runs from items 44 and 49.
    #[test]
    fn series_gate_is_strictly_stricter_than_the_pairwise_gate() {
        // Item 44: 11.98 -> 33.52, VOID under both.
        assert!(!load_drift_is_quotable(Some(11.98), Some(33.52)));
        assert!(!load_series_is_quotable(&[11.98, 20.0, 33.52]));

        // Item 49: 50.69 -> 93.30, VOID under both.
        assert!(!load_drift_is_quotable(Some(50.69), Some(93.30)));
        assert!(!load_series_is_quotable(&[50.69, 70.0, 93.30]));

        // When the extremes ARE the endpoints, the two gates coincide exactly.
        for &(a, b) in &[(8.63_f64, 9.25_f64), (23.05, 29.09), (7.10, 12.25)] {
            assert_eq!(
                load_drift_is_quotable(Some(a), Some(b)),
                load_series_is_quotable(&[a, b]),
                "gates must agree on the two-sample case ({a} -> {b})"
            );
        }
    }

    /// A missing or malformed signal must fail closed, matching the `None` arm of
    /// the pairwise gate. A partly-unreadable series must not read as clean.
    #[test]
    fn series_gate_fails_closed_on_short_or_malformed_input() {
        assert!(!load_series_is_quotable(&[]));
        assert!(!load_series_is_quotable(&[12.0]));
        assert!(!load_series_is_quotable(&[12.0, f64::NAN, 12.1]));
        assert!(!load_series_is_quotable(&[12.0, -1.0, 12.1]));
        assert!(!load_series_is_quotable(&[12.0, f64::INFINITY]));

        assert_eq!(load_series_drift(&[]), None);
        assert_eq!(load_series_drift(&[12.0]), None);
        assert_eq!(load_series_drift(&[12.0, f64::NAN]), None);
    }

    /// The 1.0 floor that keeps an idle host from reading as drift, carried over
    /// from the pairwise gate: 0.2 -> 0.5 is 2.5x and means nothing.
    #[test]
    fn series_gate_floors_an_idle_host_the_same_way() {
        assert!(load_series_is_quotable(&[0.2, 0.5, 0.31]));
        assert!(load_series_is_quotable(&[0.0, 0.9, 0.4]));
        // Above the floor the ratio bites again.
        assert!(!load_series_is_quotable(&[1.0, 1.4, 1.1]));
    }

    /// The slot's branches: self-exclusion, overlap reporting, and — the one that matters —
    /// reclaiming a slot whose holder is gone. A stale slot that is never reclaimed would make
    /// every future run report a phantom overlap and quietly disqualify itself forever.
    #[test]
    fn measurement_slots_report_live_peers_and_reclaim_dead_ones() {
        let dir = std::env::temp_dir().join(format!(
            "ft-h2h-slot-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("slot test dir");

        let write = |pid: u32, body: &str| {
            std::fs::write(dir.join(format!("{pid}.slot")), body).expect("slot write");
        };
        write(100, "pid=100 host=h elf=abc lanes=conv2d");
        write(200, "pid=200 host=h elf=abc lanes=conv2d_f32");
        write(300, "pid=300 host=h elf=abc lanes=all");

        // 100 alive, 200 and 300 gone.
        let live = |pid: u32| pid == 100;
        let others = super::scan_measurement_slots(&dir, 999, &live);
        assert_eq!(
            others.len(),
            1,
            "only the live peer is reported: {others:?}"
        );
        assert!(others[0].contains("pid=100"));
        assert!(
            !dir.join("200.slot").exists() && !dir.join("300.slot").exists(),
            "slots whose holder is gone must be reclaimed, not left to strand future runs"
        );
        assert!(
            dir.join("100.slot").exists(),
            "a live peer's slot must survive"
        );

        // A run must never report itself, even though its own slot is present.
        write(999, "pid=999 host=h elf=abc lanes=mine");
        let others = super::scan_measurement_slots(&dir, 999, &|pid| pid == 100 || pid == 999);
        assert!(
            others.iter().all(|o| !o.contains("pid=999")),
            "a run reported itself as contention: {others:?}"
        );

        // Nothing live: a clean host, with no tombstones left behind.
        let others = super::scan_measurement_slots(&dir, 999, &|_| false);
        assert!(others.is_empty(), "expected a clean host, got {others:?}");
        assert!(!dir.join("100.slot").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real liveness predicate must reject a pid that is alive but is NOT a harness, or a
    /// recycled pid would hold a slot forever. Pid 1 is always alive and never a harness.
    #[test]
    fn slot_liveness_rejects_a_live_non_harness_pid() {
        assert!(
            !super::slot_holder_is_live(1),
            "pid 1 is alive but is not an h2h harness; treating it as a holder would strand a slot"
        );
        assert!(
            !super::slot_holder_is_live(u32::MAX),
            "a pid that does not exist cannot hold a slot"
        );
    }

    /// The CPU probe underpins two separate claims — the contention detector's ACTIVE/idle split
    /// and the harness's DIRECT self-load figure — and both are silently wrong if the field
    /// indices are off, because a neighbouring field in `/proc/<pid>/stat` also parses as a
    /// number. Burning a little CPU and requiring the counter to MOVE catches an off-by-one that
    /// a "returns Some" assertion would not: `cutime` sits next to `stime` and stays at zero in a
    /// process with no children.
    #[test]
    fn cpu_ticks_tracks_work_this_process_actually_does() {
        let me = std::process::id();
        let before = super::cpu_ticks_of(me).expect("our own /proc/self/stat must parse");
        let mut spin = 0u64;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(400);
        while std::time::Instant::now() < deadline {
            spin = spin.wrapping_add(1);
        }
        assert!(spin > 0, "the spin loop must not be optimised away");
        let after = super::cpu_ticks_of(me).expect("our own /proc/self/stat must parse");
        assert!(
            after > before,
            "400ms of spinning moved the counter from {before} to {after}; the field index is \
             probably wrong (utime is field 14, at index 11 after the last parenthesis)"
        );
    }

    /// A pid that does not exist is not contention. The detector samples candidates twice and a
    /// process may exit in between, so this path is taken on a live host, not only in tests.
    #[test]
    fn cpu_ticks_of_a_dead_pid_is_none() {
        assert!(
            super::cpu_ticks_of(u32::MAX).is_none(),
            "a pid that cannot exist must not report CPU time"
        );
    }

    /// The block must always say which of the two questions it answered. Before the CPU probe it
    /// reported a NAME match as a detection, which voided runs on shell wrappers and on rch
    /// clients whose bench runs on another machine.
    #[test]
    fn contention_block_reports_activity_not_just_names() {
        let block = super::concurrent_measurement_block();
        assert!(
            block.contains("ACTIVE") || block.contains("DETECTED"),
            "the block must state whether a candidate was burning CPU, got: {block}"
        );
        assert!(
            block.starts_with("concurrent_measurements="),
            "the field name is parsed out of banked logs, got: {block}"
        );
    }
}
