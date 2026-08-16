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
}
