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
        if value.is_empty() {
            None
        } else {
            Some(value)
        }
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
pub fn version_disagreement(
    first: (&str, &str),
    observed: (&str, &str),
) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(payload_line("\n\nPT_TORCH_VERSION 2.12.1\n\n1.5\n"), Some("1.5"));
        assert_eq!(payload_line("1.5\nPT_TORCH_VERSION 2.12.1\n"), Some("1.5"));
    }

    #[test]
    fn matching_arm_versions_are_not_a_disagreement() {
        assert_eq!(
            version_disagreement(("2.12.1+cpu", "PyTorch max_pool1d"), ("2.12.1+cpu", "PyTorch conv3d")),
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
        assert!(block.contains("threads=8"), "thread count must appear: {block}");
        assert!(
            block.contains("is NOT a win"),
            "the incumbent-moved rule must travel with the block: {block}"
        );
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
