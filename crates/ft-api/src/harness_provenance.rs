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
