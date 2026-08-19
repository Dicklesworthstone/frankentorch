//! Global rayon pool WIDTH policy — `frankentorch-rayon-pool-width-qq8as`.
//!
//! # The gap this fills
//!
//! Every width result this campaign has banked was obtained by setting `RAYON_NUM_THREADS` on the
//! command line. A program that links FrankenTorch as a library and sets nothing gets rayon's
//! default, which is one worker per logical core — 64 on the measurement host, and the worst and
//! least predictable width measured:
//!
//! ```text
//! lane                    4t       8t      16t      32t      64t      (ms, arm-internal medians,
//! prelu_noshortcut    26.312   15.581   10.732   20.371   17.165       NEGATIVE_EVIDENCE item 240)
//! avg_pool2d           3.836    2.280    1.750    3.092    2.479
//! max_pool3d           1.855    1.456    1.427    2.497    2.545
//! conv3d               6.853    4.484    3.566    9.983    9.872
//! max_pool1d_nopool   16.200   10.220    7.403    8.720    8.420
//! ```
//!
//! 16 beats 64 on all five and beats 8 on all five. It also beats them in VARIANCE, which matters
//! more than the medians: repeated passes at 4, 8 and 16 agree within a few percent, while 32
//! disagrees with itself by up to 3.5x and 64 is unstable too. A wide join waits on whichever core
//! is parked at the frequency floor — this box shows a 2.87x cross-core spread at a single instant
//! — so a wide pool's cost is set by whatever else is running, while a narrow one fits inside the
//! fast set.
//!
//! Against the incumbent it is worth a standing, not just a self-speedup: `max_pool1d_nopool` is
//! PARITY with PyTorch at 8 threads and **at least 1.10x FASTER at 16**, on two runs clearing every
//! gate with the width as the only variable (item 244).
//!
//! # Why this is OFF by default
//!
//! Because the evidence is one lane against the incumbent, five lanes arm-internal, and ONE host
//! whose frequency spread is the mechanism's own precondition. `qq8as` refuses to flip a default on
//! less than "the lanes that gain nothing are shown to lose nothing", and two levers in this
//! campaign were shipped or nearly shipped on thinner evidence than this and had to be reverted.
//!
//! So this module ships the MECHANISM and leaves the POLICY to the caller. Doing nothing is the
//! default, and it is the exact behaviour of every build before this file existed.

/// What [`configure_global_pool`] did, or why it did nothing.
///
/// Returned rather than logged: a process that wanted a width and did not get one should be able to
/// find out, and a library must not print to a host application's stderr to tell it so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolOutcome {
    /// No width was requested. Rayon keeps its own default and nothing was touched.
    Unchanged,
    /// `RAYON_NUM_THREADS` is set, so rayon will honour it and we deliberately stay out of the way.
    DeferredToRayonEnv,
    /// The global pool was built with this many workers.
    Configured(usize),
    /// A width was requested but a global pool already exists, so it could not be applied.
    ///
    /// Not an error. `build_global` may be called only once per process, and losing that race to
    /// another initialiser is a legitimate state — the caller learns the request had no effect and
    /// decides whether that matters.
    AlreadyInitialized(usize),
    /// The request could not be parsed. Carries the offending value.
    Invalid(String),
}

/// Physical cores, from Linux CPU topology; `None` when it cannot be determined.
///
/// Counts distinct `(physical id, core id)` pairs in `/proc/cpuinfo`, which is how an SMT machine
/// reports two logical CPUs sharing one core. Returns `None` rather than guessing on any platform
/// or format that does not provide both fields — a wrong physical count silently selects a wrong
/// width, and [`auto_width`] falls back to the logical count where that is at least explicable.
#[must_use]
pub fn physical_cores() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut pairs = std::collections::BTreeSet::new();
    let (mut package, mut core) = (None, None);
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            // A blank line ends one CPU's block. Record whatever that block declared.
            if line.trim().is_empty()
                && let (Some(p), Some(c)) = (package.take(), core.take())
            {
                pairs.insert((p, c));
            }
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "physical id" => package = value.parse::<u32>().ok(),
            "core id" => core = value.parse::<u32>().ok(),
            _ => {}
        }
    }
    if let (Some(p), Some(c)) = (package, core) {
        pairs.insert((p, c));
    }
    (!pairs.is_empty()).then_some(pairs.len())
}

/// Width to use for `FT_RAYON_WIDTH=auto`: `min(16, physical cores)`.
///
/// # This is fitted to exactly two machines, and it says so
///
/// The first version of this function was `min(16, LOGICAL cores)`, because 16 was the measured
/// optimum on the development host (item 240). Running the same curve on a second machine refuted
/// it outright:
///
/// ```text
/// host             CPU                 physical/logical   measured best   min(16,logical) picks
/// thinkstation1    Threadripper 5975WX      32 / 64             16t              16  correct
/// fixmydocuments   Ryzen 7 5800X             8 / 16              8t              16  WORST of 4/8/16
/// ```
///
/// On the second box 16 workers was worse than 8 on five lanes out of five (0.68x-0.93x), so the
/// shipped constant would have selected the worst of the three widths tested there. `min(16,
/// PHYSICAL)` is the simplest rule consistent with both: 32 physical caps to 16, and 8 physical
/// caps to 8, each landing on that machine's measured optimum.
///
/// **Two points do not make a law.** This is a heuristic fitted to two machines that happen to
/// disagree in the informative direction, and it is why `FT_RAYON_WIDTH` stays opt-in: a caller who
/// has measured their own hardware should pass a number, and `auto` is for one who has not.
#[must_use]
pub fn auto_width(logical_cores: usize) -> usize {
    const MEASURED_CAP: usize = 16;
    let cores = physical_cores().unwrap_or(logical_cores).max(1);
    MEASURED_CAP.min(cores)
}

/// Resolve a requested width from the two environment inputs, without touching any global state.
///
/// Split out from [`configure_global_pool`] so the POLICY is testable without mutating a
/// process-wide pool — a test that calls `build_global` can only ever run once per test binary, and
/// a policy that can only be tested once is a policy that will not be tested.
///
/// `RAYON_NUM_THREADS` wins outright. An operator who set it has already chosen, every banked row
/// in this campaign was taken that way, and silently overriding it would make those rows
/// unreproducible.
///
/// An unparseable `FT_RAYON_WIDTH` is an ERROR rather than a fallback to the default. This campaign
/// has repeatedly been misled by knobs that accepted a typo and carried on: a run that believes it
/// is at width 16 and is actually at 64 produces a number that looks fine and means nothing.
#[must_use]
pub fn resolve_width(ft_width: Option<&str>, rayon_env: Option<&str>, cores: usize) -> PoolOutcome {
    if rayon_env.is_some_and(|value| !value.trim().is_empty()) {
        return PoolOutcome::DeferredToRayonEnv;
    }
    let Some(raw) = ft_width else {
        return PoolOutcome::Unchanged;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return PoolOutcome::Unchanged;
    }
    if trimmed.eq_ignore_ascii_case("auto") {
        return PoolOutcome::Configured(auto_width(cores));
    }
    match trimmed.parse::<usize>() {
        Ok(width) if width >= 1 => PoolOutcome::Configured(width),
        _ => PoolOutcome::Invalid(raw.to_owned()),
    }
}

/// Apply the width policy to rayon's global pool.
///
/// Call once, as early as possible: rayon builds its global pool lazily on first use, and after
/// that this can only report [`PoolOutcome::AlreadyInitialized`]. It is safe to call more than once
/// and safe to call after rayon has started — it never panics and never poisons anything.
///
/// Reads `FT_RAYON_WIDTH` (`auto`, or a positive integer) and `RAYON_NUM_THREADS`.
pub fn configure_global_pool() -> PoolOutcome {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let ft_width = std::env::var("FT_RAYON_WIDTH").ok();
    let rayon_env = std::env::var("RAYON_NUM_THREADS").ok();
    let decision = resolve_width(ft_width.as_deref(), rayon_env.as_deref(), cores);
    let PoolOutcome::Configured(width) = decision else {
        return decision;
    };
    match rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .build_global()
    {
        Ok(()) => PoolOutcome::Configured(width),
        // The only documented failure is a global pool that already exists. Reporting which width
        // we WANTED is what makes the outcome actionable.
        Err(_) => PoolOutcome::AlreadyInitialized(width),
    }
}

#[cfg(test)]
mod tests {
    use super::{PoolOutcome, auto_width, configure_global_pool, resolve_width};

    /// `RAYON_NUM_THREADS` is the width every banked row in this campaign was taken under.
    /// Overriding it would silently invalidate the provenance of those rows.
    #[test]
    fn rayon_env_wins_over_our_own_knob() {
        assert_eq!(
            resolve_width(Some("16"), Some("8"), 64),
            PoolOutcome::DeferredToRayonEnv
        );
        assert_eq!(
            resolve_width(Some("auto"), Some("1"), 64),
            PoolOutcome::DeferredToRayonEnv
        );
    }

    /// An empty variable is not a request. Shells export empty strings readily and treating one as
    /// "width zero" or as an error would fire on hosts that never asked for anything.
    #[test]
    fn empty_values_are_not_requests() {
        assert_eq!(resolve_width(None, None, 64), PoolOutcome::Unchanged);
        assert_eq!(resolve_width(Some(""), None, 64), PoolOutcome::Unchanged);
        assert_eq!(resolve_width(Some("  "), None, 64), PoolOutcome::Unchanged);
        assert_eq!(
            resolve_width(Some("8"), Some(""), 64),
            PoolOutcome::Configured(8)
        );
    }

    /// `auto` must never exceed the machine, and must never exceed the measured cap of 16.
    ///
    /// The exact value depends on the host's PHYSICAL core count, which the test cannot fake, so
    /// the assertions are the invariants rather than a number: bounded above by 16, bounded above
    /// by the physical count when that is readable, and at least 1 whatever the input.
    #[test]
    fn auto_is_bounded_by_the_cap_and_by_the_machine() {
        for logical in [0usize, 1, 4, 16, 64, 512] {
            let width = auto_width(logical);
            assert!(width >= 1, "a machine must have at least one worker");
            assert!(
                width <= 16,
                "the measured cap is 16, got {width} for {logical}"
            );
            if let Some(physical) = super::physical_cores() {
                assert!(
                    width <= physical.max(1),
                    "auto must not exceed the physical cores ({physical}), got {width}"
                );
            }
        }
        assert_eq!(
            resolve_width(Some("AUTO"), None, 64),
            PoolOutcome::Configured(auto_width(64)),
            "the keyword is case-insensitive; an operator typing AUTO meant auto"
        );
    }

    /// The physical count must be plausible where it is readable at all: at least one, and never
    /// more than the logical count, since SMT only ever adds logical CPUs.
    #[test]
    fn physical_cores_is_plausible_or_absent() {
        let Some(physical) = super::physical_cores() else {
            return; // not a Linux /proc/cpuinfo with topology fields; nothing to check
        };
        let logical = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        assert!(physical >= 1);
        assert!(
            physical <= logical,
            "physical ({physical}) cannot exceed logical ({logical})"
        );
    }

    /// A typo must not be read as "carry on at the default". A run that believes it is at width 16
    /// and is actually at 64 produces a number that looks fine and means nothing — which is the
    /// failure this campaign keeps re-learning.
    #[test]
    fn a_typo_is_an_error_not_a_silent_default() {
        assert_eq!(
            resolve_width(Some("sixteen"), None, 64),
            PoolOutcome::Invalid("sixteen".to_owned())
        );
        assert_eq!(
            resolve_width(Some("0"), None, 64),
            PoolOutcome::Invalid("0".to_owned()),
            "zero workers is not a pool"
        );
        assert_eq!(
            resolve_width(Some("-8"), None, 64),
            PoolOutcome::Invalid("-8".to_owned())
        );
        assert_eq!(
            resolve_width(Some("16.0"), None, 64),
            PoolOutcome::Invalid("16.0".to_owned())
        );
    }

    /// Explicit widths pass through untouched, including ones above the core count: an operator
    /// asking for oversubscription is allowed to have it, and `auto` is the opinionated path.
    #[test]
    fn explicit_widths_pass_through() {
        assert_eq!(
            resolve_width(Some("1"), None, 64),
            PoolOutcome::Configured(1)
        );
        assert_eq!(
            resolve_width(Some(" 32 "), None, 64),
            PoolOutcome::Configured(32)
        );
        assert_eq!(
            resolve_width(Some("128"), None, 8),
            PoolOutcome::Configured(128)
        );
    }

    /// The entry point must be safe to call repeatedly and after rayon is already running. This
    /// test binary uses rayon elsewhere, so in practice it exercises the losing-the-race path.
    #[test]
    fn configure_is_idempotent_and_never_panics() {
        let first = configure_global_pool();
        let second = configure_global_pool();
        assert_eq!(
            std::mem::discriminant(&first),
            std::mem::discriminant(&second),
            "two calls with the same environment must agree on what happened"
        );
        assert!(
            !matches!(first, PoolOutcome::Invalid(_)),
            "the test environment sets no width, so nothing should be invalid: {first:?}"
        );
    }
}
