//! Which incumbent level did this session draw? — `frankentorch-mdsmm`.
//!
//! WHY THIS EXISTS. On 2026-09-01, twenty-two guard-gated invocations of one ELF on one host
//! read the `conv2d_big` incumbent arm at 5.06-6.38 ms in twenty of them and 10.50-10.86 ms in
//! two, with our own arm invariant at 4.04-4.47 throughout. Three `conv2d_big_masked` rows came
//! back with ALL FOUR GATES PASS — PT A/A, FT A/A, parity `match`, drift PASS — reading 0.790,
//! 1.308 and 1.275. **The standing changed side between invocations minutes apart, and nothing
//! the harness printed could tell them apart.**
//!
//! An A/A null compares two positions INSIDE one run, so an incumbent scaled uniformly for the
//! whole run cancels out of it exactly and gives a perfect 1.000 null. Drift cannot see it
//! either: the host is stable, just stable at the wrong level. No within-run statistic can, and
//! that is the point — the comparison this needs is BETWEEN invocations, so it needs a bank.
//!
//! THE PART THAT DICTATES THE DESIGN. It is tempting to treat the banked median as the true
//! value and refuse anything far from it. That is exactly what the evidence forbids.
//! `NEGATIVE_EVIDENCE` item 219 (2026-08-19) recorded SEVEN invocations of this lane pair at
//! 11.4-12.0 ms and ONE at 6.075, and correctly, on that evidence, called the ~6 ms invocation
//! the defect. Today the prevalence is INVERTED, twenty to two the other way:
//!
//!     session        ~6 ms level    ~11 ms level
//!     2026-08-19           1              7
//!     2026-09-01          20              2
//!
//! Both readings were honest reads of their own session. So a disagreement with history is a
//! NO-VERDICT, not a correction, and this module never calls a level correct. It answers one
//! question — *which level did this row draw, and how many rows drew each* — and states the
//! consequence, which is about COMPARABILITY: a row is comparable to rows that drew its own
//! level and to no others.
//!
//! THE KEY IS THE INCUMBENT'S OWN CHECKSUM, not the harness commit. The bank has to survive our
//! own builds — the incumbent is PyTorch and does not care what we ship — but it must NOT
//! survive a change to the lane's shape, which would silently pool two different measurements.
//! The incumbent arm already returns a gradient checksum per sample, and that is precisely a
//! shape-and-semantics fingerprint: change `C2B_N`, or the loss, and it changes. It is recorded
//! to seven significant digits so a last-ulp wobble does not fork the key.

use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Default split threshold: the largest consecutive gap in the sorted population has to exceed
/// this ratio before the population is called two levels rather than one.
///
/// CALIBRATED ON THE ONE LANE WHERE BIMODALITY IS DOCUMENTED, and the margins are wide on both
/// sides. Today's `conv2d_big` corpus: the low level spans 5.06-6.38 (1.26x WITHIN a level) and
/// the high 10.50-10.86; the gap between them is 6.38 -> 10.50, **1.64x**. So the threshold has
/// to sit above 1.26 and below 1.64. 1.40 is the midpoint in log space (1.26 * 1.64 = 2.07,
/// sqrt = 1.44 — 1.40 is a touch conservative of that, which errs toward calling a population
/// SPLIT, i.e. toward flagging rather than toward silence).
///
/// The unit tests sweep it across 1.25-1.60 and assert the verdict does not move anywhere in
/// that band, so the choice is not load-bearing to two decimal places.
pub const DEFAULT_SPLIT_RATIO: f64 = 1.40;

/// Identity of a bankable incumbent observation.
///
/// `fingerprint` is the incumbent's gradient checksum, which pins the lane's SHAPE and loss;
/// `torch_version` and `host` pin the thing being measured and the thing measuring it. Our own
/// ELF is deliberately NOT part of the key: the incumbent is PyTorch, and a bank that forgot its
/// history on every rebuild of ours would never accumulate enough observations to see a level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncumbentKey {
    pub host: String,
    pub torch_version: String,
    pub lane: String,
    pub fingerprint: String,
}

impl IncumbentKey {
    /// Build a key, rendering the checksum to seven significant digits.
    ///
    /// SEVEN, not seventeen. The checksum is a deterministic sum over a deterministic input, so
    /// it *should* be bit-identical between invocations — but "should" is doing work there, and
    /// a single-ulp difference in the last place would fork the key and quietly give every run
    /// its own empty history, which is the one failure mode that would make this module useless
    /// while still appearing to work.
    #[must_use]
    pub fn new(host: &str, torch_version: &str, lane: &str, checksum: f64) -> Self {
        Self {
            host: sanitize(host),
            torch_version: sanitize(torch_version),
            lane: sanitize(lane),
            fingerprint: sanitize(&format!("{checksum:.6e}")),
        }
    }
}

/// One banked observation.
#[derive(Debug, Clone)]
pub struct IncumbentRecord {
    pub key: IncumbentKey,
    pub incumbent_ms: f64,
    pub rounds: usize,
    pub lane_count: usize,
    /// Epoch seconds. Deliberately not an ISO string: the crate has no date dependency, and a
    /// field that names a format it cannot produce is worse than an integer that means what it
    /// says. Sorts correctly, which is all the bank needs.
    pub recorded_unix: u64,
}

/// One level of the population: a cluster of observations with no large gap inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Level {
    pub n: usize,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl Level {
    /// How wide the level is internally. A level whose own spread exceeds the SPLIT THRESHOLD is
    /// a hint that there are MORE than two levels — this module only ever reports a two-way
    /// split, and this is how a third would make itself visible rather than hide inside one.
    #[must_use]
    pub fn spread(&self) -> f64 {
        if self.min > 0.0 { self.max / self.min } else { 1.0 }
    }
}

/// What the bank can say about the invocation that just ran.
#[derive(Debug, Clone, PartialEq)]
pub enum ModeReport {
    /// Nothing banked for this key yet. Not a failure; a statement that this row is one
    /// invocation short of readable.
    NoHistory { drawn_ms: f64 },
    /// One level. The row is comparable to every other row in the bank for this lane.
    Single { drawn_ms: f64, level: Level },
    /// Two levels. The row is comparable ONLY to rows that drew the same one.
    Split {
        drawn_ms: f64,
        levels: [Level; 2],
        /// 0 or 1 — which level this invocation landed in.
        drawn: usize,
        /// The consecutive ratio at the split point.
        gap_ratio: f64,
        /// The threshold this split was judged against, carried so `render` can say when a
        /// level is itself wide enough to hide another split. Comparing a level's spread
        /// against `gap_ratio` instead would almost never fire: `gap_ratio` is the LARGEST
        /// consecutive step in the population by construction.
        split_ratio: f64,
    },
}

impl ModeReport {
    /// True when the population has two levels, i.e. when this lane's standing is only
    /// meaningful against rows that drew the same level.
    #[must_use]
    pub fn is_split(&self) -> bool {
        matches!(self, Self::Split { .. })
    }

    /// A compact tag for the row line itself, so the row carries its own comparability.
    #[must_use]
    pub fn row_tag(&self) -> String {
        match self {
            Self::NoHistory { .. } => " incumbent=NO-HISTORY".to_owned(),
            Self::Single { level, .. } => format!(" incumbent=SINGLE/n{}", level.n),
            Self::Split { levels, drawn, .. } => format!(
                " incumbent=LEVEL-{}/2 n{}",
                if *drawn == 0 { "A" } else { "B" },
                levels[*drawn].n
            ),
        }
    }
}

/// Split the population at its largest consecutive gap, if that gap is wide enough.
///
/// `priors` are the previously banked observations for the key; `drawn_ms` is this invocation's.
/// The drawn value is INCLUDED in the population — the question is which level it belongs to,
/// and a point classified against a distribution it is not part of can fall between levels.
#[must_use]
pub fn classify(priors: &[f64], drawn_ms: f64, split_ratio: f64) -> ModeReport {
    if priors.is_empty() {
        return ModeReport::NoHistory { drawn_ms };
    }
    let mut population: Vec<f64> = priors.to_vec();
    population.push(drawn_ms);
    population.retain(|v| v.is_finite() && *v > 0.0);
    population.sort_by(|a, b| a.partial_cmp(b).expect("finite and positive"));
    if population.len() < 2 {
        return ModeReport::NoHistory { drawn_ms };
    }

    let mut best_index = 0usize;
    let mut best_ratio = 1.0f64;
    for i in 0..population.len() - 1 {
        let ratio = population[i + 1] / population[i];
        if ratio > best_ratio {
            best_ratio = ratio;
            best_index = i;
        }
    }

    if best_ratio <= split_ratio {
        return ModeReport::Single {
            drawn_ms,
            level: level_of(&population),
        };
    }

    let (low, high) = population.split_at(best_index + 1);
    let levels = [level_of(low), level_of(high)];
    // Which side the drawn value is on. `<=` against the low level's max is exact here: the
    // drawn value is a member of the population, so it equals one of the endpoints.
    let drawn = usize::from(drawn_ms > levels[0].max);
    ModeReport::Split {
        drawn_ms,
        levels,
        drawn,
        gap_ratio: best_ratio,
        split_ratio,
    }
}

fn level_of(sorted: &[f64]) -> Level {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    };
    Level {
        n,
        median,
        min: sorted[0],
        max: sorted[n - 1],
    }
}

/// The block the harness prints for a lane.
///
/// It says which level, how many rows drew each, and what follows for comparability — and it
/// says in as many words that NEITHER level is known to be the right one, because the one time
/// this campaign assumed the majority level was the truth, the majority flipped.
#[must_use]
pub fn render(lane: &str, report: &ModeReport, records: &[(f64, u64)]) -> String {
    let mut out = String::new();
    match report {
        ModeReport::NoHistory { drawn_ms } => {
            let _ = writeln!(
                out,
                "  incumbent_mode lane={lane} drawn={drawn_ms:.3} ms  NO HISTORY — first \
                 observation for this lane, shape and torch build. This row's incumbent has \
                 nothing to be compared against, so the row is ONE INVOCATION SHORT OF READABLE \
                 however its nulls read."
            );
        }
        ModeReport::Single { drawn_ms, level } => {
            let _ = writeln!(
                out,
                "  incumbent_mode lane={lane} drawn={drawn_ms:.3} ms  SINGLE LEVEL (n={}, \
                 median {:.3}, range [{:.3}, {:.3}], spread {:.2}x) — no second level banked for \
                 this lane, so this row is comparable to the other {}.{}",
                level.n,
                level.median,
                level.min,
                level.max,
                level.spread(),
                level.n.saturating_sub(1),
                span_caveat(records)
            );
        }
        ModeReport::Split {
            drawn_ms,
            levels,
            drawn,
            gap_ratio,
            split_ratio,
        } => {
            let _ = writeln!(
                out,
                "  incumbent_mode lane={lane} drawn={drawn_ms:.3} ms  LEVEL {} OF 2 — the banked \
                 population for this lane splits at a {gap_ratio:.2}x gap:",
                if *drawn == 0 { "A" } else { "B" }
            );
            for (i, level) in levels.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "      level {}  n={:<3} median {:8.3}  range [{:.3}, {:.3}]  spread {:.2}x{}{}",
                    if i == 0 { "A" } else { "B" },
                    level.n,
                    level.median,
                    level.min,
                    level.max,
                    level.spread(),
                    if i == *drawn { "   <- THIS RUN" } else { "" },
                    if level.spread() > *split_ratio {
                        "   [LEVEL SPREAD EXCEEDS THE GAP — suspect a third level]"
                    } else {
                        ""
                    },
                );
            }
            let _ = writeln!(
                out,
                "    This row is comparable ONLY to rows that drew level {}. NEITHER LEVEL IS \
                 KNOWN TO BE THE CORRECT ONE and the harness does not assert one: on conv2d_big \
                 the ~6 ms level was 1 of 8 observations in 2026-08 and 20 of 22 in 2026-09, so \
                 a disagreement with the majority is a NO-VERDICT, not a correction. A standing \
                 whose SIGN differs between the levels must not be banked from one invocation.{}",
                if *drawn == 0 { "A" } else { "B" },
                span_caveat(records)
            );
        }
    }
    out
}

/// Where the bank lives. `FT_H2H_INCUMBENT_BANK` overrides.
#[must_use]
pub fn bank_path() -> PathBuf {
    std::env::var("FT_H2H_INCUMBENT_BANK")
        .map_or_else(|_| PathBuf::from("artifacts/perf/incumbent_bank.jsonl"), PathBuf::from)
}

/// Prior observations for a key as `(incumbent_ms, recorded_unix)`, oldest first. A missing or
/// unreadable bank is an empty history, never an error: this is an instrument, and it must not
/// be able to fail a measurement.
#[must_use]
pub fn load_records(path: &Path, key: &IncumbentKey) -> Vec<(f64, u64)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines().filter_map(|line| match_line(line, key)).collect()
}

/// Just the times, for the classifier.
#[must_use]
pub fn load(path: &Path, key: &IncumbentKey) -> Vec<f64> {
    load_records(path, key).into_iter().map(|(ms, _)| ms).collect()
}

/// How long the banked observations span, in seconds, and how many there are.
///
/// WHY THE SPAN IS PART OF THE VERDICT. Twenty observations taken in one hour are ONE session,
/// not twenty independent draws — and the effect this module exists for is a BETWEEN-SESSION
/// one: `conv2d_big` read ~11 ms in seven of eight observations in 2026-08 and ~6 ms in twenty
/// of twenty-two in 2026-09. A bank filled inside a single hour would have called either
/// session SINGLE LEVEL with total confidence and been wrong about the lane both times. So the
/// span is reported next to the count, and a bank narrower than an hour says so.
#[must_use]
pub fn history_span_seconds(records: &[(f64, u64)]) -> u64 {
    let stamps: Vec<u64> = records.iter().map(|(_, t)| *t).filter(|t| *t > 0).collect();
    match (stamps.iter().min(), stamps.iter().max()) {
        (Some(lo), Some(hi)) => hi - lo,
        _ => 0,
    }
}

/// The sentence that qualifies a SINGLE-level verdict by how much time the bank covers.
#[must_use]
pub fn span_caveat(records: &[(f64, u64)]) -> String {
    let span = history_span_seconds(records);
    if span < 3_600 {
        format!(
            " The bank spans {span}s — under an hour, so these are ONE session's draws and a              SINGLE-level verdict from them is weak evidence: the level this lane sits at has              been observed to differ BETWEEN sessions, which a bank this narrow cannot see."
        )
    } else {
        format!(" The bank spans {:.1} days.", span as f64 / 86_400.0)
    }
}

fn match_line(line: &str, key: &IncumbentKey) -> Option<(f64, u64)> {
    let field = |name: &str| -> Option<&str> {
        let needle = format!("\"{name}\":\"");
        let start = line.find(&needle)? + needle.len();
        let rest = &line[start..];
        let end = rest.find('"')?;
        Some(&rest[..end])
    };
    if field("host")? != key.host
        || field("torch")? != key.torch_version
        || field("lane")? != key.lane
        || field("fingerprint")? != key.fingerprint
    {
        return None;
    }
    let number = |name: &str| -> Option<&str> {
        let needle = format!("\"{name}\":");
        let start = line.find(&needle)? + needle.len();
        let rest = &line[start..];
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        Some(rest[..end].trim())
    };
    let ms = number("incumbent_ms")?
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v > 0.0)?;
    // A record written before the timestamp field existed still carries a usable time.
    let stamp = number("recorded_unix").and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    Some((ms, stamp))
}

/// Append one observation. Best-effort by contract: a bank that cannot be written must not take
/// down a measurement that has already been paid for, so the caller is told and carries on.
///
/// # Errors
/// Returns the underlying I/O error when the bank cannot be created, opened or appended to.
pub fn append(path: &Path, record: &IncumbentRecord) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", render_line(record))
}

fn render_line(record: &IncumbentRecord) -> String {
    format!(
        "{{\"host\":\"{}\",\"torch\":\"{}\",\"lane\":\"{}\",\"fingerprint\":\"{}\",\
         \"incumbent_ms\":{:.6},\"rounds\":{},\"lane_count\":{},\"recorded_unix\":{}}}",
        record.key.host,
        record.key.torch_version,
        record.key.lane,
        record.key.fingerprint,
        record.incumbent_ms,
        record.rounds,
        record.lane_count,
        record.recorded_unix,
    )
}

/// Keep string fields to characters that cannot break the hand-rolled JSON above.
///
/// The crate has no serde and this file is not worth adding one for, so the escaping problem is
/// removed rather than solved: every field here is a hostname, a torch version, a lane name, a
/// formatted float or a timestamp, and all of them live inside this set already.
fn sanitize(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-' | ':'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured `conv2d_big` incumbent population, 2026-09-01, thinkstation1, one ELF
    /// (1c7bbababe8f725b), PyTorch 2.12.1+cpu, guard-gated, `concurrent_measurements=none` in
    /// every one. Twenty-two invocations: five standing attempts, six at fixed REPS=24, ten from
    /// the round-interleaved REPS ladder, one full board.
    const TODAY_CONV2D_BIG: [f64; 22] = [
        6.333, 10.856, 5.688, 5.730, 10.500, // the five standing attempts
        6.376, 5.650, 5.265, 5.290, 5.642, 6.252, // fixed-REPS probe
        5.500, 6.200, 5.064, 5.262, 5.653, // ladder pass 1
        5.489, 5.522, 5.346, 6.009, 5.653, // ladder pass 2
        5.523, // full board, 67 lanes
    ];

    #[test]
    fn the_two_high_invocations_are_separated_from_the_twenty_low_ones() {
        let priors: Vec<f64> = TODAY_CONV2D_BIG[..21].to_vec();
        let report = classify(&priors, TODAY_CONV2D_BIG[21], DEFAULT_SPLIT_RATIO);
        let ModeReport::Split { levels, drawn, .. } = report else {
            panic!("the measured population is two levels, not one: {report:?}");
        };
        assert_eq!(levels[0].n, 20, "the low level holds twenty invocations");
        assert_eq!(levels[1].n, 2, "the high level holds two");
        assert!(levels[0].max < 6.5 && levels[1].min > 10.0);
        assert_eq!(drawn, 0, "the full-board invocation drew the LOW level");
    }

    /// The row that mattered: an invocation that drew the high level must be told so, because
    /// its standing has the opposite sign to the twenty that drew the low one.
    #[test]
    fn an_invocation_that_draws_the_high_level_is_labelled_as_such() {
        let priors: Vec<f64> = TODAY_CONV2D_BIG.iter().copied().filter(|v| *v < 7.0).collect();
        let report = classify(&priors, 10.500, DEFAULT_SPLIT_RATIO);
        let ModeReport::Split { drawn, levels, .. } = report else {
            panic!("expected a split, got {report:?}");
        };
        assert_eq!(drawn, 1, "10.5 ms is the high level");
        assert_eq!(levels[1].n, 1);
        assert!(report.row_tag().contains("LEVEL-B"));
        assert!(render("conv2d_big", &report, &[]).contains("comparable ONLY to rows that drew level B"));
    }

    /// THE THRESHOLD IS NOT LOAD-BEARING. Today's low level is 1.26x wide internally and the gap
    /// to the high level is 1.64x, so any threshold strictly between them gives the same answer.
    /// Swept here rather than asserted, so a future population that narrows that margin fails
    /// this test instead of silently depending on 1.40.
    #[test]
    fn the_verdict_is_stable_across_the_whole_plausible_threshold_band() {
        let priors: Vec<f64> = TODAY_CONV2D_BIG[..21].to_vec();
        let mut checked = 0;
        let mut threshold = 1.30;
        while threshold <= 1.60 {
            let report = classify(&priors, TODAY_CONV2D_BIG[21], threshold);
            let ModeReport::Split { levels, .. } = report else {
                panic!("threshold {threshold:.2} failed to split the measured population");
            };
            assert_eq!((levels[0].n, levels[1].n), (20, 2), "at threshold {threshold:.2}");
            checked += 1;
            threshold += 0.01;
        }
        assert!(checked >= 30);
    }

    /// A unimodal population must NOT be split, or every lane on the board acquires a spurious
    /// comparability caveat and the real one stops being read. Our own arm on the same lane and
    /// the same twenty-two invocations is the natural negative control: it never moved.
    #[test]
    fn a_unimodal_population_is_not_split() {
        let ft_arm = [
            4.241, 4.473, 4.109, 4.290, 4.201, 4.192, 4.043, 4.098, 4.051, 4.170, 4.275, 4.140,
            4.421, 4.206, 4.403, 4.101, 4.150, 4.089, 4.361, 4.396, 4.167,
        ];
        let report = classify(&ft_arm, 4.128, DEFAULT_SPLIT_RATIO);
        let ModeReport::Single { level, .. } = report else {
            panic!("our arm is one level; got {report:?}");
        };
        assert_eq!(level.n, 22);
        assert!(level.spread() < 1.15);
    }

    /// August's population for this lane pair, `NEGATIVE_EVIDENCE` item 219: seven at 11.4-12.0
    /// and one at 6.075. The same classifier must split it too — and must put the SINGLE 6 ms
    /// observation in its own level rather than declaring it an outlier to be discarded. This is
    /// the prevalence-flip case: on that evidence ~11 ms was the majority, and today it is not.
    #[test]
    fn the_august_population_splits_the_same_way_with_the_majority_on_the_other_side() {
        let priors = [11.4, 11.6, 11.7, 11.8, 11.9, 12.0];
        let report = classify(&priors, 6.075, DEFAULT_SPLIT_RATIO);
        let ModeReport::Split { levels, drawn, .. } = report else {
            panic!("expected a split, got {report:?}");
        };
        assert_eq!(drawn, 0, "6.075 is the LOW level even though it is the minority here");
        assert_eq!(levels[0].n, 1);
        assert_eq!(levels[1].n, 6);
    }

    /// The verdict must never phrase itself as a correction. If this ever starts saying one
    /// level is right, the module has become the thing item 219 got wrong.
    #[test]
    fn the_rendered_verdict_never_declares_a_level_correct() {
        let report = classify(&TODAY_CONV2D_BIG[..21], 10.500, DEFAULT_SPLIT_RATIO);
        let text = render("conv2d_big", &report, &[]);
        assert!(text.contains("NEITHER LEVEL IS KNOWN TO BE THE CORRECT ONE"));
        assert!(text.contains("NO-VERDICT, not a correction"));
        for forbidden in ["outlier", "anomal", "spurious", "discard"] {
            assert!(
                !text.to_lowercase().contains(forbidden),
                "the verdict must not editorialise about which level is real: {forbidden}"
            );
        }
    }

    #[test]
    fn an_empty_bank_is_no_history_rather_than_an_error() {
        assert!(matches!(
            classify(&[], 5.0, DEFAULT_SPLIT_RATIO),
            ModeReport::NoHistory { .. }
        ));
        assert!(render("x", &classify(&[], 5.0, DEFAULT_SPLIT_RATIO), &[])
            .contains("ONE INVOCATION SHORT OF READABLE"));
    }

    /// A round-trip through the on-disk format, and — the part that matters — a record with a
    /// DIFFERENT fingerprint must not be pooled with this one. That is what stops a change to a
    /// lane's shape from silently merging two different measurements into one population.
    #[test]
    fn the_bank_round_trips_and_refuses_to_pool_across_a_shape_change() {
        let dir = std::env::temp_dir().join(format!("ft-bank-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bank.jsonl");
        let _ = std::fs::remove_file(&path);

        let key = IncumbentKey::new("thinkstation1", "2.12.1+cpu", "conv2d_big", 1234.5678);
        let other_shape = IncumbentKey::new("thinkstation1", "2.12.1+cpu", "conv2d_big", 999.0);
        assert_ne!(key.fingerprint, other_shape.fingerprint);

        for ms in [5.688, 10.500] {
            append(
                &path,
                &IncumbentRecord {
                    key: key.clone(),
                    incumbent_ms: ms,
                    rounds: 24,
                    lane_count: 2,
                    recorded_unix: 1_788_000_000,
                },
            )
            .expect("bank is writable in the temp dir");
        }
        append(
            &path,
            &IncumbentRecord {
                key: other_shape.clone(),
                incumbent_ms: 99.0,
                rounds: 24,
                lane_count: 2,
                recorded_unix: 1_788_000_000,
            },
        )
        .expect("bank is writable in the temp dir");

        let loaded = load(&path, &key);
        assert_eq!(loaded, vec![5.688, 10.500], "only this shape's observations");
        assert_eq!(load(&path, &other_shape), vec![99.0]);
        let _ = std::fs::remove_file(&path);
    }

    /// A last-ulp wobble in the checksum must not fork the key, or every invocation gets its own
    /// empty history and the module looks like it is working while banking nothing.
    /// Twenty draws inside one hour are ONE session. The verdict has to say so, because the
    /// effect this module exists for is a between-session one and a bank this narrow is blind
    /// to it — a SINGLE-level verdict from it would be exactly the overconfidence item 219 had.
    #[test]
    fn a_single_level_verdict_from_one_hour_of_history_says_it_is_weak() {
        let records: Vec<(f64, u64)> = TODAY_CONV2D_BIG
            .iter()
            .filter(|v| **v < 7.0)
            .enumerate()
            .map(|(i, ms)| (*ms, 1_788_000_000 + i as u64 * 120))
            .collect();
        let priors: Vec<f64> = records.iter().map(|(ms, _)| *ms).collect();
        assert!(history_span_seconds(&records) < 3_600);
        let report = classify(&priors, 5.523, DEFAULT_SPLIT_RATIO);
        assert!(matches!(report, ModeReport::Single { .. }));
        let text = render("conv2d_big", &report, &records);
        assert!(text.contains("ONE session's draws"), "{text}");
        assert!(text.contains("weak evidence"), "{text}");
    }

    /// A bank that spans real time reports the span instead of the warning.
    #[test]
    fn a_multi_day_bank_reports_its_span_rather_than_the_warning() {
        let records = [(5.6, 1_788_000_000), (5.7, 1_788_000_000 + 3 * 86_400)];
        let note = span_caveat(&records);
        assert!(note.contains("spans 3.0 days"), "{note}");
        assert!(!note.contains("weak evidence"));
    }

    /// The timestamp has to survive the round trip, or the span is always zero and the caveat
    /// fires forever.
    #[test]
    fn the_recorded_timestamp_round_trips() {
        let dir = std::env::temp_dir().join(format!("ft-bank-ts-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bank.jsonl");
        let _ = std::fs::remove_file(&path);
        let key = IncumbentKey::new("h", "2.12.1+cpu", "lane", 42.0);
        append(
            &path,
            &IncumbentRecord {
                key: key.clone(),
                incumbent_ms: 5.5,
                rounds: 24,
                lane_count: 2,
                recorded_unix: 1_788_123_456,
            },
        )
        .expect("temp dir is writable");
        assert_eq!(load_records(&path, &key), vec![(5.5, 1_788_123_456)]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_last_ulp_checksum_wobble_does_not_fork_the_key() {
        let a = IncumbentKey::new("h", "2.12.1+cpu", "lane", 1234.567_890_123);
        let b = IncumbentKey::new("h", "2.12.1+cpu", "lane", 1234.567_890_124);
        assert_eq!(a, b);
    }

    #[test]
    fn sanitize_strips_anything_that_could_break_the_hand_rolled_json() {
        assert_eq!(sanitize("host\"name\n\\x"), "hostnamex");
        assert_eq!(sanitize("2.12.1+cpu"), "2.12.1+cpu");
        assert_eq!(sanitize("2026-09-01T15:00:00Z"), "2026-09-01T15:00:00Z");
    }

    /// A third level would show up as a level whose own spread exceeds the gap that split it.
    /// The module reports at most a two-way split by design; this is how the design announces
    /// its own limit rather than hiding a third population inside one level.
    #[test]
    fn a_third_level_is_flagged_by_the_level_spread_note() {
        let priors = [5.0, 5.1, 5.2, 20.0, 21.0, 40.0, 41.0];
        let report = classify(&priors, 5.05, DEFAULT_SPLIT_RATIO);
        let text = render("lane", &report, &[]);
        assert!(text.contains("suspect a third level"), "{text}");
    }
}
