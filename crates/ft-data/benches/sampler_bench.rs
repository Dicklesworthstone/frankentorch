use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_data::{RandomSampler, WeightedRandomSampler};

struct BenchRng {
    state: u64,
}

impl BenchRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
}

fn finite_threshold_cmp(threshold: f64, sample: f64) -> std::cmp::Ordering {
    if threshold < sample {
        std::cmp::Ordering::Less
    } else if threshold > sample {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

fn weighted_binary_reference_indices(weights: &[f64], num_samples: usize, seed: u64) -> Vec<usize> {
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut total = 0.0;
    for &weight in weights {
        total += weight;
        cumulative.push(total);
    }
    for threshold in &mut cumulative {
        *threshold /= total;
    }

    let mut rng = BenchRng::new(seed);
    let mut result = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let sample = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let index = match cumulative.binary_search_by(|c| finite_threshold_cmp(*c, sample)) {
            Ok(i) => i,
            Err(i) => i.min(weights.len() - 1),
        };
        result.push(index);
    }
    result
}

fn bench_random_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampler");

    group.bench_function("without_replacement_repeated_passes_4096x256", |b| {
        b.iter_batched(
            || {
                RandomSampler::new(4096)
                    .with_num_samples(4096 * 256 + 1537)
                    .with_seed(0xA5A5_5A5A_0123_4567)
            },
            |sampler| black_box(sampler.indices()),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("weighted_single_positive_1x1m", |b| {
        b.iter_batched(
            || WeightedRandomSampler::new(vec![5.0], 1_000_000).with_seed(0x5151_0001),
            |sampler| black_box(sampler.indices().expect("weighted samples")),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("weighted_two_positive_2x1m", |b| {
        b.iter_batched(
            || WeightedRandomSampler::new(vec![1.0, 3.0], 1_000_000).with_seed(0x5151_0002),
            |sampler| black_box(sampler.indices().expect("weighted samples")),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("weighted_three_positive_3x1m", |b| {
        b.iter_batched(
            || WeightedRandomSampler::new(vec![1.0, 3.0, 6.0], 1_000_000).with_seed(0x5151_0003),
            |sampler| black_box(sampler.indices().expect("weighted samples")),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("weighted_four_positive_4x1m", |b| {
        b.iter_batched(
            || {
                WeightedRandomSampler::new(vec![1.0, 2.0, 3.0, 4.0], 1_000_000)
                    .with_seed(0x5151_0004)
            },
            |sampler| black_box(sampler.indices().expect("weighted samples")),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("weighted_4096_positive_4096x262k", |b| {
        b.iter_batched(
            || {
                let weights: Vec<f64> = (1..=4096).map(|i| f64::from((i % 17) + 1)).collect();
                WeightedRandomSampler::new(weights, 262_144).with_seed(0x5151_4096)
            },
            |sampler| black_box(sampler.indices().expect("weighted samples")),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("weighted_4096_positive_binary_reference_4096x262k", |b| {
        b.iter_batched(
            || {
                (1..=4096)
                    .map(|i| f64::from((i % 17) + 1))
                    .collect::<Vec<_>>()
            },
            |weights| {
                black_box(weighted_binary_reference_indices(
                    &weights,
                    262_144,
                    0x5151_4096,
                ))
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_random_sampler);
criterion_main!(benches);
