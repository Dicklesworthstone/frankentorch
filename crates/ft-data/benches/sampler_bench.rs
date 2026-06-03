use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_data::{RandomSampler, WeightedRandomSampler};

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

    group.finish();
}

criterion_group!(benches, bench_random_sampler);
criterion_main!(benches);
