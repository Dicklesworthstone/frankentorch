use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_data::{DataItem, DataLoader, DataLoaderConfig, RandomSampler, TensorDataset};

fn make_dataset(samples: usize, features: usize) -> TensorDataset {
    let items = (0..samples)
        .map(|sample| {
            let input = (0..features)
                .map(|feature| (sample * features + feature) as f64)
                .collect::<Vec<_>>();
            let target = vec![sample as f64];
            DataItem::input_target(input, vec![features], target, vec![1])
        })
        .collect::<Vec<_>>();
    TensorDataset::new(items)
}

fn bench_dataloader_epoch(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataloader");
    let dataset = make_dataset(2048, 256);

    group.bench_function("epoch_2048x256_batch128", |b| {
        b.iter_batched(
            || {
                let session = FrankenTorchSession::new(ExecutionMode::Strict);
                let loader = DataLoader::new(&dataset, DataLoaderConfig::new(128));
                (session, loader)
            },
            |(mut session, mut loader)| {
                let mut batches = 0usize;
                while let Some(batch) = loader.next_batch(&mut session).expect("batch") {
                    black_box(batch.input());
                    black_box(batch.target());
                    batches += 1;
                }
                black_box(batches)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_random_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampler");

    group.bench_function("without_replacement_size4096_samples66560", |b| {
        b.iter(|| {
            let sampler = RandomSampler::new(black_box(4096))
                .with_num_samples(black_box(66_560))
                .with_seed(black_box(0xA11C_E5EED));
            black_box(sampler.indices())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_dataloader_epoch, bench_random_sampler);
criterion_main!(benches);
