use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::ExecutionMode;
use ft_runtime::RuntimeContext;

fn bench_policy_evidence(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_policy_evidence");
    group.bench_function("new_and_switch_1024", |b| {
        b.iter(|| {
            let mut total_entries = 0_usize;
            let mut total_summary_len = 0_usize;

            for idx in 0..1024 {
                let initial_mode = if idx % 2 == 0 {
                    ExecutionMode::Strict
                } else {
                    ExecutionMode::Hardened
                };
                let next_mode = if idx % 2 == 0 {
                    ExecutionMode::Hardened
                } else {
                    ExecutionMode::Strict
                };
                let mut context = RuntimeContext::new(initial_mode);
                context.set_mode(next_mode);
                let entries = context.ledger().entries();
                total_entries += entries.len();
                total_summary_len += entries
                    .iter()
                    .map(|entry| entry.summary.len())
                    .sum::<usize>();
            }

            black_box((total_entries, total_summary_len))
        });
    });
    group.finish();
}

criterion_group!(benches, bench_policy_evidence);
criterion_main!(benches);
