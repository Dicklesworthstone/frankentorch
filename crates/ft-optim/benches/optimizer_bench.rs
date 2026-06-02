use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_autograd::{TensorBackwardReport, TensorNodeId};
use ft_core::ExecutionMode;
use ft_optim::{AdamW, Optimizer};

fn make_adamw_case() -> (
    FrankenTorchSession,
    AdamW,
    TensorBackwardReport,
    TensorNodeId,
) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut params = Vec::with_capacity(64);
    let mut total_loss = None;

    for param_idx in 0..64 {
        let values = (0..1024)
            .map(|value_idx| 0.001 * (param_idx * 1024 + value_idx) as f64)
            .collect::<Vec<_>>();
        let param = session
            .tensor_variable(values, vec![1024], true)
            .expect("param");
        let param_loss = session.tensor_sum(param).expect("param sum");
        total_loss = Some(match total_loss {
            Some(acc) => session.tensor_add(acc, param_loss).expect("loss add"),
            None => param_loss,
        });
        params.push(param);
    }

    let first_param = params[0];
    let report = session
        .tensor_backward(total_loss.expect("total loss"))
        .expect("backward");
    let optimizer = AdamW::new(params, 0.001);
    (session, optimizer, report, first_param)
}

fn bench_adamw_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("adamw");

    group.bench_function("step_64x1024", |b| {
        b.iter_batched(
            make_adamw_case,
            |(mut session, mut optimizer, report, first_param)| {
                optimizer.step(&mut session, &report).expect("step");
                black_box(session.tensor_values(first_param).expect("values")[0])
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_adamw_step);
criterion_main!(benches);
