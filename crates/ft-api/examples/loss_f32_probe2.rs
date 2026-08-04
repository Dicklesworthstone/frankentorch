use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let f32t = |s: &mut FrankenTorchSession, v: Vec<f32>, shape: Vec<usize>| {
        s.tensor_variable_f32(v, shape, false).unwrap()
    };
    let feat = f32t(&mut s, vec![0.5, 0.3, 0.2, 0.1, 0.4, 0.6], vec![2, 3]);
    let weight = f32t(&mut s, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 2]);
    let labels = f32t(&mut s, vec![0.0, 1.0], vec![2]);
    let z1 = f32t(
        &mut s,
        vec![0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.2, 0.8, 0.1, 0.3, 0.5, 0.7],
        vec![4, 3],
    );
    let z2 = f32t(
        &mut s,
        vec![0.4, 0.2, 0.3, 0.2, 0.5, 0.5, 0.3, 0.7, 0.2, 0.4, 0.4, 0.6],
        vec![4, 3],
    );
    let mm_in = f32t(&mut s, vec![0.5, 0.3, 0.2, 0.1, 0.4, 0.6], vec![2, 3]);
    let mm_tgt = f32t(&mut s, vec![0.0, 1.0], vec![2]);

    macro_rules! probe {
        ($name:expr, $call:expr) => {{
            match $call {
                Ok(t) => println!("{:<18} OK   dtype={:?}", $name, s.tensor_dtype(t).unwrap()),
                Err(e) => println!("{:<18} ERR  {:?}", $name, e),
            }
        }};
    }
    probe!(
        "arcface_loss",
        s.arcface_loss(feat, weight, labels, 30.0, 0.5)
    );
    probe!(
        "cosface_loss",
        s.cosface_loss(feat, weight, labels, 30.0, 0.35)
    );
    probe!("barlow_twins_loss", s.barlow_twins_loss(z1, z2, 0.005));
    probe!("vicreg_loss", s.vicreg_loss(z1, z2, 25.0, 25.0, 1.0));
    probe!(
        "multi_margin_loss",
        s.multi_margin_loss(mm_in, mm_tgt, 1.0, 1.0)
    );
    probe!("ring_loss", s.ring_loss(feat, 1.0));
}
