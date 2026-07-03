use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    // probability-ish [4] f32 tensors
    let mk = |s: &mut FrankenTorchSession, v: Vec<f32>, shape: Vec<usize>| {
        s.tensor_variable_f32(v, shape, false).unwrap()
    };
    let a = mk(&mut s, vec![0.6, 0.2, 0.8, 0.4], vec![4]);
    let b = mk(&mut s, vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let pred = mk(&mut s, vec![0.5, -1.0, 2.0, -0.5], vec![4]);
    let tgt = mk(&mut s, vec![1.0, -1.0, 1.0, -1.0], vec![4]);
    // boxes [2,4]: x1,y1,x2,y2
    let pb = mk(&mut s, vec![0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0], vec![2, 4]);
    let tb = mk(&mut s, vec![0.0, 0.0, 2.0, 2.0, 0.5, 0.5, 2.5, 2.5], vec![2, 4]);

    macro_rules! probe {
        ($name:expr, $call:expr) => {{
            match $call {
                Ok(t) => println!("{:<18} OK   dtype={:?}", $name, s.tensor_dtype(t).unwrap()),
                Err(e) => println!("{:<18} ERR  {:?}", $name, e),
            }
        }};
    }
    probe!("dice_loss", s.dice_loss(a, b, 1.0));
    probe!("tversky_loss", s.tversky_loss(a, b, 0.5, 0.5, 1.0));
    probe!("iou_loss", s.iou_loss(a, b, 1.0));
    probe!("hinge_loss", s.hinge_loss(pred, tgt));
    probe!("giou_loss", s.giou_loss(pb, tb));
    probe!("diou_loss", s.diou_loss(pb, tb));
    probe!("ciou_loss", s.ciou_loss(pb, tb));
    probe!("l1_reg_loss", s.l1_reg_loss(&[a], 0.01));
    probe!("l2_reg_loss", s.l2_reg_loss(&[a], 0.01));
    let _ = DType::F32;
}
