// Verify unique_consecutive preserves input dtype (f32->f32, was f32->F64 bug). cc.
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut ok = true;

    // f32: dtype must be F32; values = consecutive-unique of [1,1,2,2,2,3,1] = [1,2,3,1]
    let x = s.tensor_variable_f32(vec![1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0], vec![7], false)?;
    let (u, _, _) = s.tensor_unique_consecutive(x, false, false)?;
    let dt = s.tensor_dtype(u)?;
    let v: Vec<f64> = s.tensor_values_lossy_f64(u)?;
    let dt_ok = dt == DType::F32;
    let val_ok = v == vec![1.0, 2.0, 3.0, 1.0];
    ok &= dt_ok && val_ok;
    println!(
        "f32 unique_consecutive: dtype={dt:?} ({}) values={v:?} ({})",
        if dt_ok {
            "F32 CORRECT"
        } else {
            "WRONG (expected F32)"
        },
        if val_ok { "CORRECT" } else { "WRONG" }
    );

    // f64 must stay F64 (no regression)
    let y = s.tensor_variable(vec![5.0, 5.0, 7.0], vec![3], false)?;
    let (u2, _, _) = s.tensor_unique_consecutive(y, false, false)?;
    let dt2 = s.tensor_dtype(u2)?;
    let dt2_ok = dt2 == DType::F64;
    ok &= dt2_ok;
    println!(
        "f64 unique_consecutive: dtype={dt2:?} ({})",
        if dt2_ok { "F64 CORRECT" } else { "REGRESSION" }
    );

    println!("ALL_OK={ok}");
    Ok(())
}
