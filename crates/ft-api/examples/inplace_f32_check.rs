// Verify no-grad f32 in-place ops work (were UnsupportedDType crashes) + correct. cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
fn approx(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-6 * y.abs().max(1.0))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut ok = true;

    // maximum_ (binary movement): a = max(a,b)
    let a = s.tensor_variable_f32(vec![1.0, 5.0, 2.0, 9.0], vec![4], false)?;
    let b = s.tensor_variable_f32(vec![3.0, 4.0, 8.0, 1.0], vec![4], false)?;
    match s.tensor_maximum_(a, b) {
        Ok(()) => {
            let r = s.tensor_values_lossy_f64(a)?;
            let exp = vec![3.0, 5.0, 8.0, 9.0];
            let pass = approx(&r, &exp);
            ok &= pass;
            println!(
                "maximum_  f32: OK  result={r:?} {}",
                if pass { "CORRECT" } else { "WRONG" }
            );
        }
        Err(e) => {
            ok = false;
            println!("maximum_  f32: ERROR {e:?}");
        }
    }

    // masked_fill_ (mask, scalar): a[mask] = value
    let c = s.tensor_variable_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)?;
    let m = s.tensor_variable_f32(vec![0.0, 1.0, 0.0, 1.0], vec![4], false)?;
    match s.tensor_masked_fill_(c, m, -7.0) {
        Ok(()) => {
            let r = s.tensor_values_lossy_f64(c)?;
            let exp = vec![1.0, -7.0, 3.0, -7.0];
            let pass = approx(&r, &exp);
            ok &= pass;
            println!(
                "masked_fill_ f32: OK result={r:?} {}",
                if pass { "CORRECT" } else { "WRONG" }
            );
        }
        Err(e) => {
            ok = false;
            println!("masked_fill_ f32: ERROR {e:?}");
        }
    }

    // addcmul_ (ternary arithmetic): a += scalar * t1 * t2
    let d = s.tensor_variable_f32(vec![1.0, 1.0, 1.0], vec![3], false)?;
    let t1 = s.tensor_variable_f32(vec![2.0, 3.0, 4.0], vec![3], false)?;
    let t2 = s.tensor_variable_f32(vec![5.0, 6.0, 7.0], vec![3], false)?;
    match s.tensor_addcmul_(d, t1, t2, 1.0) {
        Ok(()) => {
            let r = s.tensor_values_lossy_f64(d)?;
            let exp = vec![11.0, 19.0, 29.0]; // 1 + 1*(2*5), 1+3*6, 1+4*7
            let pass = approx(&r, &exp);
            ok &= pass;
            println!(
                "addcmul_  f32: OK  result={r:?} {}",
                if pass { "CORRECT" } else { "WRONG" }
            );
        }
        Err(e) => {
            ok = false;
            println!("addcmul_  f32: ERROR {e:?}");
        }
    }

    println!("ALL_F32_INPLACE_OK={ok}");
    Ok(())
}
