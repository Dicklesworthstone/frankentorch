#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let n = usize::from(1 + (data[0] % MAX_DIM)); // 1..=8
    let offset = (data[1] as i32 - 128) % 4; // small offset
    let body = &data[2..];

    let input: Vec<f64> = (0..n)
        .map(|i| {
            let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
            (raw - 128) as f64 / 20.0
        })
        .collect();

    // --- diag_embed (1-D → 2-D matrix) ---
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v = match s.tensor_variable(input.clone(), vec![n], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    if let Ok(embedded) = s.tensor_diag_embed(v, offset) {
        if let Ok(emb_vals) = s.tensor_values(embedded) {
            let abs_off = offset.unsigned_abs() as usize;
            let dim = n + abs_off;
            assert_eq!(emb_vals.len(), dim * dim, "diag_embed length");

            // Verify: off-diagonal (relative to `offset`) cells are 0;
            // on-diagonal cells equal input[i].
            for r in 0..dim {
                for c in 0..dim {
                    let cell = emb_vals[r * dim + c];
                    // The k-th diagonal: c - r == offset
                    let on_target_diagonal = (c as i32 - r as i32) == offset;
                    if on_target_diagonal {
                        // Determine which input element belongs here.
                        let i = if offset >= 0 { r } else { c };
                        if i < n {
                            assert!(
                                (cell - input[i]).abs() < 1e-12,
                                "diag_embed[{r},{c}] (on diag, offset={offset}) = {cell}, expected input[{i}] = {}",
                                input[i]
                            );
                        }
                    } else {
                        // Off the target diagonal: must be 0.
                        assert!(
                            cell == 0.0,
                            "diag_embed[{r},{c}] (off diag) = {cell}, expected 0.0"
                        );
                    }
                }
            }
        }
    }

    // --- diag round-trip: diag(diag(v)) recovers v (sort of)
    // tensor_diag with 1-D input produces an (n+|off|)x(n+|off|) matrix,
    // and tensor_diag with 2-D input extracts the diagonal vector.
    // Chain: diag(diag(v, 0), 0) should give back the diagonal vector
    // which is v zero-padded if offset != 0; for offset = 0 it is v.
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v = match s.tensor_variable(input.clone(), vec![n], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    if let Ok(mat) = s.tensor_diag(v, 0) {
        if let Ok(back) = s.tensor_diag(mat, 0) {
            if let Ok(back_vals) = s.tensor_values(back) {
                assert_eq!(back_vals.len(), n, "diag(diag(v, 0), 0) length");
                for (i, (got, expected)) in back_vals.iter().zip(input.iter()).enumerate() {
                    assert!(
                        (got - expected).abs() < 1e-12,
                        "diag(diag(v, 0), 0)[{i}] = {got}, expected {expected}"
                    );
                }
            }
        }
    }

    // --- block_diag total shape = sum of input row/col counts ---
    // Build 2 random 2-D blocks and verify the output shape and that
    // the diagonal blocks contain the input values.
    let r1 = usize::from(1 + (data[0] % 3));
    let c1 = usize::from(1 + (data[1] % 3));
    let r2 = usize::from(1 + (data[2 % data.len().max(1)] % 3));
    let c2 = usize::from(1 + (body[0] % 3).max(1));
    if r1 * c1 <= 16 && r2 * c2 <= 16 {
        let block1: Vec<f64> = (0..(r1 * c1)).map(|i| (i as f64) + 1.0).collect();
        let block2: Vec<f64> = (0..(r2 * c2)).map(|i| -((i as f64) + 100.0)).collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let b1 = match s.tensor_variable(block1.clone(), vec![r1, c1], false) {
            Ok(t) => t,
            Err(_) => return,
        };
        let b2 = match s.tensor_variable(block2.clone(), vec![r2, c2], false) {
            Ok(t) => t,
            Err(_) => return,
        };
        if let Ok(out) = s.tensor_block_diag(&[b1, b2]) {
            if let Ok(out_vals) = s.tensor_values(out) {
                let total_rows = r1 + r2;
                let total_cols = c1 + c2;
                assert_eq!(
                    out_vals.len(),
                    total_rows * total_cols,
                    "block_diag length"
                );
                // Top-left block (r1, c1) equals block1.
                for r in 0..r1 {
                    for c in 0..c1 {
                        let got = out_vals[r * total_cols + c];
                        let expected = block1[r * c1 + c];
                        assert!(
                            (got - expected).abs() < 1e-12,
                            "block_diag top-left[{r},{c}] = {got}, expected {expected}"
                        );
                    }
                }
                // Bottom-right block.
                for r in 0..r2 {
                    for c in 0..c2 {
                        let got = out_vals[(r1 + r) * total_cols + c1 + c];
                        let expected = block2[r * c2 + c];
                        assert!(
                            (got - expected).abs() < 1e-12,
                            "block_diag bottom-right[{r},{c}] = {got}, expected {expected}"
                        );
                    }
                }
                // Off-block cells (top-right and bottom-left) are 0.
                for r in 0..r1 {
                    for c in c1..(c1 + c2) {
                        assert_eq!(
                            out_vals[r * total_cols + c],
                            0.0,
                            "block_diag off-block top-right not zero"
                        );
                    }
                }
            }
        }
    }
});
