//! Golden-fixture generator for the full-vector `eigh` perf investigation
//! (`frankentorch-ncwz`). Prints bit patterns for a fixed symmetric matrix so
//! exact behavior can be pinned while the implementation changes memory access.
//!
//!   rch exec -- cargo run -p ft-kernel-cpu --example eigh_golden

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{eigh_contiguous_f64, eigvalsh_contiguous_f64};

fn main() {
    let n = 8usize;
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let bij = ((i * 37 + j * 19 + 5) % 97) as f64 * 0.017 - 0.7;
            let bji = ((j * 37 + i * 19 + 5) % 97) as f64 * 0.017 - 0.7;
            a[i * n + j] = 0.5 * (bij + bji);
        }
        a[i * n + i] += n as f64;
    }

    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
    if std::env::var_os("FT_EIGVALSH_GOLDEN").is_some() {
        let r = match eigvalsh_contiguous_f64(&a, &meta) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("eigvalsh_golden failed: {err}");
                std::process::exit(1);
            }
        };

        println!("frankentorch-a9ry eigvalsh_golden");
        println!("n={n}");
        println!("eigenvalue_bits:");
        for (idx, v) in r.iter().enumerate() {
            println!("{idx}: {:#018x}", v.to_bits());
        }
        return;
    }

    let r = match eigh_contiguous_f64(&a, &meta) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("eigh_golden failed: {err}");
            std::process::exit(1);
        }
    };

    println!("frankentorch-ncwz eigh_full_golden");
    println!("n={}", r.n);
    println!("eigenvalue_bits:");
    for (idx, v) in r.eigenvalues.iter().enumerate() {
        println!("{idx}: {:#018x}", v.to_bits());
    }
    println!("eigenvector_bits:");
    for (idx, v) in r.eigenvectors.iter().enumerate() {
        println!("{idx}: {:#018x}", v.to_bits());
    }
}
