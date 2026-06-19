//! Same-worker A/B over the blocked-QR panel width NB (frankentorch-ct2yy).
//! NB=32 is the ANCHOR (current production). All variants run in ONE process so
//! a contended worker shows up as the anchor regressing too.
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example qr_nb_ab

use ft_kernel_cpu::qr_householder_panel_blocked_nb_ab;
use std::time::Instant;

fn lcg(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    let mut x: u64 = 0x9E3779B97F4A7C15;
    for slot in a.iter_mut() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *slot = (x >> 11) as f64 / 9007199254740992.0 * 2.0 - 1.0;
    }
    a
}

fn time_nb(a: &[f64], n: usize, nb: usize, it: usize) -> f64 {
    // warm
    let mut r = a.to_vec();
    let _ = qr_householder_panel_blocked_nb_ab(&mut r, n, n, n, n, nb);
    let t = Instant::now();
    for _ in 0..it {
        let mut r = a.to_vec();
        let _ = qr_householder_panel_blocked_nb_ab(&mut r, n, n, n, n, nb);
    }
    t.elapsed().as_secs_f64() * 1e3 / it as f64
}

fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[512usize, 1024] {
        let a = lcg(n);
        let it = if n <= 512 { 6 } else { 3 };
        let anchor = time_nb(&a, n, 32, it); // production NB
        print!("n={n:5} anchor(NB=32)={anchor:8.2}ms");
        for &nb in &[48usize, 64, 96, 128] {
            let t = time_nb(&a, n, nb, it);
            print!("  NB={nb}={t:7.2}ms({:.2}x)", anchor / t);
        }
        println!();
    }
}
