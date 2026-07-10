//! Report the COMPILE baseline vs the WORKER capability — the ISA-baseline check.
//! `cfg!(target_feature=…)` is what the build emitted; `is_x86_feature_detected!` is what
//! the CPU running this actually supports. A gap means the build leaves ISA on the table.
fn main() {
    macro_rules! c { ($f:literal) => { println!("  compile target_feature {:<10} = {}", $f, cfg!(target_feature = $f)); } }
    println!("host={}", std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim());
    println!("COMPILE baseline (what this binary emits):");
    c!("sse2"); c!("avx"); c!("avx2"); c!("fma"); c!("bmi2"); c!("avx512f"); c!("avx512vnni");
    #[cfg(target_arch = "x86_64")]
    {
        println!("WORKER capability (what the CPU supports at runtime):");
        for f in ["sse2","avx","avx2","fma","bmi2","avx512f","avx512vnni","avx512bw"] {
            let ok = match f {
                "sse2" => is_x86_feature_detected!("sse2"),
                "avx" => is_x86_feature_detected!("avx"),
                "avx2" => is_x86_feature_detected!("avx2"),
                "fma" => is_x86_feature_detected!("fma"),
                "bmi2" => is_x86_feature_detected!("bmi2"),
                "avx512f" => is_x86_feature_detected!("avx512f"),
                "avx512vnni" => is_x86_feature_detected!("avx512vnni"),
                "avx512bw" => is_x86_feature_detected!("avx512bw"),
                _ => false,
            };
            println!("  runtime {:<12} = {ok}", f);
        }
    }
}
