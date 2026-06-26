//! Contention-ROBUST same-process A/B for the loss no-grad fast paths (NO torch — runs on any
//! remote worker). Compares each loss's NEW API fast path against a FAITHFUL reconstruction of
//! the OLD composed path (the literal pre-fix code, via public ops), in ONE process. The
//! old/new ratio cancels worker contention; the `cat` ANCHOR sanity-checks the worker.
//! Run: cargo run --release -p ft-api --example loss_ab

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn best<F: FnMut() -> ()>(mut f: F, n: usize) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..n { let t = Instant::now(); f(); let e = t.elapsed().as_secs_f64()*1e3; if e<b {b=e;} }
    b
}

fn main() {
    let xd: Vec<f64> = (0..R*C).map(|i| 0.1 + (i%17) as f64 * 0.04).collect();
    let td: Vec<f64> = (0..R*C).map(|i| 0.1 + (i%13) as f64 * 0.05).collect();
    let tgn: Vec<f64> = (0..R*C).map(|i| if i%2==0 {1.0} else {-1.0}).collect();

    let mk = |s: &mut FrankenTorchSession, t: &[f64]| {
        let x = s.tensor_variable(xd.clone(), vec![R,C], false).unwrap();
        let y = s.tensor_variable(t.to_vec(), vec![R,C], false).unwrap();
        (x, y)
    };

    // soft_margin: OLD = mul+neg+exp+full+add+log ; NEW = API
    let sm_new = best(|| { let mut s=FrankenTorchSession::new(ExecutionMode::Strict); let (x,y)=mk(&mut s,&td);
        let _ = s.tensor_soft_margin_loss(x,y,"none").unwrap(); }, 7);
    let sm_old = best(|| { let mut s=FrankenTorchSession::new(ExecutionMode::Strict); let (x,y)=mk(&mut s,&td);
        let prod=s.tensor_mul(y,x).unwrap(); let neg=s.tensor_neg(prod).unwrap(); let e=s.tensor_exp(neg).unwrap();
        let ones=s.full(vec![R,C],1.0,false).unwrap(); let ope=s.tensor_add(ones,e).unwrap(); let _=s.tensor_log(ope).unwrap(); }, 7);

    // kl_div (log_target=false): OLD = composed exp/log+sub+mul ; NEW = API
    let kl_new = best(|| { let mut s=FrankenTorchSession::new(ExecutionMode::Strict); let (x,y)=mk(&mut s,&td);
        let _ = s.tensor_kl_div(x,y,"none",false).unwrap(); }, 7);
    let kl_old = best(|| { let mut s=FrankenTorchSession::new(ExecutionMode::Strict); let (x,y)=mk(&mut s,&td);
        let logt=s.tensor_log(y).unwrap(); let diff=s.tensor_sub(logt,x).unwrap(); let _=s.tensor_mul(y,diff).unwrap(); }, 7);

    // hinge: OLD = full x3 + sub + maximum + eq + where ; NEW = API
    let h_new = best(|| { let mut s=FrankenTorchSession::new(ExecutionMode::Strict); let (x,y)=mk(&mut s,&tgn);
        let _ = s.tensor_hinge_embedding_loss(x,y,1.0,"none").unwrap(); }, 7);
    let h_old = best(|| { let mut s=FrankenTorchSession::new(ExecutionMode::Strict); let (x,y)=mk(&mut s,&tgn);
        let zeros=s.full(vec![R,C],0.0,false).unwrap(); let margin_t=s.full(vec![R,C],1.0,false).unwrap();
        let ones=s.full(vec![R,C],1.0,false).unwrap(); let mmi=s.tensor_sub(margin_t,x).unwrap();
        let hinge=s.tensor_maximum(zeros,mmi).unwrap(); let mp=s.tensor_eq(y,ones).unwrap(); let _=s.tensor_where(mp,x,hinge).unwrap(); }, 7);

    // anchor
    let anc: Vec<f64> = (0..R*C).map(|i| (i%7) as f64).collect();
    let cat = best(|| { let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
        let x=s.tensor_variable(anc.clone(),vec![R,C],false).unwrap(); let _=s.tensor_cat(&[x,x],1).unwrap(); }, 7);

    println!("same-process A/B (OLD composed vs NEW fast path), threads={}", rayon::current_num_threads());
    println!("  soft_margin  OLD {sm_old:8.3}  NEW {sm_new:8.3}  speedup {:.2}x", sm_old/sm_new);
    println!("  kl_div       OLD {kl_old:8.3}  NEW {kl_new:8.3}  speedup {:.2}x", kl_old/kl_new);
    println!("  hinge        OLD {h_old:8.3}  NEW {h_new:8.3}  speedup {:.2}x", h_old/h_new);
    println!("  cat_anchor   {cat:8.3} ms (worker health)");
}
