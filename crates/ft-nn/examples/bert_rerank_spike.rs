//! Spike: pure-Rust frankentorch forward pass of the cross-encoder/ms-marco-MiniLM-L6-v2
//! BERT reranker, validated against numpy/ONNX reference logits.
//! Run: CARGO_HOME=$HOME/.cargo-ee-build CARGO_TARGET_DIR=$HOME/ee-build.noindex \
//!      TMPDIR=/private/tmp RCH_DISABLED=1 RCH_CARGO_WRAPPER_BYPASS=1 \
//!      command cargo run -p ft-nn --example bert_rerank_spike

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;
use std::collections::HashMap;

const MODEL: &str = "/private/tmp/ee-reranker-port/model/model_f32.safetensors";
const H: usize = 384;
const L: usize = 6;
const NH: usize = 12;
const HD: usize = H / NH; // 32
const INTER: usize = 1536;
const EPS: f64 = 1e-12;

struct Bert {
    s: FrankenTorchSession,
    w: HashMap<String, TensorNodeId>,
}

impl Bert {
    fn g(&self, name: &str) -> TensorNodeId {
        *self
            .w
            .get(name)
            .unwrap_or_else(|| panic!("missing weight {name}"))
    }

    // y = x @ Wᵀ + b   (weight stored [out,in])
    fn linear(&mut self, x: TensorNodeId, prefix: &str) -> TensorNodeId {
        let w = self.g(&format!("{prefix}.weight"));
        let b = self.g(&format!("{prefix}.bias"));
        let wt = self.s.tensor_transpose(w, 0, 1).unwrap();
        let y = self.s.tensor_matmul(x, wt).unwrap();
        self.s.tensor_add(y, b).unwrap()
    }

    fn ln(&mut self, x: TensorNodeId, prefix: &str) -> TensorNodeId {
        let w = self.g(&format!("{prefix}.weight"));
        let b = self.g(&format!("{prefix}.bias"));
        self.s
            .tensor_layer_norm(x, vec![H], Some(w), Some(b), EPS)
            .unwrap()
    }

    fn idx(&mut self, vals: &[i64]) -> TensorNodeId {
        let f: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        self.s.tensor_variable(f, vec![vals.len()], false).unwrap()
    }

    fn forward(&mut self, ids: &[i64], typ: &[i64]) -> f64 {
        let s_len = ids.len();
        // ---- embeddings: word + position + token_type, then LayerNorm ----
        let id_t = self.idx(ids);
        let pos: Vec<i64> = (0..s_len as i64).collect();
        let pos_t = self.idx(&pos);
        let typ_t = self.idx(typ);
        let we = self.g("bert.embeddings.word_embeddings.weight");
        let pe = self.g("bert.embeddings.position_embeddings.weight");
        let te = self.g("bert.embeddings.token_type_embeddings.weight");
        let e_word = self.s.tensor_embedding(id_t, we, None).unwrap();
        let e_pos = self.s.tensor_embedding(pos_t, pe, None).unwrap();
        let e_typ = self.s.tensor_embedding(typ_t, te, None).unwrap();
        let mut emb = self.s.tensor_add(e_word, e_pos).unwrap();
        emb = self.s.tensor_add(emb, e_typ).unwrap();
        emb = self.ln(emb, "bert.embeddings.LayerNorm");

        let scale = 1.0 / (HD as f64).sqrt();
        for i in 0..L {
            let p = format!("bert.encoder.layer.{i}");
            // self-attention
            let q = self.linear(emb, &format!("{p}.attention.self.query"));
            let k = self.linear(emb, &format!("{p}.attention.self.key"));
            let v = self.linear(emb, &format!("{p}.attention.self.value"));
            // [S,H] -> [S,NH,HD] -> [NH,S,HD]
            let q = self.heads(q, s_len);
            let k = self.heads(k, s_len);
            let v = self.heads(v, s_len);
            let kt = self.s.tensor_transpose(k, 1, 2).unwrap(); // [NH,HD,S]
            let mut scores = self.s.tensor_bmm(q, kt).unwrap(); // [NH,S,S]
            scores = self.s.tensor_mul_scalar(scores, scale).unwrap();
            let probs = self.s.tensor_softmax(scores, 2).unwrap();
            let ctx = self.s.tensor_bmm(probs, v).unwrap(); // [NH,S,HD]
            let ctx = self.s.tensor_transpose(ctx, 0, 1).unwrap(); // [S,NH,HD]
            let ctx = self.s.tensor_reshape(ctx, vec![s_len, H]).unwrap();
            let attn = self.linear(ctx, &format!("{p}.attention.output.dense"));
            let sum1 = self.s.tensor_add(emb, attn).unwrap();
            emb = self.ln(sum1, &format!("{p}.attention.output.LayerNorm"));
            // feed-forward
            let inter = self.linear(emb, &format!("{p}.intermediate.dense"));
            let inter = self.s.tensor_gelu(inter).unwrap();
            let ffn = self.linear(inter, &format!("{p}.output.dense"));
            let sum2 = self.s.tensor_add(emb, ffn).unwrap();
            emb = self.ln(sum2, &format!("{p}.output.LayerNorm"));
        }
        // pooler on [CLS] (row 0) + classifier
        let cls = self.s.tensor_narrow(emb, 0, 0, 1).unwrap(); // [1,H]
        let pooled = self.linear(cls, "bert.pooler.dense");
        let pooled = self.s.tensor_tanh(pooled).unwrap();
        let logit_t = self.linear(pooled, "classifier"); // [1,1]
        self.s.tensor_values(logit_t).unwrap()[0]
    }

    // [S,H] -> [NH,S,HD]
    fn heads(&mut self, x: TensorNodeId, s_len: usize) -> TensorNodeId {
        let r = self.s.tensor_reshape(x, vec![s_len, NH, HD]).unwrap();
        self.s.tensor_transpose(r, 0, 1).unwrap()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    s.no_grad_enter();
    // ---- load f32 safetensors -> f64 leaves ----
    let tensors = ft_serialize::load_safetensors(MODEL).expect("load safetensors");
    let mut w = HashMap::new();
    for (name, dense) in tensors {
        if name.contains("position_ids") {
            continue; // I64 buffer, unused
        }
        let tmp = s.tensor_variable_from_storage(dense, false);
        let shape = s.tensor_shape(tmp).unwrap();
        let vals: Vec<f64> = s
            .tensor_values_f32(tmp)
            .unwrap()
            .into_iter()
            .map(|x| x as f64)
            .collect();
        let node = s.tensor_variable(vals, shape, false).unwrap();
        w.insert(name, node);
    }
    eprintln!("loaded {} weight tensors", w.len());
    let mut m = Bert { s, w };

    // 4 parity cases (ids, token_type_ids, ref_logit)
    let cases: Vec<(Vec<i64>, Vec<i64>, f64)> = vec![
        (
            vec![101,2129,2000,8081,1037,7989,2713,2147,12314,102,1996,2713,13117,16473,2892,4132,8026,12086,1998,2039,11066,2015,2068,2000,21025,2705,12083,102],
            vec![0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            -9.808567,
        ),
        (
            vec![101,2129,2000,8081,1037,7989,2713,2147,12314,102,26191,2024,1037,2204,3120,1997,18044,1998,5510,4086,102],
            vec![0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],
            -11.332987,
        ),
        (
            vec![101,2054,2003,1996,3007,1997,2605,102,3000,2003,1996,3007,1998,2087,20151,2103,1997,2605,102],
            vec![0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],
            7.472003,
        ),
        (
            vec![101,18399,3638,3808,102,1996,17781,4638,2121,16306,2015,6095,3513,2012,4012,22090,2051,102],
            vec![0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
            -11.367251,
        ),
    ];

    println!("idx |    ft_logit |   ref_logit |     diff | ft_score");
    let mut ft_logits = Vec::new();
    let mut max_diff = 0.0f64;
    for (i, (ids, typ, refl)) in cases.iter().enumerate() {
        let logit = m.forward(ids, typ);
        let diff = (logit - refl).abs();
        max_diff = max_diff.max(diff);
        ft_logits.push(logit);
        println!("{i:3} | {logit:11.6} | {refl:11.6} | {diff:8.5} | {:.6}", sigmoid(logit));
    }

    // ranking check (descending by logit)
    let mut order: Vec<usize> = (0..ft_logits.len()).collect();
    order.sort_by(|&a, &b| ft_logits[b].partial_cmp(&ft_logits[a]).unwrap());
    let ref_order = vec![2usize, 0, 1, 3];
    let ranking_ok = order == ref_order;
    println!("\nft ranking (desc): {order:?}   ref: {ref_order:?}   ranking_ok={ranking_ok}");
    println!("max_abs_logit_diff = {max_diff:.6}");
    let pass = max_diff < 0.05 && ranking_ok;
    println!("\nPARITY: {}", if pass { "PASS" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}
