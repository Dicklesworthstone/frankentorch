# frankentorch-05upk — Arc-share leaf grad: exact paste-ready implementation plan

BlackThrush, 2026-06-21 (disk-low code-only turn — design fully specified here; APPLY +
VERIFY when disk recovers, do NOT commit to source blind — this lever is a core-public-type
change that MUST be compiler-verified before merge).

## Goal

Eliminate the per-backward LEAF-grad `to_vec` clone. Today `TensorBackwardReport.gradients`
AND `persistent_grads` each independently own every leaf grad → `accumulate_persistent_gradients`
clones (`gradient.to_vec()`) on first store. Make both hold `Arc<Vec<f64>>` so the first-backward
store is an `Arc::clone` (refcount bump, no numel copy). Accumulation across backwards uses
`Arc::make_mut` (clones only if a prior report is still alive — rare; otherwise in-place).

Why safe: the report is READ-ONLY on `gradients` (no mutation path through it), so sharing the
buffer is sound. The optimizer read path (`update_tensor_values_with_accumulated_gradient`) is
read-only on persistent. `make_mut` gives correct results in both shared (snapshot-preserving
clone) and unshared (in-place) cases.

Win: noise-buried on the gauntlet (1 leaf), but scales with #leaf-params per training step
(eliminates N param-grad clones/step — what PyTorch's caching allocator avoids).

## Preconditions
- `use std::sync::Arc;` already imported (backward closures use it).
- ft-nn must NOT be mid-rewrite (2 test callers of `gradients()` need a 1-line adapt each);
  as of this writing ft-nn carries a static 767-line peer WIP — wait for it to land.

## Exact edits (ft-autograd/src/lib.rs unless noted)

### 1. Field types
- `persistent_grads: BTreeMap<usize, Vec<f64>>`  →  `BTreeMap<usize, Arc<Vec<f64>>>`
- `struct TensorBackwardReport { gradients: Vec<Option<Vec<f64>>>, ... }`
  → `gradients: Vec<Option<Arc<Vec<f64>>>>`

### 2. Report methods
- `gradient(&self, node) -> Option<&[f64]>`:
  `self.gradients.get(node.0).and_then(|e| e.as_ref().map(|a| a.as_slice()))`
  (was `.and_then(|e| e.as_deref())` — `as_deref` on `Option<Arc<Vec>>` yields `Option<&Vec>`,
   so switch to `.as_ref().map(|a| a.as_slice())`).
- `gradients(&self) -> &[Option<Arc<Vec<f64>>>]` (return type changes; body `&self.gradients`).
- `gradient_value`: UNCHANGED (calls `self.gradient(node)` → still `&[f64]`, `.to_vec()`).
- `tensor_gradients_iter`: `.map(|(i, opt)| (i, opt.as_ref().map(|a| a.as_slice())))`.
- `scaled_clone`: build new Arcs —
  `let scaled_gradients: Vec<Option<Arc<Vec<f64>>>> = self.gradients.iter()
     .map(|opt| opt.as_ref().map(|g| Arc::new(g.iter().map(|&x| x*factor).collect::<Vec<f64>>())))
     .collect();`
  (sparse_gradients / gradient_nodes / steps / telemetry clones unchanged.)

### 3. Backward gradients build sites (wrap in Arc::new)
- first-order (~14416): `Some(grad.values)` → `Some(Arc::new(grad.values))`.
- create_graph (~17185): `gradients.push(Some(vals))` → `gradients.push(Some(Arc::new(vals)))`.

### 4. accumulate_persistent_gradients (first-order, ~19800)
- Signature: `gradients: &[Option<Arc<Vec<f64>>>]`.
- Per leaf/retain entry (after the pwjrs is_leaf||retain gate):
  ```
  match self.persistent_grads.get_mut(&idx) {
      Some(existing) => {
          let target = Arc::make_mut(existing);
          Self::accumulate_existing_tensor_gradient(node, target, gradient)?; // gradient: &[f64] via deref
      }
      None => { self.persistent_grads.insert(idx, Arc::clone(arc)); } // arc = the report's Some(Arc) — NO to_vec
  }
  ```
  NOTE: the loop must access the report's `Option<Arc<Vec<f64>>>` (not `&[f64]`) so it can
  `Arc::clone`. Iterate `gradients.iter().enumerate()` and match `Some(arc)`; use
  `arc.as_slice()` for the accumulate path and `Arc::clone(arc)` for the insert path.

### 5. create_graph persist loop (~17240, the is_leaf||retain block)
- `.and_modify(|existing| { let t = Arc::make_mut(existing); for (e,v) in t.iter_mut().zip(vals.iter()){*e+=v;} })
   .or_insert_with(|| Arc::new(vals))` (vals is the freshly-read Vec<f64>; wrap in Arc).

### 6. persistent reader/writer sites
- `tensor_accumulated_gradient` (~4531): `.map(Vec::as_slice)` → `.map(|a| a.as_slice())`.
- `tensor_accumulated_gradient_len` (~4548): `.map(Vec::len)` → `.map(|a| a.len())`.
- `zero_tensor_accumulated_gradient` (~4556): `if let Some(grad) = get_mut { Arc::make_mut(grad).fill(0.0); }`.
- `set_tensor_accumulated_gradient` (~4569): `insert(node.0, Arc::new(gradient))`.
- `update_tensor_values_with_accumulated_gradient` (~19881): `let gradient = self.persistent_grads.get(&id.0)` →
  use `gradient.as_slice()` where the closure/len-check expects `&[f64]` (read-only — no make_mut).

### 7. external callers of `gradients()` (return type changed)
- ft-autograd tests ~21184/21387: `assert_eq!(a.gradients(), b.gradients())` — Arc<Vec<f64>>: PartialEq
  compares contents → still passes, NO change needed.
- ft-nn tests ~28579/28582: `report.gradients().len()` (fine) and
  `for (i, g) in report.gradients().iter().enumerate()` where `g: &Option<Arc<Vec<f64>>>` — adapt the
  loop body's use of `g` (e.g. `g.as_ref().map(|a| a.len())`). 1-line each. DO ONLY when ft-nn not mid-rewrite.

## Verification (MANDATORY before merge — full workspace)
`cargo test -p ft-autograd` (incl GradScaler/scaled_clone, retain_graph, autograd_grad, double-backward),
`-p ft-api` (optimizers + IndexSelect sparse + the 2 known pre-existing reds expected),
`-p ft-conformance`, `cargo clippy -p ft-autograd -- -D warnings`. Expect BIT-EXACT (Arc share/make_mut
preserve values exactly). If any compile/test fails, this is a core path — fix or revert, do not force.
