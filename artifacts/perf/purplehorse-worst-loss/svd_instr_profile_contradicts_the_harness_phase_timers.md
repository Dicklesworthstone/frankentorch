# Follow-up to the instruction-grounded SVD measurement: my fixture worry is REFUTED, but the profile CONTRADICTS the harness's phase timers

Two results, one closing a doubt I raised about my own work and one opening a question about
an attribution the campaign has been trusting.

## 1. The fixture-sensitivity worry is REFUTED

I flagged that my LCG fixture might drive extra bidiagonal-QR iterations, which would make the
4.48x instruction ratio a property of the bench input rather than of the code — the "only pays
on the bench input" failure, arriving from the other direction. Tested by varying the diagonal
bump over a wide range, per-iteration instructions at n=512... measured at n=256 for time
budget:

    bump=0    379,660,526
    bump=4    378,630,830     0.3% from bump=0
    bump=64   434,143,999    +14.6%

Total spread 1.147x across a fixture range that takes the matrix from no diagonal bump to
strongly diagonally dominant, and the count goes UP with dominance, not down. The instruction
count is not fixture-sensitive in the way I feared, so the aggregate FT-vs-PyTorch instruction
ratio (4.484x at n=512, 5.046x at n=1024) is robust to this choice and stands.

## 2. The instruction profile contradicts the harness's wall-phase attribution. UNRESOLVED.

`perf record -e instructions` on the same binary and fixture, one thread, n=512:

    78.16%  ft_kernel_cpu::svd_bidiag_qr_f64::{closure#0}     <- bidiagonal QR sweep
     7.79%  matrixmultiply::dgemm_kernel::kernel_target_fma
     3.26%  ft_kernel_cpu::bidiag::dot_rows_into_f64
     2.82%  ft_kernel_cpu::bidiag::bidiag_blocked_f64
     2.09%  ft_kernel_cpu::bidiag::reduce_scaled_rows_f64

The h2h harness reports, for the same op, `reduction 69% / form_p-q 30% / QR sweep 0.270 ms 0%`,
and its timers sum to the whole lane. **A phase cannot be 0.41% of wall time and 78% of retired
instructions.** Supporting oddity: this driver takes 282 ms/SVD single-threaded against the
harness's 65.2 ms at 8 threads, a gap too large for parallelism alone.

I do not know which reading is wrong, and I am not asserting a defect in either. The two
candidate explanations, and how to tell them apart:

* **LTO inlining distorts the symbol attribution.** With `lto = true` and
  `codegen-units = 1`, code from other phases can be inlined into `svd_bidiag_qr_f64`'s
  closure, which would then "own" instructions it did not write. Discriminate by profiling a
  build with `-C lto=off -C inline-threshold=0`, or by reading `perf annotate` on that symbol
  and checking whether the hot instructions are Givens rotations or reduction code.
* **The harness's QR-sweep timer is mis-scoped**, measuring a narrower region than the sweep.
  Discriminate by reading what the timer wraps, or by the sentinel method AGENTS.md prescribes:
  poison the sweep's return and see which reported number moves.

Until one of those is done, **the PHASE-level interpretation of the instruction profile is not
safe to build on**, though the aggregate FT-vs-PyTorch ratio is — that comparison never used
the phase split.

## 3. What this does to the earlier artifact's wording

`svd_instruction_grounded_the_loss_is_work_per_instruction.md` says "It is not an
algorithmic-work problem", resting on the measured 1.46x FLOP ratio. The FLOP ratio is a direct
counter reading and stands. The SENTENCE is too strong, because an algorithmic difference can
show up as instruction DENSITY rather than FLOP count: `torch.linalg.svd` dispatches to LAPACK
`gesdd` (divide-and-conquer, GEMM-heavy) while ours is Golub-Reinsch implicit-QR, whose scalar
Givens rotations are inherently low FLOPs-per-instruction. That is an algorithmic difference
that the FLOP ratio cannot see. Read the earlier claim as "not an excess-FLOPs problem", which
is what was actually measured.

## 4. The lead this leaves

If the 78% survives the inlining check, then `dot_rows_into_f64` — the target of item 254, item
258d and `4zjaa` — is **3.26%** of instructions, and the bidiagonal QR sweep is where the op
actually lives. That would redirect the campaign more sharply than anything in the wall-clock
record, and it is cheap to settle.
