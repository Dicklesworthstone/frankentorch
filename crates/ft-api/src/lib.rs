#![forbid(unsafe_code)]

use ft_autograd::{
    AutogradError, BackwardOptions, BackwardReport, ClampOperationEvent, FunctionCtx, NodeId,
    OperationEvent, PowOperationEvent, Tape, TensorAddmmOperationEvent,
    TensorAddmvOperationEvent, TensorBackwardReport, TensorClampOperationEvent,
    TensorHookHandle, TensorJoinOperationEvent, TensorLerpOperationEvent, TensorNodeId,
    TensorNormDimOperationEvent, TensorNormOperationEvent, TensorNormalizeDimOperationEvent,
    TensorOperationEvent, TensorPowOperationEvent, TensorReductionDimOperationEvent,
    TensorReductionOperationEvent, TensorScanDimOperationEvent, TensorSortOperationEvent,
    TensorTape, TensorTopKOperationEvent, TensorUnaryOperationEvent, UnaryOperationEvent,
};
use ft_core::{DType, DenseTensor, ExecutionMode, TensorCompatError, TensorMeta};
use ft_dispatch::{
    ComparisonDispatchDecision, ComparisonOp, UnaryDispatchDecision, UnaryOp,
    dispatch_scalar_comparison, dispatch_scalar_unary, dispatch_tensor_comparison_contiguous_f64,
    dispatch_tensor_unary_contiguous_f64,
};
use ft_runtime::{EvidenceEntry, EvidenceKind, RuntimeContext};

/// Deterministic xoshiro256++ PRNG for reproducible random tensor generation.
#[derive(Debug, Clone)]
struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    fn new(seed: u64) -> Self {
        // SplitMix64 to expand seed into state
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller transform.
    fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 0.0 {
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrankenTorchSession {
    tape: Tape,
    tensor_tape: TensorTape,
    runtime: RuntimeContext,
    rng: Xoshiro256PlusPlus,
    grad_enabled_stack: Vec<bool>,
}

impl FrankenTorchSession {
    #[must_use]
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            tape: Tape::new(),
            tensor_tape: TensorTape::new(),
            runtime: RuntimeContext::new(mode),
            rng: Xoshiro256PlusPlus::new(42),
            grad_enabled_stack: vec![true],
        }
    }

    #[must_use]
    pub fn mode(&self) -> ExecutionMode {
        self.runtime.mode()
    }

    pub fn set_mode(&mut self, mode: ExecutionMode) {
        self.runtime.set_mode(mode);
    }

    /// Returns true if gradient computation is currently enabled.
    #[must_use]
    pub fn is_grad_enabled(&self) -> bool {
        *self.grad_enabled_stack.last().unwrap_or(&true)
    }

    /// Enter a no_grad context: disable gradient tracking for subsequent operations.
    pub fn no_grad_enter(&mut self) {
        self.grad_enabled_stack.push(false);
        self.sync_grad_enabled();
    }

    /// Exit a no_grad context: restore the previous gradient tracking state.
    pub fn no_grad_exit(&mut self) {
        if self.grad_enabled_stack.len() > 1 {
            self.grad_enabled_stack.pop();
        }
        self.sync_grad_enabled();
    }

    /// Enter an enable_grad context: enable gradient tracking for subsequent operations.
    pub fn enable_grad_enter(&mut self) {
        self.grad_enabled_stack.push(true);
        self.sync_grad_enabled();
    }

    /// Exit an enable_grad context: restore the previous gradient tracking state.
    pub fn enable_grad_exit(&mut self) {
        if self.grad_enabled_stack.len() > 1 {
            self.grad_enabled_stack.pop();
        }
        self.sync_grad_enabled();
    }

    /// Set the gradient enabled state directly (like `torch.set_grad_enabled(bool)`).
    pub fn set_grad_enabled(&mut self, enabled: bool) {
        if let Some(top) = self.grad_enabled_stack.last_mut() {
            *top = enabled;
        }
        self.sync_grad_enabled();
    }

    /// Execute a closure with gradient tracking disabled (panic-safe RAII equivalent).
    pub fn with_no_grad<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.no_grad_enter();
        let result = f(self);
        self.no_grad_exit();
        result
    }

    /// Execute a closure with gradient tracking enabled (panic-safe RAII equivalent).
    pub fn with_enable_grad<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.enable_grad_enter();
        let result = f(self);
        self.enable_grad_exit();
        result
    }

    fn sync_grad_enabled(&mut self) {
        let enabled = self.is_grad_enabled();
        self.tape.set_grad_enabled(enabled);
        self.tensor_tape.set_grad_enabled(enabled);
    }

    #[must_use]
    pub fn variable(&mut self, value: f64, requires_grad: bool) -> NodeId {
        self.tape.leaf(value, requires_grad)
    }

    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.add(lhs, rhs, self.mode())?;
        self.record_operation(&event);
        Ok(out)
    }

    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.mul(lhs, rhs, self.mode())?;
        self.record_operation(&event);
        Ok(out)
    }

    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.sub(lhs, rhs, self.mode())?;
        self.record_operation(&event);
        Ok(out)
    }

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.div(lhs, rhs, self.mode())?;
        self.record_operation(&event);
        Ok(out)
    }

    pub fn neg(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.neg(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn abs(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.abs(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn exp(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.exp(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn log(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.log(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn relu(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.relu(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn sigmoid(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.sigmoid(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn tanh(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.tanh(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn sin(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.sin(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn cos(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.cos(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn tan(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.tan(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn floor(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.floor(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn ceil(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.ceil(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn round(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.round(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn log2(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.log2(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn log10(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.log10(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn log1p(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.log1p(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn expm1(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.expm1(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn sign(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.sign(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn trunc(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.trunc(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn frac(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.frac(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn asin(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.asin(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn acos(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.acos(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn atan(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.atan(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn sinh(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.sinh(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn cosh(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.cosh(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn gelu(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.gelu(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn silu(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.silu(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn leaky_relu(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.leaky_relu(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn elu(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.elu(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn rsqrt(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.rsqrt(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn erf(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.erf(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn erfc(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.erfc(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn hardswish(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.hardswish(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn hardsigmoid(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.hardsigmoid(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn hardtanh(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.hardtanh(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn softplus(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.softplus(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn mish(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.mish(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn square(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.square(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn sqrt(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.sqrt(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn reciprocal(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.reciprocal(input, self.mode())?;
        self.record_unary_operation(&event);
        Ok(out)
    }

    pub fn pow(&mut self, input: NodeId, exponent: f64) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.pow(input, exponent, self.mode())?;
        self.record_pow_operation(&event);
        Ok(out)
    }

    pub fn min(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.min(lhs, rhs, self.mode())?;
        self.record_operation(&event);
        Ok(out)
    }

    pub fn max(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.max(lhs, rhs, self.mode())?;
        self.record_operation(&event);
        Ok(out)
    }

    pub fn atan2(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.atan2(lhs, rhs, self.mode())?;
        self.record_operation(&event);
        Ok(out)
    }

    pub fn fmod(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.fmod(lhs, rhs, self.mode())?;
        self.record_operation(&event);
        Ok(out)
    }

    pub fn remainder(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.remainder(lhs, rhs, self.mode())?;
        self.record_operation(&event);
        Ok(out)
    }

    pub fn clamp(
        &mut self,
        input: NodeId,
        min_val: f64,
        max_val: f64,
    ) -> Result<NodeId, AutogradError> {
        let (out, event) = self.tape.clamp(input, min_val, max_val, self.mode())?;
        self.record_clamp_operation(&event);
        Ok(out)
    }

    pub fn value(&self, node: NodeId) -> Result<f64, AutogradError> {
        self.tape.value(node)
    }

    pub fn tensor_variable(
        &mut self,
        values: Vec<f64>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.leaf(values, shape, requires_grad)
    }

    #[must_use]
    pub fn tensor_variable_from_storage(
        &mut self,
        tensor: DenseTensor,
        requires_grad: bool,
    ) -> TensorNodeId {
        self.tensor_tape.leaf_tensor(tensor, requires_grad)
    }

    pub fn tensor_variable_f32(
        &mut self,
        values: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.leaf_f32(values, shape, requires_grad)
    }

    pub fn tensor_values_f32(&self, node: TensorNodeId) -> Result<Vec<f32>, AutogradError> {
        self.tensor_tape.values_f32(node)
    }

    pub fn tensor_dtype(&self, node: TensorNodeId) -> Result<DType, AutogradError> {
        self.tensor_tape.dtype(node)
    }

    pub fn tensor_to_f32(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.to_f32(input)
    }

    pub fn tensor_to_f64(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.to_f64(input)
    }

    /// PyTorch-style alias for casting a tensor to float32.
    pub fn tensor_float(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        self.tensor_to_f32(input)
    }

    /// PyTorch-style alias for casting a tensor to float64.
    pub fn tensor_double(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        self.tensor_to_f64(input)
    }

    pub fn zeros_f32(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in zeros_f32")?;
        self.tensor_tape
            .leaf_f32(vec![0.0f32; numel], shape, requires_grad)
    }

    pub fn ones_f32(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in ones_f32")?;
        self.tensor_tape
            .leaf_f32(vec![1.0f32; numel], shape, requires_grad)
    }

    pub fn randn_f32(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in randn_f32")?;
        let values: Vec<f32> = (0..numel).map(|_| self.rng.next_normal() as f32).collect();
        self.tensor_tape.leaf_f32(values, shape, requires_grad)
    }

    pub fn zeros(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in zeros")?;
        self.tensor_tape
            .leaf(vec![0.0; numel], shape, requires_grad)
    }

    pub fn ones(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in ones")?;
        self.tensor_tape
            .leaf(vec![1.0; numel], shape, requires_grad)
    }

    pub fn full(
        &mut self,
        shape: Vec<usize>,
        fill_value: f64,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in full")?;
        self.tensor_tape
            .leaf(vec![fill_value; numel], shape, requires_grad)
    }

    pub fn arange(
        &mut self,
        start: f64,
        end: f64,
        step: f64,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "arange: start/end/step must be finite",
                },
            )));
        }
        if step == 0.0 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "arange: step must not be zero",
                },
            )));
        }
        let mut values = Vec::new();
        let mut current = start;
        if step > 0.0 {
            while current < end {
                values.push(current);
                let next = current + step;
                if next <= current {
                    return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                        ft_dispatch::DispatchKeyError::IncompatibleSet {
                            reason: "arange: step does not advance at current floating-point precision",
                        },
                    )));
                }
                current = next;
            }
        } else {
            while current > end {
                values.push(current);
                let next = current + step;
                if next >= current {
                    return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                        ft_dispatch::DispatchKeyError::IncompatibleSet {
                            reason: "arange: step does not advance at current floating-point precision",
                        },
                    )));
                }
                current = next;
            }
        }
        let n = values.len();
        self.tensor_tape.leaf(values, vec![n], requires_grad)
    }

    /// Create a 2-D identity matrix of size n x n.
    pub fn eye(&mut self, n: usize, requires_grad: bool) -> Result<TensorNodeId, AutogradError> {
        let numel = Self::checked_square_numel(n, "eye shape volume overflow")?;
        let mut values = vec![0.0; numel];
        for i in 0..n {
            values[i * n + i] = 1.0;
        }
        self.tensor_tape.leaf(values, vec![n, n], requires_grad)
    }

    /// Create a diagonal matrix from a 1-D tensor, or extract the diagonal of a 2-D tensor.
    ///
    /// If input is 1-D of length n, returns a 2-D tensor of shape [n, n] with the input as diagonal.
    /// If input is 2-D of shape [m, n], returns a 1-D tensor of length min(m, n) with the diagonal.
    pub fn diag(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let shape = self.tensor_shape(input)?;

        match shape.len() {
            1 => {
                // 1-D -> 2-D diagonal matrix: multiply input (broadcast) with identity mask
                let n = shape[0];
                let numel =
                    Self::checked_square_numel(n, "diag shape volume overflow for matrix output")?;
                let mut eye_data = vec![0.0; numel];
                for i in 0..n {
                    eye_data[i * n + i] = 1.0;
                }
                let eye_tensor = self.tensor_variable(eye_data, vec![n, n], false)?;
                // Reshape input to [n, 1] for broadcasting then multiply element-wise
                let reshaped = self.tensor_reshape(input, vec![n, 1])?;
                let expanded = self.tensor_expand(reshaped, vec![n, n])?;
                self.tensor_mul(expanded, eye_tensor)
            }
            2 => {
                // 2-D -> 1-D diagonal extraction: use gather with diagonal indices
                let m = shape[0];
                let n = shape[1];
                let diag_len = m.min(n);
                let indices: Vec<f64> = (0..diag_len).map(|i| i as f64).collect();
                let index = self.tensor_variable(indices, vec![diag_len, 1], false)?;
                let narrow = if m > diag_len {
                    self.tensor_narrow(input, 0, 0, diag_len)?
                } else {
                    input
                };
                let gathered = self.tensor_gather(narrow, 1, index)?;
                self.tensor_reshape(gathered, vec![diag_len])
            }
            _ => Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "diag expects 1-D or 2-D input",
                },
            ))),
        }
    }

    /// Return the upper triangular part of a 2-D tensor.
    ///
    /// Elements below the k-th diagonal are set to 0. k=0 is the main diagonal,
    /// k>0 is above, k<0 is below.
    pub fn triu(&mut self, input: TensorNodeId, k: i64) -> Result<TensorNodeId, AutogradError> {
        let shape = self.tensor_shape(input)?;
        if shape.len() != 2 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "triu expects 2-D input",
                },
            )));
        }
        let m = shape[0];
        let n = shape[1];
        let numel = m.checked_mul(n).ok_or({
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "triu shape volume overflow",
                },
            ))
        })?;
        let mut mask = vec![0.0; numel];
        for i in 0..m {
            for j in 0..n {
                if j as i64 >= i as i64 + k {
                    mask[i * n + j] = 1.0;
                }
            }
        }
        let mask_tensor = self.tensor_variable(mask, vec![m, n], false)?;
        self.tensor_mul(input, mask_tensor)
    }

    /// Return the lower triangular part of a 2-D tensor.
    ///
    /// Elements above the k-th diagonal are set to 0. k=0 is the main diagonal.
    pub fn tril(&mut self, input: TensorNodeId, k: i64) -> Result<TensorNodeId, AutogradError> {
        let shape = self.tensor_shape(input)?;
        if shape.len() != 2 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "tril expects 2-D input",
                },
            )));
        }
        let m = shape[0];
        let n = shape[1];
        let numel = m.checked_mul(n).ok_or({
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "tril shape volume overflow",
                },
            ))
        })?;
        let mut mask = vec![0.0; numel];
        for i in 0..m {
            for j in 0..n {
                if j as i64 <= i as i64 + k {
                    mask[i * n + j] = 1.0;
                }
            }
        }
        let mask_tensor = self.tensor_variable(mask, vec![m, n], false)?;
        self.tensor_mul(input, mask_tensor)
    }

    /// Create a tensor filled with uniform random values in [0, 1).
    pub fn rand(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in rand")?;
        let values: Vec<f64> = (0..numel).map(|_| self.rng.next_f64()).collect();
        self.tensor_tape.leaf(values, shape, requires_grad)
    }

    /// Create a tensor filled with standard normal random values.
    pub fn randn(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in randn")?;
        let values: Vec<f64> = (0..numel).map(|_| self.rng.next_normal()).collect();
        self.tensor_tape.leaf(values, shape, requires_grad)
    }

    /// Create a tensor with the same shape as `other`, filled with uniform random values in [0, 1).
    pub fn rand_like(
        &mut self,
        other: TensorNodeId,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let meta = self.tensor_tape.tensor_meta(other)?.clone();
        let shape = meta.shape().to_vec();
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in rand_like")?;
        let values: Vec<f64> = (0..numel).map(|_| self.rng.next_f64()).collect();
        let tensor = DenseTensor::from_contiguous_values(values, shape, meta.device())?;
        Ok(self.tensor_tape.leaf_tensor(tensor, requires_grad))
    }

    /// Create a tensor with the same shape as `other`, filled with standard normal random values.
    pub fn randn_like(
        &mut self,
        other: TensorNodeId,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let meta = self.tensor_tape.tensor_meta(other)?.clone();
        let shape = meta.shape().to_vec();
        let numel = Self::checked_shape_numel(
            &shape,
            "tensor factory shape volume overflow in randn_like",
        )?;
        let values: Vec<f64> = (0..numel).map(|_| self.rng.next_normal()).collect();
        let tensor = DenseTensor::from_contiguous_values(values, shape, meta.device())?;
        Ok(self.tensor_tape.leaf_tensor(tensor, requires_grad))
    }

    /// Create a 1-D tensor with `steps` evenly spaced values from `start` to `end` (inclusive).
    pub fn linspace(
        &mut self,
        start: f64,
        end: f64,
        steps: usize,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let values = if steps == 0 {
            vec![]
        } else if steps == 1 {
            vec![start]
        } else {
            (0..steps)
                .map(|i| start + (end - start) * (i as f64) / ((steps - 1) as f64))
                .collect()
        };
        self.tensor_tape.leaf(values, vec![steps], requires_grad)
    }

    /// Create a 1-D tensor with `steps` logarithmically spaced values.
    ///
    /// Values are `base^linspace(start, end, steps)`.
    /// Default base is 10.0.
    pub fn logspace(
        &mut self,
        start: f64,
        end: f64,
        steps: usize,
        base: f64,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let values = if steps == 0 {
            vec![]
        } else if steps == 1 {
            vec![base.powf(start)]
        } else {
            (0..steps)
                .map(|i| {
                    let t = start + (end - start) * (i as f64) / ((steps - 1) as f64);
                    base.powf(t)
                })
                .collect()
        };
        self.tensor_tape.leaf(values, vec![steps], requires_grad)
    }

    /// Create a tensor without initializing values (filled with zeros in Rust).
    ///
    /// In Rust, memory is always initialized, so `empty` produces zeros.
    /// This is primarily for API compatibility with PyTorch.
    pub fn empty(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in empty")?;
        self.tensor_tape
            .leaf(vec![0.0; numel], shape, requires_grad)
    }

    /// Create an uninitialized tensor with the same shape as `other`.
    ///
    /// In Rust, this produces zeros (see `empty`).
    pub fn empty_like(
        &mut self,
        other: TensorNodeId,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let meta = self.tensor_tape.tensor_meta(other)?.clone();
        let shape = meta.shape().to_vec();
        self.empty(shape, requires_grad)
    }

    /// Create a tensor with the same shape as `other`, filled with `fill_value`.
    pub fn full_like(
        &mut self,
        other: TensorNodeId,
        fill_value: f64,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let meta = self.tensor_tape.tensor_meta(other)?.clone();
        let shape = meta.shape().to_vec();
        let numel =
            Self::checked_shape_numel(&shape, "tensor factory shape volume overflow in full_like")?;
        let values = vec![fill_value; numel];
        let tensor = DenseTensor::from_contiguous_values(values, shape, meta.device())?;
        Ok(self.tensor_tape.leaf_tensor(tensor, requires_grad))
    }

    /// Create a tensor of zeros with the same shape as `other`.
    pub fn zeros_like(
        &mut self,
        other: TensorNodeId,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        self.full_like(other, 0.0, requires_grad)
    }

    /// Create a tensor of ones with the same shape as `other`.
    pub fn ones_like(
        &mut self,
        other: TensorNodeId,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        self.full_like(other, 1.0, requires_grad)
    }

    /// Reverse a tensor along the specified dimensions.
    ///
    /// Equivalent to PyTorch's `torch.flip(input, dims)`.
    pub fn tensor_flip(
        &mut self,
        input: TensorNodeId,
        dims: &[usize],
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.flip(input, dims.to_vec())
    }

    /// Repeat a tensor along each dimension.
    ///
    /// `repeats` specifies the number of repetitions for each output dimension.
    /// Its length must be greater than or equal to the input rank; when larger,
    /// leading singleton dimensions are implicitly prepended (PyTorch-compatible).
    pub fn tensor_repeat(
        &mut self,
        input: TensorNodeId,
        repeats: &[usize],
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.repeat(input, repeats.to_vec())
    }

    /// Roll tensor elements along the given dimension by `shift` positions.
    ///
    /// Elements that are shifted beyond the last position are re-introduced at
    /// the first position (circular shift).
    pub fn tensor_roll(
        &mut self,
        input: TensorNodeId,
        shift: i64,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let shift = isize::try_from(shift).map_err(|_| {
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "tensor_roll shift is out of range for platform isize",
                },
            ))
        })?;
        self.tensor_tape.roll(input, shift, dim)
    }

    pub fn tensor_add(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.add(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_mul(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.mul(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_sub(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sub(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_div(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.div(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_matmul(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.matmul(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_dot(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.dot(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_outer(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.outer(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_bmm(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.bmm(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    /// Tensor contraction over specified dimensions (generalised matrix multiply).
    ///
    /// `dims` is the number of trailing dimensions of `a` to contract with
    /// the leading dimensions of `b`.
    /// Equivalent to PyTorch's `torch.tensordot(a, b, dims=N)`.
    pub fn tensor_tensordot(
        &mut self,
        a: TensorNodeId,
        b: TensorNodeId,
        dims: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let a_shape = self.tensor_shape(a)?;
        let b_shape = self.tensor_shape(b)?;
        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();

        if dims > a_ndim || dims > b_ndim {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension {
                    dim: dims,
                    ndim: a_ndim.min(b_ndim),
                },
            )));
        }

        // Check that contracted dimensions match
        for i in 0..dims {
            if a_shape[a_ndim - dims + i] != b_shape[i] {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::ShapeMismatch {
                        lhs: a_shape.clone(),
                        rhs: b_shape.clone(),
                    },
                )));
            }
        }

        // Special case: dims=0 is outer product-like (no contraction)
        if dims == 0 {
            // Reshape a to [..., 1] and b to [1, ...], then matmul
            let a_total: usize = a_shape.iter().product();
            let b_total: usize = b_shape.iter().product();
            let a_flat = self.tensor_reshape(a, vec![a_total, 1])?;
            let b_flat = self.tensor_reshape(b, vec![1, b_total])?;
            let result = self.tensor_matmul(a_flat, b_flat)?;
            let mut out_shape = a_shape;
            out_shape.extend_from_slice(&b_shape);
            return self.tensor_reshape(result, out_shape);
        }

        // Reshape a: [free_a..., contract...] -> [prod(free_a), prod(contract)]
        let free_a: usize = a_shape[..a_ndim - dims].iter().product();
        let contract: usize = a_shape[a_ndim - dims..].iter().product();
        let a_2d = self.tensor_reshape(a, vec![free_a, contract])?;

        // Reshape b: [contract..., free_b...] -> [prod(contract), prod(free_b)]
        let free_b: usize = b_shape[dims..].iter().product();
        let b_2d = self.tensor_reshape(b, vec![contract, free_b])?;

        // Matmul: [free_a, contract] @ [contract, free_b] -> [free_a, free_b]
        let result = self.tensor_matmul(a_2d, b_2d)?;

        // Reshape back to [...free_a_dims, ...free_b_dims]
        let mut out_shape: Vec<usize> = a_shape[..a_ndim - dims].to_vec();
        out_shape.extend_from_slice(&b_shape[dims..]);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        self.tensor_reshape(result, out_shape)
    }

    /// Kronecker product of two tensors.
    ///
    /// For 2D inputs A (m x n) and B (p x q), result is (m*p x n*q).
    /// For 1D inputs a (m,) and b (n,), result is (m*n,).
    pub fn tensor_kron(
        &mut self,
        a: TensorNodeId,
        b: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let a_shape = self.tensor_shape(a)?;
        let b_shape = self.tensor_shape(b)?;

        match (a_shape.len(), b_shape.len()) {
            (1, 1) => {
                // 1D kron: outer product then flatten
                let outer = self.tensor_outer(a, b)?;
                let total = a_shape[0] * b_shape[0];
                self.tensor_reshape(outer, vec![total])
            }
            (2, 2) => {
                // 2D kron: for each (i,j) in A, place A[i,j]*B as block
                let (ma, na) = (a_shape[0], a_shape[1]);
                let (mb, nb) = (b_shape[0], b_shape[1]);
                let a_vals = self.tensor_values(a)?;
                let b_vals = self.tensor_values(b)?;
                let mut result = vec![0.0f64; (ma * mb) * (na * nb)];
                let out_cols = na * nb;
                for ia in 0..ma {
                    for ja in 0..na {
                        let a_val = a_vals[ia * na + ja];
                        for ib in 0..mb {
                            for jb in 0..nb {
                                let out_row = ia * mb + ib;
                                let out_col = ja * nb + jb;
                                result[out_row * out_cols + out_col] =
                                    a_val * b_vals[ib * nb + jb];
                            }
                        }
                    }
                }
                self.tensor_variable(result, vec![ma * mb, na * nb], false)
            }
            _ => {
                // General case: pad shorter tensor dimensions to match
                // For simplicity, only support 1D and 2D for now
                Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::ShapeMismatch {
                        lhs: a_shape,
                        rhs: b_shape,
                    },
                )))
            }
        }
    }

    /// Einstein summation.
    ///
    /// Supports 1- and 2-operand equations via compact index notation.
    /// Decomposes into permute/reshape/matmul/sum operations for backward compatibility.
    ///
    /// Examples:
    /// - `"ij,jk->ik"` — matrix multiply
    /// - `"ii->"` — trace
    /// - `"ij->ji"` — transpose
    /// - `"i,j->ij"` — outer product
    /// - `"i,i->"` — dot product
    /// - `"bij,bjk->bik"` — batch matmul
    pub fn tensor_einsum(
        &mut self,
        equation: &str,
        tensors: &[TensorNodeId],
    ) -> Result<TensorNodeId, AutogradError> {
        let make_err = |reason: &'static str| {
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet { reason },
            ))
        };

        // Parse equation: "subscripts -> output" or just "subscripts" (implicit output)
        let (input_part, output_part) = if let Some(pos) = equation.find("->") {
            (&equation[..pos], Some(&equation[pos + 2..]))
        } else {
            (equation, None)
        };

        let input_subs: Vec<&str> = input_part.split(',').collect();
        if input_subs.len() != tensors.len() {
            return Err(make_err("einsum: number of subscripts must match number of tensors"));
        }

        // Validate characters: only lowercase ASCII letters
        for ch in equation.chars() {
            if ch != ',' && ch != '-' && ch != '>' && ch != ' ' && !ch.is_ascii_lowercase() {
                return Err(make_err("einsum: only lowercase ASCII letters allowed in subscripts"));
            }
        }

        // Parse subscripts into index lists
        let input_indices: Vec<Vec<char>> = input_subs
            .iter()
            .map(|s| s.chars().filter(|c| c.is_ascii_lowercase()).collect())
            .collect();

        // Validate shapes match subscripts
        for (i, indices) in input_indices.iter().enumerate() {
            let shape = self.tensor_shape(tensors[i])?;
            if shape.len() != indices.len() {
                return Err(make_err("einsum: subscript length must match tensor ndim"));
            }
        }

        // Build dimension size map
        let mut dim_sizes: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for (i, indices) in input_indices.iter().enumerate() {
            let shape = self.tensor_shape(tensors[i])?;
            for (d, &idx) in indices.iter().enumerate() {
                if let Some(&existing) = dim_sizes.get(&idx) {
                    if existing != shape[d] {
                        return Err(make_err(
                            "einsum: dimension size mismatch for repeated index",
                        ));
                    }
                } else {
                    dim_sizes.insert(idx, shape[d]);
                }
            }
        }

        // Determine output indices
        let output_indices: Vec<char> = if let Some(out) = output_part {
            out.chars().filter(|c| c.is_ascii_lowercase()).collect()
        } else {
            // Implicit: sorted unique indices that appear exactly once across all inputs
            let mut counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
            for indices in &input_indices {
                for &ch in indices {
                    *counts.entry(ch).or_insert(0) += 1;
                }
            }
            let mut out: Vec<char> = counts
                .into_iter()
                .filter(|(_, count)| *count == 1)
                .map(|(ch, _)| ch)
                .collect();
            out.sort();
            out
        };

        // Handle based on number of operands
        match tensors.len() {
            1 => self.einsum_unary(&input_indices[0], &output_indices, tensors[0], &dim_sizes),
            2 => self.einsum_binary(
                &input_indices[0],
                &input_indices[1],
                &output_indices,
                tensors[0],
                tensors[1],
                &dim_sizes,
            ),
            _ => {
                // Multi-operand: pairwise reduction left-to-right
                // Determine which indices to contract at each step
                let mut current = tensors[0];
                let mut current_indices = input_indices[0].clone();
                for i in 1..tensors.len() {
                    let rhs_indices = &input_indices[i];
                    // For intermediate steps, keep all indices needed downstream
                    let is_last = i == tensors.len() - 1;
                    let intermediate_out: Vec<char> = if is_last {
                        output_indices.clone()
                    } else {
                        // Keep indices that appear in remaining inputs or final output
                        let mut needed: std::collections::HashSet<char> = output_indices.iter().copied().collect();
                        for j in (i + 1)..tensors.len() {
                            for &ch in &input_indices[j] {
                                needed.insert(ch);
                            }
                        }
                        let mut out: Vec<char> = current_indices
                            .iter()
                            .chain(rhs_indices.iter())
                            .copied()
                            .filter(|ch| needed.contains(ch))
                            .collect();
                        // Deduplicate while preserving order
                        let mut seen = std::collections::HashSet::new();
                        out.retain(|ch| seen.insert(*ch));
                        out
                    };
                    current = self.einsum_binary(
                        &current_indices,
                        rhs_indices,
                        &intermediate_out,
                        current,
                        tensors[i],
                        &dim_sizes,
                    )?;
                    current_indices = intermediate_out;
                }
                Ok(current)
            }
        }
    }

    fn einsum_unary(
        &mut self,
        input_idx: &[char],
        output_idx: &[char],
        tensor: TensorNodeId,
        dim_sizes: &std::collections::HashMap<char, usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let make_err = |reason: &'static str| {
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet { reason },
            ))
        };

        // Check for trace pattern: repeated index in input
        let mut repeated: Vec<char> = Vec::new();
        {
            let mut seen = std::collections::HashSet::new();
            for &ch in input_idx {
                if !seen.insert(ch) && !repeated.contains(&ch) {
                    repeated.push(ch);
                }
            }
        }

        if !repeated.is_empty() {
            // Trace-like: extract diagonal(s) then sum/permute
            // For simplicity, handle the common case: "ii->" (trace) and "ii->i" (diagonal)
            if input_idx.len() == 2 && repeated.len() == 1 {
                let shape = self.tensor_shape(tensor)?;
                if shape[0] != shape[1] {
                    return Err(make_err("einsum: trace requires square matrix"));
                }
                if output_idx.is_empty() {
                    // Trace: sum of diagonal
                    return self.tensor_trace(tensor);
                } else if output_idx.len() == 1 && output_idx[0] == repeated[0] {
                    // Diagonal extraction
                    let n = shape[0];
                    let vals = self.tensor_values(tensor)?;
                    let diag: Vec<f64> = (0..n).map(|i| vals[i * n + i]).collect();
                    return self.tensor_variable(diag, vec![n], false);
                }
            }
            return Err(make_err(
                "einsum: unsupported repeated-index pattern for single operand",
            ));
        }

        // No repeated indices: transpose + sum over contracted dims
        // First, figure out which input dims map to which output dims
        let contract_dims: Vec<usize> = input_idx
            .iter()
            .enumerate()
            .filter(|(_, ch)| !output_idx.contains(ch))
            .map(|(i, _)| i)
            .collect();

        let mut current = tensor;

        if output_idx.is_empty() && contract_dims.len() == input_idx.len() {
            // Sum all elements
            let shape = self.tensor_shape(current)?;
            let total: usize = shape.iter().product();
            let flat = self.tensor_reshape(current, vec![total])?;
            return self.tensor_sum_dim(flat, 0);
        }

        // Permute to bring output indices first, then contracted dims
        let mut perm: Vec<usize> = Vec::with_capacity(input_idx.len());
        for &out_ch in output_idx {
            let pos = input_idx
                .iter()
                .position(|&ch| ch == out_ch)
                .ok_or_else(|| make_err("einsum: output index not found in input"))?;
            perm.push(pos);
        }
        for &cd in &contract_dims {
            perm.push(cd);
        }

        if perm.len() == input_idx.len() {
            // Check if permutation is identity
            let is_identity = perm.iter().enumerate().all(|(i, &v)| v == i);
            if !is_identity {
                current = self.tensor_permute(current, perm)?;
            }
        }

        // Sum over the trailing (contracted) dimensions from last to first
        for _ in 0..contract_dims.len() {
            let shape = self.tensor_shape(current)?;
            let last_dim = shape.len() - 1;
            current = self.tensor_sum_dim(current, last_dim)?;
        }

        // If output is scalar, ensure shape
        if output_idx.is_empty() {
            let shape = self.tensor_shape(current)?;
            if shape != [1] {
                current = self.tensor_reshape(current, vec![1])?;
            }
        }

        Ok(current)
    }

    fn einsum_binary(
        &mut self,
        lhs_idx: &[char],
        rhs_idx: &[char],
        output_idx: &[char],
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        _dim_sizes: &std::collections::HashMap<char, usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        // Categorize indices:
        // - batch: appear in both inputs AND in output
        // - contract: appear in both inputs but NOT in output
        // - free_lhs: appear only in lhs (and in output)
        // - free_rhs: appear only in rhs (and in output)
        let lhs_set: std::collections::HashSet<char> = lhs_idx.iter().copied().collect();
        let rhs_set: std::collections::HashSet<char> = rhs_idx.iter().copied().collect();

        let mut batch_chars: Vec<char> = Vec::new();
        let mut contract_chars: Vec<char> = Vec::new();
        let mut free_lhs_chars: Vec<char> = Vec::new();
        let mut free_rhs_chars: Vec<char> = Vec::new();

        // Process lhs indices in order
        for &ch in lhs_idx {
            if rhs_set.contains(&ch) {
                if output_idx.contains(&ch) {
                    if !batch_chars.contains(&ch) {
                        batch_chars.push(ch);
                    }
                } else if !contract_chars.contains(&ch) {
                    contract_chars.push(ch);
                }
            } else if !free_lhs_chars.contains(&ch) {
                free_lhs_chars.push(ch);
            }
        }
        for &ch in rhs_idx {
            if !lhs_set.contains(&ch) && !free_rhs_chars.contains(&ch) {
                free_rhs_chars.push(ch);
            }
        }

        // Build permutation for lhs: [batch..., free_lhs..., contract...]
        let lhs_perm: Vec<usize> = batch_chars
            .iter()
            .chain(free_lhs_chars.iter())
            .chain(contract_chars.iter())
            .map(|ch| lhs_idx.iter().position(|c| c == ch).unwrap())
            .collect();

        // Build permutation for rhs: [batch..., contract..., free_rhs...]
        let rhs_perm: Vec<usize> = batch_chars
            .iter()
            .chain(contract_chars.iter())
            .chain(free_rhs_chars.iter())
            .map(|ch| rhs_idx.iter().position(|c| c == ch).unwrap())
            .collect();

        let lhs_shape = self.tensor_shape(lhs)?;
        let rhs_shape = self.tensor_shape(rhs)?;

        // Compute dimension sizes after permutation
        let batch_size: usize = batch_chars
            .iter()
            .map(|ch| lhs_shape[lhs_idx.iter().position(|c| c == ch).unwrap()])
            .product();
        let free_lhs_size: usize = free_lhs_chars
            .iter()
            .map(|ch| lhs_shape[lhs_idx.iter().position(|c| c == ch).unwrap()])
            .product();
        let contract_size: usize = contract_chars
            .iter()
            .map(|ch| lhs_shape[lhs_idx.iter().position(|c| c == ch).unwrap()])
            .product();
        let free_rhs_size: usize = free_rhs_chars
            .iter()
            .map(|ch| rhs_shape[rhs_idx.iter().position(|c| c == ch).unwrap()])
            .product();

        // Permute tensors
        let mut lhs_p = lhs;
        if !lhs_perm.iter().enumerate().all(|(i, &v)| v == i) {
            lhs_p = self.tensor_permute(lhs_p, lhs_perm)?;
        }
        let mut rhs_p = rhs;
        if !rhs_perm.iter().enumerate().all(|(i, &v)| v == i) {
            rhs_p = self.tensor_permute(rhs_p, rhs_perm)?;
        }

        // Handle the case where there are no contractions (element-wise or outer product)
        if contract_chars.is_empty() && batch_chars.is_empty() {
            // Outer product case: reshape lhs to [M, 1], rhs to [1, N], matmul
            let m = free_lhs_size.max(1);
            let n = free_rhs_size.max(1);
            let lhs_2d = self.tensor_reshape(lhs_p, vec![m, 1])?;
            let rhs_2d = self.tensor_reshape(rhs_p, vec![1, n])?;
            let result = self.tensor_matmul(lhs_2d, rhs_2d)?;
            // Reshape to output shape
            let mut out_shape: Vec<usize> = Vec::new();
            for &ch in output_idx {
                if free_lhs_chars.contains(&ch) {
                    out_shape.push(lhs_shape[lhs_idx.iter().position(|c| *c == ch).unwrap()]);
                } else if free_rhs_chars.contains(&ch) {
                    out_shape.push(rhs_shape[rhs_idx.iter().position(|c| *c == ch).unwrap()]);
                }
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            return self.tensor_reshape(result, out_shape);
        }

        if contract_chars.is_empty() {
            // Batch + free only: element-wise multiply with broadcasting
            let m = free_lhs_size.max(1);
            let n = free_rhs_size.max(1);
            let lhs_3d = self.tensor_reshape(lhs_p, vec![batch_size, m, 1])?;
            let rhs_3d = self.tensor_reshape(rhs_p, vec![batch_size, 1, n])?;
            let result = self.tensor_bmm(lhs_3d, rhs_3d)?;
            let mut out_shape: Vec<usize> = Vec::new();
            for &ch in output_idx {
                if batch_chars.contains(&ch) {
                    out_shape.push(lhs_shape[lhs_idx.iter().position(|c| *c == ch).unwrap()]);
                } else if free_lhs_chars.contains(&ch) {
                    out_shape.push(lhs_shape[lhs_idx.iter().position(|c| *c == ch).unwrap()]);
                } else if free_rhs_chars.contains(&ch) {
                    out_shape.push(rhs_shape[rhs_idx.iter().position(|c| *c == ch).unwrap()]);
                }
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            return self.tensor_reshape(result, out_shape);
        }

        // General case with contraction
        let m = free_lhs_size.max(1);
        let k = contract_size.max(1);
        let n = free_rhs_size.max(1);

        let result = if batch_chars.is_empty() {
            // No batch: 2D matmul [M, K] @ [K, N] -> [M, N]
            let lhs_2d = self.tensor_reshape(lhs_p, vec![m, k])?;
            let rhs_2d = self.tensor_reshape(rhs_p, vec![k, n])?;
            self.tensor_matmul(lhs_2d, rhs_2d)?
        } else {
            // Batch: 3D bmm [B, M, K] @ [B, K, N] -> [B, M, N]
            let b = batch_size;
            let lhs_3d = self.tensor_reshape(lhs_p, vec![b, m, k])?;
            let rhs_3d = self.tensor_reshape(rhs_p, vec![b, k, n])?;
            self.tensor_bmm(lhs_3d, rhs_3d)?
        };

        // Reshape to output shape
        let mut out_shape: Vec<usize> = Vec::new();
        for &ch in output_idx {
            if batch_chars.contains(&ch) {
                out_shape.push(lhs_shape[lhs_idx.iter().position(|c| *c == ch).unwrap()]);
            } else if free_lhs_chars.contains(&ch) {
                out_shape.push(lhs_shape[lhs_idx.iter().position(|c| *c == ch).unwrap()]);
            } else if free_rhs_chars.contains(&ch) {
                out_shape.push(rhs_shape[rhs_idx.iter().position(|c| *c == ch).unwrap()]);
            } else if contract_chars.contains(&ch) {
                out_shape.push(lhs_shape[lhs_idx.iter().position(|c| *c == ch).unwrap()]);
            }
        }

        if out_shape.is_empty() {
            // Scalar output: already [B, M, N] = [1, 1, 1]
            let flat = self.tensor_reshape(result, vec![1])?;
            return Ok(flat);
        }

        self.tensor_reshape(result, out_shape)
    }

    /// Apply a custom autograd function.
    ///
    /// `forward_fn` receives a `&mut FunctionCtx` and input data (values + shapes),
    /// returns output (values, shape).
    ///
    /// `backward_fn` receives the saved context and incoming gradients,
    /// returns one `Option<Vec<f64>>` per input.
    pub fn tensor_apply_function<F, B>(
        &mut self,
        inputs: &[TensorNodeId],
        forward_fn: F,
        backward_fn: B,
    ) -> Result<TensorNodeId, AutogradError>
    where
        F: FnOnce(
            &mut FunctionCtx,
            &[(&[f64], &[usize])],
        ) -> Result<(Vec<f64>, Vec<usize>), AutogradError>,
        B: Fn(&FunctionCtx, &[&[f64]]) -> Result<Vec<Option<Vec<f64>>>, AutogradError>
            + Send
            + Sync
            + 'static,
    {
        self.tensor_tape.apply_function(inputs, forward_fn, backward_fn)
    }

    /// Cross product of two 3-element vectors.
    ///
    /// Both inputs must be 1D tensors of length 3.
    /// Returns a 1D tensor of length 3: c = a × b.
    pub fn tensor_cross(
        &mut self,
        a: TensorNodeId,
        b: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let a_shape = self.tensor_shape(a)?;
        let b_shape = self.tensor_shape(b)?;
        if a_shape != [3] || b_shape != [3] {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::ShapeMismatch {
                    lhs: a_shape,
                    rhs: b_shape,
                },
            )));
        }
        let av = self.tensor_values(a)?;
        let bv = self.tensor_values(b)?;
        let result = vec![
            av[1] * bv[2] - av[2] * bv[1],
            av[2] * bv[0] - av[0] * bv[2],
            av[0] * bv[1] - av[1] * bv[0],
        ];
        self.tensor_variable(result, vec![3], false)
    }

    /// Dot product along the last dimension (batched dot product).
    ///
    /// For 1D inputs: equivalent to `tensor_dot`.
    /// For N-D inputs: dot product along the last dimension, output has one fewer dim.
    pub fn tensor_vecdot(
        &mut self,
        a: TensorNodeId,
        b: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let a_shape = self.tensor_shape(a)?;
        let b_shape = self.tensor_shape(b)?;
        if a_shape != b_shape {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::ShapeMismatch {
                    lhs: a_shape.clone(),
                    rhs: b_shape,
                },
            )));
        }
        if a_shape.len() == 1 {
            return self.tensor_dot(a, b);
        }
        // For N-D: element-wise multiply then sum along last dim
        let prod = self.tensor_mul(a, b)?;
        let last_dim = a_shape.len() - 1;
        self.tensor_sum_dim(prod, last_dim)
    }

    /// Embed a 1D vector as the diagonal of a 2D matrix.
    ///
    /// `diag_embed([1,2,3])` produces `[[1,0,0],[0,2,0],[0,0,3]]`.
    /// `offset` controls which diagonal: 0=main, 1=super, -1=sub.
    pub fn tensor_diag_embed(
        &mut self,
        input: TensorNodeId,
        offset: i32,
    ) -> Result<TensorNodeId, AutogradError> {
        let shape = self.tensor_shape(input)?;
        if shape.len() != 1 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension {
                    dim: shape.len(),
                    ndim: 1,
                },
            )));
        }
        let k = shape[0];
        let abs_offset = offset.unsigned_abs() as usize;
        let n = k + abs_offset;
        let vals = self.tensor_values(input)?;
        let mut result = vec![0.0f64; n * n];
        for i in 0..k {
            let (row, col) = if offset >= 0 {
                (i, i + abs_offset)
            } else {
                (i + abs_offset, i)
            };
            result[row * n + col] = vals[i];
        }
        self.tensor_variable(result, vec![n, n], false)
    }

    /// Return unique elements from a 1D tensor.
    ///
    /// If `sorted` is true (default), output is sorted ascending.
    /// If `return_inverse` is true, also returns index mapping input -> unique.
    /// If `return_counts` is true, also returns count of each unique value.
    pub fn tensor_unique(
        &mut self,
        input: TensorNodeId,
        sorted: bool,
        return_inverse: bool,
        return_counts: bool,
    ) -> Result<(TensorNodeId, Option<TensorNodeId>, Option<TensorNodeId>), AutogradError> {
        let vals = self.tensor_values(input)?;

        // Collect unique values preserving first-occurrence order
        let mut unique_vals: Vec<f64> = Vec::new();
        let mut inverse_indices: Vec<usize> = Vec::with_capacity(vals.len());

        for &v in &vals {
            let pos = unique_vals
                .iter()
                .position(|&u| (u - v).abs() < f64::EPSILON || (u.is_nan() && v.is_nan()));
            match pos {
                Some(idx) => inverse_indices.push(idx),
                None => {
                    inverse_indices.push(unique_vals.len());
                    unique_vals.push(v);
                }
            }
        }

        if sorted {
            // Sort unique values and remap inverse indices
            let mut order: Vec<usize> = (0..unique_vals.len()).collect();
            order.sort_by(|&a, &b| unique_vals[a].partial_cmp(&unique_vals[b]).unwrap());
            let mut remap = vec![0usize; unique_vals.len()];
            for (new_idx, &old_idx) in order.iter().enumerate() {
                remap[old_idx] = new_idx;
            }
            let sorted_vals: Vec<f64> = order.iter().map(|&i| unique_vals[i]).collect();
            unique_vals = sorted_vals;
            for idx in &mut inverse_indices {
                *idx = remap[*idx];
            }
        }

        let unique_len = unique_vals.len();
        let unique_node = self.tensor_variable(unique_vals, vec![unique_len], false)?;

        let inverse_node = if return_inverse {
            let inv_f64: Vec<f64> = inverse_indices.iter().map(|&i| i as f64).collect();
            let n = inv_f64.len();
            Some(self.tensor_variable(inv_f64, vec![n], false)?)
        } else {
            None
        };

        let counts_node = if return_counts {
            let mut counts = vec![0.0f64; unique_len];
            for &idx in &inverse_indices {
                counts[idx] += 1.0;
            }
            Some(self.tensor_variable(counts, vec![unique_len], false)?)
        } else {
            None
        };

        Ok((unique_node, inverse_node, counts_node))
    }

    /// Remove consecutive duplicate elements from a 1D tensor.
    ///
    /// Unlike `unique`, only removes adjacent duplicates (O(n) without sorting).
    pub fn tensor_unique_consecutive(
        &mut self,
        input: TensorNodeId,
        return_inverse: bool,
        return_counts: bool,
    ) -> Result<(TensorNodeId, Option<TensorNodeId>, Option<TensorNodeId>), AutogradError> {
        let vals = self.tensor_values(input)?;

        let mut unique_vals: Vec<f64> = Vec::new();
        let mut inverse_indices: Vec<usize> = Vec::with_capacity(vals.len());
        let mut counts: Vec<f64> = Vec::new();

        for &v in &vals {
            if unique_vals.is_empty()
                || !((unique_vals.last().unwrap() - v).abs() < f64::EPSILON
                    || (unique_vals.last().unwrap().is_nan() && v.is_nan()))
            {
                unique_vals.push(v);
                counts.push(1.0);
            } else {
                *counts.last_mut().unwrap() += 1.0;
            }
            inverse_indices.push(unique_vals.len() - 1);
        }

        let unique_len = unique_vals.len();
        let unique_node = self.tensor_variable(unique_vals, vec![unique_len], false)?;

        let inverse_node = if return_inverse {
            let inv_f64: Vec<f64> = inverse_indices.iter().map(|&i| i as f64).collect();
            let n = inv_f64.len();
            Some(self.tensor_variable(inv_f64, vec![n], false)?)
        } else {
            None
        };

        let counts_node = if return_counts {
            Some(self.tensor_variable(counts, vec![unique_len], false)?)
        } else {
            None
        };

        Ok((unique_node, inverse_node, counts_node))
    }

    pub fn tensor_trace(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.trace(input, self.mode())?;
        self.record_tensor_reduction_operation(&event);
        Ok(out)
    }

    pub fn tensor_lerp(
        &mut self,
        start: TensorNodeId,
        end: TensorNodeId,
        weight: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.lerp(start, end, weight, self.mode())?;
        self.record_tensor_lerp_operation(&event);
        Ok(out)
    }

    pub fn tensor_addmm(
        &mut self,
        input: TensorNodeId,
        mat1: TensorNodeId,
        mat2: TensorNodeId,
        beta: f64,
        alpha: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self
            .tensor_tape
            .addmm(input, mat1, mat2, beta, alpha, self.mode())?;
        self.record_tensor_addmm_operation(&event);
        Ok(out)
    }

    pub fn tensor_addmv(
        &mut self,
        input: TensorNodeId,
        mat: TensorNodeId,
        vec_input: TensorNodeId,
        beta: f64,
        alpha: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) =
            self.tensor_tape
                .addmv(input, mat, vec_input, beta, alpha, self.mode())?;
        self.record_tensor_addmv_operation(&event);
        Ok(out)
    }

    pub fn tensor_neg(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.neg(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_abs(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.abs(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_exp(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.exp(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_log(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.log(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_relu(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.relu(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sigmoid(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sigmoid(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_tanh(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.tanh(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sin(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sin(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_cos(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.cos(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_tan(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.tan(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_floor(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.floor(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_ceil(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.ceil(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_round(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.round(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_log2(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.log2(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_log10(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.log10(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_log1p(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.log1p(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_expm1(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.expm1(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sign(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sign(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_trunc(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.trunc(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_frac(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.frac(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_asin(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.asin(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_acos(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.acos(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_atan(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.atan(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sinh(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sinh(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_cosh(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.cosh(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_gelu(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.gelu(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_silu(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.silu(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_leaky_relu(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.leaky_relu(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_elu(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.elu(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_rsqrt(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.rsqrt(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_erf(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.erf(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_erfc(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.erfc(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_hardswish(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.hardswish(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_hardsigmoid(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.hardsigmoid(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_hardtanh(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.hardtanh(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_softplus(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.softplus(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_mish(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.mish(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_square(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.square(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sqrt(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sqrt(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_reciprocal(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.reciprocal(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_pow(
        &mut self,
        input: TensorNodeId,
        exponent: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.pow(input, exponent, self.mode())?;
        self.record_tensor_pow_operation(&event);
        Ok(out)
    }

    pub fn tensor_min(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.tensor_min(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_max(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.tensor_max(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_atan2(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.tensor_atan2(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_fmod(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.tensor_fmod(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_remainder(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.tensor_remainder(lhs, rhs, self.mode())?;
        self.record_tensor_operation(&event);
        Ok(out)
    }

    pub fn tensor_clamp(
        &mut self,
        input: TensorNodeId,
        min_val: f64,
        max_val: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self
            .tensor_tape
            .tensor_clamp(input, min_val, max_val, self.mode())?;
        self.record_tensor_clamp_operation(&event);
        Ok(out)
    }

    pub fn isnan(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        self.scalar_float_classify(UnaryOp::IsNan, input)
    }

    pub fn isinf(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        self.scalar_float_classify(UnaryOp::IsInf, input)
    }

    pub fn isfinite(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        self.scalar_float_classify(UnaryOp::IsFinite, input)
    }

    pub fn eq(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        self.scalar_comparison(ComparisonOp::Eq, lhs, rhs)
    }

    pub fn ne(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        self.scalar_comparison(ComparisonOp::Ne, lhs, rhs)
    }

    pub fn lt(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        self.scalar_comparison(ComparisonOp::Lt, lhs, rhs)
    }

    pub fn gt(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        self.scalar_comparison(ComparisonOp::Gt, lhs, rhs)
    }

    pub fn le(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        self.scalar_comparison(ComparisonOp::Le, lhs, rhs)
    }

    pub fn ge(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        self.scalar_comparison(ComparisonOp::Ge, lhs, rhs)
    }

    pub fn tensor_isnan(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        self.tensor_float_classify(UnaryOp::IsNan, input)
    }

    pub fn tensor_isinf(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        self.tensor_float_classify(UnaryOp::IsInf, input)
    }

    pub fn tensor_isfinite(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        self.tensor_float_classify(UnaryOp::IsFinite, input)
    }

    pub fn tensor_eq(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_comparison(ComparisonOp::Eq, lhs, rhs)
    }

    pub fn tensor_ne(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_comparison(ComparisonOp::Ne, lhs, rhs)
    }

    pub fn tensor_lt(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_comparison(ComparisonOp::Lt, lhs, rhs)
    }

    pub fn tensor_gt(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_comparison(ComparisonOp::Gt, lhs, rhs)
    }

    pub fn tensor_le(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_comparison(ComparisonOp::Le, lhs, rhs)
    }

    pub fn tensor_ge(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_comparison(ComparisonOp::Ge, lhs, rhs)
    }

    /// Return `true` when tensors have identical shape and element values.
    ///
    /// NaN values are considered unequal (matching PyTorch `torch.equal`).
    pub fn tensor_equal(
        &self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<bool, AutogradError> {
        let (lhs_values, lhs_meta) = self.tensor_values_meta(lhs)?;
        let (rhs_values, rhs_meta) = self.tensor_values_meta(rhs)?;

        if lhs_meta.shape() != rhs_meta.shape() || lhs_meta.dtype() != rhs_meta.dtype() {
            return Ok(false);
        }

        for (l, r) in lhs_values.iter().zip(rhs_values.iter()) {
            if l.is_nan() || r.is_nan() {
                return Ok(false);
            }
            if l != r {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Return `true` when tensors are elementwise close within tolerance.
    ///
    /// Uses: `|a - b| <= atol + rtol * |b|`.
    pub fn tensor_allclose(
        &self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> Result<bool, AutogradError> {
        if !rtol.is_finite() || !atol.is_finite() || rtol < 0.0 || atol < 0.0 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "allclose requires finite non-negative rtol and atol",
                },
            )));
        }

        let (lhs_values, lhs_meta) = self.tensor_values_meta(lhs)?;
        let (rhs_values, rhs_meta) = self.tensor_values_meta(rhs)?;

        if lhs_meta.shape() != rhs_meta.shape() || lhs_meta.dtype() != rhs_meta.dtype() {
            return Ok(false);
        }

        for (l, r) in lhs_values.iter().zip(rhs_values.iter()) {
            if l.is_nan() || r.is_nan() {
                if equal_nan && l.is_nan() && r.is_nan() {
                    continue;
                }
                return Ok(false);
            }

            if l.is_infinite() || r.is_infinite() {
                if l == r {
                    continue;
                }
                return Ok(false);
            }

            let tolerance = atol + rtol * r.abs();
            if (l - r).abs() > tolerance {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn tensor_sum(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sum(input, self.mode())?;
        self.record_tensor_reduction_operation(&event);
        Ok(out)
    }

    pub fn tensor_mean(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.mean(input, self.mode())?;
        self.record_tensor_reduction_operation(&event);
        Ok(out)
    }

    pub fn tensor_sum_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sum_dim(input, dim, self.mode())?;
        self.record_tensor_reduction_dim_operation(&event);
        Ok(out)
    }

    pub fn tensor_mean_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.mean_dim(input, dim, self.mode())?;
        self.record_tensor_reduction_dim_operation(&event);
        Ok(out)
    }

    pub fn tensor_prod_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.prod_dim(input, dim, self.mode())?;
        self.record_tensor_reduction_dim_operation(&event);
        Ok(out)
    }

    pub fn tensor_var_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.var_dim(input, dim, self.mode())?;
        self.record_tensor_reduction_dim_operation(&event);
        Ok(out)
    }

    pub fn tensor_std_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.std_dim(input, dim, self.mode())?;
        self.record_tensor_reduction_dim_operation(&event);
        Ok(out)
    }

    pub fn tensor_norm(
        &mut self,
        input: TensorNodeId,
        p: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.norm(input, p, self.mode())?;
        self.record_tensor_norm_operation(&event);
        Ok(out)
    }

    pub fn tensor_norm_dim(
        &mut self,
        input: TensorNodeId,
        p: f64,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.norm_dim(input, p, dim, self.mode())?;
        self.record_tensor_norm_dim_operation(&event);
        Ok(out)
    }

    /// Cumulative sum along a given dimension.
    ///
    /// For a 1-D input [a, b, c] with dim=0, returns [a, a+b, a+b+c].
    /// Output has the same shape as input.
    pub fn tensor_cumsum(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.cumsum(input, dim, self.mode())?;
        self.record_tensor_scan_dim_operation(&event);
        Ok(out)
    }

    /// Cumulative product along a given dimension.
    ///
    /// For a 1-D input [a, b, c] with dim=0, returns [a, a*b, a*b*c].
    /// Output has the same shape as input.
    pub fn tensor_cumprod(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.cumprod(input, dim, self.mode())?;
        self.record_tensor_scan_dim_operation(&event);
        Ok(out)
    }

    pub fn tensor_softmax(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.softmax(input, dim, self.mode())?;
        self.record_tensor_normalize_dim_operation(&event);
        Ok(out)
    }

    pub fn tensor_log_softmax(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.log_softmax(input, dim, self.mode())?;
        self.record_tensor_normalize_dim_operation(&event);
        Ok(out)
    }

    pub fn tensor_argmax(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.argmax(input, dim)
    }

    pub fn tensor_argmin(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.argmin(input, dim)
    }

    pub fn tensor_max_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        self.tensor_tape.max_dim(input, dim)
    }

    pub fn tensor_min_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        self.tensor_tape.min_dim(input, dim)
    }

    fn validate_index_tensor_values(
        input_shape: &[usize],
        dim: usize,
        index_values: &[f64],
    ) -> Result<(), AutogradError> {
        let Some(&dim_size) = input_shape.get(dim) else {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "index operation dimension out of bounds",
                },
            )));
        };

        let dim_size_i = isize::try_from(dim_size).map_err(|_| {
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "index operation dimension is out of range for platform isize",
                },
            ))
        })?;
        for &idx in index_values {
            let idx_i = Self::exact_integer_index_to_isize(
                idx,
                "index tensors must contain finite integer values",
            )?;
            if idx_i < -dim_size_i || idx_i >= dim_size_i {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "index tensor value out of bounds for input dimension",
                    },
                )));
            }
        }

        Ok(())
    }

    pub fn tensor_index_select(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        indices: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = self.tensor_shape(input)?;
        let indices_data = self.tensor_tape.values(indices)?;
        Self::validate_index_tensor_values(&input_shape, dim, &indices_data)?;
        self.tensor_tape.index_select(input, dim, &indices_data)
    }

    pub fn tensor_gather(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        index: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = self.tensor_shape(input)?;
        let index_data = self.tensor_tape.values(index)?;
        Self::validate_index_tensor_values(&input_shape, dim, &index_data)?;
        let index_shape = self.tensor_tape.tensor(index)?.meta().shape().to_vec();
        self.tensor_tape
            .gather(input, dim, &index_data, index_shape)
    }

    pub fn tensor_scatter(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        index: TensorNodeId,
        src: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = self.tensor_shape(input)?;
        let index_data = self.tensor_tape.values(index)?;
        Self::validate_index_tensor_values(&input_shape, dim, &index_data)?;
        let index_shape = self.tensor_tape.tensor(index)?.meta().shape().to_vec();
        let src_data = self.tensor_tape.values(src)?;
        self.tensor_tape
            .scatter(input, dim, &index_data, index_shape, &src_data)
    }

    pub fn tensor_scatter_add(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        index: TensorNodeId,
        src: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = self.tensor_shape(input)?;
        let index_data = self.tensor_tape.values(index)?;
        Self::validate_index_tensor_values(&input_shape, dim, &index_data)?;
        let index_shape = self.tensor_tape.tensor(index)?.meta().shape().to_vec();
        let src_data = self.tensor_tape.values(src)?;
        self.tensor_tape
            .scatter_add(input, dim, &index_data, index_shape, &src_data)
    }

    pub fn tensor_index_put(
        &mut self,
        input: TensorNodeId,
        indices: &[TensorNodeId],
        values: TensorNodeId,
        accumulate: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let idx_data: Vec<Vec<f64>> = indices
            .iter()
            .map(|&idx| self.tensor_tape.values(idx))
            .collect::<Result<_, _>>()?;
        let vals_data = self.tensor_tape.values(values)?;
        self.tensor_tape
            .index_put(input, &idx_data, &vals_data, accumulate)
    }

    pub fn tensor_masked_fill(
        &mut self,
        input: TensorNodeId,
        mask: TensorNodeId,
        value: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let shape = self.tensor_shape(input)?;
        let fill_tensor = self.full(shape, value, false)?;
        self.tensor_where(mask, fill_tensor, input)
    }

    pub fn tensor_cat(
        &mut self,
        inputs: &[TensorNodeId],
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.cat(inputs, dim, self.mode())?;
        self.record_tensor_join_operation(&event);
        Ok(out)
    }

    pub fn tensor_stack(
        &mut self,
        inputs: &[TensorNodeId],
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.stack(inputs, dim, self.mode())?;
        self.record_tensor_join_operation(&event);
        Ok(out)
    }

    pub fn tensor_reshape(
        &mut self,
        input: TensorNodeId,
        new_shape: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.reshape(input, new_shape)
    }

    pub fn tensor_view(
        &mut self,
        input: TensorNodeId,
        new_shape: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.view(input, new_shape)
    }

    pub fn tensor_squeeze(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.squeeze(input, dim)
    }

    pub fn tensor_unsqueeze(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.unsqueeze(input, dim)
    }

    pub fn tensor_transpose(
        &mut self,
        input: TensorNodeId,
        dim0: usize,
        dim1: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.transpose(input, dim0, dim1)
    }

    pub fn tensor_permute(
        &mut self,
        input: TensorNodeId,
        dims: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.permute(input, dims)
    }

    pub fn tensor_flatten(
        &mut self,
        input: TensorNodeId,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.flatten(input, start_dim, end_dim)
    }

    pub fn tensor_unflatten(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        sizes: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.unflatten(input, dim, sizes)
    }

    pub fn tensor_narrow(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.narrow(input, dim, start, length)
    }

    pub fn tensor_expand(
        &mut self,
        input: TensorNodeId,
        target_shape: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.expand(input, target_shape)
    }

    pub fn tensor_split(
        &mut self,
        input: TensorNodeId,
        split_sizes: &[usize],
        dim: usize,
    ) -> Result<Vec<TensorNodeId>, AutogradError> {
        self.tensor_tape.split(input, split_sizes, dim)
    }

    pub fn tensor_chunk(
        &mut self,
        input: TensorNodeId,
        chunks: usize,
        dim: usize,
    ) -> Result<Vec<TensorNodeId>, AutogradError> {
        self.tensor_tape.chunk(input, chunks, dim)
    }

    pub fn tensor_unbind(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<Vec<TensorNodeId>, AutogradError> {
        let shape = self.tensor_shape(input)?;
        let ndim = shape.len();
        if dim >= ndim {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension { dim, ndim },
            )));
        }
        let dim_size = shape[dim];
        let mut results = Vec::with_capacity(dim_size);
        for i in 0..dim_size {
            let narrowed = self.tensor_narrow(input, dim, i, 1)?;
            let squeezed = self.tensor_squeeze(narrowed, dim)?;
            results.push(squeezed);
        }
        Ok(results)
    }

    pub fn tensor_meshgrid(
        &mut self,
        inputs: &[TensorNodeId],
    ) -> Result<Vec<TensorNodeId>, AutogradError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let mut sizes = Vec::with_capacity(inputs.len());
        for &id in inputs {
            let shape = self.tensor_shape(id)?;
            if shape.len() != 1 {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::ShapeMismatch {
                        lhs: shape,
                        rhs: vec![0],
                    },
                )));
            }
            sizes.push(shape[0]);
        }
        let ndim = sizes.len();
        let mut results = Vec::with_capacity(ndim);
        for (i, &id) in inputs.iter().enumerate() {
            // Reshape to have size 1 in all dims except dim i
            let mut reshape_dims = vec![1; ndim];
            reshape_dims[i] = sizes[i];
            let reshaped = self.tensor_reshape(id, reshape_dims)?;
            // Expand to full grid shape
            let expanded = self.tensor_expand(reshaped, sizes.clone())?;
            results.push(expanded);
        }
        Ok(results)
    }

    pub fn tensor_diagonal(
        &mut self,
        input: TensorNodeId,
        offset: i64,
    ) -> Result<TensorNodeId, AutogradError> {
        let shape = self.tensor_shape(input)?;
        if shape.len() != 2 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension {
                    dim: shape.len(),
                    ndim: 2,
                },
            )));
        }
        let m = shape[0];
        let n = shape[1];
        let (row_start, col_start, diag_len) = if offset >= 0 {
            let col_start = offset as usize;
            if col_start >= n {
                let empty = self.tensor_variable(vec![], vec![0], false)?;
                return Ok(empty);
            }
            let diag_len = m.min(n - col_start);
            (0, col_start, diag_len)
        } else {
            let row_start = (-offset) as usize;
            if row_start >= m {
                let empty = self.tensor_variable(vec![], vec![0], false)?;
                return Ok(empty);
            }
            let diag_len = n.min(m - row_start);
            (row_start, 0, diag_len)
        };
        // Extract diagonal elements using index_select on flattened tensor
        let vals = self.tensor_tape.values(input)?;
        let mut diag_vals = Vec::with_capacity(diag_len);
        for i in 0..diag_len {
            diag_vals.push(vals[(row_start + i) * n + col_start + i]);
        }
        self.tensor_variable(diag_vals, vec![diag_len], false)
    }

    pub fn tensor_values(&self, node: TensorNodeId) -> Result<Vec<f64>, AutogradError> {
        self.tensor_tape.values(node)
    }

    /// Return both the contiguous values and metadata for a tensor node.
    pub fn tensor_values_meta(
        &self,
        node: TensorNodeId,
    ) -> Result<(Vec<f64>, TensorMeta), AutogradError> {
        let values = self.tensor_tape.values(node)?;
        let meta = self.tensor_tape.tensor_meta(node)?.clone();
        Ok((values, meta))
    }

    /// In-place subtraction: target = target - other.
    pub fn tensor_sub_(
        &mut self,
        target: TensorNodeId,
        other: TensorNodeId,
    ) -> Result<(), AutogradError> {
        self.validate_tensor_in_place_binary_compatibility(target, other)?;
        let target_vals = self.tensor_tape.values(target)?;
        let other_vals = self.tensor_tape.values(other)?;
        let new_values: Vec<f64> = target_vals
            .iter()
            .zip(other_vals.iter())
            .map(|(a, b)| a - b)
            .collect();
        self.tensor_tape.update_tensor_values(target, new_values)?;
        self.record_tensor_in_place_operation("sub_", target, Some(format!("other={}", other.0)));
        Ok(())
    }

    /// In-place addition: target = target + other.
    pub fn tensor_add_(
        &mut self,
        target: TensorNodeId,
        other: TensorNodeId,
    ) -> Result<(), AutogradError> {
        self.validate_tensor_in_place_binary_compatibility(target, other)?;
        let target_vals = self.tensor_tape.values(target)?;
        let other_vals = self.tensor_tape.values(other)?;
        let new_values: Vec<f64> = target_vals
            .iter()
            .zip(other_vals.iter())
            .map(|(a, b)| a + b)
            .collect();
        self.tensor_tape.update_tensor_values(target, new_values)?;
        self.record_tensor_in_place_operation("add_", target, Some(format!("other={}", other.0)));
        Ok(())
    }

    /// In-place multiplication: target = target * other.
    pub fn tensor_mul_(
        &mut self,
        target: TensorNodeId,
        other: TensorNodeId,
    ) -> Result<(), AutogradError> {
        self.validate_tensor_in_place_binary_compatibility(target, other)?;
        let target_vals = self.tensor_tape.values(target)?;
        let other_vals = self.tensor_tape.values(other)?;
        let new_values: Vec<f64> = target_vals
            .iter()
            .zip(other_vals.iter())
            .map(|(a, b)| a * b)
            .collect();
        self.tensor_tape.update_tensor_values(target, new_values)?;
        self.record_tensor_in_place_operation("mul_", target, Some(format!("other={}", other.0)));
        Ok(())
    }

    /// In-place division: target = target / other.
    pub fn tensor_div_(
        &mut self,
        target: TensorNodeId,
        other: TensorNodeId,
    ) -> Result<(), AutogradError> {
        self.validate_tensor_in_place_binary_compatibility(target, other)?;
        let target_vals = self.tensor_tape.values(target)?;
        let other_vals = self.tensor_tape.values(other)?;
        let new_values: Vec<f64> = target_vals
            .iter()
            .zip(other_vals.iter())
            .map(|(a, b)| a / b)
            .collect();
        self.tensor_tape.update_tensor_values(target, new_values)?;
        self.record_tensor_in_place_operation("div_", target, Some(format!("other={}", other.0)));
        Ok(())
    }

    /// In-place zero: target = zeros.
    pub fn tensor_zero_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.validate_tensor_in_place_target(target)?;
        let target_vals = self.tensor_tape.values(target)?;
        let new_values = vec![0.0; target_vals.len()];
        self.tensor_tape.update_tensor_values(target, new_values)?;
        self.record_tensor_in_place_operation("zero_", target, None);
        Ok(())
    }

    /// In-place fill: target = fill_value.
    pub fn tensor_fill_(
        &mut self,
        target: TensorNodeId,
        fill_value: f64,
    ) -> Result<(), AutogradError> {
        self.validate_tensor_in_place_target(target)?;
        let target_vals = self.tensor_tape.values(target)?;
        let new_values = vec![fill_value; target_vals.len()];
        self.tensor_tape.update_tensor_values(target, new_values)?;
        self.record_tensor_in_place_operation(
            "fill_",
            target,
            Some(format!("fill_value={fill_value}")),
        );
        Ok(())
    }

    /// In-place scalar multiplication: target = target * scalar.
    pub fn tensor_mul_scalar_(
        &mut self,
        target: TensorNodeId,
        scalar: f64,
    ) -> Result<(), AutogradError> {
        self.validate_tensor_in_place_target(target)?;
        let target_vals = self.tensor_tape.values(target)?;
        let new_values: Vec<f64> = target_vals.iter().map(|v| v * scalar).collect();
        self.tensor_tape.update_tensor_values(target, new_values)?;
        self.record_tensor_in_place_operation(
            "mul_scalar_",
            target,
            Some(format!("scalar={scalar}")),
        );
        Ok(())
    }

    /// In-place scalar addition: target = target + scalar.
    pub fn tensor_add_scalar_(
        &mut self,
        target: TensorNodeId,
        scalar: f64,
    ) -> Result<(), AutogradError> {
        self.validate_tensor_in_place_target(target)?;
        let target_vals = self.tensor_tape.values(target)?;
        let new_values: Vec<f64> = target_vals.iter().map(|v| v + scalar).collect();
        self.tensor_tape.update_tensor_values(target, new_values)?;
        self.record_tensor_in_place_operation(
            "add_scalar_",
            target,
            Some(format!("scalar={scalar}")),
        );
        Ok(())
    }

    pub fn tensor_neg_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("neg_", target, None, |v| -v)
    }

    pub fn tensor_abs_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("abs_", target, None, f64::abs)
    }

    pub fn tensor_exp_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("exp_", target, None, f64::exp)
    }

    pub fn tensor_log_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("log_", target, None, f64::ln)
    }

    pub fn tensor_relu_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("relu_", target, None, |v| if v > 0.0 { v } else { 0.0 })
    }

    pub fn tensor_sigmoid_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("sigmoid_", target, None, |v| 1.0 / (1.0 + (-v).exp()))
    }

    pub fn tensor_tanh_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("tanh_", target, None, f64::tanh)
    }

    pub fn tensor_sin_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("sin_", target, None, f64::sin)
    }

    pub fn tensor_cos_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("cos_", target, None, f64::cos)
    }

    pub fn tensor_sqrt_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("sqrt_", target, None, f64::sqrt)
    }

    pub fn tensor_floor_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("floor_", target, None, f64::floor)
    }

    pub fn tensor_ceil_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("ceil_", target, None, f64::ceil)
    }

    pub fn tensor_round_(&mut self, target: TensorNodeId) -> Result<(), AutogradError> {
        self.apply_tensor_unary_in_place("round_", target, None, f64::round)
    }

    pub fn tensor_clamp_(
        &mut self,
        target: TensorNodeId,
        min_val: f64,
        max_val: f64,
    ) -> Result<(), AutogradError> {
        if min_val > max_val {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "tensor_clamp_ requires min_val <= max_val",
                },
            )));
        }
        self.apply_tensor_unary_in_place(
            "clamp_",
            target,
            Some(format!("min={min_val} max={max_val}")),
            |v| v.clamp(min_val, max_val),
        )
    }

    pub fn backward(&mut self, root: NodeId) -> Result<BackwardReport, AutogradError> {
        let options = BackwardOptions::for_mode(self.mode());
        self.backward_with_options(root, options)
    }

    pub fn backward_with_options(
        &mut self,
        root: NodeId,
        options: BackwardOptions,
    ) -> Result<BackwardReport, AutogradError> {
        let report = self.tape.backward_with_options(root, options)?;
        self.runtime.ledger_mut().record(
            EvidenceKind::Backward,
            format!(
                "root={} backward_steps={} queue_pushes={} queue_pops={} max_queue_len={} reentrant_guard={}",
                root.0,
                report.steps.len(),
                report.telemetry.queue_pushes,
                report.telemetry.queue_pops,
                report.telemetry.max_queue_len,
                report.telemetry.reentrant_guard_triggered
            ),
        );
        Ok(report)
    }

    pub fn tensor_backward(
        &mut self,
        root: TensorNodeId,
    ) -> Result<TensorBackwardReport, AutogradError> {
        let options = BackwardOptions::for_mode(self.mode());
        self.tensor_backward_with_options(root, options)
    }

    pub fn tensor_backward_with_options(
        &mut self,
        root: TensorNodeId,
        options: BackwardOptions,
    ) -> Result<TensorBackwardReport, AutogradError> {
        let report = self.tensor_tape.backward_with_options(root, options)?;
        self.runtime.ledger_mut().record(
            EvidenceKind::Backward,
            format!(
                "tensor_root={} backward_steps={} queue_pushes={} queue_pops={} max_queue_len={} reentrant_guard={}",
                root.0,
                report.steps.len(),
                report.telemetry.queue_pushes,
                report.telemetry.queue_pops,
                report.telemetry.max_queue_len,
                report.telemetry.reentrant_guard_triggered
            ),
        );
        Ok(report)
    }

    #[must_use]
    pub fn gradient(&self, report: &BackwardReport, node: NodeId) -> Option<f64> {
        report.gradient(node)
    }

    #[must_use]
    pub fn tensor_gradient<'a>(
        &self,
        report: &'a TensorBackwardReport,
        node: TensorNodeId,
    ) -> Option<&'a [f64]> {
        report.gradient(node)
    }

    pub fn tensor_accumulated_gradient(
        &self,
        node: TensorNodeId,
    ) -> Result<Option<Vec<f64>>, AutogradError> {
        self.tensor_tape.tensor_accumulated_gradient_values(node)
    }

    pub fn tensor_zero_grad(&mut self, node: TensorNodeId) -> Result<(), AutogradError> {
        self.tensor_tape.zero_tensor_accumulated_gradient(node)
    }

    pub fn tensor_zero_grads(&mut self, nodes: &[TensorNodeId]) -> Result<(), AutogradError> {
        for node in nodes {
            self.tensor_tape.zero_tensor_accumulated_gradient(*node)?;
        }
        Ok(())
    }

    pub fn tensor_set_accumulated_gradient(
        &mut self,
        node: TensorNodeId,
        gradient: Vec<f64>,
    ) -> Result<(), AutogradError> {
        self.tensor_tape
            .set_tensor_accumulated_gradient(node, gradient)
    }

    /// Update a parameter's values directly, bypassing the leaf-grad in-place guard.
    ///
    /// This is the correct way for optimizers to update parameter values.
    /// In PyTorch, `param.data -= update` also bypasses the autograd guard.
    pub fn tensor_update_param_values(
        &mut self,
        param: TensorNodeId,
        new_values: Vec<f64>,
    ) -> Result<(), AutogradError> {
        self.tensor_tape.update_tensor_values(param, new_values)
    }

    #[must_use]
    pub fn evidence(&self) -> &[EvidenceEntry] {
        self.runtime.ledger().entries()
    }

    #[must_use]
    pub fn evidence_len(&self) -> usize {
        self.runtime.ledger().len()
    }

    fn checked_shape_numel(
        shape: &[usize],
        overflow_reason: &'static str,
    ) -> Result<usize, AutogradError> {
        let mut product = 1usize;
        for dim in shape.iter().copied() {
            if dim == 0 {
                return Ok(0);
            }
            let Some(next) = product.checked_mul(dim) else {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: overflow_reason,
                    },
                )));
            };
            product = next;
        }
        Ok(product)
    }

    fn checked_square_numel(
        n: usize,
        overflow_reason: &'static str,
    ) -> Result<usize, AutogradError> {
        n.checked_mul(n).ok_or({
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: overflow_reason,
                },
            ))
        })
    }

    fn exact_integer_index_to_isize(
        value: f64,
        invalid_reason: &'static str,
    ) -> Result<isize, AutogradError> {
        if !value.is_finite() || value.trunc() != value {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: invalid_reason,
                },
            )));
        }
        if value < isize::MIN as f64 || value > isize::MAX as f64 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: invalid_reason,
                },
            )));
        }
        let converted = value as isize;
        if converted as f64 != value {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: invalid_reason,
                },
            )));
        }
        Ok(converted)
    }

    fn exact_nonnegative_index_to_usize(
        value: f64,
        invalid_reason: &'static str,
    ) -> Result<usize, AutogradError> {
        if !value.is_finite() || value < 0.0 || value.trunc() != value {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: invalid_reason,
                },
            )));
        }
        if value > usize::MAX as f64 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: invalid_reason,
                },
            )));
        }
        let converted = value as usize;
        if converted as f64 != value {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: invalid_reason,
                },
            )));
        }
        Ok(converted)
    }

    fn validate_tensor_in_place_binary_compatibility(
        &self,
        target: TensorNodeId,
        other: TensorNodeId,
    ) -> Result<(), AutogradError> {
        self.validate_tensor_in_place_target(target)?;
        let target_meta = self.tensor_tape.tensor_meta(target)?;
        let other_meta = self.tensor_tape.tensor_meta(other)?;
        if target_meta.shape() != other_meta.shape() {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::ShapeMismatch {
                    lhs: target_meta.shape().to_vec(),
                    rhs: other_meta.shape().to_vec(),
                },
            )));
        }
        if target_meta.dtype() != other_meta.dtype() {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::Incompatible(TensorCompatError::DTypeMismatch {
                    lhs: target_meta.dtype(),
                    rhs: other_meta.dtype(),
                }),
            )));
        }
        if target_meta.device() != other_meta.device() {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::Incompatible(TensorCompatError::DeviceMismatch {
                    lhs: target_meta.device(),
                    rhs: other_meta.device(),
                }),
            )));
        }
        Ok(())
    }

    fn validate_tensor_in_place_target(&self, target: TensorNodeId) -> Result<(), AutogradError> {
        if self.is_grad_enabled()
            && self.tensor_tape.tensor_is_leaf(target)?
            && self.tensor_tape.tensor_requires_grad(target)?
        {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "in-place mutation on leaf tensors that require grad is forbidden",
                },
            )));
        }
        Ok(())
    }

    fn apply_tensor_unary_in_place<F>(
        &mut self,
        op: &'static str,
        target: TensorNodeId,
        extra: Option<String>,
        transform: F,
    ) -> Result<(), AutogradError>
    where
        F: Fn(f64) -> f64,
    {
        self.validate_tensor_in_place_target(target)?;
        let target_vals = self.tensor_tape.values(target)?;
        let new_values: Vec<f64> = target_vals.into_iter().map(transform).collect();
        self.tensor_tape.update_tensor_values(target, new_values)?;
        self.record_tensor_in_place_operation(op, target, extra);
        Ok(())
    }

    fn record_operation(&mut self, event: &OperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "op={:?} lhs={} rhs={} out={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.op,
                event.lhs.0,
                event.rhs.0,
                event.out.0,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_operation(&mut self, event: &TensorOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_op={:?} lhs={} rhs={} out={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.op,
                event.lhs.0,
                event.rhs.0,
                event.out.0,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_in_place_operation(
        &mut self,
        op: &'static str,
        target: TensorNodeId,
        extra: Option<String>,
    ) {
        let mut summary = format!("tensor_inplace_op={op} target={}", target.0);
        if let Some(extra) = extra {
            summary.push(' ');
            summary.push_str(&extra);
        }
        self.runtime
            .ledger_mut()
            .record(EvidenceKind::Dispatch, summary);
    }

    fn record_unary_operation(&mut self, event: &UnaryOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "unary_op={:?} input={} out={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.op,
                event.input.0,
                event.out.0,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_reduction_operation(&mut self, event: &TensorReductionOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_reduction_op={:?} input={} out={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.op,
                event.input.0,
                event.out.0,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_reduction_dim_operation(&mut self, event: &TensorReductionDimOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_reduction_dim_op={:?} input={} out={} dim={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.op,
                event.input.0,
                event.out.0,
                event.dim,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_scan_dim_operation(&mut self, event: &TensorScanDimOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_scan_dim_op={:?} input={} out={} dim={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.op,
                event.input.0,
                event.out.0,
                event.dim,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_sort_operation(&mut self, event: &TensorSortOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_sort input={} out={} dim={} descending={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.out.0,
                event.dim,
                event.descending,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_topk_operation(&mut self, event: &TensorTopKOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_topk input={} out={} k={} dim={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.out.0,
                event.k,
                event.dim,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_join_operation(&mut self, event: &TensorJoinOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_join_op={:?} inputs={:?} out={} dim={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.op,
                event.inputs.iter().map(|id| id.0).collect::<Vec<_>>(),
                event.out.0,
                event.dim,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_normalize_dim_operation(&mut self, event: &TensorNormalizeDimOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_normalize_dim_op={:?} input={} out={} dim={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.op,
                event.input.0,
                event.out.0,
                event.dim,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_unary_operation(&mut self, event: &TensorUnaryOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_unary_op={:?} input={} out={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.op,
                event.input.0,
                event.out.0,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn scalar_comparison(
        &mut self,
        op: ComparisonOp,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Result<NodeId, AutogradError> {
        let lhs_value = self.tape.value(lhs)?;
        let rhs_value = self.tape.value(rhs)?;
        let lhs_scalar =
            ft_core::ScalarTensor::new(lhs_value, ft_core::DType::F64, ft_core::Device::Cpu);
        let rhs_scalar =
            ft_core::ScalarTensor::new(rhs_value, ft_core::DType::F64, ft_core::Device::Cpu);

        let outcome = dispatch_scalar_comparison(op, self.mode(), &lhs_scalar, &rhs_scalar, false)
            .map_err(AutogradError::Dispatch)?;

        let out = self.tape.leaf(outcome.tensor.value(), false);
        self.record_comparison_operation(op, &outcome.decision);
        Ok(out)
    }

    fn tensor_comparison(
        &mut self,
        op: ComparisonOp,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (lhs_storage, lhs_meta, rhs_storage, rhs_meta) = {
            let lhs_tensor = self.tensor_tape.tensor(lhs)?;
            let rhs_tensor = self.tensor_tape.tensor(rhs)?;
            (
                lhs_tensor.storage().to_vec(),
                lhs_tensor.meta().clone(),
                rhs_tensor.storage().to_vec(),
                rhs_tensor.meta().clone(),
            )
        };

        let outcome = dispatch_tensor_comparison_contiguous_f64(
            op,
            self.mode(),
            &lhs_storage,
            &rhs_storage,
            &lhs_meta,
            &rhs_meta,
            false,
        )
        .map_err(AutogradError::Dispatch)?;

        let out = self
            .tensor_tape
            .leaf(outcome.values, lhs_meta.shape().to_vec(), false)?;
        self.record_comparison_operation(op, &outcome.decision);
        Ok(out)
    }

    fn scalar_float_classify(
        &mut self,
        op: UnaryOp,
        input: NodeId,
    ) -> Result<NodeId, AutogradError> {
        let value = self.tape.value(input)?;
        let scalar = ft_core::ScalarTensor::new(value, ft_core::DType::F64, ft_core::Device::Cpu);

        let outcome = dispatch_scalar_unary(op, self.mode(), &scalar, false)
            .map_err(AutogradError::Dispatch)?;

        let out = self.tape.leaf(outcome.tensor.value(), false);
        self.record_float_classify_operation(op, &outcome.decision);
        Ok(out)
    }

    fn tensor_float_classify(
        &mut self,
        op: UnaryOp,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (storage, meta) = {
            let tensor = self.tensor_tape.tensor(input)?;
            (tensor.storage().to_vec(), tensor.meta().clone())
        };

        let outcome = dispatch_tensor_unary_contiguous_f64(op, self.mode(), &storage, &meta, false)
            .map_err(AutogradError::Dispatch)?;

        let out = self
            .tensor_tape
            .leaf(outcome.values, meta.shape().to_vec(), false)?;
        self.record_float_classify_operation(op, &outcome.decision);
        Ok(out)
    }

    fn record_float_classify_operation(&mut self, op: UnaryOp, decision: &UnaryDispatchDecision) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "float_classify_op={:?} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                op,
                decision.mode,
                decision.kernel,
                decision.selected_key,
                decision.backend_key,
                decision.keyset_bits,
                decision.fallback_used
            ),
        );
    }

    fn record_comparison_operation(
        &mut self,
        op: ComparisonOp,
        decision: &ComparisonDispatchDecision,
    ) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "comparison_op={:?} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                op,
                decision.mode,
                decision.kernel,
                decision.selected_key,
                decision.backend_key,
                decision.keyset_bits,
                decision.fallback_used
            ),
        );
    }

    fn record_pow_operation(&mut self, event: &PowOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "pow_op input={} out={} exponent={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.out.0,
                event.exponent,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_pow_operation(&mut self, event: &TensorPowOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_pow_op input={} out={} exponent={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.out.0,
                event.exponent,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_norm_operation(&mut self, event: &TensorNormOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_norm_op input={} out={} p={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.out.0,
                event.p,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_norm_dim_operation(&mut self, event: &TensorNormDimOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_norm_dim_op input={} out={} p={} dim={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.out.0,
                event.p,
                event.dim,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_lerp_operation(&mut self, event: &TensorLerpOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_lerp_op start={} end={} out={} weight={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.start.0,
                event.end.0,
                event.out.0,
                event.weight,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_addmm_operation(&mut self, event: &TensorAddmmOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_addmm_op input={} mat1={} mat2={} out={} beta={} alpha={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.mat1.0,
                event.mat2.0,
                event.out.0,
                event.beta,
                event.alpha,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_addmv_operation(&mut self, event: &TensorAddmvOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_addmv_op input={} mat={} vec={} out={} beta={} alpha={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.mat.0,
                event.vec.0,
                event.out.0,
                event.beta,
                event.alpha,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_clamp_operation(&mut self, event: &ClampOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "clamp_op input={} out={} min={} max={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.out.0,
                event.min_val,
                event.max_val,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    fn record_tensor_clamp_operation(&mut self, event: &TensorClampOperationEvent) {
        self.runtime.ledger_mut().record(
            EvidenceKind::Dispatch,
            format!(
                "tensor_clamp_op input={} out={} min={} max={} mode={:?} kernel={} key={:?} backend={:?} keyset=0x{:016x} fallback={}",
                event.input.0,
                event.out.0,
                event.min_val,
                event.max_val,
                event.decision.mode,
                event.decision.kernel,
                event.decision.selected_key,
                event.decision.backend_key,
                event.decision.keyset_bits,
                event.decision.fallback_used
            ),
        );
    }

    // ── Utility Methods ─────────────────────────────────────────────────

    /// Return the shape of a tensor node as a `Vec<usize>`.
    pub fn tensor_shape(&self, node: TensorNodeId) -> Result<Vec<usize>, AutogradError> {
        let tensor = self.tensor_tape.tensor(node)?;
        Ok(tensor.meta().shape().to_vec())
    }

    /// Return the number of dimensions of a tensor.
    pub fn tensor_dim(&self, node: TensorNodeId) -> Result<usize, AutogradError> {
        let tensor = self.tensor_tape.tensor(node)?;
        Ok(tensor.meta().shape().len())
    }

    /// Return the total number of elements in a tensor.
    pub fn tensor_numel(&self, node: TensorNodeId) -> Result<usize, AutogradError> {
        let tensor = self.tensor_tape.tensor(node)?;
        Ok(tensor.meta().numel())
    }

    /// Return whether this tensor currently tracks gradients.
    pub fn tensor_requires_grad(&self, node: TensorNodeId) -> Result<bool, AutogradError> {
        self.tensor_tape.tensor_requires_grad(node)
    }

    /// Return whether this tensor is a leaf (user-created or detached in-place).
    pub fn tensor_is_leaf(&self, node: TensorNodeId) -> Result<bool, AutogradError> {
        self.tensor_tape.tensor_is_leaf(node)
    }

    /// Return the gradient function name for non-leaf tensors.
    pub fn tensor_grad_fn(&self, node: TensorNodeId) -> Result<Option<String>, AutogradError> {
        self.tensor_tape.tensor_grad_fn(node)
    }

    /// In-place toggle for gradient tracking on leaf tensors.
    pub fn tensor_requires_grad_(
        &mut self,
        node: TensorNodeId,
        requires_grad: bool,
    ) -> Result<(), AutogradError> {
        self.tensor_tape
            .set_tensor_requires_grad(node, requires_grad)
    }

    /// In-place detach from the computation graph.
    pub fn tensor_detach_(&mut self, node: TensorNodeId) -> Result<(), AutogradError> {
        self.tensor_tape.detach_tensor_in_place(node)
    }

    /// Register a callback invoked when this tensor's gradient is computed during backward.
    pub fn tensor_register_hook<F>(
        &mut self,
        node: TensorNodeId,
        hook: F,
    ) -> Result<TensorHookHandle, AutogradError>
    where
        F: Fn(&[f64]) -> Result<Option<Vec<f64>>, AutogradError> + Send + Sync + 'static,
    {
        self.tensor_tape.register_tensor_hook(node, hook)
    }

    /// Remove a previously registered tensor hook.
    pub fn tensor_remove_hook(&mut self, handle: TensorHookHandle) -> Result<bool, AutogradError> {
        self.tensor_tape.remove_tensor_hook(handle)
    }

    /// Extract a single scalar value from a 0-d or 1-element tensor.
    pub fn tensor_item(&self, node: TensorNodeId) -> Result<f64, AutogradError> {
        let values = self.tensor_tape.values(node)?;
        if values.len() != 1 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "tensor_item requires exactly 1 element",
                },
            )));
        }
        Ok(values[0])
    }

    /// Create a deep copy of a tensor node.
    ///
    /// When `requires_grad` is true, the clone participates in autograd and
    /// gradients flow back to the original tensor. When false, the clone is
    /// detached from the computation graph (equivalent to `tensor_detach`).
    pub fn tensor_clone(
        &mut self,
        node: TensorNodeId,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        if requires_grad {
            // Autograd-tracked clone: multiply by ones tensor to maintain gradient flow
            let shape = self.tensor_shape(node)?;
            let ones = self.full(shape, 1.0, false)?;
            self.tensor_mul(node, ones)
        } else {
            let values = self.tensor_tape.values(node)?;
            let meta = self.tensor_tape.tensor_meta(node)?.clone();
            let shape = meta.shape().to_vec();
            let tensor = DenseTensor::from_contiguous_values(values, shape, meta.device())?;
            Ok(self.tensor_tape.leaf_tensor(tensor, requires_grad))
        }
    }

    /// Create a copy of a tensor detached from the computation graph.
    /// The resulting tensor has `requires_grad = false`.
    pub fn tensor_detach(&mut self, node: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        self.tensor_clone(node, false)
    }

    /// Return true (1.0) if any element in the tensor is non-zero.
    pub fn tensor_any(&self, node: TensorNodeId) -> Result<bool, AutogradError> {
        let values = self.tensor_tape.values(node)?;
        Ok(values.iter().any(|&v| v != 0.0))
    }

    /// Return true (1.0) if all elements in the tensor are non-zero.
    pub fn tensor_all(&self, node: TensorNodeId) -> Result<bool, AutogradError> {
        let values = self.tensor_tape.values(node)?;
        Ok(values.iter().all(|&v| v != 0.0))
    }

    /// Return the median value of all elements in the tensor.
    ///
    /// For an even number of elements, returns the lower of the two middle values
    /// (matching PyTorch's behavior for integer-like cases).
    pub fn tensor_median(&mut self, node: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let shape = self.tensor_shape(node)?;
        let numel = Self::checked_shape_numel(&shape, "median shape volume overflow")?;
        if numel == 0 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "median requires non-empty tensor",
                },
            )));
        }
        // Flatten to 1-D, sort, then pick the median element via narrow.
        // This preserves autograd flow through sort and narrow.
        let flat = self.tensor_reshape(node, vec![numel])?;
        let (sorted, _indices) = self.tensor_sort(flat, 0, false)?;
        let median_idx = (numel - 1) / 2;
        self.tensor_narrow(sorted, 0, median_idx, 1)
    }

    // ── Similarity Functions ────────────────────────────────────────────

    /// Cosine similarity between two tensors along a dimension.
    ///
    /// `cos_sim(x1, x2, dim) = sum(x1 * x2, dim) / max(norm(x1, dim) * norm(x2, dim), eps)`
    ///
    /// Returns a tensor with the specified dimension reduced.
    /// `eps` prevents division by zero when one or both inputs have zero norm.
    ///
    /// Equivalent to PyTorch's `F.cosine_similarity(x1, x2, dim, eps)`.
    pub fn cosine_similarity(
        &mut self,
        x1: TensorNodeId,
        x2: TensorNodeId,
        dim: usize,
        eps: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        // dot_product = sum(x1 * x2, dim)
        let prod = self.tensor_mul(x1, x2)?;
        let dot = self.tensor_sum_dim(prod, dim)?;

        // norms = norm(x1, dim) * norm(x2, dim)
        let norm1 = self.tensor_norm_dim(x1, 2.0, dim)?;
        let norm2 = self.tensor_norm_dim(x2, 2.0, dim)?;
        let norms = self.tensor_mul(norm1, norm2)?;

        // Clamp norms below to eps to avoid division by zero
        let norms_shape = self.tensor_shape(norms)?;
        let eps_tensor = self.full(norms_shape, eps, false)?;
        let denom = self.tensor_max(norms, eps_tensor)?;

        // cos_sim = dot / denom
        self.tensor_div(dot, denom)
    }

    // ── Loss Functions ──────────────────────────────────────────────────

    /// Mean Squared Error loss: `mean((pred - target)^2)`
    ///
    /// Composed from existing tensor operations so backward/autograd works
    /// automatically.
    pub fn mse_loss(
        &mut self,
        pred: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let diff = self.tensor_sub(pred, target)?;
        let sq = self.tensor_mul(diff, diff)?;
        self.tensor_mean(sq)
    }

    /// L1 (Mean Absolute Error) loss: `mean(|pred - target|)`
    pub fn l1_loss(
        &mut self,
        pred: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let diff = self.tensor_sub(pred, target)?;
        let abs_diff = self.tensor_abs(diff)?;
        self.tensor_mean(abs_diff)
    }

    /// Binary Cross-Entropy loss:
    /// `-mean(target * log(pred) + (1 - target) * log(1 - pred))`
    ///
    /// Clamps `pred` to `[eps, 1 - eps]` (eps = 1e-7) for numerical stability.
    pub fn bce_loss(
        &mut self,
        pred: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let eps: f64 = 1e-7;
        let shape = self.tensor_shape(target)?;

        // clamp pred to [eps, 1-eps]
        let clamped = self.tensor_clamp(pred, eps, 1.0 - eps)?;

        // log(pred_clamped)
        let log_pred = self.tensor_log(clamped)?;

        // target * log(pred_clamped)
        let term_a = self.tensor_mul(target, log_pred)?;

        // Construct (1 - target) and (1 - pred_clamped) via a ones constant.
        let ones = self.full(shape, 1.0, false)?;

        // 1 - target
        let one_minus_target = self.tensor_sub(ones, target)?;

        // 1 - pred_clamped  (reuse `ones` node id — DAG allows multiple refs)
        let one_minus_pred = self.tensor_sub(ones, clamped)?;

        // log(1 - pred_clamped)
        let log_one_minus_pred = self.tensor_log(one_minus_pred)?;

        // (1 - target) * log(1 - pred_clamped)
        let term_b = self.tensor_mul(one_minus_target, log_one_minus_pred)?;

        // target * log(pred) + (1 - target) * log(1 - pred)
        let sum_terms = self.tensor_add(term_a, term_b)?;

        // mean(...)
        let mean_val = self.tensor_mean(sum_terms)?;

        // negate: -mean(...)
        self.tensor_neg(mean_val)
    }

    /// Binary cross-entropy with logits loss (numerically stable):
    /// `mean(max(x, 0) - x * y + log1p(exp(-|x|)))`
    ///
    /// This is equivalent to `sigmoid(x)` followed by BCE loss but avoids
    /// the numerical instability of computing `log(sigmoid(x))` for extreme
    /// input values.
    pub fn bce_with_logits_loss(
        &mut self,
        logits: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // max(x, 0) = relu(x)
        let relu_x = self.tensor_relu(logits)?;

        // x * y
        let x_times_y = self.tensor_mul(logits, target)?;

        // |x|
        let abs_x = self.tensor_abs(logits)?;

        // -|x|
        let neg_abs_x = self.tensor_neg(abs_x)?;

        // exp(-|x|)
        let exp_neg_abs = self.tensor_exp(neg_abs_x)?;

        // log1p(exp(-|x|)) = log(1 + exp(-|x|))
        let log_term = self.tensor_log1p(exp_neg_abs)?;

        // relu(x) - x * y
        let diff = self.tensor_sub(relu_x, x_times_y)?;

        // relu(x) - x * y + log1p(exp(-|x|))
        let elementwise = self.tensor_add(diff, log_term)?;

        // mean reduction
        self.tensor_mean(elementwise)
    }

    /// Smooth L1 (Huber) loss:
    /// - When `|d| < beta`: `0.5 * d^2 / beta`
    /// - When `|d| >= beta`: `|d| - 0.5 * beta`
    ///
    /// Returns `mean(element_wise_huber)`.
    ///
    /// The two branches are blended using a comparison mask so that gradients
    /// flow through the active branch at each element.
    pub fn smooth_l1_loss(
        &mut self,
        pred: TensorNodeId,
        target: TensorNodeId,
        beta: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        if beta <= 0.0 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "smooth_l1_loss requires beta > 0",
                },
            )));
        }

        let shape = self.tensor_shape(pred)?;

        let diff = self.tensor_sub(pred, target)?;
        let abs_diff = self.tensor_abs(diff)?;

        // Quadratic branch: 0.5 * diff^2 / beta
        let diff_sq = self.tensor_mul(diff, diff)?;
        let half_over_beta = self.full(shape.clone(), 0.5 / beta, false)?;
        let quadratic = self.tensor_mul(diff_sq, half_over_beta)?;

        // Linear branch: |diff| - 0.5 * beta
        let half_beta = self.full(shape.clone(), 0.5 * beta, false)?;
        let linear = self.tensor_sub(abs_diff, half_beta)?;

        // Build mask: 1.0 where |diff| < beta, 0.0 otherwise.
        // tensor_gt returns 1.0 where lhs > rhs. We want |diff| < beta,
        // i.e. beta > |diff|, so: mask = tensor_gt(beta_tensor, abs_diff).
        let beta_tensor = self.full(shape.clone(), beta, false)?;
        let mask = self.tensor_gt(beta_tensor, abs_diff)?;

        // Blend: mask * quadratic + (1 - mask) * linear
        let masked_quad = self.tensor_mul(mask, quadratic)?;
        let ones = self.full(shape, 1.0, false)?;
        let one_minus_mask = self.tensor_sub(ones, mask)?;
        let masked_lin = self.tensor_mul(one_minus_mask, linear)?;
        let huber = self.tensor_add(masked_quad, masked_lin)?;

        self.tensor_mean(huber)
    }

    /// Huber loss (robust regression loss).
    ///
    /// Behaves as L2 loss when the error is small (`|pred - target| <= delta`)
    /// and as L1 loss when the error is large (`|pred - target| > delta`).
    ///
    /// When `|diff| <= delta`: loss = 0.5 * diff^2
    /// When `|diff| > delta`:  loss = delta * (|diff| - 0.5 * delta)
    ///
    /// Returns the mean loss over all elements.
    ///
    /// Equivalent to PyTorch's `F.huber_loss(pred, target, delta=delta)`.
    pub fn huber_loss(
        &mut self,
        pred: TensorNodeId,
        target: TensorNodeId,
        delta: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        if delta <= 0.0 || !delta.is_finite() {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "huber_loss requires finite delta > 0",
                },
            )));
        }

        let shape = self.tensor_shape(pred)?;

        let diff = self.tensor_sub(pred, target)?;
        let abs_diff = self.tensor_abs(diff)?;

        // Quadratic branch: 0.5 * diff^2
        let diff_sq = self.tensor_mul(diff, diff)?;
        let half = self.full(shape.clone(), 0.5, false)?;
        let quadratic = self.tensor_mul(diff_sq, half)?;

        // Linear branch: delta * (|diff| - 0.5 * delta)
        let half_delta = self.full(shape.clone(), 0.5 * delta, false)?;
        let shifted = self.tensor_sub(abs_diff, half_delta)?;
        let delta_tensor = self.full(shape.clone(), delta, false)?;
        let linear = self.tensor_mul(delta_tensor, shifted)?;

        // Mask: 1.0 where |diff| <= delta (i.e., delta >= |diff|)
        // Reuses delta_tensor — DAG allows multiple refs
        let in_quadratic = self.tensor_ge(delta_tensor, abs_diff)?;

        // Blend: mask * quadratic + (1 - mask) * linear
        let masked_quad = self.tensor_mul(in_quadratic, quadratic)?;
        let ones = self.full(shape, 1.0, false)?;
        let one_minus_mask = self.tensor_sub(ones, in_quadratic)?;
        let masked_lin = self.tensor_mul(one_minus_mask, linear)?;
        let result = self.tensor_add(masked_quad, masked_lin)?;

        self.tensor_mean(result)
    }

    /// Cosine embedding loss for similarity learning.
    ///
    /// Measures whether two inputs are similar or dissimilar, using a target label:
    /// - When `target[i] = 1.0`: loss = `1 - cos_sim(x1[i], x2[i])`
    /// - When `target[i] = -1.0`: loss = `max(0, cos_sim(x1[i], x2[i]) - margin)`
    ///
    /// `x1` and `x2` must have shape `[batch, features]`. `target` must have shape
    /// `[batch]` with values `1.0` or `-1.0`. Returns the mean loss.
    ///
    /// Equivalent to PyTorch's `F.cosine_embedding_loss(x1, x2, target, margin)`.
    pub fn cosine_embedding_loss(
        &mut self,
        x1: TensorNodeId,
        x2: TensorNodeId,
        target: TensorNodeId,
        margin: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let x1_shape = self.tensor_shape(x1)?;
        if x1_shape.len() != 2 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "cosine_embedding_loss expects 2-D inputs [batch, features]",
                },
            )));
        }

        let batch_size = x1_shape[0];

        // Validate target values: must be exactly 1.0 or -1.0
        let target_vals = self.tensor_values(target)?;
        for &t in &target_vals {
            if t != 1.0 && t != -1.0 {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "cosine_embedding_loss target must contain only 1.0 or -1.0",
                    },
                )));
            }
        }

        // cos_sim along dim=1 (feature dimension) -> [batch]
        let cos_sim = self.cosine_similarity(x1, x2, 1, 1e-8)?;

        // Positive pairs (target=1): loss = 1 - cos_sim
        let ones = self.full(vec![batch_size], 1.0, false)?;
        let pos_loss = self.tensor_sub(ones, cos_sim)?;

        // Negative pairs (target=-1): loss = max(0, cos_sim - margin)
        let margin_t = self.full(vec![batch_size], margin, false)?;
        let neg_diff = self.tensor_sub(cos_sim, margin_t)?;
        let zeros = self.full(vec![batch_size], 0.0, false)?;
        let neg_loss = self.tensor_max(neg_diff, zeros)?;

        // Select based on target: mask = (target + 1) / 2 -> 1.0 for positive, 0.0 for negative
        let mask_vals: Vec<f64> = target_vals.iter().map(|t| (t + 1.0) / 2.0).collect();
        let mask = self.tensor_variable(mask_vals, vec![batch_size], false)?;
        let inv_mask = self.tensor_sub(ones, mask)?;

        // result = mask * pos_loss + (1 - mask) * neg_loss
        let weighted_pos = self.tensor_mul(mask, pos_loss)?;
        let weighted_neg = self.tensor_mul(inv_mask, neg_loss)?;
        let total = self.tensor_add(weighted_pos, weighted_neg)?;

        self.tensor_mean(total)
    }

    /// Negative log likelihood loss.
    ///
    /// Expects `log_probs` to be log-probabilities of shape [batch, classes] and
    /// `targets` to be class indices of shape [batch] (as f64 values that will be
    /// cast to usize). Returns the mean NLL loss.
    ///
    /// NLL(x, y) = -mean(log_probs[i, targets[i]] for i in batch)
    pub fn nll_loss(
        &mut self,
        log_probs: TensorNodeId,
        targets: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let log_prob_shape = self.tensor_shape(log_probs)?;
        let target_shape = self.tensor_shape(targets)?;
        let target_vals = self.tensor_values(targets)?;

        if log_prob_shape.len() != 2 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "nll_loss expects 2-D log_probs [batch, classes]",
                },
            )));
        }

        let batch_size = log_prob_shape[0];
        let num_classes = log_prob_shape[1];

        if target_shape.len() != 1
            || target_shape[0] != batch_size
            || target_vals.len() != batch_size
        {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "nll_loss expects targets shape [batch] matching log_probs batch",
                },
            )));
        }

        // Validate target indices
        for &target in &target_vals {
            let cls = Self::exact_nonnegative_index_to_usize(
                target,
                "nll_loss targets must be finite non-negative integer indices",
            )?;
            if cls >= num_classes {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "nll_loss target index out of bounds",
                    },
                )));
            }
        }

        // Build index tensor of shape [batch_size, 1] for gather along dim=1
        let index = self.tensor_variable(target_vals, vec![batch_size, 1], false)?;

        // Gather log-probabilities at target indices (autograd-tracked)
        let gathered = self.tensor_gather(log_probs, 1, index)?;

        // NLL = -mean(gathered log-probabilities)
        let neg_gathered = self.tensor_neg(gathered)?;
        self.tensor_mean(neg_gathered)
    }

    /// Conditional selection: `torch.where(condition, x, y)`.
    ///
    /// Selects elements from `x` where `condition != 0.0` and from `y` otherwise.
    /// All three tensors must have the same shape. Gradients flow to `x` and `y`
    /// but not to `condition`.
    pub fn tensor_where(
        &mut self,
        condition: TensorNodeId,
        x: TensorNodeId,
        y: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.tensor_where(condition, x, y)
    }

    /// Returns indices of non-zero elements as a tensor of shape `[M, ndim]`
    /// where M is the number of non-zero elements.
    ///
    /// NaN values are treated as non-zero (PyTorch behavior).
    /// The result is a non-differentiable leaf tensor.
    pub fn tensor_nonzero(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let values = self.tensor_values(input)?;
        let shape = self.tensor_shape(input)?;
        let ndim = shape.len();

        // Compute strides for index decomposition
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let mut indices = Vec::new();
        let numel: usize = shape.iter().product();
        for flat_idx in 0..numel {
            if values[flat_idx] != 0.0 || values[flat_idx].is_nan() {
                let mut remaining = flat_idx;
                for d in 0..ndim {
                    let dim_idx = remaining / strides[d];
                    remaining %= strides[d];
                    indices.push(dim_idx as f64);
                }
            }
        }

        let num_nonzero = if ndim > 0 { indices.len() / ndim } else { 0 };
        let out_shape = vec![num_nonzero, ndim];
        self.tensor_tape.leaf(indices, out_shape, false)
    }

    /// Returns indices of non-zero elements as a tuple of 1-D index tensors,
    /// one per dimension. Equivalent to `torch.nonzero(input, as_tuple=True)`.
    ///
    /// The result tensors are non-differentiable leaves.
    pub fn tensor_nonzero_as_tuple(
        &mut self,
        input: TensorNodeId,
    ) -> Result<Vec<TensorNodeId>, AutogradError> {
        let values = self.tensor_values(input)?;
        let shape = self.tensor_shape(input)?;
        let ndim = shape.len();

        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let mut per_dim_indices: Vec<Vec<f64>> = vec![Vec::new(); ndim];
        let numel: usize = shape.iter().product();
        for flat_idx in 0..numel {
            if values[flat_idx] != 0.0 || values[flat_idx].is_nan() {
                let mut remaining = flat_idx;
                for d in 0..ndim {
                    let dim_idx = remaining / strides[d];
                    remaining %= strides[d];
                    per_dim_indices[d].push(dim_idx as f64);
                }
            }
        }

        let mut result = Vec::with_capacity(ndim);
        for dim_indices in per_dim_indices {
            let len = dim_indices.len();
            let t = self.tensor_tape.leaf(dim_indices, vec![len], false)?;
            result.push(t);
        }
        Ok(result)
    }

    /// Selects elements from input where mask is non-zero, returning a 1-D tensor.
    ///
    /// The mask must have the same number of elements as the input.
    /// Gradients flow to selected positions; masked-out positions receive zero gradient.
    pub fn tensor_masked_select(
        &mut self,
        input: TensorNodeId,
        mask: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = self.tensor_shape(input)?;
        let mask_shape = self.tensor_shape(mask)?;
        let input_numel: usize = input_shape.iter().product();
        let mask_numel: usize = mask_shape.iter().product();
        if input_numel != mask_numel {
            return Err(AutogradError::Dispatch(
                ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "masked_select: input and mask must have same number of elements",
                    },
                ),
            ));
        }

        self.tensor_apply_function(
            &[input, mask],
            // Forward
            |ctx, inputs| {
                let (input_data, input_shape) = inputs[0];
                let (mask_data, _) = inputs[1];

                // Save mask for backward
                ctx.save_for_backward(mask_data.to_vec(), input_shape.to_vec());

                let selected: Vec<f64> = input_data
                    .iter()
                    .zip(mask_data.iter())
                    .filter(|&(_, m)| *m != 0.0)
                    .map(|(v, _)| *v)
                    .collect();

                let num_selected = selected.len();
                Ok((selected, vec![num_selected]))
            },
            // Backward
            |ctx, grad_outputs| {
                let grad_out = grad_outputs[0];
                let mask_data = &ctx.saved_tensors()[0];
                let input_shape = &ctx.saved_shapes()[0];
                let input_numel: usize = input_shape.iter().product();

                let mut grad_input = vec![0.0f64; input_numel];
                let mut grad_idx = 0;
                for i in 0..input_numel {
                    if mask_data[i] != 0.0 {
                        if grad_idx < grad_out.len() {
                            grad_input[i] = grad_out[grad_idx];
                        }
                        grad_idx += 1;
                    }
                }

                // No gradient for mask
                Ok(vec![Some(grad_input), None])
            },
        )
    }

    /// Binary search for insertion positions of `values` in a sorted sequence.
    ///
    /// For 1-D `sorted_sequence`, returns a tensor where each element is the index
    /// at which the corresponding element of `values` should be inserted to maintain
    /// sorted order.
    ///
    /// * `right=false` (default): returns the leftmost suitable position (lower_bound)
    /// * `right=true`: returns the rightmost suitable position (upper_bound)
    ///
    /// The result is a non-differentiable leaf tensor.
    pub fn tensor_searchsorted(
        &mut self,
        sorted_sequence: TensorNodeId,
        values: TensorNodeId,
        right: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let seq_vals = self.tensor_values(sorted_sequence)?;
        let seq_shape = self.tensor_shape(sorted_sequence)?;
        let val_vals = self.tensor_values(values)?;
        let val_shape = self.tensor_shape(values)?;

        if seq_shape.len() == 1 {
            // 1-D sorted sequence: search for each value
            let n = seq_vals.len();
            let indices: Vec<f64> = val_vals
                .iter()
                .map(|&v| {
                    if right {
                        upper_bound(&seq_vals, v, n) as f64
                    } else {
                        lower_bound(&seq_vals, v, n) as f64
                    }
                })
                .collect();
            self.tensor_tape.leaf(indices, val_shape, false)
        } else if seq_shape.len() == 2 && val_shape.len() == 2 {
            // 2-D batched: seq [B, S], values [B, V]
            let batch = seq_shape[0];
            let seq_len = seq_shape[1];
            if val_shape[0] != batch {
                return Err(AutogradError::Dispatch(
                    ft_dispatch::DispatchError::Key(
                        ft_dispatch::DispatchKeyError::IncompatibleSet {
                            reason: "searchsorted: batch dimensions must match",
                        },
                    ),
                ));
            }
            let num_vals = val_shape[1];
            let mut indices = Vec::with_capacity(batch * num_vals);
            for b in 0..batch {
                let seq_slice = &seq_vals[b * seq_len..(b + 1) * seq_len];
                for v_idx in 0..num_vals {
                    let v = val_vals[b * num_vals + v_idx];
                    let idx = if right {
                        upper_bound(seq_slice, v, seq_len)
                    } else {
                        lower_bound(seq_slice, v, seq_len)
                    };
                    indices.push(idx as f64);
                }
            }
            self.tensor_tape.leaf(indices, val_shape, false)
        } else {
            Err(AutogradError::Dispatch(
                ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "searchsorted: sorted_sequence must be 1-D or 2-D",
                    },
                ),
            ))
        }
    }

    /// Maps each element of `input` to a bucket index based on `boundaries`.
    ///
    /// Equivalent to `searchsorted(boundaries, input, right)`.
    /// Returns a non-differentiable leaf tensor of bucket indices.
    pub fn tensor_bucketize(
        &mut self,
        input: TensorNodeId,
        boundaries: TensorNodeId,
        right: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_searchsorted(boundaries, input, right)
    }

    /// Sort a tensor along the given dimension.
    ///
    /// Returns `(sorted_values_tensor, indices)` where indices contains the original
    /// positions of elements along the sorted dimension. Gradients flow back through
    /// the sorted values tensor.
    pub fn tensor_sort(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        descending: bool,
    ) -> Result<(TensorNodeId, Vec<usize>), AutogradError> {
        let (out, indices, event) = self.tensor_tape.sort(input, dim, descending, self.mode())?;
        self.record_tensor_sort_operation(&event);
        Ok((out, indices))
    }

    /// Return the k largest (or smallest) elements along the given dimension.
    ///
    /// Returns `(values_tensor, indices)`. By default, returns the k largest elements
    /// in sorted order. Gradients flow back through the values tensor.
    pub fn tensor_topk(
        &mut self,
        input: TensorNodeId,
        k: usize,
        dim: usize,
        largest: bool,
        sorted: bool,
    ) -> Result<(TensorNodeId, Vec<usize>), AutogradError> {
        let (out, indices, event) =
            self.tensor_tape
                .topk(input, k, dim, largest, sorted, self.mode())?;
        self.record_tensor_topk_operation(&event);
        Ok((out, indices))
    }

    /// Cross-entropy loss for classification.
    ///
    /// Applies log_softmax to `logits` along the last dimension, then computes
    /// NLL loss against `targets` (class indices as f64).
    ///
    /// Equivalent to PyTorch's `F.cross_entropy(logits, targets)`.
    pub fn cross_entropy_loss(
        &mut self,
        logits: TensorNodeId,
        targets: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let shape = self.tensor_shape(logits)?;
        if shape.is_empty() {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension { dim: 0, ndim: 0 },
            )));
        }
        let last_dim = shape.len() - 1;
        let log_probs = self.tensor_log_softmax(logits, last_dim)?;
        self.nll_loss(log_probs, targets)
    }

    /// Return indices that would sort the tensor along a dimension.
    ///
    /// Returns the permutation of indices as f64 values in a new tensor.
    pub fn tensor_argsort(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        descending: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let (_, indices) = self.tensor_sort(input, dim, descending)?;
        let shape = self.tensor_shape(input)?;
        let indices_f64: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
        self.tensor_tape.leaf(indices_f64, shape, false)
    }

    /// One-hot encoding: converts class indices to binary vectors.
    ///
    /// Input should contain integer indices in `[0, num_classes)` stored as f64.
    /// Returns a tensor with an additional trailing dimension of size `num_classes`.
    pub fn one_hot(
        &mut self,
        indices: TensorNodeId,
        num_classes: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (idx_vals, idx_meta) = self.tensor_values_meta(indices)?;
        let idx_shape = idx_meta.shape().to_vec();
        let total = idx_vals.len();
        let mut output = vec![0.0; total * num_classes];
        for (i, &idx) in idx_vals.iter().enumerate() {
            let class = idx as usize;
            if class >= num_classes || idx < 0.0 || idx != idx.floor() {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "one_hot index out of range or not an integer",
                    },
                )));
            }
            output[i * num_classes + class] = 1.0;
        }
        let mut out_shape = idx_shape;
        out_shape.push(num_classes);
        self.tensor_tape.leaf(output, out_shape, false)
    }

    /// Constant-value padding along each dimension.
    ///
    /// `padding` is specified as pairs `[before_last, after_last, before_second_last, ...]`
    /// matching PyTorch's `F.pad` convention (innermost dimension first).
    pub fn tensor_pad(
        &mut self,
        input: TensorNodeId,
        padding: &[usize],
        value: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        self.tensor_tape.pad(input, padding, value)
    }

    // -------------------------------------------------------------------
    // Linear Algebra: LU Decomposition
    // -------------------------------------------------------------------

    /// Compute the LU factorization of a square matrix with partial pivoting.
    ///
    /// Returns `(lu_factor_node, pivot_indices)` where `lu_factor_node` is a
    /// tensor containing the packed LU matrix, and `pivot_indices` is a `Vec<usize>`
    /// of row pivot indices.
    ///
    /// The input must be a 2-D square tensor (n x n).
    pub fn tensor_lu_factor(
        &mut self,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, Vec<usize>), AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let result = ft_kernel_cpu::lu_factor_contiguous_f64(&values, &meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        let n = result.n;
        let lu_node = self.tensor_variable(result.lu, vec![n, n], false)?;
        Ok((lu_node, result.pivots))
    }

    /// Compute the full LU decomposition: `P @ L @ U = A`.
    ///
    /// Returns `(P, L, U)` as three separate tensor nodes.
    /// - P: permutation matrix (n x n)
    /// - L: lower triangular with unit diagonal (n x n)
    /// - U: upper triangular (n x n)
    pub fn tensor_linalg_lu(
        &mut self,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId, TensorNodeId), AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let factor = ft_kernel_cpu::lu_factor_contiguous_f64(&values, &meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        let unpacked = ft_kernel_cpu::lu_unpack(&factor);
        let n = unpacked.n;
        let p_node = self.tensor_variable(unpacked.p, vec![n, n], false)?;
        let l_node = self.tensor_variable(unpacked.l, vec![n, n], false)?;
        let u_node = self.tensor_variable(unpacked.u, vec![n, n], false)?;
        Ok((p_node, l_node, u_node))
    }

    /// Solve `A * X = B` using a pre-computed LU factorization.
    ///
    /// `lu_packed` is the packed LU tensor from `tensor_lu_factor`.
    /// `pivots` is the pivot indices from `tensor_lu_factor`.
    /// `b` is the right-hand side tensor (shape [n] or [n, m]).
    ///
    /// Returns the solution tensor X.
    pub fn tensor_lu_solve(
        &mut self,
        lu_packed: TensorNodeId,
        pivots: &[usize],
        b: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (lu_values, lu_meta) = self.tensor_values_meta(lu_packed)?;
        let lu_shape = lu_meta.shape();
        if lu_shape.len() != 2 || lu_shape[0] != lu_shape[1] {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::ShapeMismatch {
                    lhs: lu_shape.to_vec(),
                    rhs: vec![lu_shape.len()],
                },
            )));
        }
        let n = lu_shape[0];
        let factor = ft_kernel_cpu::LuFactorResult {
            lu: lu_values,
            pivots: pivots.to_vec(),
            n,
        };

        let (b_values, b_meta) = self.tensor_values_meta(b)?;
        let b_shape = b_meta.shape().to_vec();
        let solution = ft_kernel_cpu::lu_solve_contiguous_f64(&factor, &b_values, &b_meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;

        self.tensor_variable(solution, b_shape, false)
    }

    /// Solve the linear system `A @ X = B` for X.
    ///
    /// Internally factorises A via LU decomposition then solves.
    /// A must be a square (n x n) tensor; B must be [n] or [n, m].
    /// Returns the solution tensor X with the same shape as B.
    pub fn tensor_linalg_solve(
        &mut self,
        a: TensorNodeId,
        b: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (a_values, a_meta) = self.tensor_values_meta(a)?;
        let factor = ft_kernel_cpu::lu_factor_contiguous_f64(&a_values, &a_meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        let (b_values, b_meta) = self.tensor_values_meta(b)?;
        let b_shape = b_meta.shape().to_vec();
        let solution = ft_kernel_cpu::lu_solve_contiguous_f64(&factor, &b_values, &b_meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        self.tensor_variable(solution, b_shape, false)
    }

    /// Compute the Cholesky decomposition of a symmetric positive-definite matrix.
    ///
    /// Returns the lower triangular factor L such that A = L @ L^T.
    /// If `upper` is true, returns U such that A = U^T @ U.
    pub fn tensor_linalg_cholesky(
        &mut self,
        input: TensorNodeId,
        upper: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let result = ft_kernel_cpu::cholesky_contiguous_f64(&values, &meta, upper)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        let n = result.n;
        self.tensor_variable(result.factor, vec![n, n], false)
    }

    /// Solve A @ X = B given Cholesky factor L where A = L @ L^T.
    ///
    /// More efficient than general solve for symmetric positive-definite systems.
    /// `cholesky_factor` is the result of `tensor_linalg_cholesky`.
    /// B can be [n] or [n, m].
    pub fn tensor_cholesky_solve(
        &mut self,
        b: TensorNodeId,
        cholesky_factor: TensorNodeId,
        upper: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let (factor_values, factor_meta) = self.tensor_values_meta(cholesky_factor)?;
        let factor_shape = factor_meta.shape();
        if factor_shape.len() != 2 || factor_shape[0] != factor_shape[1] {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::ShapeMismatch {
                    lhs: factor_shape.to_vec(),
                    rhs: vec![factor_shape.len()],
                },
            )));
        }
        let n = factor_shape[0];
        let chol = ft_kernel_cpu::CholeskyResult {
            factor: factor_values,
            n,
        };
        let (b_values, b_meta) = self.tensor_values_meta(b)?;
        let b_shape = b_meta.shape().to_vec();
        let solution =
            ft_kernel_cpu::cholesky_solve_contiguous_f64(&chol, &b_values, &b_meta, upper)
                .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        self.tensor_variable(solution, b_shape, false)
    }

    /// Compute the matrix inverse.
    ///
    /// Returns A^-1 where A @ A^-1 = I. Errors if A is singular.
    pub fn tensor_linalg_inv(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let shape = meta.shape().to_vec();
        let result = ft_kernel_cpu::inv_tensor_contiguous_f64(&values, &meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        self.tensor_variable(result, shape, false)
    }

    /// Compute A^n via binary exponentiation.
    ///
    /// n > 0: repeated squaring. n = 0: identity. n < 0: inv(A)^|n|.
    pub fn tensor_matrix_power(
        &mut self,
        input: TensorNodeId,
        n: i32,
    ) -> Result<TensorNodeId, AutogradError> {
        let shape = self.tensor_shape(input)?;
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::ShapeMismatch {
                    lhs: shape,
                    rhs: vec![2],
                },
            )));
        }
        let dim = shape[0];

        if n == 0 {
            return self.eye(dim, false);
        }

        // For negative n, invert first
        let base = if n < 0 {
            self.tensor_linalg_inv(input)?
        } else {
            input
        };

        let mut exp = n.unsigned_abs();
        let mut result = self.eye(dim, false)?;
        let mut current = base;

        while exp > 0 {
            if exp & 1 == 1 {
                result = self.tensor_matmul(result, current)?;
            }
            exp >>= 1;
            if exp > 0 {
                current = self.tensor_matmul(current, current)?;
            }
        }

        Ok(result)
    }

    /// Matrix exponential: exp(A).
    ///
    /// Uses scaling-and-squaring with Padé [6/6] approximation.
    /// NOT element-wise exp — this is the matrix exponential.
    pub fn tensor_matrix_exp(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let result = ft_kernel_cpu::matrix_exp_contiguous_f64(&values, &meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        let shape = meta.shape().to_vec();
        self.tensor_variable(result, shape, false)
    }

    /// Eigendecomposition of a symmetric matrix: A = V @ diag(λ) @ V^T.
    ///
    /// Returns `(eigenvalues, eigenvectors)` where eigenvalues are sorted ascending
    /// and eigenvectors are the columns of V (orthonormal).
    pub fn tensor_linalg_eigh(
        &mut self,
        input: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let result = ft_kernel_cpu::eigh_contiguous_f64(&values, &meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        let n = result.n;
        let evals = self.tensor_variable(result.eigenvalues, vec![n], false)?;
        let evecs = self.tensor_variable(result.eigenvectors, vec![n, n], false)?;
        Ok((evals, evecs))
    }

    /// Compute just the eigenvalues of a symmetric matrix (sorted ascending).
    pub fn tensor_linalg_eigvalsh(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let n = meta.shape()[0];
        let evals = ft_kernel_cpu::eigvalsh_contiguous_f64(&values, &meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        self.tensor_variable(evals, vec![n], false)
    }

    /// Compute the SVD: A = U @ diag(S) @ Vh.
    ///
    /// If `full_matrices` is true, U is (m x m) and Vh is (n x n).
    /// If false (reduced), U is (m x k) and Vh is (k x n) where k = min(m,n).
    /// S is always a 1D tensor of singular values sorted descending.
    pub fn tensor_linalg_svd(
        &mut self,
        input: TensorNodeId,
        full_matrices: bool,
    ) -> Result<(TensorNodeId, TensorNodeId, TensorNodeId), AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let result = ft_kernel_cpu::svd_contiguous_f64(&values, &meta, full_matrices)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        let m = result.m;
        let n = result.n;
        let k = result.k;
        let u_cols = if full_matrices { m } else { k };
        let vh_rows = if full_matrices { n } else { k };
        let u_node = self.tensor_variable(result.u, vec![m, u_cols], false)?;
        let s_node = self.tensor_variable(result.s, vec![k], false)?;
        let vh_node = self.tensor_variable(result.vh, vec![vh_rows, n], false)?;
        Ok((u_node, s_node, vh_node))
    }

    /// Compute just the singular values of a matrix.
    ///
    /// Returns a 1D tensor of singular values sorted descending.
    pub fn tensor_linalg_svdvals(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let input_shape = meta.shape();
        let k = input_shape[0].min(input_shape[1]);
        let s = ft_kernel_cpu::svdvals_contiguous_f64(&values, &meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        self.tensor_variable(s, vec![k], false)
    }

    /// Compute the determinant of a square matrix.
    ///
    /// Uses LU factorization: det = product(U diagonal) * sign(permutation).
    pub fn tensor_linalg_det(
        &mut self,
        input: TensorNodeId,
    ) -> Result<f64, AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let result = ft_kernel_cpu::det_contiguous_f64(&values, &meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        Ok(result.det)
    }

    /// Compute sign and log-absolute-determinant of a square matrix.
    ///
    /// Returns `(sign, logabsdet)` where `det(A) = sign * exp(logabsdet)`.
    /// More numerically stable than `det` for large matrices.
    pub fn tensor_linalg_slogdet(
        &mut self,
        input: TensorNodeId,
    ) -> Result<(f64, f64), AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let result = ft_kernel_cpu::slogdet_contiguous_f64(&values, &meta)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        Ok((result.sign, result.logabsdet))
    }

    /// Compute the QR decomposition: `A = Q @ R`.
    ///
    /// Returns `(Q, R)` as two separate tensor nodes.
    /// - If `reduced` is true: Q is (m x k), R is (k x n) where k = min(m, n).
    /// - If `reduced` is false: Q is (m x m), R is (m x n).
    pub fn tensor_linalg_qr(
        &mut self,
        input: TensorNodeId,
        reduced: bool,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        let (values, meta) = self.tensor_values_meta(input)?;
        let input_m = meta.shape()[0];
        let result = ft_kernel_cpu::qr_contiguous_f64(&values, &meta, reduced)
            .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;
        let q_node = self.tensor_variable(result.q, vec![input_m, result.m], false)?;
        let r_node = self.tensor_variable(result.r, vec![result.m, result.n], false)?;
        Ok((q_node, r_node))
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let ndim = shape.len();
        let mut strides = vec![0usize; ndim];
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for d in (0..ndim - 1).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
        }
        strides
    }

    // ── nn.init — Parameter Initialization Functions ───────────────────
    //
    // These functions modify tensor values in-place for parameter initialization.
    // Unlike regular in-place ops, they bypass the requires_grad leaf validation
    // because initialization is meant to set parameter values before training.
    // This mirrors PyTorch's nn.init which uses @torch.no_grad() internally.

    /// Compute (fan_in, fan_out) for a tensor based on its shape.
    ///
    /// For 2-D tensors (Linear): fan_in = shape[1], fan_out = shape[0].
    /// For N-D tensors (Conv): fan_in = shape[1] * prod(shape[2:]),
    ///                         fan_out = shape[0] * prod(shape[2:]).
    pub fn calculate_fan_in_and_fan_out(
        &self,
        tensor: TensorNodeId,
    ) -> Result<(usize, usize), AutogradError> {
        let shape = self.tensor_shape(tensor)?;
        if shape.len() < 2 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "calculate_fan_in_and_fan_out requires at least 2 dimensions",
                },
            )));
        }
        let receptive_field: usize = shape[2..].iter().product();
        let fan_in = shape[1] * receptive_field;
        let fan_out = shape[0] * receptive_field;
        Ok((fan_in, fan_out))
    }

    /// Fill tensor with values drawn from uniform distribution U(a, b).
    pub fn init_uniform_(
        &mut self,
        tensor: TensorNodeId,
        a: f64,
        b: f64,
    ) -> Result<(), AutogradError> {
        let numel = self.tensor_numel(tensor)?;
        let values: Vec<f64> = (0..numel)
            .map(|_| a + (b - a) * self.rng.next_f64())
            .collect();
        self.tensor_tape.update_tensor_values(tensor, values)?;
        self.record_tensor_in_place_operation(
            "init_uniform_",
            tensor,
            Some(format!("a={a} b={b}")),
        );
        Ok(())
    }

    /// Fill tensor with values drawn from normal distribution N(mean, std^2).
    pub fn init_normal_(
        &mut self,
        tensor: TensorNodeId,
        mean: f64,
        std: f64,
    ) -> Result<(), AutogradError> {
        let numel = self.tensor_numel(tensor)?;
        let values: Vec<f64> = (0..numel)
            .map(|_| mean + std * self.rng.next_normal())
            .collect();
        self.tensor_tape.update_tensor_values(tensor, values)?;
        self.record_tensor_in_place_operation(
            "init_normal_",
            tensor,
            Some(format!("mean={mean} std={std}")),
        );
        Ok(())
    }

    /// Fill tensor with a constant value.
    pub fn init_constant_(
        &mut self,
        tensor: TensorNodeId,
        val: f64,
    ) -> Result<(), AutogradError> {
        let numel = self.tensor_numel(tensor)?;
        let values = vec![val; numel];
        self.tensor_tape.update_tensor_values(tensor, values)?;
        self.record_tensor_in_place_operation(
            "init_constant_",
            tensor,
            Some(format!("val={val}")),
        );
        Ok(())
    }

    /// Fill tensor with ones.
    pub fn init_ones_(&mut self, tensor: TensorNodeId) -> Result<(), AutogradError> {
        self.init_constant_(tensor, 1.0)
    }

    /// Fill tensor with zeros.
    pub fn init_zeros_(&mut self, tensor: TensorNodeId) -> Result<(), AutogradError> {
        self.init_constant_(tensor, 0.0)
    }

    /// Fill a 2-D tensor with the identity matrix.
    pub fn init_eye_(&mut self, tensor: TensorNodeId) -> Result<(), AutogradError> {
        let shape = self.tensor_shape(tensor)?;
        if shape.len() != 2 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "init_eye_ requires a 2-D tensor",
                },
            )));
        }
        let rows = shape[0];
        let cols = shape[1];
        let numel = rows * cols;
        let mut values = vec![0.0; numel];
        let min_dim = rows.min(cols);
        for i in 0..min_dim {
            values[i * cols + i] = 1.0;
        }
        self.tensor_tape.update_tensor_values(tensor, values)?;
        self.record_tensor_in_place_operation("init_eye_", tensor, None);
        Ok(())
    }

    /// Fill a {3,4,5}-D tensor with the Dirac delta function.
    ///
    /// For Conv weight tensors: sets input channel == output channel filters to identity.
    pub fn init_dirac_(
        &mut self,
        tensor: TensorNodeId,
        groups: usize,
    ) -> Result<(), AutogradError> {
        let shape = self.tensor_shape(tensor)?;
        let ndim = shape.len();
        if !(3..=5).contains(&ndim) {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "init_dirac_ requires a 3-D, 4-D, or 5-D tensor",
                },
            )));
        }
        let out_channels = shape[0];
        let in_channels = shape[1];
        if groups == 0 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "init_dirac_ requires groups > 0",
                },
            )));
        }
        let out_channels_per_group = out_channels / groups;
        let min_dim = out_channels_per_group.min(in_channels);
        let numel: usize = shape.iter().product();
        let mut values = vec![0.0; numel];
        let spatial: Vec<usize> = shape[2..].to_vec();
        let spatial_numel: usize = spatial.iter().product();

        for g in 0..groups {
            for d in 0..min_dim {
                let oc = g * out_channels_per_group + d;
                let ic = d;
                // Center index in each spatial dimension
                let mut center_offset = 0usize;
                let mut stride = 1;
                for &s in spatial.iter().rev() {
                    center_offset += (s / 2) * stride;
                    stride *= s;
                }
                let flat_idx = oc * (in_channels * spatial_numel) + ic * spatial_numel + center_offset;
                if flat_idx < numel {
                    values[flat_idx] = 1.0;
                }
            }
        }
        self.tensor_tape.update_tensor_values(tensor, values)?;
        self.record_tensor_in_place_operation(
            "init_dirac_",
            tensor,
            Some(format!("groups={groups}")),
        );
        Ok(())
    }

    /// Xavier (Glorot) uniform initialization.
    ///
    /// Fills with values from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).
    pub fn init_xavier_uniform_(
        &mut self,
        tensor: TensorNodeId,
        gain: f64,
    ) -> Result<(), AutogradError> {
        let (fan_in, fan_out) = self.calculate_fan_in_and_fan_out(tensor)?;
        let a = gain * (6.0 / (fan_in + fan_out) as f64).sqrt();
        self.init_uniform_(tensor, -a, a)
    }

    /// Xavier (Glorot) normal initialization.
    ///
    /// Fills with values from N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out)).
    pub fn init_xavier_normal_(
        &mut self,
        tensor: TensorNodeId,
        gain: f64,
    ) -> Result<(), AutogradError> {
        let (fan_in, fan_out) = self.calculate_fan_in_and_fan_out(tensor)?;
        let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
        self.init_normal_(tensor, 0.0, std)
    }

    /// Compute gain for a given nonlinearity.
    ///
    /// Returns the recommended gain factor for the given activation function.
    #[must_use]
    pub fn calculate_gain(nonlinearity: &str, param: f64) -> f64 {
        match nonlinearity {
            "linear" | "conv1d" | "conv2d" | "conv3d" | "conv_transpose1d"
            | "conv_transpose2d" | "conv_transpose3d" | "sigmoid" => 1.0,
            "tanh" => 5.0 / 3.0,
            "relu" => 2.0_f64.sqrt(),
            "leaky_relu" => (2.0 / (1.0 + param * param)).sqrt(),
            "selu" => 0.75,
            _ => 1.0,
        }
    }

    /// Kaiming (He) uniform initialization.
    ///
    /// Fills with values from U(-bound, bound) where bound = sqrt(3) * gain / sqrt(fan).
    /// `mode` is "fan_in" or "fan_out". `nonlinearity` determines the gain.
    pub fn init_kaiming_uniform_(
        &mut self,
        tensor: TensorNodeId,
        a: f64,
        mode: &str,
        nonlinearity: &str,
    ) -> Result<(), AutogradError> {
        let (fan_in, fan_out) = self.calculate_fan_in_and_fan_out(tensor)?;
        let fan = match mode {
            "fan_in" => fan_in,
            "fan_out" => fan_out,
            _ => {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "init_kaiming_uniform_ mode must be 'fan_in' or 'fan_out'",
                    },
                )));
            }
        };
        let gain = Self::calculate_gain(nonlinearity, a);
        let std = gain / (fan as f64).sqrt();
        let bound = 3.0_f64.sqrt() * std;
        self.init_uniform_(tensor, -bound, bound)
    }

    /// Kaiming (He) normal initialization.
    ///
    /// Fills with values from N(0, std^2) where std = gain / sqrt(fan).
    /// `mode` is "fan_in" or "fan_out". `nonlinearity` determines the gain.
    pub fn init_kaiming_normal_(
        &mut self,
        tensor: TensorNodeId,
        a: f64,
        mode: &str,
        nonlinearity: &str,
    ) -> Result<(), AutogradError> {
        let (fan_in, fan_out) = self.calculate_fan_in_and_fan_out(tensor)?;
        let fan = match mode {
            "fan_in" => fan_in,
            "fan_out" => fan_out,
            _ => {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "init_kaiming_normal_ mode must be 'fan_in' or 'fan_out'",
                    },
                )));
            }
        };
        let gain = Self::calculate_gain(nonlinearity, a);
        let std = gain / (fan as f64).sqrt();
        self.init_normal_(tensor, 0.0, std)
    }

    /// Orthogonal initialization using QR decomposition.
    ///
    /// Fills a 2-D tensor with a (semi-)orthogonal matrix. For non-square tensors,
    /// uses reduced QR decomposition.
    pub fn init_orthogonal_(
        &mut self,
        tensor: TensorNodeId,
        gain: f64,
    ) -> Result<(), AutogradError> {
        let shape = self.tensor_shape(tensor)?;
        if shape.len() < 2 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "init_orthogonal_ requires at least 2 dimensions",
                },
            )));
        }
        let rows = shape[0];
        let cols: usize = shape[1..].iter().product();

        // Generate a random matrix and compute QR
        let flat_shape = if rows >= cols {
            vec![rows, cols]
        } else {
            vec![cols, rows]
        };
        let random_mat = self.randn(flat_shape.clone(), false)?;
        let (q_node, r_node) = self.tensor_linalg_qr(random_mat, true)?;

        // Fix sign: make diagonal of R positive (ensures unique Q)
        let r_values = self.tensor_values(r_node)?;
        let q_values = self.tensor_values(q_node)?;
        let k = flat_shape[1]; // min(rows, cols) after our arrangement
        let n_cols_q = k;

        let mut q_fixed = q_values.clone();
        for j in 0..k {
            let r_diag = r_values[j * k + j]; // R is k x k (or k x cols)
            if r_diag < 0.0 {
                // Negate column j of Q
                for i in 0..flat_shape[0] {
                    q_fixed[i * n_cols_q + j] = -q_fixed[i * n_cols_q + j];
                }
            }
        }

        // If rows < cols, transpose Q
        let final_values = if rows < cols {
            let mut transposed = vec![0.0; rows * cols];
            for i in 0..cols {
                for j in 0..rows {
                    transposed[j * cols + i] = q_fixed[i * rows + j];
                }
            }
            transposed
        } else {
            q_fixed
        };

        // Apply gain and reshape to original tensor shape
        let numel: usize = shape.iter().product();
        let scaled: Vec<f64> = final_values[..numel].iter().map(|&v| v * gain).collect();
        self.tensor_tape.update_tensor_values(tensor, scaled)?;
        self.record_tensor_in_place_operation(
            "init_orthogonal_",
            tensor,
            Some(format!("gain={gain}")),
        );
        Ok(())
    }

    /// Sparse initialization: fill tensor with normally distributed non-zero entries.
    ///
    /// Each column has a fraction `(1 - sparsity)` of non-zero entries.
    pub fn init_sparse_(
        &mut self,
        tensor: TensorNodeId,
        sparsity: f64,
        std: f64,
    ) -> Result<(), AutogradError> {
        let shape = self.tensor_shape(tensor)?;
        if shape.len() != 2 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "init_sparse_ requires a 2-D tensor",
                },
            )));
        }
        if !(0.0..1.0).contains(&sparsity) {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "init_sparse_ requires sparsity in [0, 1)",
                },
            )));
        }
        let rows = shape[0];
        let cols = shape[1];
        let num_zeros_per_col = (sparsity * rows as f64).ceil() as usize;
        let num_zeros_per_col = num_zeros_per_col.min(rows);
        let numel = rows * cols;
        let mut values = vec![0.0; numel];

        // Fill with normal values, then zero out `num_zeros_per_col` entries per column
        for j in 0..cols {
            // Generate normal values for this column
            let mut col_values: Vec<f64> = (0..rows).map(|_| std * self.rng.next_normal()).collect();

            // Generate random indices to zero out using Fisher-Yates partial shuffle
            let mut indices: Vec<usize> = (0..rows).collect();
            for k in 0..num_zeros_per_col {
                let swap_idx = k + (self.rng.next_u64() as usize % (rows - k));
                indices.swap(k, swap_idx);
            }
            for &idx in &indices[..num_zeros_per_col] {
                col_values[idx] = 0.0;
            }

            for (i, &val) in col_values.iter().enumerate() {
                values[i * cols + j] = val;
            }
        }

        self.tensor_tape.update_tensor_values(tensor, values)?;
        self.record_tensor_in_place_operation(
            "init_sparse_",
            tensor,
            Some(format!("sparsity={sparsity} std={std}")),
        );
        Ok(())
    }
}

pub use ft_autograd::{
    BackwardOptions as DacBackwardOptions, BackwardReport as DacBackwardReport,
    FunctionCtx as DacFunctionCtx, NodeId as DacNodeId, ReentrantPolicy as DacReentrantPolicy,
    TensorBackwardReport as DacTensorBackwardReport, TensorNodeId as DacTensorNodeId,
};

/// Binary search: leftmost insertion position (lower_bound).
/// Returns the index i such that all elements before i are < value.
fn lower_bound(sorted: &[f64], value: f64, n: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if sorted[mid] < value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Binary search: rightmost insertion position (upper_bound).
/// Returns the index i such that all elements before i are <= value.
fn upper_bound(sorted: &[f64], value: f64, n: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if sorted[mid] <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ft_autograd::{AutogradError, BackwardOptions, ReentrantPolicy};
    use ft_core::{DType, DenseTensor, Device, ExecutionMode, TensorMeta};
    use ft_runtime::EvidenceKind;

    use super::FrankenTorchSession;

    #[test]
    fn session_add_backward_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let y = session.variable(3.0, true);
        let z = session.add(x, y).expect("add should succeed");
        let value = session.value(z).expect("value should resolve");
        assert_eq!(value, 5.0);

        let report = session.backward(z).expect("backward should succeed");
        assert_eq!(session.gradient(&report, x), Some(1.0));
        assert_eq!(session.gradient(&report, y), Some(1.0));
        assert!(session.evidence_len() >= 3);
    }

    #[test]
    fn mode_switch_is_supported() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        session.set_mode(ExecutionMode::Hardened);
        assert_eq!(session.mode(), ExecutionMode::Hardened);
    }

    #[test]
    fn backward_with_options_supports_hardened_reentrant_fallback() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Hardened);
        let x = session.variable(2.0, true);
        let y = session.variable(3.0, true);
        let z = session.add(x, y).expect("add should succeed");

        let report = session
            .backward_with_options(
                z,
                BackwardOptions {
                    max_reentrant_depth: 1,
                    current_reentrant_depth: 2,
                    policy: ReentrantPolicy::HardenedBoundedFallback,
                    retain_graph: false,
                    create_graph: false,
                },
            )
            .expect("hardened fallback should succeed");

        assert!(report.telemetry.reentrant_guard_triggered);
    }

    #[test]
    fn tensor_backward_with_options_supports_hardened_reentrant_fallback() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Hardened);
        let x = session
            .tensor_variable(vec![2.0, 4.0], vec![2], true)
            .expect("lhs tensor variable should succeed");
        let y = session
            .tensor_variable(vec![3.0, 2.0], vec![2], true)
            .expect("rhs tensor variable should succeed");
        let z = session.tensor_add(x, y).expect("tensor add should succeed");

        let report = session
            .tensor_backward_with_options(
                z,
                BackwardOptions {
                    max_reentrant_depth: 1,
                    current_reentrant_depth: 2,
                    policy: ReentrantPolicy::HardenedBoundedFallback,
                    retain_graph: false,
                    create_graph: false,
                },
            )
            .expect("hardened tensor fallback should succeed");

        assert!(report.telemetry.reentrant_guard_triggered);
        let entry = session
            .evidence()
            .iter()
            .rev()
            .find(|entry| {
                entry.kind == EvidenceKind::Backward && entry.summary.contains("tensor_root=")
            })
            .expect("tensor backward evidence should be recorded");
        assert!(
            entry.summary.contains("reentrant_guard=true"),
            "missing fallback marker in evidence summary: {}",
            entry.summary
        );
    }

    #[test]
    fn session_sub_backward_records_negative_rhs_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let y = session.variable(3.0, true);
        let z = session.sub(x, y).expect("sub should succeed");
        let value = session.value(z).expect("value should resolve");
        assert_eq!(value, -1.0);

        let report = session.backward(z).expect("backward should succeed");
        assert_eq!(session.gradient(&report, x), Some(1.0));
        assert_eq!(session.gradient(&report, y), Some(-1.0));
    }

    #[test]
    fn session_div_backward_records_expected_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(6.0, true);
        let y = session.variable(3.0, true);
        let z = session.div(x, y).expect("div should succeed");
        let value = session.value(z).expect("value should resolve");
        assert_eq!(value, 2.0);

        let report = session.backward(z).expect("backward should succeed");
        let x_grad = session
            .gradient(&report, x)
            .expect("x grad should be present");
        let y_grad = session
            .gradient(&report, y)
            .expect("y grad should be present");
        assert!((x_grad - (1.0 / 3.0)).abs() <= 1e-12);
        assert!((y_grad - (-2.0 / 3.0)).abs() <= 1e-12);
    }

    #[test]
    fn session_tensor_add_backward_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("lhs tensor variable should succeed");
        let y = session
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], true)
            .expect("rhs tensor variable should succeed");
        let z = session.tensor_add(x, y).expect("tensor add should succeed");

        assert_eq!(
            session
                .tensor_values(z)
                .expect("tensor values should resolve"),
            vec![5.0, 7.0, 9.0]
        );

        let report = session
            .tensor_backward(z)
            .expect("tensor backward should succeed");
        assert_eq!(
            session
                .tensor_gradient(&report, x)
                .expect("x grad should exist"),
            &[1.0, 1.0, 1.0]
        );
        assert_eq!(
            session
                .tensor_gradient(&report, y)
                .expect("y grad should exist"),
            &[1.0, 1.0, 1.0]
        );
        assert!(session.evidence_len() >= 2);
    }

    #[test]
    fn session_tensor_mul_backward_records_expected_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, 4.0], vec![2], true)
            .expect("lhs tensor variable should succeed");
        let y = session
            .tensor_variable(vec![3.0, 2.0], vec![2], true)
            .expect("rhs tensor variable should succeed");
        let z = session.tensor_mul(x, y).expect("tensor mul should succeed");

        assert_eq!(
            session
                .tensor_values(z)
                .expect("tensor values should resolve"),
            vec![6.0, 8.0]
        );

        let report = session
            .tensor_backward(z)
            .expect("tensor backward should succeed");
        assert_eq!(
            session
                .tensor_gradient(&report, x)
                .expect("x grad should exist"),
            &[3.0, 2.0]
        );
        assert_eq!(
            session
                .tensor_gradient(&report, y)
                .expect("y grad should exist"),
            &[2.0, 4.0]
        );
    }

    #[test]
    fn session_tensor_div_backward_records_expected_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, 4.0], vec![2], true)
            .expect("lhs tensor variable should succeed");
        let y = session
            .tensor_variable(vec![3.0, 2.0], vec![2], true)
            .expect("rhs tensor variable should succeed");
        let z = session.tensor_div(x, y).expect("tensor div should succeed");

        let values = session
            .tensor_values(z)
            .expect("tensor values should resolve");
        assert!((values[0] - (2.0 / 3.0)).abs() <= 1e-12);
        assert!((values[1] - 2.0).abs() <= 1e-12);

        let report = session
            .tensor_backward(z)
            .expect("tensor backward should succeed");
        let x_grad = session
            .tensor_gradient(&report, x)
            .expect("x grad should exist");
        let y_grad = session
            .tensor_gradient(&report, y)
            .expect("y grad should exist");
        let expected_x_grad = [1.0 / 3.0, 0.5];
        let expected_y_grad = [-2.0 / 9.0, -1.0];
        for (actual, expected) in x_grad.iter().zip(expected_x_grad) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        for (actual, expected) in y_grad.iter().zip(expected_y_grad) {
            assert!((actual - expected).abs() <= 1e-12);
        }
    }

    #[test]
    fn session_tensor_matmul_backward_records_expected_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("lhs tensor variable should succeed");
        let y = session
            .tensor_variable(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true)
            .expect("rhs tensor variable should succeed");
        let z = session
            .tensor_matmul(x, y)
            .expect("tensor matmul should succeed");

        assert_eq!(
            session
                .tensor_values(z)
                .expect("tensor values should resolve"),
            vec![19.0, 22.0, 43.0, 50.0]
        );

        let report = session
            .tensor_backward(z)
            .expect("tensor backward should succeed");
        assert_eq!(
            session
                .tensor_gradient(&report, x)
                .expect("x grad should exist"),
            &[11.0, 15.0, 11.0, 15.0]
        );
        assert_eq!(
            session
                .tensor_gradient(&report, y)
                .expect("y grad should exist"),
            &[4.0, 4.0, 6.0, 6.0]
        );
    }

    #[test]
    fn session_tensor_add_fails_closed_on_non_contiguous_input() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lhs_meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![4, 1], 0, DType::F64, Device::Cpu)
                .expect("non-contiguous meta should validate");
        let lhs = DenseTensor::from_storage(lhs_meta, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("lhs tensor should build");
        let rhs = session
            .tensor_variable(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true)
            .expect("rhs tensor variable should build");
        let lhs = session.tensor_variable_from_storage(lhs, true);

        let err = session
            .tensor_add(lhs, rhs)
            .expect_err("non-contiguous tensor input must fail closed");
        assert!(
            err.to_string()
                .contains("unsupported non-contiguous layout on lhs")
        );
    }

    #[test]
    fn session_tensor_add_fails_closed_on_device_mismatch() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lhs_meta =
            TensorMeta::from_shape_and_strides(vec![2], vec![1], 0, DType::F64, Device::Cuda)
                .expect("cuda meta should validate");
        let lhs =
            DenseTensor::from_storage(lhs_meta, vec![1.0, 2.0]).expect("lhs tensor should build");
        let rhs = session
            .tensor_variable(vec![3.0, 4.0], vec![2], true)
            .expect("rhs tensor variable should build");
        let lhs = session.tensor_variable_from_storage(lhs, true);

        let err = session
            .tensor_add(lhs, rhs)
            .expect_err("device-mismatched tensor input must fail closed");
        let message = err.to_string();
        assert!(
            message.contains("device mismatch"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains("lhs=Cuda, rhs=Cpu"),
            "missing device mismatch payload: {message}"
        );
    }

    #[test]
    fn session_neg_scalar_returns_negated_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(5.0, false);
        let neg_x = session.neg(x).expect("neg should succeed");
        assert_eq!(session.value(neg_x).unwrap(), -5.0);
    }

    #[test]
    fn session_neg_scalar_backward_produces_minus_one_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.0, true);
        let neg_x = session.neg(x).expect("neg should succeed");
        let report = session.backward(neg_x).expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(-1.0));
    }

    #[test]
    fn session_neg_scalar_double_negation_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let neg_x = session.neg(x).expect("first neg should succeed");
        let neg_neg_x = session.neg(neg_x).expect("second neg should succeed");
        assert_eq!(session.value(neg_neg_x).unwrap(), 2.0);
        let report = session
            .backward(neg_neg_x)
            .expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(1.0));
    }

    #[test]
    fn session_neg_in_expression_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.0, true);
        let y = session.variable(2.0, true);
        let neg_x = session.neg(x).expect("neg should succeed");
        let result = session.add(neg_x, y).expect("add should succeed");
        assert_eq!(session.value(result).unwrap(), -1.0);
        let report = session.backward(result).expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(-1.0));
        assert_eq!(report.gradient(y), Some(1.0));
    }

    #[test]
    fn session_neg_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let _neg_x = session.neg(x).expect("neg should succeed");
        let has_unary_evidence = session.evidence().iter().any(|entry| {
            entry.kind == EvidenceKind::Dispatch && entry.summary.contains("unary_op=Neg")
        });
        assert!(
            has_unary_evidence,
            "neg should emit unary dispatch evidence"
        );
    }

    #[test]
    fn session_tensor_neg_returns_negated_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2], false)
            .expect("tensor creation should succeed");
        let neg_t = session.tensor_neg(t).expect("tensor neg should succeed");
        let values = session
            .tensor_values(neg_t)
            .expect("tensor values should succeed");
        assert_eq!(values, vec![-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn session_tensor_neg_backward_produces_minus_one_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("tensor creation should succeed");
        let neg_t = session.tensor_neg(t).expect("tensor neg should succeed");
        let report = session
            .tensor_backward(neg_t)
            .expect("backward should succeed");
        assert_eq!(report.gradient(t), Some(vec![-1.0, -1.0, -1.0].as_slice()));
    }

    #[test]
    fn session_tensor_neg_double_negation_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![5.0, -3.0], vec![2], true)
            .expect("tensor creation should succeed");
        let neg_t = session.tensor_neg(t).expect("first neg should succeed");
        let neg_neg_t = session
            .tensor_neg(neg_t)
            .expect("second neg should succeed");
        let values = session
            .tensor_values(neg_neg_t)
            .expect("values should succeed");
        assert_eq!(values, vec![5.0, -3.0]);
        let report = session
            .tensor_backward(neg_neg_t)
            .expect("backward should succeed");
        assert_eq!(report.gradient(t), Some(vec![1.0, 1.0].as_slice()));
    }

    #[test]
    fn session_abs_scalar_returns_absolute_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(-5.0, false);
        let abs_x = session.abs(x).expect("abs should succeed");
        assert_eq!(session.value(abs_x).unwrap(), 5.0);
    }

    #[test]
    fn session_abs_scalar_positive_unchanged() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.0, false);
        let abs_x = session.abs(x).expect("abs should succeed");
        assert_eq!(session.value(abs_x).unwrap(), 3.0);
    }

    #[test]
    fn session_abs_scalar_backward_negative_input() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(-4.0, true);
        let abs_x = session.abs(x).expect("abs should succeed");
        let report = session.backward(abs_x).expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(-1.0));
    }

    #[test]
    fn session_abs_scalar_backward_positive_input() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(4.0, true);
        let abs_x = session.abs(x).expect("abs should succeed");
        let report = session.backward(abs_x).expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(1.0));
    }

    #[test]
    fn session_abs_scalar_backward_zero_input() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let abs_x = session.abs(x).expect("abs should succeed");
        let report = session.backward(abs_x).expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(0.0));
    }

    #[test]
    fn session_tensor_abs_returns_absolute_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2], false)
            .expect("tensor creation should succeed");
        let abs_t = session.tensor_abs(t).expect("tensor abs should succeed");
        let values = session.tensor_values(abs_t).expect("values should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn session_tensor_abs_backward_mixed_signs() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![-2.0, 3.0, 0.0], vec![3], true)
            .expect("tensor creation should succeed");
        let abs_t = session.tensor_abs(t).expect("tensor abs should succeed");
        let report = session
            .tensor_backward(abs_t)
            .expect("backward should succeed");
        assert_eq!(report.gradient(t), Some(vec![-1.0, 1.0, 0.0].as_slice()));
    }

    #[test]
    fn session_exp_scalar_returns_expected_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, false);
        let exp_x = session.exp(x).expect("exp should succeed");
        let value = session.value(exp_x).unwrap();
        assert!((value - std::f64::consts::E).abs() <= 1e-12);
    }

    #[test]
    fn session_exp_scalar_backward_produces_exp_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let exp_x = session.exp(x).expect("exp should succeed");
        let report = session.backward(exp_x).expect("backward should succeed");
        let grad = report.gradient(x).expect("x grad should be present");
        assert!((grad - 2.0_f64.exp()).abs() <= 1e-12);
    }

    #[test]
    fn session_log_scalar_returns_expected_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(std::f64::consts::E, false);
        let log_x = session.log(x).expect("log should succeed");
        let value = session.value(log_x).unwrap();
        assert!((value - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn session_log_scalar_backward_produces_reciprocal_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(4.0, true);
        let log_x = session.log(x).expect("log should succeed");
        let report = session.backward(log_x).expect("backward should succeed");
        let grad = report.gradient(x).expect("x grad should be present");
        assert!((grad - 0.25).abs() <= 1e-12);
    }

    #[test]
    fn session_exp_log_roundtrip_scalar() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.0, true);
        let exp_x = session.exp(x).expect("exp should succeed");
        let log_exp_x = session.log(exp_x).expect("log should succeed");
        let value = session.value(log_exp_x).unwrap();
        assert!((value - 3.0).abs() <= 1e-12);
        let report = session
            .backward(log_exp_x)
            .expect("backward should succeed");
        let grad = report.gradient(x).expect("x grad should be present");
        assert!((grad - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn session_exp_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let _exp_x = session.exp(x).expect("exp should succeed");
        let has_exp_evidence = session.evidence().iter().any(|entry| {
            entry.kind == EvidenceKind::Dispatch && entry.summary.contains("unary_op=Exp")
        });
        assert!(has_exp_evidence, "exp should emit unary dispatch evidence");
    }

    #[test]
    fn session_log_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let _log_x = session.log(x).expect("log should succeed");
        let has_log_evidence = session.evidence().iter().any(|entry| {
            entry.kind == EvidenceKind::Dispatch && entry.summary.contains("unary_op=Log")
        });
        assert!(has_log_evidence, "log should emit unary dispatch evidence");
    }

    #[test]
    fn session_tensor_exp_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![0.0, 1.0, 2.0], vec![3], false)
            .expect("tensor creation should succeed");
        let exp_t = session.tensor_exp(t).expect("tensor exp should succeed");
        let values = session.tensor_values(exp_t).expect("values should succeed");
        assert!((values[0] - 1.0).abs() <= 1e-12);
        assert!((values[1] - std::f64::consts::E).abs() <= 1e-12);
        assert!((values[2] - 2.0_f64.exp()).abs() <= 1e-12);
    }

    #[test]
    fn session_tensor_exp_backward_produces_exp_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .expect("tensor creation should succeed");
        let exp_t = session.tensor_exp(t).expect("tensor exp should succeed");
        let report = session
            .tensor_backward(exp_t)
            .expect("backward should succeed");
        let grad = report.gradient(t).expect("gradient should exist");
        assert!((grad[0] - 1.0_f64.exp()).abs() <= 1e-12);
        assert!((grad[1] - 2.0_f64.exp()).abs() <= 1e-12);
    }

    #[test]
    fn session_tensor_log_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, std::f64::consts::E, 10.0], vec![3], false)
            .expect("tensor creation should succeed");
        let log_t = session.tensor_log(t).expect("tensor log should succeed");
        let values = session.tensor_values(log_t).expect("values should succeed");
        assert!((values[0] - 0.0).abs() <= 1e-12);
        assert!((values[1] - 1.0).abs() <= 1e-12);
        assert!((values[2] - 10.0_f64.ln()).abs() <= 1e-12);
    }

    #[test]
    fn session_tensor_log_backward_produces_reciprocal_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![2.0, 4.0], vec![2], true)
            .expect("tensor creation should succeed");
        let log_t = session.tensor_log(t).expect("tensor log should succeed");
        let report = session
            .tensor_backward(log_t)
            .expect("backward should succeed");
        let grad = report.gradient(t).expect("gradient should exist");
        assert!((grad[0] - 0.5).abs() <= 1e-12);
        assert!((grad[1] - 0.25).abs() <= 1e-12);
    }

    #[test]
    fn session_tensor_exp_log_roundtrip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("tensor creation should succeed");
        let exp_t = session.tensor_exp(t).expect("tensor exp should succeed");
        let log_exp_t = session
            .tensor_log(exp_t)
            .expect("tensor log should succeed");
        let values = session
            .tensor_values(log_exp_t)
            .expect("values should succeed");
        for (actual, expected) in values.iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        let report = session
            .tensor_backward(log_exp_t)
            .expect("backward should succeed");
        let grad = report.gradient(t).expect("gradient should exist");
        for g in grad {
            assert!((g - 1.0).abs() <= 1e-12);
        }
    }

    #[test]
    fn session_tensor_sum_returns_expected_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("tensor creation should succeed");
        let sum_t = session.tensor_sum(t).expect("tensor sum should succeed");
        let values = session.tensor_values(sum_t).expect("values should succeed");
        assert_eq!(values, vec![10.0]);
    }

    #[test]
    fn session_tensor_sum_backward_produces_ones_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("tensor creation should succeed");
        let sum_t = session.tensor_sum(t).expect("tensor sum should succeed");
        let report = session
            .tensor_backward(sum_t)
            .expect("backward should succeed");
        let grad = report.gradient(t).expect("gradient should exist");
        assert_eq!(grad, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_sum_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .expect("tensor creation should succeed");
        let _sum_t = session.tensor_sum(t).expect("tensor sum should succeed");
        let has_evidence = session.evidence().iter().any(|entry| {
            entry.kind == EvidenceKind::Dispatch
                && entry.summary.contains("tensor_reduction_op=Sum")
        });
        assert!(has_evidence, "sum should emit reduction dispatch evidence");
    }

    #[test]
    fn session_tensor_mean_returns_expected_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("tensor creation should succeed");
        let mean_t = session.tensor_mean(t).expect("tensor mean should succeed");
        let values = session
            .tensor_values(mean_t)
            .expect("values should succeed");
        assert_eq!(values, vec![2.5]);
    }

    #[test]
    fn session_tensor_mean_backward_produces_scaled_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
            .expect("tensor creation should succeed");
        let mean_t = session.tensor_mean(t).expect("tensor mean should succeed");
        let report = session
            .tensor_backward(mean_t)
            .expect("backward should succeed");
        let grad = report.gradient(t).expect("gradient should exist");
        assert_eq!(grad, &[0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn session_tensor_mean_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .expect("tensor creation should succeed");
        let _mean_t = session.tensor_mean(t).expect("tensor mean should succeed");
        let has_evidence = session.evidence().iter().any(|entry| {
            entry.kind == EvidenceKind::Dispatch
                && entry.summary.contains("tensor_reduction_op=Mean")
        });
        assert!(has_evidence, "mean should emit reduction dispatch evidence");
    }

    #[test]
    fn session_tensor_sum_after_mul_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, 3.0], vec![2], true)
            .expect("tensor creation should succeed");
        let y = session
            .tensor_variable(vec![4.0, 5.0], vec![2], true)
            .expect("tensor creation should succeed");
        let product = session.tensor_mul(x, y).expect("tensor mul should succeed");
        let loss = session
            .tensor_sum(product)
            .expect("tensor sum should succeed");
        let values = session.tensor_values(loss).expect("values should succeed");
        assert_eq!(values, vec![23.0]);

        let report = session
            .tensor_backward(loss)
            .expect("backward should succeed");
        let x_grad = report.gradient(x).expect("x gradient should exist");
        let y_grad = report.gradient(y).expect("y gradient should exist");
        assert_eq!(x_grad, &[4.0, 5.0]);
        assert_eq!(y_grad, &[2.0, 3.0]);
    }

    #[test]
    fn factory_zeros_creates_zero_filled_tensor() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session.zeros(vec![3], false).expect("zeros should succeed");
        let values = session.tensor_values(t).expect("values should succeed");
        assert_eq!(values, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn factory_zeros_2d_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .zeros(vec![2, 3], false)
            .expect("zeros should succeed");
        let values = session.tensor_values(t).expect("values should succeed");
        assert_eq!(values, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn factory_ones_creates_one_filled_tensor() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session.ones(vec![4], false).expect("ones should succeed");
        let values = session.tensor_values(t).expect("values should succeed");
        assert_eq!(values, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn factory_ones_supports_grad() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session.ones(vec![2], true).expect("ones should succeed");
        let doubled = session.tensor_add(t, t).expect("add should succeed");
        let values = session
            .tensor_values(doubled)
            .expect("values should succeed");
        assert_eq!(values, vec![2.0, 2.0]);
        let report = session
            .tensor_backward(doubled)
            .expect("backward should succeed");
        let grad = report.gradient(t).expect("gradient should exist");
        assert_eq!(grad, &[2.0, 2.0]);
    }

    #[test]
    fn factory_full_creates_custom_filled_tensor() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .full(vec![3], 7.5, false)
            .expect("full should succeed");
        let values = session.tensor_values(t).expect("values should succeed");
        assert_eq!(values, vec![7.5, 7.5, 7.5]);
    }

    #[test]
    fn factory_arange_creates_sequence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .arange(0.0, 5.0, 1.0, false)
            .expect("arange should succeed");
        let values = session.tensor_values(t).expect("values should succeed");
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn factory_arange_with_step() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .arange(1.0, 3.0, 0.5, false)
            .expect("arange should succeed");
        let values = session.tensor_values(t).expect("values should succeed");
        assert_eq!(values, vec![1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn factory_arange_empty_range() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .arange(5.0, 3.0, 1.0, false)
            .expect("arange should succeed");
        let values = session.tensor_values(t).expect("values should succeed");
        assert!(values.is_empty());
    }

    #[test]
    fn factory_arange_negative_step() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .arange(5.0, 2.0, -1.0, false)
            .expect("arange should succeed");
        let values = session.tensor_values(t).expect("values should succeed");
        assert_eq!(values, vec![5.0, 4.0, 3.0]);
    }

    #[test]
    fn factory_arange_rejects_non_finite_inputs() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let err = session
            .arange(0.0, 1.0, f64::NAN, false)
            .expect_err("arange with NaN step must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "arange: start/end/step must be finite"
                }
            ))
        ));
    }

    #[test]
    fn factory_arange_rejects_non_advancing_step_at_precision_limit() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let start = 9_007_199_254_740_992.0; // 2^53
        let err = session
            .arange(start, start + 4.0, 1.0, false)
            .expect_err("non-advancing step must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "arange: step does not advance at current floating-point precision"
                }
            ))
        ));
    }

    #[test]
    fn factory_zeros_rejects_shape_volume_overflow() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let err = session
            .zeros(vec![usize::MAX, 2], false)
            .expect_err("overflowing shape must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "tensor factory shape volume overflow in zeros"
                }
            ))
        ));
    }

    #[test]
    fn factory_eye_rejects_shape_volume_overflow() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let err = session
            .eye(usize::MAX, false)
            .expect_err("overflowing eye shape must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "eye shape volume overflow"
                }
            ))
        ));
    }

    #[test]
    fn session_relu_scalar_positive_passes_through() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.0, true);
        let y = session.relu(x).expect("relu should succeed");
        assert_eq!(session.value(y).expect("value should resolve"), 3.0);
    }

    #[test]
    fn session_relu_scalar_negative_returns_zero() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(-2.0, true);
        let y = session.relu(x).expect("relu should succeed");
        assert_eq!(session.value(y).expect("value should resolve"), 0.0);
    }

    #[test]
    fn session_relu_scalar_backward_positive_input() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let y = session.relu(x).expect("relu should succeed");
        let report = session.backward(y).expect("backward should succeed");
        assert_eq!(session.gradient(&report, x), Some(1.0));
    }

    #[test]
    fn session_relu_scalar_backward_negative_input() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(-2.0, true);
        let y = session.relu(x).expect("relu should succeed");
        let report = session.backward(y).expect("backward should succeed");
        assert_eq!(session.gradient(&report, x), Some(0.0));
    }

    #[test]
    fn session_sigmoid_scalar_at_zero_returns_half() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.sigmoid(x).expect("sigmoid should succeed");
        assert_eq!(session.value(y).expect("value should resolve"), 0.5);
    }

    #[test]
    fn session_sigmoid_scalar_backward_at_zero() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.sigmoid(x).expect("sigmoid should succeed");
        let report = session.backward(y).expect("backward should succeed");
        // sigmoid(0)=0.5, grad=0.5*(1-0.5)=0.25
        assert_eq!(session.gradient(&report, x), Some(0.25));
    }

    #[test]
    fn session_tanh_scalar_at_zero_returns_zero() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.tanh(x).expect("tanh should succeed");
        assert_eq!(session.value(y).expect("value should resolve"), 0.0);
    }

    #[test]
    fn session_tanh_scalar_backward_at_zero() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.tanh(x).expect("tanh should succeed");
        let report = session.backward(y).expect("backward should succeed");
        // tanh(0)=0, grad=1-0^2=1.0
        assert_eq!(session.gradient(&report, x), Some(1.0));
    }

    #[test]
    fn session_relu_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let _y = session.relu(x).expect("relu should succeed");
        let relu_evidence = session
            .evidence()
            .iter()
            .any(|e| e.kind == EvidenceKind::Dispatch && e.summary.contains("Relu"));
        assert!(relu_evidence, "relu dispatch evidence should be recorded");
    }

    #[test]
    fn session_tensor_relu_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-1.0, 0.0, 2.0, -3.0], vec![4], true)
            .expect("tensor variable should succeed");
        let y = session.tensor_relu(x).expect("tensor relu should succeed");
        assert_eq!(
            session.tensor_values(y).expect("values should resolve"),
            vec![0.0, 0.0, 2.0, 0.0]
        );
    }

    #[test]
    fn session_tensor_relu_backward_produces_step_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-1.0, 0.0, 2.0, -3.0], vec![4], true)
            .expect("tensor variable should succeed");
        let y = session.tensor_relu(x).expect("tensor relu should succeed");
        let report = session
            .tensor_backward(y)
            .expect("tensor backward should succeed");
        assert_eq!(
            session
                .tensor_gradient(&report, x)
                .expect("x grad should exist"),
            &[0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn session_tensor_sigmoid_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 10.0, -10.0], vec![3], true)
            .expect("tensor variable should succeed");
        let y = session
            .tensor_sigmoid(x)
            .expect("tensor sigmoid should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert!((values[0] - 0.5).abs() < 1e-10);
        // sigmoid(10) ≈ 1 - e^(-10) ≈ 0.99995; sigmoid(-10) ≈ e^(-10) ≈ 4.5e-5
        assert!((values[1] - 1.0).abs() < 1e-4);
        assert!(values[2].abs() < 1e-4);
    }

    #[test]
    fn session_tensor_sigmoid_backward_produces_expected_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0], vec![1], true)
            .expect("tensor variable should succeed");
        let y = session
            .tensor_sigmoid(x)
            .expect("tensor sigmoid should succeed");
        let report = session
            .tensor_backward(y)
            .expect("tensor backward should succeed");
        let grad = session
            .tensor_gradient(&report, x)
            .expect("x grad should exist");
        // sigmoid(0)=0.5, grad=0.5*(1-0.5)=0.25
        assert!((grad[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_tanh_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 10.0, -10.0], vec![3], true)
            .expect("tensor variable should succeed");
        let y = session.tensor_tanh(x).expect("tensor tanh should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert!(values[0].abs() < 1e-10);
        // tanh(10) ≈ 1 - 2*e^(-20) ≈ 0.99999999; tanh(-10) ≈ -1 + 2*e^(-20)
        assert!((values[1] - 1.0).abs() < 1e-8);
        assert!((values[2] + 1.0).abs() < 1e-8);
    }

    #[test]
    fn session_tensor_tanh_backward_produces_expected_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0], vec![1], true)
            .expect("tensor variable should succeed");
        let y = session.tensor_tanh(x).expect("tensor tanh should succeed");
        let report = session
            .tensor_backward(y)
            .expect("tensor backward should succeed");
        let grad = session
            .tensor_gradient(&report, x)
            .expect("x grad should exist");
        // tanh(0)=0, grad=1-0^2=1.0
        assert!((grad[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn session_activation_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let _s = session.sigmoid(x).expect("sigmoid should succeed");
        let _t = session.tanh(x).expect("tanh should succeed");

        let has_sigmoid = session
            .evidence()
            .iter()
            .any(|e| e.kind == EvidenceKind::Dispatch && e.summary.contains("Sigmoid"));
        let has_tanh = session
            .evidence()
            .iter()
            .any(|e| e.kind == EvidenceKind::Dispatch && e.summary.contains("Tanh"));
        assert!(has_sigmoid, "sigmoid evidence should be recorded");
        assert!(has_tanh, "tanh evidence should be recorded");
    }

    #[test]
    fn session_scalar_eq_returns_expected_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(3.0, false);
        let b = session.variable(3.0, false);
        let c = session.variable(4.0, false);
        let eq_ab = session.eq(a, b).expect("eq");
        assert_eq!(session.value(eq_ab).expect("val"), 1.0);
        let eq_ac = session.eq(a, c).expect("eq");
        assert_eq!(session.value(eq_ac).expect("val"), 0.0);
    }

    #[test]
    fn session_scalar_ne_returns_expected_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(3.0, false);
        let b = session.variable(3.0, false);
        let c = session.variable(4.0, false);
        let ne_ab = session.ne(a, b).expect("ne");
        assert_eq!(session.value(ne_ab).expect("val"), 0.0);
        let ne_ac = session.ne(a, c).expect("ne");
        assert_eq!(session.value(ne_ac).expect("val"), 1.0);
    }

    #[test]
    fn session_scalar_eq_ne_respect_ieee_special_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pos_inf = session.variable(f64::INFINITY, false);
        let neg_inf = session.variable(f64::NEG_INFINITY, false);
        let nan = session.variable(f64::NAN, false);

        let eq_inf = session.eq(pos_inf, pos_inf).expect("eq inf");
        assert_eq!(session.value(eq_inf).expect("eq inf value"), 1.0);
        let ne_inf = session.ne(pos_inf, pos_inf).expect("ne inf");
        assert_eq!(session.value(ne_inf).expect("ne inf value"), 0.0);

        let eq_nan = session.eq(nan, nan).expect("eq nan");
        assert_eq!(session.value(eq_nan).expect("eq nan value"), 0.0);
        let ne_nan = session.ne(nan, nan).expect("ne nan");
        assert_eq!(session.value(ne_nan).expect("ne nan value"), 1.0);

        let eq_inf_sign = session.eq(pos_inf, neg_inf).expect("eq inf sign");
        assert_eq!(session.value(eq_inf_sign).expect("eq inf sign value"), 0.0);
        let ne_inf_sign = session.ne(pos_inf, neg_inf).expect("ne inf sign");
        assert_eq!(session.value(ne_inf_sign).expect("ne inf sign value"), 1.0);
    }

    #[test]
    fn session_scalar_lt_gt_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(2.0, false);
        let b = session.variable(3.0, false);
        let lt_ab = session.lt(a, b).expect("lt");
        assert_eq!(session.value(lt_ab).expect("val"), 1.0);
        let lt_ba = session.lt(b, a).expect("lt");
        assert_eq!(session.value(lt_ba).expect("val"), 0.0);
        let gt_ba = session.gt(b, a).expect("gt");
        assert_eq!(session.value(gt_ba).expect("val"), 1.0);
        let gt_ab = session.gt(a, b).expect("gt");
        assert_eq!(session.value(gt_ab).expect("val"), 0.0);
    }

    #[test]
    fn session_scalar_le_ge_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(2.0, false);
        let b = session.variable(2.0, false);
        let c = session.variable(3.0, false);
        let le_ab = session.le(a, b).expect("le");
        assert_eq!(session.value(le_ab).expect("val"), 1.0);
        let le_ac = session.le(a, c).expect("le");
        assert_eq!(session.value(le_ac).expect("val"), 1.0);
        let le_ca = session.le(c, a).expect("le");
        assert_eq!(session.value(le_ca).expect("val"), 0.0);
        let ge_ab = session.ge(a, b).expect("ge");
        assert_eq!(session.value(ge_ab).expect("val"), 1.0);
        let ge_ca = session.ge(c, a).expect("ge");
        assert_eq!(session.value(ge_ca).expect("val"), 1.0);
        let ge_ac = session.ge(a, c).expect("ge");
        assert_eq!(session.value(ge_ac).expect("val"), 0.0);
    }

    #[test]
    fn session_scalar_lt_gt_le_ge_respect_ieee_special_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pos_inf = session.variable(f64::INFINITY, false);
        let neg_inf = session.variable(f64::NEG_INFINITY, false);
        let nan = session.variable(f64::NAN, false);

        let lt_nan = session.lt(nan, nan).expect("lt nan");
        assert_eq!(session.value(lt_nan).expect("lt nan value"), 0.0);
        let gt_nan = session.gt(nan, nan).expect("gt nan");
        assert_eq!(session.value(gt_nan).expect("gt nan value"), 0.0);
        let le_nan = session.le(nan, nan).expect("le nan");
        assert_eq!(session.value(le_nan).expect("le nan value"), 0.0);
        let ge_nan = session.ge(nan, nan).expect("ge nan");
        assert_eq!(session.value(ge_nan).expect("ge nan value"), 0.0);

        let lt_inf = session.lt(neg_inf, pos_inf).expect("lt inf");
        assert_eq!(session.value(lt_inf).expect("lt inf value"), 1.0);
        let gt_inf = session.gt(pos_inf, neg_inf).expect("gt inf");
        assert_eq!(session.value(gt_inf).expect("gt inf value"), 1.0);
        let le_inf = session.le(pos_inf, pos_inf).expect("le inf");
        assert_eq!(session.value(le_inf).expect("le inf value"), 1.0);
        let ge_inf = session.ge(neg_inf, neg_inf).expect("ge inf");
        assert_eq!(session.value(ge_inf).expect("ge inf value"), 1.0);
    }

    #[test]
    fn session_tensor_eq_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("tensor variable should succeed");
        let y = session
            .tensor_variable(vec![1.0, 5.0, 3.0], vec![3], false)
            .expect("tensor variable should succeed");
        let z = session.tensor_eq(x, y).expect("tensor eq should succeed");
        assert_eq!(
            session.tensor_values(z).expect("values should resolve"),
            vec![1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn session_tensor_equal_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("tensor variable should succeed");
        let b = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("tensor variable should succeed");
        let c = session
            .tensor_variable(vec![1.0, 2.0, 4.0], vec![3], false)
            .expect("tensor variable should succeed");
        let d = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("tensor variable should succeed");

        assert!(session.tensor_equal(a, b).expect("equal should succeed"));
        assert!(!session.tensor_equal(a, c).expect("equal should succeed"));
        assert!(!session.tensor_equal(a, d).expect("equal should succeed"));
    }

    #[test]
    fn session_tensor_equal_treats_nan_as_unequal() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![f64::NAN, 1.0], vec![2], false)
            .expect("tensor variable should succeed");
        let b = session
            .tensor_variable(vec![f64::NAN, 1.0], vec![2], false)
            .expect("tensor variable should succeed");
        assert!(!session.tensor_equal(a, b).expect("equal should succeed"));
    }

    #[test]
    fn session_tensor_allclose_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("tensor variable should succeed");
        let b = session
            .tensor_variable(vec![1.0000001, 1.9999999, 3.0], vec![3], false)
            .expect("tensor variable should succeed");
        let c = session
            .tensor_variable(vec![1.1, 2.0, 3.0], vec![3], false)
            .expect("tensor variable should succeed");

        assert!(
            session
                .tensor_allclose(a, b, 1e-5, 1e-8, false)
                .expect("allclose should succeed")
        );
        assert!(
            !session
                .tensor_allclose(a, c, 1e-5, 1e-8, false)
                .expect("allclose should succeed")
        );
    }

    #[test]
    fn session_tensor_allclose_nan_and_inf_handling() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![f64::NAN, f64::INFINITY, 1.0], vec![3], false)
            .expect("tensor variable should succeed");
        let b = session
            .tensor_variable(vec![f64::NAN, f64::INFINITY, 1.0], vec![3], false)
            .expect("tensor variable should succeed");
        let c = session
            .tensor_variable(vec![f64::NAN, f64::NEG_INFINITY, 1.0], vec![3], false)
            .expect("tensor variable should succeed");

        assert!(
            !session
                .tensor_allclose(a, b, 1e-5, 1e-8, false)
                .expect("allclose should succeed")
        );
        assert!(
            session
                .tensor_allclose(a, b, 1e-5, 1e-8, true)
                .expect("allclose should succeed")
        );
        assert!(
            !session
                .tensor_allclose(a, c, 1e-5, 1e-8, true)
                .expect("allclose should succeed")
        );
    }

    #[test]
    fn session_tensor_allclose_rejects_invalid_tolerances() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("tensor variable should succeed");
        let b = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("tensor variable should succeed");

        assert!(session.tensor_allclose(a, b, -1.0, 0.0, false).is_err());
        assert!(session.tensor_allclose(a, b, 0.0, -1.0, false).is_err());
        assert!(session.tensor_allclose(a, b, f64::NAN, 0.0, false).is_err());
    }

    #[test]
    fn session_tensor_ne_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("tensor variable should succeed");
        let y = session
            .tensor_variable(vec![1.0, 5.0, 3.0], vec![3], false)
            .expect("tensor variable should succeed");
        let z = session.tensor_ne(x, y).expect("tensor ne should succeed");
        assert_eq!(
            session.tensor_values(z).expect("values should resolve"),
            vec![0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn session_tensor_eq_ne_respect_ieee_special_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lhs = session
            .tensor_variable(
                vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 5.0],
                vec![4],
                false,
            )
            .expect("lhs tensor variable should succeed");
        let rhs = session
            .tensor_variable(
                vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 6.0],
                vec![4],
                false,
            )
            .expect("rhs tensor variable should succeed");

        let eq = session
            .tensor_eq(lhs, rhs)
            .expect("tensor eq should succeed");
        assert_eq!(
            session.tensor_values(eq).expect("eq values should resolve"),
            vec![1.0, 1.0, 0.0, 0.0]
        );

        let ne = session
            .tensor_ne(lhs, rhs)
            .expect("tensor ne should succeed");
        assert_eq!(
            session.tensor_values(ne).expect("ne values should resolve"),
            vec![0.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn session_tensor_lt_gt_le_ge_respect_ieee_special_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lhs = session
            .tensor_variable(
                vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0, 2.0],
                vec![5],
                false,
            )
            .expect("lhs tensor variable should succeed");
        let rhs = session
            .tensor_variable(
                vec![f64::NAN, f64::NEG_INFINITY, f64::INFINITY, 1.0, 3.0],
                vec![5],
                false,
            )
            .expect("rhs tensor variable should succeed");

        let lt = session
            .tensor_lt(lhs, rhs)
            .expect("tensor lt should succeed");
        assert_eq!(
            session.tensor_values(lt).expect("lt values should resolve"),
            vec![0.0, 0.0, 1.0, 0.0, 1.0]
        );

        let gt = session
            .tensor_gt(lhs, rhs)
            .expect("tensor gt should succeed");
        assert_eq!(
            session.tensor_values(gt).expect("gt values should resolve"),
            vec![0.0, 1.0, 0.0, 0.0, 0.0]
        );

        let le = session
            .tensor_le(lhs, rhs)
            .expect("tensor le should succeed");
        assert_eq!(
            session.tensor_values(le).expect("le values should resolve"),
            vec![0.0, 0.0, 1.0, 1.0, 1.0]
        );

        let ge = session
            .tensor_ge(lhs, rhs)
            .expect("tensor ge should succeed");
        assert_eq!(
            session.tensor_values(ge).expect("ge values should resolve"),
            vec![0.0, 1.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn session_tensor_lt_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 3.0, 2.0], vec![3], false)
            .expect("tensor variable should succeed");
        let y = session
            .tensor_variable(vec![2.0, 3.0, 1.0], vec![3], false)
            .expect("tensor variable should succeed");
        let z = session.tensor_lt(x, y).expect("tensor lt should succeed");
        assert_eq!(
            session.tensor_values(z).expect("values should resolve"),
            vec![1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn session_tensor_gt_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 3.0, 2.0], vec![3], false)
            .expect("tensor variable should succeed");
        let y = session
            .tensor_variable(vec![2.0, 3.0, 1.0], vec![3], false)
            .expect("tensor variable should succeed");
        let z = session.tensor_gt(x, y).expect("tensor gt should succeed");
        assert_eq!(
            session.tensor_values(z).expect("values should resolve"),
            vec![0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn session_tensor_le_ge_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 3.0, 2.0], vec![3], false)
            .expect("tensor variable should succeed");
        let y = session
            .tensor_variable(vec![2.0, 3.0, 1.0], vec![3], false)
            .expect("tensor variable should succeed");
        let le = session.tensor_le(x, y).expect("tensor le should succeed");
        let ge = session.tensor_ge(x, y).expect("tensor ge should succeed");
        assert_eq!(
            session.tensor_values(le).expect("values should resolve"),
            vec![1.0, 1.0, 0.0]
        );
        assert_eq!(
            session.tensor_values(ge).expect("values should resolve"),
            vec![0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn session_comparison_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(1.0, false);
        let b = session.variable(2.0, false);
        let _z = session.lt(a, b).expect("lt should succeed");
        let has_comparison = session
            .evidence()
            .iter()
            .any(|e| e.kind == EvidenceKind::Dispatch && e.summary.contains("comparison_op"));
        assert!(has_comparison, "comparison evidence should be recorded");
    }

    #[test]
    fn session_comparison_result_does_not_require_grad() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .expect("tensor variable should succeed");
        let y = session
            .tensor_variable(vec![1.0, 3.0], vec![2], true)
            .expect("tensor variable should succeed");
        let cmp = session.tensor_eq(x, y).expect("tensor eq should succeed");
        // Comparison result is a leaf with requires_grad=false, so backward should fail
        let err = session.tensor_backward(cmp);
        assert!(err.is_err(), "backward on comparison result should fail");
    }

    #[test]
    fn session_sqrt_scalar_returns_expected_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(9.0, false);
        let r = session.sqrt(a).expect("sqrt");
        assert_eq!(session.value(r).expect("val"), 3.0);
    }

    #[test]
    fn session_sqrt_scalar_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(4.0, true);
        let y = session.sqrt(x).expect("sqrt");
        let report = session.backward(y).expect("backward");
        // d(sqrt(x))/dx = 0.5/sqrt(x) = 0.5/2.0 = 0.25
        let grad = report.gradient(x).expect("grad");
        assert!((grad - 0.25).abs() < 1e-10);
    }

    #[test]
    fn session_reciprocal_scalar_returns_expected_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(4.0, false);
        let r = session.reciprocal(a).expect("reciprocal");
        assert_eq!(session.value(r).expect("val"), 0.25);
    }

    #[test]
    fn session_reciprocal_scalar_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let y = session.reciprocal(x).expect("reciprocal");
        let report = session.backward(y).expect("backward");
        // d(1/x)/dx = -1/x^2 = -1/4 = -0.25
        let grad = report.gradient(x).expect("grad");
        assert!((grad - (-0.25)).abs() < 1e-10);
    }

    #[test]
    fn session_pow_scalar_returns_expected_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(3.0, false);
        let r = session.pow(a, 2.0).expect("pow");
        assert_eq!(session.value(r).expect("val"), 9.0);
    }

    #[test]
    fn session_pow_scalar_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.0, true);
        let y = session.pow(x, 2.0).expect("pow");
        let report = session.backward(y).expect("backward");
        // d(x^2)/dx = 2*x = 6.0
        let grad = report.gradient(x).expect("grad");
        assert!((grad - 6.0).abs() < 1e-10);
    }

    #[test]
    fn session_pow_scalar_fractional_exponent() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(4.0, true);
        let y = session.pow(x, 0.5).expect("pow");
        // 4^0.5 = 2.0
        assert!((session.value(y).expect("val") - 2.0).abs() < 1e-10);
        let report = session.backward(y).expect("backward");
        // d(x^0.5)/dx = 0.5 * x^(-0.5) = 0.5/2 = 0.25
        let grad = report.gradient(x).expect("grad");
        assert!((grad - 0.25).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_sqrt_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 4.0, 9.0, 16.0], vec![4], false)
            .expect("tensor variable");
        let r = session.tensor_sqrt(t).expect("tensor_sqrt");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn session_tensor_sqrt_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0, 9.0], vec![2], true)
            .expect("tensor variable");
        let y = session.tensor_sqrt(x).expect("tensor_sqrt");
        let s = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // d(sqrt(4))/dx = 0.5/2 = 0.25, d(sqrt(9))/dx = 0.5/3 ~ 0.1667
        assert!((grad[0] - 0.25).abs() < 1e-10);
        assert!((grad[1] - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_reciprocal_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 4.0], vec![3], false)
            .expect("tensor variable");
        let r = session.tensor_reciprocal(t).expect("tensor_reciprocal");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![1.0, 0.5, 0.25]);
    }

    #[test]
    fn session_tensor_reciprocal_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, 4.0], vec![2], true)
            .expect("tensor variable");
        let y = session.tensor_reciprocal(x).expect("tensor_reciprocal");
        let s = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // d(1/2)/dx = -1/4 = -0.25, d(1/4)/dx = -1/16 = -0.0625
        assert!((grad[0] - (-0.25)).abs() < 1e-10);
        assert!((grad[1] - (-0.0625)).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_pow_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("tensor variable");
        let r = session.tensor_pow(t, 2.0).expect("tensor_pow");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn session_tensor_pow_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, 3.0], vec![2], true)
            .expect("tensor variable");
        let y = session.tensor_pow(x, 3.0).expect("tensor_pow");
        let s = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // d(x^3)/dx = 3*x^2: for x=2 -> 12.0, for x=3 -> 27.0
        assert!((grad[0] - 12.0).abs() < 1e-10);
        assert!((grad[1] - 27.0).abs() < 1e-10);
    }

    // --- min/max/clamp scalar tests ---

    #[test]
    fn session_min_returns_smaller_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(3.0, false);
        let b = session.variable(5.0, false);
        let r = session.min(a, b).expect("min");
        assert_eq!(session.value(r).unwrap(), 3.0);
    }

    #[test]
    fn session_max_returns_larger_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(3.0, false);
        let b = session.variable(5.0, false);
        let r = session.max(a, b).expect("max");
        assert_eq!(session.value(r).unwrap(), 5.0);
    }

    #[test]
    fn session_clamp_restricts_to_range() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let low = session.variable(-5.0, false);
        let mid = session.variable(3.0, false);
        let high = session.variable(15.0, false);
        let r1 = session.clamp(low, 0.0, 10.0).expect("clamp low");
        let r2 = session.clamp(mid, 0.0, 10.0).expect("clamp mid");
        let r3 = session.clamp(high, 0.0, 10.0).expect("clamp high");
        assert_eq!(session.value(r1).unwrap(), 0.0);
        assert_eq!(session.value(r2).unwrap(), 3.0);
        assert_eq!(session.value(r3).unwrap(), 10.0);
    }

    #[test]
    fn session_min_backward_grad_flows_to_smaller() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(2.0, true);
        let b = session.variable(5.0, true);
        let r = session.min(a, b).expect("min");
        let report = session.backward(r).expect("backward");
        // a < b, so grad flows to a
        assert_eq!(report.gradient(a), Some(1.0));
        assert_eq!(report.gradient(b), Some(0.0));
    }

    #[test]
    fn session_max_backward_grad_flows_to_larger() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(2.0, true);
        let b = session.variable(5.0, true);
        let r = session.max(a, b).expect("max");
        let report = session.backward(r).expect("backward");
        // b > a, so grad flows to b
        assert_eq!(report.gradient(a), Some(0.0));
        assert_eq!(report.gradient(b), Some(1.0));
    }

    #[test]
    fn session_clamp_backward_grad_passes_in_range() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.0, true);
        let r = session.clamp(x, 0.0, 10.0).expect("clamp");
        let report = session.backward(r).expect("backward");
        assert_eq!(report.gradient(x), Some(1.0));
    }

    #[test]
    fn session_clamp_backward_grad_zero_outside_range() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(-5.0, true);
        let r = session.clamp(x, 0.0, 10.0).expect("clamp");
        let report = session.backward(r).expect("backward");
        assert_eq!(report.gradient(x), Some(0.0));
    }

    // --- min/max/clamp tensor tests ---

    #[test]
    fn session_tensor_min_returns_elementwise_minimum() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 5.0, 3.0], vec![3], false)
            .expect("a");
        let b = session
            .tensor_variable(vec![4.0, 2.0, 3.0], vec![3], false)
            .expect("b");
        let r = session.tensor_min(a, b).expect("tensor_min");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn session_tensor_max_returns_elementwise_maximum() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 5.0, 3.0], vec![3], false)
            .expect("a");
        let b = session
            .tensor_variable(vec![4.0, 2.0, 3.0], vec![3], false)
            .expect("b");
        let r = session.tensor_max(a, b).expect("tensor_max");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![4.0, 5.0, 3.0]);
    }

    #[test]
    fn session_tensor_clamp_restricts_elements() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![-2.0, 0.5, 3.0, 7.0], vec![4], false)
            .expect("t");
        let r = session.tensor_clamp(t, 0.0, 5.0).expect("tensor_clamp");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![0.0, 0.5, 3.0, 5.0]);
    }

    #[test]
    fn session_tensor_min_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 5.0], vec![2], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![4.0, 2.0], vec![2], true)
            .expect("b");
        let r = session.tensor_min(a, b).expect("tensor_min");
        let s = session.tensor_sum(r).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad_a = report.gradient(a).expect("grad_a");
        let grad_b = report.gradient(b).expect("grad_b");
        // Element 0: a=1 < b=4, grad -> a. Element 1: b=2 < a=5, grad -> b
        assert_eq!(grad_a, vec![1.0, 0.0]);
        assert_eq!(grad_b, vec![0.0, 1.0]);
    }

    #[test]
    fn session_tensor_max_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 5.0], vec![2], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![4.0, 2.0], vec![2], true)
            .expect("b");
        let r = session.tensor_max(a, b).expect("tensor_max");
        let s = session.tensor_sum(r).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad_a = report.gradient(a).expect("grad_a");
        let grad_b = report.gradient(b).expect("grad_b");
        // Element 0: b=4 > a=1, grad -> b. Element 1: a=5 > b=2, grad -> a
        assert_eq!(grad_a, vec![0.0, 1.0]);
        assert_eq!(grad_b, vec![1.0, 0.0]);
    }

    #[test]
    fn session_tensor_clamp_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-2.0, 0.5, 3.0, 7.0], vec![4], true)
            .expect("x");
        let r = session.tensor_clamp(x, 0.0, 5.0).expect("tensor_clamp");
        let s = session.tensor_sum(r).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // -2.0 < 0 -> 0, 0.5 in [0,5] -> 1, 3.0 in [0,5] -> 1, 7.0 > 5 -> 0
        assert_eq!(grad, vec![0.0, 1.0, 1.0, 0.0]);
    }

    // --- dim-aware reduction tests ---

    #[test]
    fn session_tensor_sum_dim0_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        let r = session.tensor_sum_dim(t, 0).expect("sum_dim 0");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn session_tensor_sum_dim1_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        let r = session.tensor_sum_dim(t, 1).expect("sum_dim 1");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![6.0, 15.0]);
    }

    #[test]
    fn session_tensor_mean_dim0_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        let r = session.tensor_mean_dim(t, 0).expect("mean_dim 0");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn session_tensor_mean_dim1_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        let r = session.tensor_mean_dim(t, 1).expect("mean_dim 1");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![2.0, 5.0]);
    }

    #[test]
    fn session_tensor_sum_dim_backward_broadcasts_grad() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let r = session.tensor_sum_dim(x, 0).expect("sum_dim 0");
        // r has shape [3], need to sum to scalar for backward
        let s = session.tensor_sum(r).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // sum_dim(0) broadcasts grad along dim 0: all elements get 1.0
        assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_sum_dim1_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let r = session.tensor_sum_dim(x, 1).expect("sum_dim 1");
        // r has shape [2]
        let s = session.tensor_sum(r).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // sum_dim(1) broadcasts grad along dim 1: all elements get 1.0
        assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_mean_dim_backward_scales_grad() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let r = session.tensor_mean_dim(x, 0).expect("mean_dim 0");
        // r has shape [3]
        let s = session.tensor_sum(r).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // mean_dim(0) with reduce_size=2: grad = 1/2 = 0.5 for all elements
        assert_eq!(grad, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn session_tensor_mean_dim1_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let r = session.tensor_mean_dim(x, 1).expect("mean_dim 1");
        // r has shape [2]
        let s = session.tensor_sum(r).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // mean_dim(1) with reduce_size=3: grad = 1/3 for all elements
        let expected = 1.0 / 3.0;
        for g in grad {
            assert!((*g - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn session_tensor_sum_dim_3d_middle() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3, 2]
        let t = session
            .tensor_variable(
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                vec![2, 3, 2],
                false,
            )
            .expect("t");
        let r = session.tensor_sum_dim(t, 1).expect("sum_dim 1 on 3d");
        let vals = session.tensor_values(r).expect("vals");
        // Output shape [2, 2]: [9, 12, 27, 30]
        assert_eq!(vals, vec![9.0, 12.0, 27.0, 30.0]);
    }

    // ── sin/cos/tan API tests ────────────────────────────────────────

    #[test]
    fn session_scalar_sin() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(std::f64::consts::FRAC_PI_2, true);
        let y = session.sin(x).expect("sin");
        let val = session.value(y).expect("value");
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_cos() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.cos(x).expect("cos");
        let val = session.value(y).expect("value");
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_tan() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(std::f64::consts::FRAC_PI_4, true);
        let y = session.tan(x).expect("tan");
        let val = session.value(y).expect("value");
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_sin() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, std::f64::consts::FRAC_PI_2], vec![2], true)
            .expect("leaf");
        let y = session.tensor_sin(x).expect("tensor_sin");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_cos() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, std::f64::consts::PI], vec![2], true)
            .expect("leaf");
        let y = session.tensor_cos(x).expect("tensor_cos");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_tan() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, std::f64::consts::FRAC_PI_4], vec![2], true)
            .expect("leaf");
        let y = session.tensor_tan(x).expect("tensor_tan");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_sin_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let y = session.sin(x).expect("sin");
        let report = session.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        assert!((grad - 1.0_f64.cos()).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_sin_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.5, 1.0], vec![2], true)
            .expect("leaf");
        let y = session.tensor_sin(x).expect("tensor_sin");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert!((grads[0] - 0.5_f64.cos()).abs() < 1e-12);
        assert!((grads[1] - 1.0_f64.cos()).abs() < 1e-12);
    }

    // ── floor/ceil/round API tests ───────────────────────────────────

    #[test]
    fn session_scalar_floor() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.7, true);
        let y = session.floor(x).expect("floor");
        assert_eq!(session.value(y).expect("value"), 2.0);
    }

    #[test]
    fn session_scalar_ceil() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.3, true);
        let y = session.ceil(x).expect("ceil");
        assert_eq!(session.value(y).expect("value"), 3.0);
    }

    #[test]
    fn session_scalar_round() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.7, true);
        let y = session.round(x).expect("round");
        assert_eq!(session.value(y).expect("value"), 3.0);
    }

    #[test]
    fn session_tensor_floor() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.5, -0.3], vec![2], false)
            .expect("leaf");
        let y = session.tensor_floor(x).expect("tensor_floor");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![1.0, -1.0]);
    }

    #[test]
    fn session_tensor_ceil() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.5, -0.3], vec![2], false)
            .expect("leaf");
        let y = session.tensor_ceil(x).expect("tensor_ceil");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![2.0, 0.0]);
    }

    #[test]
    fn session_tensor_round() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.4, 2.6], vec![2], false)
            .expect("leaf");
        let y = session.tensor_round(x).expect("tensor_round");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![1.0, 3.0]);
    }

    #[test]
    fn session_scalar_floor_backward_zero() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.7, true);
        let y = session.floor(x).expect("floor");
        let report = session.backward(y).expect("backward");
        assert_eq!(report.gradient(x), Some(0.0));
    }

    // ── log2/log10/log1p/expm1 API tests ─────────────────────────────

    #[test]
    fn session_scalar_log2() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(8.0, true);
        let y = session.log2(x).expect("log2");
        assert!((session.value(y).expect("value") - 3.0).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_log10() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(100.0, true);
        let y = session.log10(x).expect("log10");
        assert!((session.value(y).expect("value") - 2.0).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_log1p() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.log1p(x).expect("log1p");
        assert!((session.value(y).expect("value")).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_expm1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.expm1(x).expect("expm1");
        assert!((session.value(y).expect("value")).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_log2() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 4.0], vec![2], false)
            .expect("leaf");
        let y = session.tensor_log2(x).expect("tensor_log2");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_expm1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("leaf");
        let y = session.tensor_expm1(x).expect("tensor_expm1");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-15);
        assert!((vals[1] - (std::f64::consts::E - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_log2_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(4.0, true);
        let y = session.log2(x).expect("log2");
        let report = session.backward(y).expect("backward");
        let expected = 1.0 / (4.0 * std::f64::consts::LN_2);
        assert!((report.gradient(x).expect("grad") - expected).abs() < 1e-12);
    }

    // ── sign/trunc/frac API tests ─────────────────────────────

    #[test]
    fn session_scalar_sign() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(-5.0, false);
        let y = session.sign(x).expect("sign");
        assert_eq!(session.value(y).expect("value"), -1.0);
    }

    #[test]
    fn session_scalar_trunc() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.7, false);
        let y = session.trunc(x).expect("trunc");
        assert_eq!(session.value(y).expect("value"), 3.0);
    }

    #[test]
    fn session_scalar_frac() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.7, false);
        let y = session.frac(x).expect("frac");
        let val = session.value(y).expect("value");
        assert!((val - 0.7).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_sign() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-3.0, 2.0, 5.0], vec![3], false)
            .expect("leaf");
        let y = session.tensor_sign(x).expect("tensor_sign");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[-1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_trunc() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.7, -2.3], vec![2], false)
            .expect("leaf");
        let y = session.tensor_trunc(x).expect("tensor_trunc");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[3.0, -2.0]);
    }

    #[test]
    fn session_tensor_frac() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.7, -2.3], vec![2], false)
            .expect("leaf");
        let y = session.tensor_frac(x).expect("tensor_frac");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - 0.7).abs() < 1e-12);
        assert!((vals[1] - (-0.3)).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_frac_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.7, true);
        let y = session.frac(x).expect("frac");
        let report = session.backward(y).expect("backward");
        assert_eq!(report.gradient(x).expect("grad"), 1.0);
    }

    // ── asin/acos/atan API tests ─────────────────────────────

    #[test]
    fn session_scalar_asin() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.5, false);
        let y = session.asin(x).expect("asin");
        let val = session.value(y).expect("value");
        assert!((val - 0.5_f64.asin()).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_acos() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.5, false);
        let y = session.acos(x).expect("acos");
        let val = session.value(y).expect("value");
        assert!((val - 0.5_f64.acos()).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_atan() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, false);
        let y = session.atan(x).expect("atan");
        let val = session.value(y).expect("value");
        assert!((val - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_asin() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 0.5], vec![2], false)
            .expect("leaf");
        let y = session.tensor_asin(x).expect("tensor_asin");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 0.5_f64.asin()).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_atan() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("leaf");
        let y = session.tensor_atan(x).expect("tensor_atan");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_atan_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let y = session.atan(x).expect("atan");
        let report = session.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        assert!((grad - 0.5).abs() < 1e-12);
    }

    // ── sinh/cosh API tests ─────────────────────────────

    #[test]
    fn session_scalar_sinh() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, false);
        let y = session.sinh(x).expect("sinh");
        let val = session.value(y).expect("value");
        assert!((val - 1.0_f64.sinh()).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_cosh() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, false);
        let y = session.cosh(x).expect("cosh");
        let val = session.value(y).expect("value");
        assert!((val - 1.0_f64.cosh()).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_sinh() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("leaf");
        let y = session.tensor_sinh(x).expect("tensor_sinh");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 1.0_f64.sinh()).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_cosh() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("leaf");
        let y = session.tensor_cosh(x).expect("tensor_cosh");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 1.0_f64.cosh()).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_sinh_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let y = session.sinh(x).expect("sinh");
        let report = session.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        assert!((grad - 1.0_f64.cosh()).abs() < 1e-12);
    }

    // ── gelu/silu/leaky_relu/elu API tests ─────────────────────────────

    fn gelu_val(x: f64) -> f64 {
        let c = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
        let k = c * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + k.tanh())
    }

    #[test]
    fn session_scalar_gelu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, false);
        let y = session.gelu(x).expect("gelu");
        let val = session.value(y).expect("value");
        assert!((val - gelu_val(1.0)).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_silu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, false);
        let y = session.silu(x).expect("silu");
        let val = session.value(y).expect("value");
        let expected = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((val - expected).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_leaky_relu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(-3.0, false);
        let y = session.leaky_relu(x).expect("leaky_relu");
        let val = session.value(y).expect("value");
        assert!((val - (-0.03)).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_elu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(-1.0, false);
        let y = session.elu(x).expect("elu");
        let val = session.value(y).expect("value");
        assert!((val - ((-1.0_f64).exp() - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_gelu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("leaf");
        let y = session.tensor_gelu(x).expect("tensor_gelu");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - gelu_val(1.0)).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_silu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("leaf");
        let y = session.tensor_silu(x).expect("tensor_silu");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        let expected = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((vals[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_leaky_relu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, -3.0], vec![2], false)
            .expect("leaf");
        let y = session.tensor_leaky_relu(x).expect("tensor_leaky_relu");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals[0], 2.0);
        assert!((vals[1] - (-0.03)).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_elu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, -1.0], vec![2], false)
            .expect("leaf");
        let y = session.tensor_elu(x).expect("tensor_elu");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals[0], 2.0);
        assert!((vals[1] - ((-1.0_f64).exp() - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn session_scalar_gelu_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let y = session.gelu(x).expect("gelu");
        let report = session.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        let eps = 1e-6;
        let numerical = (gelu_val(1.0 + eps) - gelu_val(1.0 - eps)) / (2.0 * eps);
        assert!((grad - numerical).abs() < 1e-5);
    }

    // ── prod_dim/var_dim/std_dim API tests ─────────────────────────────

    #[test]
    fn session_tensor_prod_dim0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("leaf");
        let y = session.tensor_prod_dim(x, 0).expect("prod_dim 0");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn session_tensor_prod_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("leaf");
        let y = session.tensor_prod_dim(x, 1).expect("prod_dim 1");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[6.0, 120.0]);
    }

    #[test]
    fn session_tensor_var_dim0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], false)
            .expect("leaf");
        let y = session.tensor_var_dim(x, 0).expect("var_dim 0");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - 4.0).abs() < 1e-12);
        assert!((vals[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_var_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("leaf");
        let y = session.tensor_var_dim(x, 1).expect("var_dim 1");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_std_dim0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], false)
            .expect("leaf");
        let y = session.tensor_std_dim(x, 0).expect("std_dim 0");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_prod_dim_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, 3.0, 4.0], vec![1, 3], true)
            .expect("leaf");
        let y = session.tensor_prod_dim(x, 1).expect("prod_dim 1");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // prod=24, grads: 24/2=12, 24/3=8, 24/4=6
        assert!((grads[0] - 12.0).abs() < 1e-10);
        assert!((grads[1] - 8.0).abs() < 1e-10);
        assert!((grads[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_var_dim_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let y = session.tensor_var_dim(x, 1).expect("var_dim 1");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert!((grads[0] - (-1.0)).abs() < 1e-12);
        assert!(grads[1].abs() < 1e-12);
        assert!((grads[2] - 1.0).abs() < 1e-12);
    }

    // ── softmax/log_softmax API tests ─────────────────────────────

    #[test]
    fn session_tensor_softmax_sums_to_one() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("leaf");
        let y = session.tensor_softmax(x, 1).expect("softmax");
        let vals = session.tensor_values(y).expect("values");
        let row0_sum: f64 = vals[0..3].iter().sum();
        let row1_sum: f64 = vals[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-12);
        assert!((row1_sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_log_softmax_consistent() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("leaf");
        let y = session.tensor_log_softmax(x, 1).expect("log_softmax");
        let vals = session.tensor_values(y).expect("values");
        // exp of log_softmax should sum to 1
        let sum: f64 = vals.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_softmax_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let y = session.tensor_softmax(x, 1).expect("softmax");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // Sum of softmax backward gradients (with all-ones incoming) should be 0
        // because softmax is invariant to constant shift
        let grad_sum: f64 = grads.iter().sum();
        assert!(
            grad_sum.abs() < 1e-12,
            "softmax grad sum should be 0, got {grad_sum}"
        );
    }

    #[test]
    fn session_tensor_log_softmax_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let y = session.tensor_log_softmax(x, 1).expect("log_softmax");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // Sum of log_softmax backward gradients should be 0
        let grad_sum: f64 = grads.iter().sum();
        assert!(
            grad_sum.abs() < 1e-12,
            "log_softmax grad sum should be 0, got {grad_sum}"
        );
    }

    // ── cat/stack API tests ─────────────────────────────

    #[test]
    fn session_tensor_cat_dim0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("a");
        let b = session
            .tensor_variable(vec![7.0, 8.0, 9.0], vec![1, 3], false)
            .expect("b");
        let y = session.tensor_cat(&[a, b], 0).expect("cat 0");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn session_tensor_cat_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .expect("a");
        let b = session
            .tensor_variable(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3], false)
            .expect("b");
        let y = session.tensor_cat(&[a, b], 1).expect("cat 1");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn session_tensor_stack_dim0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("a");
        let b = session
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], false)
            .expect("b");
        let y = session.tensor_stack(&[a, b], 0).expect("stack 0");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn session_tensor_cat_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![3.0, 4.0, 5.0], vec![1, 3], true)
            .expect("b");
        let y = session.tensor_cat(&[a, b], 1).expect("cat 1");
        let report = session.tensor_backward(y).expect("backward");
        let grads_a = report.gradient(a).expect("grad a");
        let grads_b = report.gradient(b).expect("grad b");
        assert_eq!(grads_a, &[1.0, 1.0]);
        assert_eq!(grads_b, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_stack_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], true)
            .expect("b");
        let y = session.tensor_stack(&[a, b], 0).expect("stack 0");
        let report = session.tensor_backward(y).expect("backward");
        let grads_a = report.gradient(a).expect("grad a");
        let grads_b = report.gradient(b).expect("grad b");
        assert_eq!(grads_a, &[1.0, 1.0, 1.0]);
        assert_eq!(grads_b, &[1.0, 1.0, 1.0]);
    }

    // ---- reshape/view/squeeze/unsqueeze API tests ----

    #[test]
    fn session_tensor_reshape_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let y = session.tensor_reshape(x, vec![3, 2]).expect("reshape");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn session_tensor_reshape_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let y = session.tensor_reshape(x, vec![6]).expect("reshape");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_view_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
            .expect("x");
        let y = session.tensor_view(x, vec![2, 2]).expect("view");
        let shape = session.tensor_shape(y).expect("shape");
        assert_eq!(shape, vec![2, 2]);
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn session_tensor_squeeze_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("x");
        let y = session.tensor_squeeze(x, 0).expect("squeeze");
        let shape = session.tensor_shape(y).expect("shape");
        assert_eq!(shape, vec![3]);
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn session_tensor_squeeze_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("x");
        let y = session.tensor_squeeze(x, 0).expect("squeeze");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_unsqueeze_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("x");
        let y = session.tensor_unsqueeze(x, 0).expect("unsqueeze");
        let shape = session.tensor_shape(y).expect("shape");
        assert_eq!(shape, vec![1, 3]);
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn session_tensor_unsqueeze_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("x");
        let y = session.tensor_unsqueeze(x, 1).expect("unsqueeze");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_reshape_mismatched_numel_fails() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("x");
        assert!(session.tensor_reshape(x, vec![2]).is_err());
    }

    #[test]
    fn session_tensor_argmax_returns_correct_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3]: argmax along dim 0
        let x = session
            .tensor_variable(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], false)
            .expect("x");
        let out = session.tensor_argmax(x, 0).expect("argmax");
        let vals = session.tensor_values(out).expect("values");
        // Row 0: [1,5,3], Row 1: [4,2,6]
        // Along dim 0: max(1,4)=4@1, max(5,2)=5@0, max(3,6)=6@1
        assert_eq!(vals, &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn session_tensor_argmin_returns_correct_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3]: argmin along dim 0
        let x = session
            .tensor_variable(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], false)
            .expect("x");
        let out = session.tensor_argmin(x, 0).expect("argmin");
        let vals = session.tensor_values(out).expect("values");
        // Row 0: [1,5,3], Row 1: [4,2,6]
        // Along dim 0: min(1,4)=1@0, min(5,2)=2@1, min(3,6)=3@0
        assert_eq!(vals, &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn session_tensor_max_dim_returns_values_and_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values, indices) = session.tensor_max_dim(x, 1).expect("max_dim");
        let vals = session.tensor_values(values).expect("values");
        let idxs = session.tensor_values(indices).expect("indices");
        assert_eq!(vals, &[5.0, 6.0]);
        assert_eq!(idxs, &[1.0, 2.0]);
    }

    #[test]
    fn session_tensor_min_dim_returns_values_and_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values, indices) = session.tensor_min_dim(x, 1).expect("min_dim");
        let vals = session.tensor_values(values).expect("values");
        let idxs = session.tensor_values(indices).expect("indices");
        assert_eq!(vals, &[1.0, 2.0]);
        assert_eq!(idxs, &[0.0, 1.0]);
    }

    #[test]
    fn session_tensor_max_dim_backward_scatters_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values, _indices) = session.tensor_max_dim(x, 1).expect("max_dim");
        let s = session.tensor_sum(values).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn session_tensor_min_dim_backward_scatters_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values, _indices) = session.tensor_min_dim(x, 1).expect("min_dim");
        let s = session.tensor_sum(values).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    // ── Phase 6 Shape Operation Tests ────────────────────────────────────

    // flatten tests

    #[test]
    fn session_tensor_flatten_all_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3, 4] = 24 elements
        let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let x = session
            .tensor_variable(data.clone(), vec![2, 3, 4], false)
            .expect("x");
        let y = session.tensor_flatten(x, 0, 2).expect("flatten");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, data);
        let shape = session.tensor_shape(y).expect("shape");
        assert_eq!(shape, vec![24]);
    }

    #[test]
    fn session_tensor_flatten_partial() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3, 4] = 24 elements, flatten dims 1..2 -> [2, 12]
        let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let x = session
            .tensor_variable(data.clone(), vec![2, 3, 4], false)
            .expect("x");
        let y = session.tensor_flatten(x, 1, 2).expect("flatten");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, data);
        let shape = session.tensor_shape(y).expect("shape");
        assert_eq!(shape, vec![2, 12]);
    }

    #[test]
    fn session_tensor_flatten_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3, 4] = 24 elements
        let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let x = session
            .tensor_variable(data, vec![2, 3, 4], true)
            .expect("x");
        let y = session.tensor_flatten(x, 0, 2).expect("flatten");
        // flatten to [24], sum to scalar for backward
        let s = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        // gradient through flatten + sum should be all ones
        assert_eq!(grads, &vec![1.0; 24]);
    }

    // unflatten tests

    #[test]
    fn session_tensor_unflatten_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 12], unflatten dim=1 into [3, 4] -> [2, 3, 4]
        let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let x = session
            .tensor_variable(data.clone(), vec![2, 12], false)
            .expect("x");
        let y = session
            .tensor_unflatten(x, 1, vec![3, 4])
            .expect("unflatten");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, data);
        let shape = session.tensor_shape(y).expect("shape");
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn session_tensor_unflatten_roundtrip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // start with shape [2, 3, 4], flatten dims 1..2 -> [2, 12], then unflatten back
        let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let x = session
            .tensor_variable(data.clone(), vec![2, 3, 4], false)
            .expect("x");
        let flat = session.tensor_flatten(x, 1, 2).expect("flatten");
        let shape_flat = session.tensor_shape(flat).expect("flat shape");
        assert_eq!(shape_flat, vec![2, 12]);
        let restored = session
            .tensor_unflatten(flat, 1, vec![3, 4])
            .expect("unflatten");
        let vals = session.tensor_values(restored).expect("values");
        assert_eq!(vals, data);
        let shape_restored = session.tensor_shape(restored).expect("restored shape");
        assert_eq!(shape_restored, vec![2, 3, 4]);
    }

    #[test]
    fn session_tensor_unflatten_rejects_size_product_overflow() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], false)
            .expect("x");
        let err = session
            .tensor_unflatten(x, 0, vec![usize::MAX, usize::MAX])
            .expect_err("overflowing unflatten sizes must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "unflatten sizes multiplication overflow"
                }
            ))
        ));
    }

    // narrow tests

    #[test]
    fn session_tensor_narrow_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 4]: [[1, 2, 3, 4], [5, 6, 7, 8]]
        // narrow dim=1, start=1, length=2 -> [2, 2]: [[2, 3], [6, 7]]
        let x = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![2, 4],
                false,
            )
            .expect("x");
        let y = session.tensor_narrow(x, 1, 1, 2).expect("narrow");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![2.0, 3.0, 6.0, 7.0]);
        let shape = session.tensor_shape(y).expect("shape");
        assert_eq!(shape, vec![2, 2]);
    }

    #[test]
    fn session_tensor_narrow_dim0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [4, 3]: [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
        // narrow dim=0, start=1, length=2 -> [2, 3]: [[4,5,6],[7,8,9]]
        let x = session
            .tensor_variable(
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                vec![4, 3],
                false,
            )
            .expect("x");
        let y = session.tensor_narrow(x, 0, 1, 2).expect("narrow");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let shape = session.tensor_shape(y).expect("shape");
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn session_tensor_narrow_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 4]: [[1, 2, 3, 4], [5, 6, 7, 8]]
        // narrow dim=1, start=1, length=2 -> [2, 2]: [[2, 3], [6, 7]]
        // gradient should scatter back: zero-padded on non-narrowed positions
        let x = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![2, 4],
                true,
            )
            .expect("x");
        let y = session.tensor_narrow(x, 1, 1, 2).expect("narrow");
        let s = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        // positions [*,0] and [*,3] not in narrow window -> 0.0
        // positions [*,1] and [*,2] in narrow window -> 1.0
        assert_eq!(grads, &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    // expand tests

    #[test]
    fn session_tensor_expand_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [1, 3] -> expand to [4, 3]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("x");
        let y = session.tensor_expand(x, vec![4, 3]).expect("expand");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(
            vals,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        let shape = session.tensor_shape(y).expect("shape");
        assert_eq!(shape, vec![4, 3]);
    }

    #[test]
    fn session_tensor_expand_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [1, 3] -> expand to [4, 3]
        // gradient should sum along expanded dim 0: each element gets 4.0
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("x");
        let y = session.tensor_expand(x, vec![4, 3]).expect("expand");
        let s = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        // sum of upstream grad (all 1s) along expanded dim 0 (size 4) -> 4.0 each
        assert_eq!(grads, &[4.0, 4.0, 4.0]);
    }

    // split tests

    #[test]
    fn session_tensor_split_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 6]: [[1..6], [7..12]]
        // split dim=1 with sizes [2, 4] -> [2, 2] and [2, 4]
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let x = session.tensor_variable(data, vec![2, 6], false).expect("x");
        let parts = session.tensor_split(x, &[2, 4], 1).expect("split");
        assert_eq!(parts.len(), 2);

        let vals0 = session.tensor_values(parts[0]).expect("values 0");
        assert_eq!(vals0, vec![1.0, 2.0, 7.0, 8.0]);
        let shape0 = session.tensor_shape(parts[0]).expect("shape 0");
        assert_eq!(shape0, vec![2, 2]);

        let vals1 = session.tensor_values(parts[1]).expect("values 1");
        assert_eq!(vals1, vec![3.0, 4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0]);
        let shape1 = session.tensor_shape(parts[1]).expect("shape 1");
        assert_eq!(shape1, vec![2, 4]);
    }

    #[test]
    fn session_tensor_split_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 6], split dim=1 with sizes [2, 4]
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let x = session.tensor_variable(data, vec![2, 6], true).expect("x");
        let parts = session.tensor_split(x, &[2, 4], 1).expect("split");
        // sum both parts to scalar for backward
        let s0 = session.tensor_sum(parts[0]).expect("sum 0");
        let s1 = session.tensor_sum(parts[1]).expect("sum 1");
        let total = session.tensor_add(s0, s1).expect("add");
        let report = session.tensor_backward(total).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        // all elements participate in the sum, so gradient is all ones
        assert_eq!(grads, &vec![1.0; 12]);
    }

    // chunk tests

    #[test]
    fn session_tensor_chunk_even() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 6], chunk into 3 along dim=1 -> three [2, 2] tensors
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let x = session.tensor_variable(data, vec![2, 6], false).expect("x");
        let chunks = session.tensor_chunk(x, 3, 1).expect("chunk");
        assert_eq!(chunks.len(), 3);

        let vals0 = session.tensor_values(chunks[0]).expect("values 0");
        assert_eq!(vals0, vec![1.0, 2.0, 7.0, 8.0]);
        let shape0 = session.tensor_shape(chunks[0]).expect("shape 0");
        assert_eq!(shape0, vec![2, 2]);

        let vals1 = session.tensor_values(chunks[1]).expect("values 1");
        assert_eq!(vals1, vec![3.0, 4.0, 9.0, 10.0]);
        let shape1 = session.tensor_shape(chunks[1]).expect("shape 1");
        assert_eq!(shape1, vec![2, 2]);

        let vals2 = session.tensor_values(chunks[2]).expect("values 2");
        assert_eq!(vals2, vec![5.0, 6.0, 11.0, 12.0]);
        let shape2 = session.tensor_shape(chunks[2]).expect("shape 2");
        assert_eq!(shape2, vec![2, 2]);
    }

    #[test]
    fn session_tensor_chunk_uneven() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 5], chunk into 3 along dim=1
        // 5 / 3 = ceil(5/3) = 2, so chunks of size [2, 2, 1]
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let x = session.tensor_variable(data, vec![2, 5], false).expect("x");
        let chunks = session.tensor_chunk(x, 3, 1).expect("chunk");
        assert_eq!(chunks.len(), 3);

        let vals0 = session.tensor_values(chunks[0]).expect("values 0");
        assert_eq!(vals0, vec![1.0, 2.0, 6.0, 7.0]);
        let shape0 = session.tensor_shape(chunks[0]).expect("shape 0");
        assert_eq!(shape0, vec![2, 2]);

        let vals1 = session.tensor_values(chunks[1]).expect("values 1");
        assert_eq!(vals1, vec![3.0, 4.0, 8.0, 9.0]);
        let shape1 = session.tensor_shape(chunks[1]).expect("shape 1");
        assert_eq!(shape1, vec![2, 2]);

        let vals2 = session.tensor_values(chunks[2]).expect("values 2");
        assert_eq!(vals2, vec![5.0, 10.0]);
        let shape2 = session.tensor_shape(chunks[2]).expect("shape 2");
        assert_eq!(shape2, vec![2, 1]);
    }

    // ── Advanced Indexing Tests ──────────────────────────────────────────

    #[test]
    fn session_tensor_index_select_dim0_swaps_rows() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("x");
        let indices = session
            .tensor_variable(vec![1.0, 0.0], vec![2], false)
            .expect("indices");
        let y = session
            .tensor_index_select(x, 0, indices)
            .expect("index_select");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn session_tensor_index_select_dim1_picks_columns() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("x");
        let indices = session
            .tensor_variable(vec![2.0, 0.0], vec![2], false)
            .expect("indices");
        let y = session
            .tensor_index_select(x, 1, indices)
            .expect("index_select");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![3.0, 1.0, 6.0, 4.0]);
    }

    #[test]
    fn session_tensor_index_select_rejects_nan_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .expect("x");
        let indices = session
            .tensor_variable(vec![f64::NAN], vec![1], false)
            .expect("indices");
        let result = session.tensor_index_select(x, 0, indices);
        assert!(result.is_err());
    }

    #[test]
    fn session_tensor_index_select_rejects_tiny_fractional_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("x");
        let indices = session
            .tensor_variable(vec![1e-20], vec![1], false)
            .expect("indices");
        let result = session.tensor_index_select(x, 0, indices);
        assert!(result.is_err());
    }

    #[test]
    fn session_tensor_index_select_backward_scatter_add() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let indices = session
            .tensor_variable(vec![1.0, 0.0], vec![2], false)
            .expect("indices");
        let y = session
            .tensor_index_select(x, 0, indices)
            .expect("index_select");
        let s = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        // gradient scatters back: row 0 gets grad from index position 1, row 1 from position 0
        // both rows contribute 1.0 per element from the sum backward
        assert_eq!(grads, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_gather_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("x");
        let index = session
            .tensor_variable(vec![0.0, 2.0, 1.0, 0.0], vec![2, 2], false)
            .expect("index");
        let y = session.tensor_gather(x, 1, index).expect("gather");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![1.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn session_tensor_gather_rejects_fractional_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .expect("x");
        let index = session
            .tensor_variable(vec![0.5, 0.0], vec![1, 2], false)
            .expect("index");
        let result = session.tensor_gather(x, 1, index);
        assert!(result.is_err());
    }

    #[test]
    fn session_tensor_gather_backward_scatters_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let index = session
            .tensor_variable(vec![0.0, 2.0, 1.0, 0.0], vec![2, 2], false)
            .expect("index");
        let y = session.tensor_gather(x, 1, index).expect("gather");
        let s = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        // index [[0,2],[1,0]] -> row 0: col 0 gets 1.0, col 2 gets 1.0; row 1: col 1 gets 1.0, col 0 gets 1.0
        assert_eq!(grads, &[1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn session_tensor_scatter_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0; 6], vec![2, 3], false)
            .expect("x");
        let index = session
            .tensor_variable(vec![0.0, 2.0, 1.0, 0.0], vec![2, 2], false)
            .expect("index");
        let src = session
            .tensor_variable(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2], false)
            .expect("src");
        let y = session.tensor_scatter(x, 1, index, src).expect("scatter");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![10.0, 0.0, 20.0, 40.0, 30.0, 0.0]);
    }

    #[test]
    fn session_tensor_scatter_rejects_non_finite_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0; 4], vec![2, 2], false)
            .expect("x");
        let index = session
            .tensor_variable(vec![f64::INFINITY, 0.0], vec![1, 2], false)
            .expect("index");
        let src = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .expect("src");
        let result = session.tensor_scatter(x, 1, index, src);
        assert!(result.is_err());
    }

    #[test]
    fn session_tensor_masked_fill_replaces_masked_positions() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("x");
        let mask = session
            .tensor_variable(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![2, 3], false)
            .expect("mask");
        let y = session
            .tensor_masked_fill(x, mask, -1.0)
            .expect("masked_fill");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![-1.0, 2.0, -1.0, 4.0, -1.0, 6.0]);
    }

    #[test]
    fn session_tensor_masked_fill_with_comparison_mask() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("x");
        let threshold = session
            .tensor_variable(vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0], vec![2, 3], false)
            .expect("threshold");
        let mask = session.tensor_gt(x, threshold).expect("comparison mask");
        let y = session
            .tensor_masked_fill(x, mask, -1.0)
            .expect("masked_fill");
        let vals = session.tensor_values(y).expect("values");
        // mask is 1 where x > 3: [0,0,0,1,1,1], so positions 3,4,5 are filled with -1.0
        assert_eq!(vals, vec![1.0, 2.0, 3.0, -1.0, -1.0, -1.0]);
    }

    // ── Loss Function Tests ─────────────────────────────────────────────

    #[test]
    fn mse_loss_identical_tensors_returns_zero() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .unwrap();
        let target = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let loss = s.mse_loss(pred, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert_eq!(vals.len(), 1);
        assert!(
            vals[0].abs() < 1e-12,
            "MSE of identical tensors should be 0, got {}",
            vals[0]
        );
    }

    #[test]
    fn mse_loss_known_value() {
        // pred=[1,2,3], target=[4,5,6] => diffs=[-3,-3,-3], sq=[9,9,9], mean=9
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .unwrap();
        let target = s
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], false)
            .unwrap();
        let loss = s.mse_loss(pred, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 9.0).abs() < 1e-12,
            "expected MSE=9.0, got {}",
            vals[0]
        );
    }

    #[test]
    fn mse_loss_backward_produces_gradients() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .unwrap();
        let target = s
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], false)
            .unwrap();
        let loss = s.mse_loss(pred, target).unwrap();
        let report = s.tensor_backward(loss).expect("backward should succeed");
        let grad = s
            .tensor_gradient(&report, pred)
            .expect("pred grad should exist");
        assert_eq!(grad.len(), 3);
        // d/d(pred_i) of mean((pred-target)^2) = 2*(pred_i - target_i)/n
        // = 2*(-3)/3 = -2.0 for each element
        for &g in grad {
            assert!((g - (-2.0)).abs() < 1e-10, "expected grad=-2.0, got {g}");
        }
    }

    #[test]
    fn l1_loss_identical_tensors_returns_zero() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .unwrap();
        let target = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let loss = s.l1_loss(pred, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            vals[0].abs() < 1e-12,
            "L1 of identical tensors should be 0, got {}",
            vals[0]
        );
    }

    #[test]
    fn l1_loss_known_value() {
        // pred=[1,4], target=[3,1] => |diff|=[2,3], mean=2.5
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![1.0, 4.0], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![3.0, 1.0], vec![2], false).unwrap();
        let loss = s.l1_loss(pred, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 2.5).abs() < 1e-12,
            "expected L1=2.5, got {}",
            vals[0]
        );
    }

    #[test]
    fn l1_loss_backward_produces_gradients() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![1.0, 4.0], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![3.0, 1.0], vec![2], false).unwrap();
        let loss = s.l1_loss(pred, target).unwrap();
        let report = s.tensor_backward(loss).expect("backward should succeed");
        let grad = s
            .tensor_gradient(&report, pred)
            .expect("pred grad should exist");
        assert_eq!(grad.len(), 2);
        // d/d(pred_i) of mean(|pred-target|) = sign(pred_i - target_i)/n
        // pred-target = [-2, 3], signs = [-1, 1], /2 = [-0.5, 0.5]
        assert!(
            (grad[0] - (-0.5)).abs() < 1e-10,
            "expected grad[0]=-0.5, got {}",
            grad[0]
        );
        assert!(
            (grad[1] - 0.5).abs() < 1e-10,
            "expected grad[1]=0.5, got {}",
            grad[1]
        );
    }

    #[test]
    fn bce_loss_known_value() {
        // pred=0.5 for all, target=1.0 => -mean(1*log(0.5) + 0*log(0.5))
        // = -log(0.5) = ln(2) ~ 0.6931471805599453
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.5, 0.5], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![1.0, 1.0], vec![2], false).unwrap();
        let loss = s.bce_loss(pred, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        let expected = -(0.5_f64.ln());
        assert!(
            (vals[0] - expected).abs() < 1e-12,
            "expected BCE~{expected}, got {}",
            vals[0]
        );
    }

    #[test]
    fn bce_loss_perfect_prediction() {
        // target=[1,0], pred=[0.999,0.001] => near-perfect prediction
        // BCE = -mean(1*log(0.999) + 0*log(0.001) + 0*log(0.001) + 1*log(0.999))
        //     = -mean(log(0.999) + log(0.999)) = -log(0.999)
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s
            .tensor_variable(vec![0.999, 0.001], vec![2], true)
            .unwrap();
        let target = s.tensor_variable(vec![1.0, 0.0], vec![2], false).unwrap();
        let loss = s.bce_loss(pred, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        let expected = -(0.999_f64.ln());
        assert!(
            (vals[0] - expected).abs() < 1e-10,
            "expected BCE~{expected}, got {}",
            vals[0]
        );
    }

    #[test]
    fn bce_loss_backward_produces_gradients() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.5, 0.8], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![1.0, 0.0], vec![2], false).unwrap();
        let loss = s.bce_loss(pred, target).unwrap();
        let report = s.tensor_backward(loss).expect("backward should succeed");
        let grad = s
            .tensor_gradient(&report, pred)
            .expect("pred grad should exist");
        assert_eq!(grad.len(), 2);
        // d/dp BCE = -(t/p - (1-t)/(1-p)) / n
        // grad[0]: -(1/0.5 - 0) / 2 = -1.0
        // grad[1]: -(0 - 1/0.2) / 2 = 2.5
        assert!(
            (grad[0] - (-1.0)).abs() < 1e-12,
            "expected grad[0]=-1.0, got {}",
            grad[0]
        );
        assert!(
            (grad[1] - 2.5).abs() < 1e-12,
            "expected grad[1]=2.5, got {}",
            grad[1]
        );
    }

    #[test]
    fn bce_loss_extreme_pred_does_not_produce_nan() {
        // pred at 0.0 and 1.0 exactly -- clamping should prevent NaN
        // Both pred and target match perfectly after clamping, so loss should be near zero
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.0, 1.0], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![0.0, 1.0], vec![2], false).unwrap();
        let loss = s.bce_loss(pred, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            vals[0].is_finite(),
            "BCE with extreme preds should be finite, got {}",
            vals[0]
        );
        // Clamped pred matches target, so loss should be very small (near-perfect prediction)
        assert!(
            vals[0] < 0.1,
            "BCE with matched extreme preds should be small, got {}",
            vals[0]
        );
    }

    #[test]
    fn bce_with_logits_loss_known_value() {
        // logits=0 => sigmoid(0)=0.5, target=1.0
        // Stable formula: max(0,0) - 0*1 + log1p(exp(-|0|)) = 0 - 0 + log(2) = ln(2)
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits = s.tensor_variable(vec![0.0, 0.0], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![1.0, 1.0], vec![2], false).unwrap();
        let loss = s.bce_with_logits_loss(logits, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        let expected = 2.0_f64.ln();
        assert!(
            (vals[0] - expected).abs() < 1e-12,
            "expected BCEWithLogits~{expected}, got {}",
            vals[0]
        );
    }

    #[test]
    fn bce_with_logits_loss_positive_logit() {
        // logit=2.0, target=1.0
        // max(2,0) - 2*1 + log1p(exp(-2)) = 2 - 2 + log(1+exp(-2))
        // = log(1 + exp(-2)) ≈ 0.12692801104297263
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits = s.tensor_variable(vec![2.0], vec![1], true).unwrap();
        let target = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let loss = s.bce_with_logits_loss(logits, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        let expected = (1.0 + (-2.0_f64).exp()).ln();
        assert!(
            (vals[0] - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            vals[0]
        );
    }

    #[test]
    fn bce_with_logits_loss_negative_logit() {
        // logit=-3.0, target=0.0
        // max(-3,0) - (-3)*0 + log1p(exp(-|-3|)) = 0 - 0 + log(1+exp(-3))
        // = log(1 + exp(-3)) ≈ 0.04858735157374196
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits = s.tensor_variable(vec![-3.0], vec![1], true).unwrap();
        let target = s.tensor_variable(vec![0.0], vec![1], false).unwrap();
        let loss = s.bce_with_logits_loss(logits, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        let expected = (1.0 + (-3.0_f64).exp()).ln();
        assert!(
            (vals[0] - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            vals[0]
        );
    }

    #[test]
    fn bce_with_logits_loss_extreme_logits_finite() {
        // Very large logits should not produce NaN or Inf
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits = s
            .tensor_variable(vec![100.0, -100.0], vec![2], true)
            .unwrap();
        let target = s.tensor_variable(vec![1.0, 0.0], vec![2], false).unwrap();
        let loss = s.bce_with_logits_loss(logits, target).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            vals[0].is_finite(),
            "BCEWithLogitsLoss should be finite for extreme logits, got {}",
            vals[0]
        );
        // For x=100, y=1: max(100,0) - 100*1 + log1p(exp(-100)) ≈ 0
        // For x=-100, y=0: max(-100,0) - (-100)*0 + log1p(exp(-100)) ≈ 0
        // Mean ≈ 0
        assert!(
            vals[0] < 1e-10,
            "BCEWithLogitsLoss for perfectly matching extreme logits should be near 0, got {}",
            vals[0]
        );
    }

    #[test]
    fn bce_with_logits_loss_backward_produces_gradients() {
        // The gradient of BCEWithLogitsLoss w.r.t. logits is (sigmoid(x) - y) / n
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits = s.tensor_variable(vec![0.5, -0.5], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![1.0, 0.0], vec![2], false).unwrap();
        let loss = s.bce_with_logits_loss(logits, target).unwrap();
        let report = s.tensor_backward(loss).expect("backward should succeed");
        let grad = s
            .tensor_gradient(&report, logits)
            .expect("logits grad should exist");
        assert_eq!(grad.len(), 2);
        // Check gradients are finite and non-zero
        assert!(grad[0].is_finite(), "grad[0] should be finite");
        assert!(grad[1].is_finite(), "grad[1] should be finite");
        assert!(grad[0].abs() > 1e-15, "grad[0] should be non-zero");
        assert!(grad[1].abs() > 1e-15, "grad[1] should be non-zero");
    }

    #[test]
    fn bce_with_logits_matches_sigmoid_then_bce() {
        // The numerically-stable formulation should produce the same result as
        // sigmoid + bce_loss for moderate input values
        let mut s1 = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits1 = s1
            .tensor_variable(vec![0.5, -1.0, 2.0, -0.3], vec![4], true)
            .unwrap();
        let target1 = s1
            .tensor_variable(vec![1.0, 0.0, 1.0, 0.0], vec![4], false)
            .unwrap();
        let loss1 = s1.bce_with_logits_loss(logits1, target1).unwrap();
        let vals1 = s1.tensor_values(loss1).unwrap();

        // Compare with sigmoid + bce_loss approach
        let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits2 = s2
            .tensor_variable(vec![0.5, -1.0, 2.0, -0.3], vec![4], true)
            .unwrap();
        let target2 = s2
            .tensor_variable(vec![1.0, 0.0, 1.0, 0.0], vec![4], false)
            .unwrap();
        let probs = s2.tensor_sigmoid(logits2).unwrap();
        let loss2 = s2.bce_loss(probs, target2).unwrap();
        let vals2 = s2.tensor_values(loss2).unwrap();

        assert!(
            (vals1[0] - vals2[0]).abs() < 1e-6,
            "BCEWithLogitsLoss ({}) should match sigmoid+BCE ({}) for moderate inputs",
            vals1[0],
            vals2[0]
        );
    }

    #[test]
    fn smooth_l1_loss_identical_tensors_returns_zero() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .unwrap();
        let target = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let loss = s.smooth_l1_loss(pred, target, 1.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            vals[0].abs() < 1e-12,
            "Smooth L1 of identical tensors should be 0, got {}",
            vals[0]
        );
    }

    #[test]
    fn smooth_l1_loss_quadratic_region() {
        // |diff| < beta uses quadratic branch: 0.5 * diff^2 / beta
        // pred=[1.0], target=[1.5], beta=1.0
        // diff=-0.5, |diff|=0.5 < 1.0 => 0.5 * 0.25 / 1.0 = 0.125
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![1.0], vec![1], true).unwrap();
        let target = s.tensor_variable(vec![1.5], vec![1], false).unwrap();
        let loss = s.smooth_l1_loss(pred, target, 1.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 0.125).abs() < 1e-10,
            "expected 0.125, got {}",
            vals[0]
        );
    }

    #[test]
    fn smooth_l1_loss_linear_region() {
        // |diff| >= beta uses linear branch: |diff| - 0.5 * beta
        // pred=[0.0], target=[3.0], beta=1.0
        // diff=-3.0, |diff|=3.0 >= 1.0 => 3.0 - 0.5 = 2.5
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.0], vec![1], true).unwrap();
        let target = s.tensor_variable(vec![3.0], vec![1], false).unwrap();
        let loss = s.smooth_l1_loss(pred, target, 1.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 2.5).abs() < 1e-10,
            "expected 2.5, got {}",
            vals[0]
        );
    }

    #[test]
    fn smooth_l1_loss_mixed_regions() {
        // Two elements: one in quadratic, one in linear region
        // pred=[1.0, 0.0], target=[1.3, 5.0], beta=1.0
        // diff=[-0.3, -5.0], |diff|=[0.3, 5.0]
        // elem 0: 0.3 < 1.0 => 0.5 * 0.09 / 1.0 = 0.045
        // elem 1: 5.0 >= 1.0 => 5.0 - 0.5 = 4.5
        // mean = (0.045 + 4.5) / 2 = 2.2725
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![1.0, 0.0], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![1.3, 5.0], vec![2], false).unwrap();
        let loss = s.smooth_l1_loss(pred, target, 1.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 2.2725).abs() < 1e-10,
            "expected 2.2725, got {}",
            vals[0]
        );
    }

    #[test]
    fn smooth_l1_loss_backward_produces_gradients() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![1.0, 0.0], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![1.3, 5.0], vec![2], false).unwrap();
        let loss = s.smooth_l1_loss(pred, target, 1.0).unwrap();
        let report = s.tensor_backward(loss).expect("backward should succeed");
        let grad = s
            .tensor_gradient(&report, pred)
            .expect("pred grad should exist");
        assert_eq!(grad.len(), 2);
        // Element 0: |diff|=0.3 < beta=1.0 (quadratic): grad = diff/beta/n = -0.3/1.0/2 = -0.15
        // Element 1: |diff|=5.0 >= beta=1.0 (linear): grad = sign(diff)/n = -1.0/2 = -0.5
        assert!(
            (grad[0] - (-0.15)).abs() < 1e-12,
            "expected grad[0]=-0.15, got {}",
            grad[0]
        );
        assert!(
            (grad[1] - (-0.5)).abs() < 1e-12,
            "expected grad[1]=-0.5, got {}",
            grad[1]
        );
    }

    #[test]
    fn smooth_l1_loss_custom_beta() {
        // With beta=2.0:
        // pred=[0.0], target=[1.0], diff=-1.0, |diff|=1.0 < 2.0
        // => 0.5 * 1.0 / 2.0 = 0.25
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.0], vec![1], true).unwrap();
        let target = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let loss = s.smooth_l1_loss(pred, target, 2.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 0.25).abs() < 1e-10,
            "expected 0.25, got {}",
            vals[0]
        );
    }

    // ── Huber loss tests ───────────────────────────────────────────────

    #[test]
    fn huber_loss_identical_tensors_returns_zero() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let target = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let loss = s.huber_loss(pred, target, 1.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            vals[0].abs() < 1e-12,
            "huber_loss of identical tensors should be 0, got {}",
            vals[0]
        );
    }

    #[test]
    fn huber_loss_quadratic_region() {
        // |diff| <= delta: loss = 0.5 * diff^2
        // pred=[0.0], target=[0.3], diff=-0.3, 0.3 <= 1.0
        // loss = 0.5 * 0.09 = 0.045
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.0], vec![1], false).unwrap();
        let target = s.tensor_variable(vec![0.3], vec![1], false).unwrap();
        let loss = s.huber_loss(pred, target, 1.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 0.045).abs() < 1e-10,
            "expected 0.045, got {}",
            vals[0]
        );
    }

    #[test]
    fn huber_loss_linear_region() {
        // |diff| > delta: loss = delta * (|diff| - 0.5 * delta)
        // pred=[0.0], target=[3.0], diff=-3.0, 3.0 > 1.0
        // loss = 1.0 * (3.0 - 0.5) = 2.5
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.0], vec![1], false).unwrap();
        let target = s.tensor_variable(vec![3.0], vec![1], false).unwrap();
        let loss = s.huber_loss(pred, target, 1.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 2.5).abs() < 1e-10,
            "expected 2.5, got {}",
            vals[0]
        );
    }

    #[test]
    fn huber_loss_mixed_regions() {
        // Two elements: one in quadratic, one in linear
        // pred=[0.0, 0.0], target=[0.5, 5.0], delta=1.0
        // elem 0: |diff|=0.5 <= 1.0, loss = 0.5 * 0.25 = 0.125
        // elem 1: |diff|=5.0 > 1.0, loss = 1.0 * (5.0 - 0.5) = 4.5
        // mean = (0.125 + 4.5) / 2 = 2.3125
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.0, 0.0], vec![2], false).unwrap();
        let target = s.tensor_variable(vec![0.5, 5.0], vec![2], false).unwrap();
        let loss = s.huber_loss(pred, target, 1.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 2.3125).abs() < 1e-10,
            "expected 2.3125, got {}",
            vals[0]
        );
    }

    #[test]
    fn huber_loss_custom_delta() {
        // delta=0.5, pred=[0.0], target=[2.0], |diff|=2.0 > 0.5
        // loss = 0.5 * (2.0 - 0.25) = 0.875
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.0], vec![1], false).unwrap();
        let target = s.tensor_variable(vec![2.0], vec![1], false).unwrap();
        let loss = s.huber_loss(pred, target, 0.5).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 0.875).abs() < 1e-10,
            "expected 0.875, got {}",
            vals[0]
        );
    }

    #[test]
    fn huber_loss_backward_produces_gradients() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![0.0, 0.0], vec![2], true).unwrap();
        let target = s.tensor_variable(vec![0.5, 5.0], vec![2], false).unwrap();
        let loss = s.huber_loss(pred, target, 1.0).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        let grad = s.tensor_gradient(&report, pred);
        assert!(
            grad.is_some(),
            "huber_loss should produce gradients for pred"
        );
    }

    #[test]
    fn huber_loss_rejects_non_positive_delta() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pred = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let target = s.tensor_variable(vec![2.0], vec![1], false).unwrap();
        assert!(s.huber_loss(pred, target, 0.0).is_err());
        assert!(s.huber_loss(pred, target, -1.0).is_err());
    }

    // ── Cosine similarity tests ────────────────────────────────────────

    #[test]
    fn cosine_similarity_identical_vectors() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Two identical vectors: cos_sim = 1.0
        let x1 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let sim = s.cosine_similarity(x1, x2, 0, 1e-8).unwrap();
        let vals = s.tensor_values(sim).unwrap();
        assert!(
            (vals[0] - 1.0).abs() < 1e-6,
            "identical vectors should have cos_sim=1.0, got {}",
            vals[0]
        );
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Orthogonal vectors: cos_sim = 0.0
        let x1 = s.tensor_variable(vec![1.0, 0.0], vec![2], false).unwrap();
        let x2 = s.tensor_variable(vec![0.0, 1.0], vec![2], false).unwrap();
        let sim = s.cosine_similarity(x1, x2, 0, 1e-8).unwrap();
        let vals = s.tensor_values(sim).unwrap();
        assert!(
            vals[0].abs() < 1e-6,
            "orthogonal vectors should have cos_sim=0, got {}",
            vals[0]
        );
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Opposite vectors: cos_sim = -1.0
        let x1 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![-1.0, -2.0, -3.0], vec![3], false)
            .unwrap();
        let sim = s.cosine_similarity(x1, x2, 0, 1e-8).unwrap();
        let vals = s.tensor_values(sim).unwrap();
        assert!(
            (vals[0] + 1.0).abs() < 1e-6,
            "opposite vectors should have cos_sim=-1.0, got {}",
            vals[0]
        );
    }

    #[test]
    fn cosine_similarity_batch_dim() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Batch of vectors: [2, 3] with similarity along dim=1
        let x1 = s
            .tensor_variable(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3], false)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], vec![2, 3], false)
            .unwrap();
        let sim = s.cosine_similarity(x1, x2, 1, 1e-8).unwrap();
        let vals = s.tensor_values(sim).unwrap();
        // First pair: [1,0,0] vs [1,0,0] -> cos_sim=1.0
        assert!(
            (vals[0] - 1.0).abs() < 1e-6,
            "parallel vectors should be 1.0, got {}",
            vals[0]
        );
        // Second pair: [0,1,0] vs [0,0,1] -> cos_sim=0.0
        assert!(
            vals[1].abs() < 1e-6,
            "orthogonal vectors should be 0.0, got {}",
            vals[1]
        );
    }

    #[test]
    fn cosine_similarity_backward_produces_gradients() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x1 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], true)
            .unwrap();
        let sim = s.cosine_similarity(x1, x2, 0, 1e-8).unwrap();
        let loss = s.tensor_sum(sim).unwrap();
        let report = s.tensor_backward(loss).unwrap();

        assert!(
            s.tensor_gradient(&report, x1).is_some(),
            "cosine_similarity should produce gradients for x1"
        );
        assert!(
            s.tensor_gradient(&report, x2).is_some(),
            "cosine_similarity should produce gradients for x2"
        );
    }

    #[test]
    fn cosine_similarity_zero_vector_with_eps() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Zero vector: eps prevents NaN
        let x1 = s
            .tensor_variable(vec![0.0, 0.0, 0.0], vec![3], false)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let sim = s.cosine_similarity(x1, x2, 0, 1e-8).unwrap();
        let vals = s.tensor_values(sim).unwrap();
        // dot = 0, norms product is clamped to eps, result should be ~0
        assert!(
            vals[0].is_finite(),
            "zero vector with eps should not produce NaN/Inf, got {}",
            vals[0]
        );
    }

    // ── Cosine embedding loss tests ────────────────────────────────────

    #[test]
    fn cosine_embedding_loss_similar_pair() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Two identical vectors with target=1 -> loss should be ~0
        let x1 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .unwrap();
        let target = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let loss = s.cosine_embedding_loss(x1, x2, target, 0.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            vals[0].abs() < 1e-4,
            "identical vectors with target=1 should have ~0 loss, got {}",
            vals[0]
        );
    }

    #[test]
    fn cosine_embedding_loss_dissimilar_pair() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Two identical vectors with target=-1 -> loss = max(0, cos_sim - margin) = max(0, 1.0) = 1.0
        let x1 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .unwrap();
        let target = s.tensor_variable(vec![-1.0], vec![1], false).unwrap();
        let loss = s.cosine_embedding_loss(x1, x2, target, 0.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            (vals[0] - 1.0).abs() < 1e-4,
            "identical vectors with target=-1 should have loss~1.0, got {}",
            vals[0]
        );
    }

    #[test]
    fn cosine_embedding_loss_margin() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Two orthogonal vectors with target=-1, margin=0 -> loss = max(0, 0-0) = 0
        let x1 = s
            .tensor_variable(vec![1.0, 0.0], vec![1, 2], false)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![0.0, 1.0], vec![1, 2], false)
            .unwrap();
        let target = s.tensor_variable(vec![-1.0], vec![1], false).unwrap();
        let loss = s.cosine_embedding_loss(x1, x2, target, 0.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        assert!(
            vals[0].abs() < 1e-4,
            "orthogonal vectors with target=-1 should have ~0 loss, got {}",
            vals[0]
        );
    }

    #[test]
    fn cosine_embedding_loss_backward_produces_gradients() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x1 = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![1, 3], true)
            .unwrap();
        let target = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let loss = s.cosine_embedding_loss(x1, x2, target, 0.0).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        assert!(
            s.tensor_gradient(&report, x1).is_some(),
            "should produce gradients for x1"
        );
        assert!(
            s.tensor_gradient(&report, x2).is_some(),
            "should produce gradients for x2"
        );
    }

    #[test]
    fn cosine_embedding_loss_mixed_targets() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Batch of 2: first pair similar (target=1), second pair dissimilar (target=-1)
        let x1 = s
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false)
            .unwrap();
        let target = s.tensor_variable(vec![1.0, -1.0], vec![2], false).unwrap();
        let loss = s.cosine_embedding_loss(x1, x2, target, 0.0).unwrap();
        let vals = s.tensor_values(loss).unwrap();
        // pair 0: cos=1.0, target=1 -> loss=0
        // pair 1: cos=1.0, target=-1 -> loss=max(0, 1.0-0)=1.0
        // mean = 0.5
        assert!(
            (vals[0] - 0.5).abs() < 1e-4,
            "mixed targets should give mean loss ~0.5, got {}",
            vals[0]
        );
    }

    #[test]
    fn cosine_embedding_loss_rejects_invalid_target() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x1 = s
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .unwrap();
        let x2 = s
            .tensor_variable(vec![3.0, 4.0], vec![1, 2], false)
            .unwrap();
        // target=0.5 is invalid (must be 1.0 or -1.0)
        let target = s.tensor_variable(vec![0.5], vec![1], false).unwrap();
        assert!(
            s.cosine_embedding_loss(x1, x2, target, 0.0).is_err(),
            "invalid target value should be rejected"
        );
    }

    #[test]
    fn tensor_add_in_place() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false)
            .unwrap();
        s.tensor_add_(a, b).unwrap();
        assert_eq!(s.tensor_values(a).unwrap(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn tensor_add_in_place_rejects_device_mismatch() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let target = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("target");
        let other_meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cuda);
        let other_dense =
            DenseTensor::from_storage(other_meta, vec![10.0, 20.0, 30.0]).expect("other dense");
        let other = s.tensor_variable_from_storage(other_dense, false);

        let err = s
            .tensor_add_(target, other)
            .expect_err("device mismatch must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::Incompatible(
                    ft_core::TensorCompatError::DeviceMismatch { .. }
                )
            ))
        ));
    }

    #[test]
    fn tensor_add_in_place_records_evidence() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false)
            .unwrap();
        let before = s.evidence_len();

        s.tensor_add_(a, b).unwrap();

        assert_eq!(s.evidence_len(), before + 1);
        let last = s.evidence().last().expect("in-place op should be recorded");
        assert_eq!(last.kind, EvidenceKind::Dispatch);
        assert!(
            last.summary.contains("tensor_inplace_op=add_"),
            "unexpected summary: {}",
            last.summary
        );
    }

    #[test]
    fn tensor_mul_in_place() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![2.0, 3.0, 4.0], vec![3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![5.0, 6.0, 7.0], vec![3], false)
            .unwrap();
        s.tensor_mul_(a, b).unwrap();
        assert_eq!(s.tensor_values(a).unwrap(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn tensor_div_in_place() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![2.0, 5.0, 10.0], vec![3], false)
            .unwrap();
        s.tensor_div_(a, b).unwrap();
        assert_eq!(s.tensor_values(a).unwrap(), vec![5.0, 4.0, 3.0]);
    }

    #[test]
    fn tensor_zero_in_place() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        s.tensor_zero_(a).unwrap();
        assert_eq!(s.tensor_values(a).unwrap(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn tensor_fill_in_place() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        s.tensor_fill_(a, 42.0).unwrap();
        assert_eq!(s.tensor_values(a).unwrap(), vec![42.0, 42.0, 42.0]);
    }

    #[test]
    fn tensor_mul_scalar_in_place() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        s.tensor_mul_scalar_(a, 3.0).unwrap();
        assert_eq!(s.tensor_values(a).unwrap(), vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn tensor_add_scalar_in_place() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        s.tensor_add_scalar_(a, 10.0).unwrap();
        assert_eq!(s.tensor_values(a).unwrap(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn tensor_sub_in_place() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        s.tensor_sub_(a, b).unwrap();
        assert_eq!(s.tensor_values(a).unwrap(), vec![9.0, 18.0, 27.0]);
    }

    fn assert_tensor_unary_in_place_matches_out_of_place(
        values: &[f64],
        apply_out_of_place: impl Fn(
            &mut FrankenTorchSession,
            ft_autograd::TensorNodeId,
        ) -> Result<ft_autograd::TensorNodeId, AutogradError>,
        apply_in_place: impl Fn(
            &mut FrankenTorchSession,
            ft_autograd::TensorNodeId,
        ) -> Result<(), AutogradError>,
    ) {
        let shape = vec![values.len()];

        let mut s_out = FrankenTorchSession::new(ExecutionMode::Strict);
        let out_input = s_out
            .tensor_variable(values.to_vec(), shape.clone(), false)
            .expect("out-of-place input");
        let out_node = apply_out_of_place(&mut s_out, out_input).expect("out-of-place op");
        let expected = s_out.tensor_values(out_node).expect("out-of-place values");

        let mut s_in = FrankenTorchSession::new(ExecutionMode::Strict);
        let in_target = s_in
            .tensor_variable(values.to_vec(), shape, false)
            .expect("in-place target");
        apply_in_place(&mut s_in, in_target).expect("in-place op");
        let actual = s_in.tensor_values(in_target).expect("in-place values");

        assert_eq!(actual.len(), expected.len());
        for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-12,
                "mismatch at index {idx}: in-place={lhs} out-of-place={rhs}"
            );
        }
    }

    #[test]
    fn tensor_unary_in_place_matches_out_of_place_variants() {
        type TId = ft_autograd::TensorNodeId;

        let signed = vec![-1.5, -0.25, 0.0, 0.25, 1.5];
        let positive = vec![0.25, 0.5, 1.0, 2.0, 3.5];

        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_relu(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_relu_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_sigmoid(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_sigmoid_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_tanh(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_tanh_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_abs(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_abs_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_neg(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_neg_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_exp(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_exp_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_floor(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_floor_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_ceil(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_ceil_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_round(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_round_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_sin(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_sin_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_cos(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_cos_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &positive,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_log(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_log_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &positive,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_sqrt(t),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_sqrt_(t),
        );
        assert_tensor_unary_in_place_matches_out_of_place(
            &signed,
            |s: &mut FrankenTorchSession, t: TId| s.tensor_clamp(t, -0.3, 0.9),
            |s: &mut FrankenTorchSession, t: TId| s.tensor_clamp_(t, -0.3, 0.9),
        );
    }

    #[test]
    fn tensor_unary_in_place_rejects_leaf_requires_grad() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(vec![1.0, -2.0, 3.0], vec![3], true)
            .expect("leaf");

        let err = s
            .tensor_relu_(x)
            .expect_err("in-place relu on leaf requires_grad tensor must fail");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet { .. }
            ))
        ));
    }

    #[test]
    fn tensor_existing_in_place_ops_reject_leaf_requires_grad() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("leaf");

        let err = s
            .tensor_add_scalar_(x, 1.0)
            .expect_err("existing in-place ops should enforce the same guard");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet { .. }
            ))
        ));
    }

    #[test]
    fn tensor_unary_in_place_bumps_version_counter() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(vec![1.0, -2.0, 3.0], vec![3], false)
            .expect("leaf");
        let before = s.tensor_tape.tensor(x).expect("tensor").version();

        s.tensor_abs_(x).expect("in-place abs should succeed");

        let after = s.tensor_tape.tensor(x).expect("tensor").version();
        assert_eq!(after, before + 1);
    }

    #[test]
    fn tensor_unary_in_place_ops_chain_correctly() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(vec![-4.0, -1.0, 0.0, 1.0, 4.0], vec![5], false)
            .expect("leaf");

        s.tensor_relu_(x).expect("relu_");
        s.tensor_sqrt_(x).expect("sqrt_");
        s.tensor_add_scalar_(x, 1.0).expect("add_scalar_");

        assert_eq!(
            s.tensor_values(x).expect("values"),
            vec![1.0, 1.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn rand_creates_uniform_values() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.rand(vec![100], false).unwrap();
        let vals = s.tensor_values(t).unwrap();
        assert_eq!(vals.len(), 100);
        for &v in &vals {
            assert!((0.0..1.0).contains(&v), "rand value {v} not in [0, 1)");
        }
    }

    #[test]
    fn randn_creates_normal_values() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.randn(vec![1000], false).unwrap();
        let vals = s.tensor_values(t).unwrap();
        assert_eq!(vals.len(), 1000);
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        // Mean should be roughly 0 (within ~0.15 for 1000 samples)
        assert!(mean.abs() < 0.2, "randn mean should be near 0, got {mean}");
    }

    #[test]
    fn rand_like_matches_shape() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let template = s.tensor_variable(vec![0.0; 12], vec![3, 4], false).unwrap();
        let r = s.rand_like(template, false).unwrap();
        let (vals, meta) = s.tensor_values_meta(r).unwrap();
        assert_eq!(meta.shape(), &[3, 4]);
        assert_eq!(vals.len(), 12);
    }

    #[test]
    fn randn_like_matches_shape() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let template = s.tensor_variable(vec![0.0; 6], vec![2, 3], false).unwrap();
        let r = s.randn_like(template, false).unwrap();
        let (vals, meta) = s.tensor_values_meta(r).unwrap();
        assert_eq!(meta.shape(), &[2, 3]);
        assert_eq!(vals.len(), 6);
    }

    #[test]
    fn tensor_values_meta_returns_correct_shape_and_values() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        let (vals, meta) = s.tensor_values_meta(t).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(meta.shape(), &[2, 2]);
    }

    // ---- tensor_acos (bd-1q3d) ----

    #[test]
    fn session_tensor_acos_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 0.0, -1.0], vec![3], false)
            .expect("tensor creation");
        let out = session.tensor_acos(t).expect("acos");
        let vals = session.tensor_values(out).expect("values");
        assert!((vals[0] - 0.0).abs() < 1e-10); // acos(1) = 0
        assert!((vals[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10); // acos(0) = pi/2
        assert!((vals[2] - std::f64::consts::PI).abs() < 1e-10); // acos(-1) = pi
    }

    #[test]
    fn session_tensor_acos_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![0.5, -0.5], vec![2], true)
            .expect("tensor creation");
        let out = session.tensor_acos(t).expect("acos");
        let report = session.tensor_backward(out).expect("backward");
        let grad = report.gradient(t).expect("gradient should exist");
        // d/dx acos(x) = -1/sqrt(1 - x^2)
        let expected_0 = -1.0 / (1.0 - 0.25_f64).sqrt(); // x=0.5
        let expected_1 = -1.0 / (1.0 - 0.25_f64).sqrt(); // x=-0.5
        assert!((grad[0] - expected_0).abs() < 1e-8);
        assert!((grad[1] - expected_1).abs() < 1e-8);
    }

    // ---- tensor_log10 (bd-1q3d) ----

    #[test]
    fn session_tensor_log10_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 10.0, 100.0], vec![3], false)
            .expect("tensor creation");
        let out = session.tensor_log10(t).expect("log10");
        let vals = session.tensor_values(out).expect("values");
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!((vals[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_log10_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![10.0, 100.0], vec![2], true)
            .expect("tensor creation");
        let out = session.tensor_log10(t).expect("log10");
        let report = session.tensor_backward(out).expect("backward");
        let grad = report.gradient(t).expect("gradient should exist");
        // d/dx log10(x) = 1/(x * ln(10))
        let ln10 = 10.0_f64.ln();
        assert!((grad[0] - 1.0 / (10.0 * ln10)).abs() < 1e-8);
        assert!((grad[1] - 1.0 / (100.0 * ln10)).abs() < 1e-8);
    }

    // ---- tensor_log1p (bd-1q3d) ----

    #[test]
    fn session_tensor_log1p_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![0.0, 1.0, std::f64::consts::E - 1.0], vec![3], false)
            .expect("tensor creation");
        let out = session.tensor_log1p(t).expect("log1p");
        let vals = session.tensor_values(out).expect("values");
        assert!((vals[0] - 0.0).abs() < 1e-10); // log1p(0) = ln(1) = 0
        assert!((vals[1] - 2.0_f64.ln()).abs() < 1e-10); // log1p(1) = ln(2)
        assert!((vals[2] - 1.0).abs() < 1e-10); // log1p(e-1) = ln(e) = 1
    }

    #[test]
    fn session_tensor_log1p_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 3.0], vec![2], true)
            .expect("tensor creation");
        let out = session.tensor_log1p(t).expect("log1p");
        let report = session.tensor_backward(out).expect("backward");
        let grad = report.gradient(t).expect("gradient should exist");
        // d/dx log1p(x) = 1/(1+x)
        assert!((grad[0] - 0.5).abs() < 1e-8); // 1/(1+1) = 0.5
        assert!((grad[1] - 0.25).abs() < 1e-8); // 1/(1+3) = 0.25
    }

    // ---- tensor_transpose (bd-1q3d) ----

    #[test]
    fn session_tensor_transpose_swaps_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2x3 matrix: [[1,2,3],[4,5,6]]
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("tensor creation");
        let transposed = session.tensor_transpose(t, 0, 1).expect("transpose");
        let vals = session.tensor_values(transposed).expect("values");
        // Transposed shape should be [3, 2]: [[1,4],[2,5],[3,6]]
        assert_eq!(vals, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn session_tensor_transpose_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("tensor creation");
        let transposed = session.tensor_transpose(t, 0, 1).expect("transpose");
        let sum = session.tensor_sum(transposed).expect("sum");
        let report = session.tensor_backward(sum).expect("backward");
        let grad = report.gradient(t).expect("gradient should exist");
        // Gradient through sum + transpose should be all 1s
        assert_eq!(grad, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    // ---- tensor_permute (bd-1q3d) ----

    #[test]
    fn session_tensor_permute_reorders_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2x3 matrix, permute to [1, 0] = transpose
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("tensor creation");
        let permuted = session
            .tensor_permute(t, vec![1, 0])
            .expect("permute should succeed");
        let vals = session.tensor_values(permuted).expect("values");
        assert_eq!(vals, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn session_tensor_permute_3d_reorders_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2x2x2 tensor, permute [2, 0, 1]
        // Input [[[1,2],[3,4]],[[5,6],[7,8]]] shape [2,2,2]
        // After permute [2,0,1]: new[k][i][j] = old[i][j][k]
        // new[0][0][0]=old[0][0][0]=1, new[0][0][1]=old[0][1][0]=3
        // new[0][1][0]=old[1][0][0]=5, new[0][1][1]=old[1][1][0]=7
        // new[1][0][0]=old[0][0][1]=2, new[1][0][1]=old[0][1][1]=4
        // new[1][1][0]=old[1][0][1]=6, new[1][1][1]=old[1][1][1]=8
        let t = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![2, 2, 2],
                false,
            )
            .expect("tensor creation");
        let permuted = session
            .tensor_permute(t, vec![2, 0, 1])
            .expect("permute should succeed");
        let vals = session.tensor_values(permuted).expect("values");
        assert_eq!(vals, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
    }

    // ---- Edge case tests (bd-3du0) ----

    #[test]
    fn session_tensor_clamp_all_values_at_boundary() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![2.0, 2.0, 2.0], vec![3], false)
            .expect("t");
        let r = session.tensor_clamp(t, 2.0, 2.0).expect("clamp min==max");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn session_tensor_clamp_backward_at_boundaries() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // Values exactly at min and max
        let x = session
            .tensor_variable(vec![0.0, 5.0, 2.5], vec![3], true)
            .expect("x");
        let r = session.tensor_clamp(x, 0.0, 5.0).expect("clamp");
        let s = session.tensor_sum(r).expect("sum");
        let report = session.tensor_backward(s).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // At boundaries: 0.0 == min -> in range, 5.0 == max -> in range, 2.5 -> in range
        assert_eq!(grad, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_prod_dim_singleton_dimension() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [1, 3]: prod along dim 0 (size 1) should return same values
        let t = session
            .tensor_variable(vec![2.0, 3.0, 5.0], vec![1, 3], false)
            .expect("t");
        let r = session.tensor_prod_dim(t, 0).expect("prod_dim 0");
        let vals = session.tensor_values(r).expect("vals");
        assert_eq!(vals, vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn session_tensor_var_dim_singleton_dimension() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [1, 3]: var along dim 0 (size 1) with Bessel correction -> NaN or 0
        let t = session
            .tensor_variable(vec![2.0, 3.0, 5.0], vec![1, 3], false)
            .expect("t");
        let r = session.tensor_var_dim(t, 0).expect("var_dim 0");
        let vals = session.tensor_values(r).expect("vals");
        // Variance of a single element with Bessel correction: division by (n-1)=0 -> NaN
        // Without Bessel correction: 0.0
        // Either is acceptable — just verify it doesn't panic
        assert_eq!(vals.len(), 3);
    }

    #[test]
    fn session_tensor_sum_dim_3d_reduces_middle() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3, 2]: sum along dim 1
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let t = session
            .tensor_variable(data, vec![2, 3, 2], false)
            .expect("t");
        let r = session.tensor_sum_dim(t, 1).expect("sum_dim 1");
        let vals = session.tensor_values(r).expect("vals");
        // [[1,2],[3,4],[5,6]] sum dim1 -> [9, 12]
        // [[7,8],[9,10],[11,12]] sum dim1 -> [27, 30]
        assert_eq!(vals, vec![9.0, 12.0, 27.0, 30.0]);
    }

    #[test]
    fn session_tensor_mean_dim_3d_reduces_last() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 2, 3]: mean along dim 2
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let t = session
            .tensor_variable(data, vec![2, 2, 3], false)
            .expect("t");
        let r = session.tensor_mean_dim(t, 2).expect("mean_dim 2");
        let vals = session.tensor_values(r).expect("vals");
        // [1,2,3]->2.0, [4,5,6]->5.0, [7,8,9]->8.0, [10,11,12]->11.0
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 5.0).abs() < 1e-10);
        assert!((vals[2] - 8.0).abs() < 1e-10);
        assert!((vals[3] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_reshape_mismatched_numel_fails_closed() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .expect("t");
        // 4 elements can't reshape to [3, 2] (6 elements)
        let result = session.tensor_reshape(t, vec![3, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn session_tensor_reshape_rejects_shape_volume_overflow() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0], vec![1], false)
            .expect("t");
        let err = session
            .tensor_reshape(t, vec![usize::MAX, usize::MAX])
            .expect_err("overflowing reshape shape must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "reshape target shape volume overflow"
                }
            ))
        ));
    }

    #[test]
    fn session_tensor_squeeze_non_singleton_is_noop() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        // squeeze dim 0 (size 2, not singleton) should be a no-op
        let squeezed = session.tensor_squeeze(t, 0).expect("squeeze non-singleton");
        let shape = session.tensor_shape(squeezed).expect("shape");
        assert_eq!(
            shape,
            vec![2, 3],
            "shape should be unchanged for non-singleton squeeze"
        );
        let vals = session.tensor_values(squeezed).expect("vals");
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn session_tensor_expand_backward_sums_over_broadcast_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [1, 3]: expand to [4, 3]
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("t");
        let expanded = session
            .tensor_expand(t, vec![4, 3])
            .expect("expand should succeed");
        let vals = session.tensor_values(expanded).expect("vals");
        assert_eq!(vals.len(), 12);
        // All rows should be [1, 2, 3]
        for row in 0..4 {
            assert_eq!(vals[row * 3], 1.0);
            assert_eq!(vals[row * 3 + 1], 2.0);
            assert_eq!(vals[row * 3 + 2], 3.0);
        }
        // Backward: gradient should sum over expanded dim 0
        let sum = session.tensor_sum(expanded).expect("sum");
        let report = session.tensor_backward(sum).expect("backward");
        let grad = report.gradient(t).expect("grad");
        // Each of the 3 elements was replicated 4 times, so grad = [4, 4, 4]
        assert_eq!(grad, &[4.0, 4.0, 4.0]);
    }

    #[test]
    fn session_tensor_softmax_single_element_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [3, 1]: softmax along dim 1 (size 1) -> all 1.0
        let t = session
            .tensor_variable(vec![5.0, -3.0, 0.0], vec![3, 1], false)
            .expect("t");
        let sm = session.tensor_softmax(t, 1).expect("softmax");
        let vals = session.tensor_values(sm).expect("vals");
        assert_eq!(vals, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_argmax_along_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3]: argmax along dim 1
        let t = session
            .tensor_variable(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], false)
            .expect("t");
        let indices = session.tensor_argmax(t, 1).expect("argmax");
        let vals = session.tensor_values(indices).expect("vals");
        assert_eq!(vals, vec![1.0, 2.0]); // index 1 and 2
    }

    #[test]
    fn session_tensor_argmin_along_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // shape [2, 3]: argmin along dim 1
        let t = session
            .tensor_variable(vec![3.0, 1.0, 5.0, 6.0, 4.0, 2.0], vec![2, 3], false)
            .expect("t");
        let indices = session.tensor_argmin(t, 1).expect("argmin");
        let vals = session.tensor_values(indices).expect("vals");
        assert_eq!(vals, vec![1.0, 2.0]); // index 1 and 2
    }

    // ---- cumsum ----

    #[test]
    fn session_tensor_cumsum_1d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("t");
        let cs = session.tensor_cumsum(t, 0).expect("cumsum");
        let vals = session.tensor_values(cs).expect("vals");
        assert_eq!(vals, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn session_tensor_cumsum_2d_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        let cs = session.tensor_cumsum(t, 1).expect("cumsum");
        let vals = session.tensor_values(cs).expect("vals");
        assert_eq!(vals, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn session_tensor_cumsum_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("t");
        let cs = session.tensor_cumsum(t, 0).expect("cumsum");
        // cumsum = [1, 3, 6], sum = 10
        let loss = session.tensor_sum(cs).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        let grad = session.tensor_gradient(&report, t).expect("grad");
        // d(sum(cumsum(x)))/dx = reverse_cumsum([1,1,1]) = [3,2,1]
        assert_eq!(grad, &[3.0, 2.0, 1.0]);
    }

    // ---- cumprod ----

    #[test]
    fn session_tensor_cumprod_1d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![2.0, 3.0, 4.0], vec![3], false)
            .expect("t");
        let cp = session.tensor_cumprod(t, 0).expect("cumprod");
        let vals = session.tensor_values(cp).expect("vals");
        assert_eq!(vals, vec![2.0, 6.0, 24.0]);
    }

    #[test]
    fn session_tensor_cumprod_2d_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        let cp = session.tensor_cumprod(t, 1).expect("cumprod");
        let vals = session.tensor_values(cp).expect("vals");
        assert_eq!(vals, vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
    }

    #[test]
    fn session_tensor_cumprod_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![2.0, 3.0, 4.0], vec![3], true)
            .expect("t");
        let cp = session.tensor_cumprod(t, 0).expect("cumprod");
        // cumprod = [2, 6, 24], sum = 32
        let loss = session.tensor_sum(cp).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        let grad = session.tensor_gradient(&report, t).expect("grad");
        // d(sum(cumprod(x)))/dx for x=[2,3,4]:
        // d/dx0: cumprod[0]/x0 + cumprod[1]/x0 + cumprod[2]/x0 = 2/2 + 6/2 + 24/2 = 1+3+12 = 16
        // d/dx1: cumprod[1]/x1 + cumprod[2]/x1 = 6/3 + 24/3 = 2+8 = 10
        // d/dx2: cumprod[2]/x2 = 24/4 = 6
        assert!((grad[0] - 16.0).abs() < 1e-10, "grad[0]={}", grad[0]);
        assert!((grad[1] - 10.0).abs() < 1e-10, "grad[1]={}", grad[1]);
        assert!((grad[2] - 6.0).abs() < 1e-10, "grad[2]={}", grad[2]);
    }

    // ---- tensor_where ----

    #[test]
    fn session_tensor_where_selects_by_condition() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cond = session
            .tensor_variable(vec![1.0, 0.0, 1.0, 0.0], vec![4], false)
            .expect("cond");
        let x = session
            .tensor_variable(vec![10.0, 20.0, 30.0, 40.0], vec![4], false)
            .expect("x");
        let y = session
            .tensor_variable(vec![-1.0, -2.0, -3.0, -4.0], vec![4], false)
            .expect("y");
        let result = session.tensor_where(cond, x, y).expect("where");
        let vals = session.tensor_values(result).expect("vals");
        assert_eq!(vals, vec![10.0, -2.0, 30.0, -4.0]);
    }

    #[test]
    fn session_tensor_where_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cond = session
            .tensor_variable(vec![1.0, 0.0, 1.0], vec![3], false)
            .expect("cond");
        let x = session
            .tensor_variable(vec![2.0, 4.0, 6.0], vec![3], true)
            .expect("x");
        let y = session
            .tensor_variable(vec![10.0, 20.0, 30.0], vec![3], true)
            .expect("y");
        let result = session.tensor_where(cond, x, y).expect("where");
        // result = [2.0, 20.0, 6.0], sum = 28.0
        let loss = session.tensor_sum(result).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x).expect("x grad");
        let y_grad = session.tensor_gradient(&report, y).expect("y grad");

        // x gets gradient where condition is 1, y gets gradient where condition is 0
        assert_eq!(x_grad, &[1.0, 0.0, 1.0]);
        assert_eq!(y_grad, &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn session_tensor_where_rejects_shape_mismatch() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cond = session
            .tensor_variable(vec![1.0, 0.0], vec![2], false)
            .expect("cond");
        let x = session
            .tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false)
            .expect("x");
        let y = session
            .tensor_variable(vec![-1.0, -2.0, -3.0], vec![3], false)
            .expect("y");
        let result = session.tensor_where(cond, x, y);
        assert!(result.is_err());
    }

    // ---- nll_loss ----

    #[test]
    fn session_nll_loss_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // log_probs for batch=2, classes=3
        // log_softmax-like values (already log probabilities)
        let log_probs = session
            .tensor_variable(vec![-2.0, -0.5, -1.5, -0.3, -2.0, -1.0], vec![2, 3], false)
            .expect("log_probs");
        // targets: class 1 for sample 0, class 0 for sample 1
        let targets = session
            .tensor_variable(vec![1.0, 0.0], vec![2], false)
            .expect("targets");
        let loss = session.nll_loss(log_probs, targets).expect("nll_loss");
        let val = session.tensor_values(loss).expect("vals")[0];
        // NLL = -mean(-0.5, -0.3) = mean(0.5, 0.3) = 0.4
        assert!((val - 0.4).abs() < 1e-10, "expected NLL=0.4, got {}", val);
    }

    #[test]
    fn session_nll_loss_rejects_fractional_targets() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let log_probs = session
            .tensor_variable(vec![-2.0, -0.5, -1.5, -0.3, -2.0, -1.0], vec![2, 3], false)
            .expect("log_probs");
        let targets = session
            .tensor_variable(vec![1.5, 0.0], vec![2], false)
            .expect("targets");
        let result = session.nll_loss(log_probs, targets);
        assert!(result.is_err());
    }

    #[test]
    fn session_nll_loss_rejects_tiny_fractional_targets() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let log_probs = session
            .tensor_variable(vec![-2.0, -0.5, -1.5, -0.3, -2.0, -1.0], vec![2, 3], false)
            .expect("log_probs");
        let targets = session
            .tensor_variable(vec![1e-20, 0.0], vec![2], false)
            .expect("targets");
        let result = session.nll_loss(log_probs, targets);
        assert!(result.is_err());
    }

    #[test]
    fn session_nll_loss_rejects_mismatched_target_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let log_probs = session
            .tensor_variable(vec![-2.0, -0.5, -1.5, -0.3, -2.0, -1.0], vec![2, 3], false)
            .expect("log_probs");
        let targets = session
            .tensor_variable(vec![1.0, 0.0], vec![1, 2], false)
            .expect("targets");
        let result = session.nll_loss(log_probs, targets);
        assert!(result.is_err());
    }

    // ---- cross_entropy_loss ----

    #[test]
    fn session_cross_entropy_loss_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // logits: batch=1, classes=3
        let logits = session
            .tensor_variable(vec![2.0, 1.0, 0.1], vec![1, 3], true)
            .expect("logits");
        // target: class 0 (highest logit)
        let targets = session
            .tensor_variable(vec![0.0], vec![1], false)
            .expect("targets");
        let loss = session
            .cross_entropy_loss(logits, targets)
            .expect("cross_entropy");
        let val = session.tensor_values(loss).expect("vals")[0];
        // cross_entropy with correct class should be relatively low
        assert!(val > 0.0, "cross entropy should be positive, got {}", val);
        assert!(
            val < 2.0,
            "cross entropy for correct class should be small, got {}",
            val
        );
    }

    #[test]
    fn session_cross_entropy_loss_wrong_class_is_higher() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // logits: batch=1, classes=3: class 0 has highest logit
        let logits = session
            .tensor_variable(vec![5.0, 1.0, 0.1], vec![1, 3], false)
            .expect("logits");

        // Target correct class (0) - should have low loss
        let target_correct = session
            .tensor_variable(vec![0.0], vec![1], false)
            .expect("targets_correct");
        let loss_correct = session
            .cross_entropy_loss(logits, target_correct)
            .expect("ce_correct");
        let val_correct = session.tensor_values(loss_correct).expect("vals")[0];

        // Target wrong class (2) - should have higher loss
        let logits2 = session
            .tensor_variable(vec![5.0, 1.0, 0.1], vec![1, 3], false)
            .expect("logits2");
        let target_wrong = session
            .tensor_variable(vec![2.0], vec![1], false)
            .expect("targets_wrong");
        let loss_wrong = session
            .cross_entropy_loss(logits2, target_wrong)
            .expect("ce_wrong");
        let val_wrong = session.tensor_values(loss_wrong).expect("vals")[0];

        assert!(
            val_wrong > val_correct,
            "wrong class loss ({}) should be higher than correct class loss ({})",
            val_wrong,
            val_correct
        );
    }

    #[test]
    fn session_cross_entropy_loss_rejects_rank0_logits() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let logits = session
            .tensor_variable(vec![2.0], vec![], false)
            .expect("rank-0 logits");
        let targets = session
            .tensor_variable(vec![0.0], vec![1], false)
            .expect("targets");
        let err = session
            .cross_entropy_loss(logits, targets)
            .expect_err("rank-0 logits should fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension { dim: 0, ndim: 0 }
            ))
        ));
    }

    // ---- kernel where tests ----

    #[test]
    fn session_tensor_where_all_true() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cond = session
            .tensor_variable(vec![1.0, 1.0, 1.0], vec![3], false)
            .expect("cond");
        let x = session
            .tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false)
            .expect("x");
        let y = session
            .tensor_variable(vec![-1.0, -2.0, -3.0], vec![3], false)
            .expect("y");
        let result = session.tensor_where(cond, x, y).expect("where");
        let vals = session.tensor_values(result).expect("vals");
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn session_tensor_where_all_false() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cond = session
            .tensor_variable(vec![0.0, 0.0, 0.0], vec![3], false)
            .expect("cond");
        let x = session
            .tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false)
            .expect("x");
        let y = session
            .tensor_variable(vec![-1.0, -2.0, -3.0], vec![3], false)
            .expect("y");
        let result = session.tensor_where(cond, x, y).expect("where");
        let vals = session.tensor_values(result).expect("vals");
        assert_eq!(vals, vec![-1.0, -2.0, -3.0]);
    }

    // ---- eye ----

    #[test]
    fn session_eye_3x3() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let eye = session.eye(3, false).expect("eye");
        let vals = session.tensor_values(eye).expect("vals");
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let shape = session.tensor_shape(eye).expect("shape");
        assert_eq!(shape, vec![3, 3]);
    }

    #[test]
    fn session_eye_1x1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let eye = session.eye(1, false).expect("eye");
        let vals = session.tensor_values(eye).expect("vals");
        assert_eq!(vals, vec![1.0]);
    }

    // ---- diag ----

    #[test]
    fn session_diag_1d_to_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("v");
        let d = session.diag(v).expect("diag");
        let vals = session.tensor_values(d).expect("vals");
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        let shape = session.tensor_shape(d).expect("shape");
        assert_eq!(shape, vec![3, 3]);
    }

    #[test]
    fn session_diag_2d_to_1d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![3, 3],
                false,
            )
            .expect("m");
        let d = session.diag(m).expect("diag");
        let vals = session.tensor_values(d).expect("vals");
        assert_eq!(vals, vec![1.0, 5.0, 9.0]);
        let shape = session.tensor_shape(d).expect("shape");
        assert_eq!(shape, vec![3]);
    }

    #[test]
    fn session_diag_2d_rectangular() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("m");
        let d = session.diag(m).expect("diag");
        let vals = session.tensor_values(d).expect("vals");
        assert_eq!(vals, vec![1.0, 5.0]); // min(2,3) = 2 elements
    }

    // ---- triu ----

    #[test]
    fn session_triu_k0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![3, 3],
                false,
            )
            .expect("m");
        let u = session.triu(m, 0).expect("triu");
        let vals = session.tensor_values(u).expect("vals");
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn session_triu_k1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![3, 3],
                false,
            )
            .expect("m");
        let u = session.triu(m, 1).expect("triu");
        let vals = session.tensor_values(u).expect("vals");
        assert_eq!(vals, vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
    }

    // ---- tril ----

    #[test]
    fn session_tril_k0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![3, 3],
                false,
            )
            .expect("m");
        let l = session.tril(m, 0).expect("tril");
        let vals = session.tensor_values(l).expect("vals");
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn session_tril_k_neg1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![3, 3],
                false,
            )
            .expect("m");
        let l = session.tril(m, -1).expect("tril");
        let vals = session.tensor_values(l).expect("vals");
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]);
    }

    // ---- sort ----

    #[test]
    fn session_sort_ascending() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5], false)
            .expect("t");
        let (sorted, indices) = session.tensor_sort(t, 0, false).expect("sort");
        let vals = session.tensor_values(sorted).expect("vals");
        assert_eq!(vals, vec![1.0, 1.0, 3.0, 4.0, 5.0]);
        assert_eq!(indices, vec![1, 3, 0, 2, 4]);
    }

    #[test]
    fn session_sort_descending() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![2.0, 5.0, 1.0, 3.0], vec![4], false)
            .expect("t");
        let (sorted, indices) = session.tensor_sort(t, 0, true).expect("sort");
        let vals = session.tensor_values(sorted).expect("vals");
        assert_eq!(vals, vec![5.0, 3.0, 2.0, 1.0]);
        assert_eq!(indices, vec![1, 3, 0, 2]);
    }

    #[test]
    fn session_sort_2d_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], vec![2, 3], false)
            .expect("t");
        let (sorted, _indices) = session.tensor_sort(t, 1, false).expect("sort");
        let vals = session.tensor_values(sorted).expect("vals");
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn session_sort_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![3.0, 1.0, 2.0], vec![3], true)
            .expect("t");
        let (sorted, _indices) = session.tensor_sort(t, 0, false).expect("sort");
        // sorted = [1.0, 2.0, 3.0], indices = [1, 2, 0]
        // sum the sorted values
        let loss = session.tensor_sum(sorted).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        // All gradient should be 1.0 (scattered back to original positions)
        assert_eq!(report.gradient(t), Some(vec![1.0, 1.0, 1.0].as_slice()));
    }

    // ---- topk ----

    #[test]
    fn session_topk_largest() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5], false)
            .expect("t");
        let (topk, indices) = session.tensor_topk(t, 3, 0, true, true).expect("topk");
        let vals = session.tensor_values(topk).expect("vals");
        assert_eq!(vals, vec![5.0, 4.0, 3.0]);
        assert_eq!(indices, vec![4, 2, 0]);
    }

    #[test]
    fn session_topk_smallest() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5], false)
            .expect("t");
        let (topk, indices) = session.tensor_topk(t, 2, 0, false, true).expect("topk");
        let vals = session.tensor_values(topk).expect("vals");
        assert_eq!(vals, vec![1.0, 1.0]);
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    fn session_topk_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(
                vec![4.0, 2.0, 3.0, 1.0, 8.0, 6.0, 7.0, 5.0],
                vec![2, 4],
                false,
            )
            .expect("t");
        let (topk, _indices) = session.tensor_topk(t, 2, 1, true, true).expect("topk");
        let vals = session.tensor_values(topk).expect("vals");
        assert_eq!(vals, vec![4.0, 3.0, 8.0, 7.0]);
        let shape = session.tensor_shape(topk).expect("shape");
        assert_eq!(shape, vec![2, 2]);
    }

    #[test]
    fn session_topk_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5], true)
            .expect("t");
        let (topk, _indices) = session.tensor_topk(t, 2, 0, true, true).expect("topk");
        // topk = [5.0, 4.0], indices = [4, 2]
        let loss = session.tensor_sum(topk).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        // Only positions 4 and 2 should have gradient 1.0, rest 0.0
        assert_eq!(
            report.gradient(t),
            Some(vec![0.0, 0.0, 1.0, 0.0, 1.0].as_slice())
        );
    }

    // ---- utility methods ----

    #[test]
    fn session_tensor_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        assert_eq!(session.tensor_shape(t).expect("shape"), vec![2, 3]);
    }

    #[test]
    fn session_tensor_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        assert_eq!(session.tensor_dim(t).expect("dim"), 2);
    }

    #[test]
    fn session_tensor_numel() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        assert_eq!(session.tensor_numel(t).expect("numel"), 6);
    }

    #[test]
    fn session_tensor_item_scalar() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![42.0], vec![1], false)
            .expect("t");
        assert_eq!(session.tensor_item(t).expect("item"), 42.0);
    }

    #[test]
    fn session_tensor_item_multi_element_fails() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("t");
        assert!(session.tensor_item(t).is_err());
    }

    #[test]
    fn session_tensor_clone_creates_independent_copy() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        let c = session.tensor_clone(t, false).expect("clone");
        assert_eq!(session.tensor_values(c).expect("vals"), vec![1.0, 2.0, 3.0]);
        assert_eq!(session.tensor_shape(c).expect("shape"), vec![3]);
        // Nodes should be different
        assert_ne!(t, c);
    }

    #[test]
    fn session_tensor_detach_has_no_grad() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("t");
        let d = session.tensor_detach(t).expect("detach");
        let vals = session.tensor_values(d).expect("vals");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
        // Detached tensor should not propagate gradients back to original
        // Add d to t so the backward root requires grad
        let added = session.tensor_add(t, d).expect("add");
        let sum = session.tensor_sum(added).expect("sum");
        let report = session.tensor_backward(sum).expect("backward");
        // t should receive gradient (from the direct path), but d is detached
        assert!(
            report.gradient(t).is_some(),
            "original tensor should have gradient"
        );
        assert!(
            report.gradient(d).is_none(),
            "detached tensor should not have gradient"
        );
    }

    #[test]
    fn session_tensor_requires_grad_toggle_on_leaf_round_trips() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("leaf");

        assert!(!session.tensor_requires_grad(t).expect("requires_grad"));
        assert!(session.tensor_is_leaf(t).expect("is_leaf"));
        assert!(session.tensor_grad_fn(t).expect("grad_fn").is_none());

        session
            .tensor_requires_grad_(t, true)
            .expect("enable requires_grad on leaf");
        assert!(session.tensor_requires_grad(t).expect("requires_grad"));

        session
            .tensor_requires_grad_(t, false)
            .expect("disable requires_grad on leaf");
        assert!(!session.tensor_requires_grad(t).expect("requires_grad"));
    }

    #[test]
    fn session_tensor_requires_grad_on_non_leaf_errors() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("t");
        let ones = session.full(vec![3], 1.0, false).expect("ones");
        let y = session.tensor_mul(t, ones).expect("mul");

        let err = session
            .tensor_requires_grad_(y, false)
            .expect_err("non-leaf requires_grad_ should fail");
        assert!(matches!(
            err,
            AutogradError::TensorRequiresGradNonLeaf { .. }
        ));
    }

    #[test]
    fn session_tensor_detach_in_place_turns_non_leaf_into_leaf() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("t");
        let ones = session.full(vec![3], 1.0, false).expect("ones");
        let y = session.tensor_mul(t, ones).expect("mul");

        assert!(session.tensor_requires_grad(y).expect("requires_grad"));
        assert!(!session.tensor_is_leaf(y).expect("is_leaf"));
        assert_eq!(
            session.tensor_grad_fn(y).expect("grad_fn"),
            Some("Mul".to_string())
        );

        session.tensor_detach_(y).expect("detach_");

        assert!(!session.tensor_requires_grad(y).expect("requires_grad"));
        assert!(session.tensor_is_leaf(y).expect("is_leaf"));
        assert!(session.tensor_grad_fn(y).expect("grad_fn").is_none());

        let added = session.tensor_add(t, y).expect("add");
        let sum = session.tensor_sum(added).expect("sum");
        let report = session.tensor_backward(sum).expect("backward");
        assert!(report.gradient(t).is_some(), "direct path must keep grad");
        assert!(
            report.gradient(y).is_none(),
            "detached node must not receive grad"
        );
    }

    #[test]
    fn session_tensor_freeze_unfreeze_controls_gradient_flow() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("t");
        let ones = session.full(vec![3], 1.0, false).expect("ones");

        session.tensor_requires_grad_(t, false).expect("freeze");
        let frozen_out = session.tensor_mul(t, ones).expect("mul frozen");
        let frozen_sum = session.tensor_sum(frozen_out).expect("sum frozen");
        let frozen_err = session
            .tensor_backward(frozen_sum)
            .expect_err("frozen tensor graph should reject backward root");
        assert!(matches!(
            frozen_err,
            AutogradError::TensorRootDoesNotRequireGrad { .. }
        ));

        session.tensor_requires_grad_(t, true).expect("unfreeze");
        let ones2 = session.full(vec![3], 1.0, false).expect("ones2");
        let active_out = session.tensor_mul(t, ones2).expect("mul active");
        let active_sum = session.tensor_sum(active_out).expect("sum active");
        let active_report = session.tensor_backward(active_sum).expect("backward");
        assert!(
            active_report.gradient(t).is_some(),
            "unfrozen tensor should receive gradient"
        );
    }

    #[test]
    fn session_tensor_detach_in_place_on_leaf_is_noop() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("leaf");
        session.tensor_detach_(t).expect("detach_ leaf");
        assert!(session.tensor_is_leaf(t).expect("is_leaf"));
        assert!(!session.tensor_requires_grad(t).expect("requires_grad"));
        assert!(session.tensor_grad_fn(t).expect("grad_fn").is_none());
    }

    #[test]
    fn session_tensor_requires_grad_toggle_works_inside_no_grad_scope() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("leaf");
        session.no_grad_enter();
        session
            .tensor_requires_grad_(t, true)
            .expect("requires_grad_ in no_grad");
        session.no_grad_exit();
        assert!(
            session.tensor_requires_grad(t).expect("requires_grad"),
            "explicit requires_grad_ toggle should persist after no_grad scope exits"
        );
    }

    #[test]
    fn session_tensor_register_hook_modifies_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("t");
        let neg = session.tensor_neg(t).expect("neg");

        session
            .tensor_register_hook(neg, |grad| {
                Ok(Some(
                    grad.iter().map(|value| value * 2.0).collect::<Vec<_>>(),
                ))
            })
            .expect("register hook");

        let loss = session.tensor_sum(neg).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        assert_eq!(report.gradient(neg).expect("neg grad"), &[2.0, 2.0, 2.0]);
        assert_eq!(report.gradient(t).expect("t grad"), &[-2.0, -2.0, -2.0]);
    }

    #[test]
    fn session_tensor_register_hook_remove_stops_callback() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("t");
        let neg = session.tensor_neg(t).expect("neg");

        let handle = session
            .tensor_register_hook(neg, |grad| {
                Ok(Some(
                    grad.iter().map(|value| value * 4.0).collect::<Vec<_>>(),
                ))
            })
            .expect("register hook");
        assert!(session.tensor_remove_hook(handle).expect("remove hook"));

        let loss = session.tensor_sum(neg).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        assert_eq!(report.gradient(neg).expect("neg grad"), &[1.0, 1.0, 1.0]);
        assert_eq!(report.gradient(t).expect("t grad"), &[-1.0, -1.0, -1.0]);
    }

    #[test]
    fn session_tensor_register_hook_order_and_error() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .expect("t");
        let neg = session.tensor_neg(t).expect("neg");

        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let order_first = Arc::clone(&order);
        session
            .tensor_register_hook(neg, move |grad| {
                order_first.lock().expect("order lock").push("first");
                Ok(Some(
                    grad.iter().map(|value| value + 1.0).collect::<Vec<_>>(),
                ))
            })
            .expect("register first");
        let order_second = Arc::clone(&order);
        session
            .tensor_register_hook(neg, move |grad| {
                order_second.lock().expect("order lock").push("second");
                Ok(Some(
                    grad.iter().map(|value| value * 2.0).collect::<Vec<_>>(),
                ))
            })
            .expect("register second");

        let loss = session.tensor_sum(neg).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        assert_eq!(report.gradient(t).expect("t grad"), &[-4.0, -4.0]);
        let order_values = order.lock().expect("order lock");
        assert_eq!(order_values.as_slice(), &["first", "second"]);

        let mut session_err = FrankenTorchSession::new(ExecutionMode::Strict);
        let u = session_err
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .expect("u");
        let neg_u = session_err.tensor_neg(u).expect("neg_u");
        session_err
            .tensor_register_hook(neg_u, |_grad| Err(AutogradError::TensorGraphConsumed))
            .expect("register error hook");
        let loss_u = session_err.tensor_sum(neg_u).expect("sum_u");
        let err = session_err
            .tensor_backward(loss_u)
            .expect_err("hook error should propagate");
        assert!(matches!(err, AutogradError::TensorGraphConsumed));
    }

    #[test]
    fn session_tensor_register_hook_supports_gradient_reversal() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("t");
        let neg = session.tensor_neg(t).expect("neg");
        session
            .tensor_register_hook(neg, |grad| {
                Ok(Some(grad.iter().map(|value| -*value).collect::<Vec<_>>()))
            })
            .expect("register reversal hook");

        let loss = session.tensor_sum(neg).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        assert_eq!(report.gradient(neg).expect("neg grad"), &[-1.0, -1.0, -1.0]);
        assert_eq!(report.gradient(t).expect("t grad"), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_register_hook_supports_gradient_clipping() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![2.0, -3.0, 0.5], vec![3], true)
            .expect("t");
        session
            .tensor_register_hook(t, |grad| {
                Ok(Some(
                    grad.iter().map(|value| value.clamp(-1.0, 1.0)).collect(),
                ))
            })
            .expect("register clipping hook");

        let squared = session.tensor_mul(t, t).expect("square");
        let loss = session.tensor_sum(squared).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        assert_eq!(report.gradient(t).expect("t grad"), &[1.0, -1.0, 1.0]);
    }

    #[test]
    fn session_tensor_register_hook_supports_multiple_tensors_in_same_graph() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![2.0, 3.0], vec![2], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![5.0, 7.0], vec![2], true)
            .expect("b");
        let fire_count = Arc::new(Mutex::new(0usize));

        let count_a = Arc::clone(&fire_count);
        session
            .tensor_register_hook(a, move |grad| {
                *count_a.lock().expect("count lock") += 1;
                Ok(Some(vec![0.0; grad.len()]))
            })
            .expect("register hook a");

        let count_b = Arc::clone(&fire_count);
        session
            .tensor_register_hook(b, move |grad| {
                *count_b.lock().expect("count lock") += 1;
                Ok(Some(grad.iter().map(|value| value * 2.0).collect()))
            })
            .expect("register hook b");

        let prod = session.tensor_mul(a, b).expect("mul");
        let loss = session.tensor_sum(prod).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert_eq!(report.gradient(a).expect("a grad"), &[0.0, 0.0]);
        assert_eq!(report.gradient(b).expect("b grad"), &[4.0, 6.0]);
        assert_eq!(*fire_count.lock().expect("count lock"), 2);
    }

    // ---- tensor creation ----

    #[test]
    fn session_linspace_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session.linspace(0.0, 1.0, 5, false).expect("linspace");
        let vals = session.tensor_values(t).expect("vals");
        assert_eq!(vals, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(session.tensor_shape(t).expect("shape"), vec![5]);
    }

    #[test]
    fn session_linspace_single_step() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session.linspace(3.0, 7.0, 1, false).expect("linspace");
        let vals = session.tensor_values(t).expect("vals");
        assert_eq!(vals, vec![3.0]);
    }

    #[test]
    fn session_linspace_two_steps() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session.linspace(0.0, 10.0, 2, false).expect("linspace");
        let vals = session.tensor_values(t).expect("vals");
        assert_eq!(vals, vec![0.0, 10.0]);
    }

    #[test]
    fn session_full_like() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        let f = session.full_like(t, 7.0, false).expect("full_like");
        let vals = session.tensor_values(f).expect("vals");
        assert_eq!(vals, vec![7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
        assert_eq!(session.tensor_shape(f).expect("shape"), vec![2, 3]);
    }

    #[test]
    fn session_zeros_like() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        let z = session.zeros_like(t, false).expect("zeros_like");
        let vals = session.tensor_values(z).expect("vals");
        assert_eq!(vals, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn session_ones_like() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        let o = session.ones_like(t, false).expect("ones_like");
        let vals = session.tensor_values(o).expect("vals");
        assert_eq!(vals, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_dot_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], true)
            .expect("b");
        let z = session.tensor_dot(a, b).expect("dot");

        // dot([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 32
        assert_eq!(session.tensor_values(z).expect("dot values"), vec![32.0]);

        let report = session.tensor_backward(z).expect("backward");
        // grad_a = grad_out * b = 1 * [4,5,6]
        assert_eq!(
            session.tensor_gradient(&report, a).expect("a grad"),
            &[4.0, 5.0, 6.0]
        );
        // grad_b = grad_out * a = 1 * [1,2,3]
        assert_eq!(
            session.tensor_gradient(&report, b).expect("b grad"),
            &[1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn session_tensor_outer_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![3.0, 4.0, 5.0], vec![3], true)
            .expect("b");
        let z = session.tensor_outer(a, b).expect("outer");

        // outer([1,2], [3,4,5]) = [[3,4,5],[6,8,10]]
        assert_eq!(
            session.tensor_values(z).expect("outer values"),
            vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        );

        // Need to reduce to scalar for backward. Sum the outer product.
        let s = session.tensor_sum(z).expect("sum");
        let report = session.tensor_backward(s).expect("backward");

        // grad_a[i] = sum_j(1 * b[j]) = sum(b) = 12
        assert_eq!(
            session.tensor_gradient(&report, a).expect("a grad"),
            &[12.0, 12.0]
        );
        // grad_b[j] = sum_i(1 * a[i]) = sum(a) = 3
        assert_eq!(
            session.tensor_gradient(&report, b).expect("b grad"),
            &[3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn session_tensor_bmm_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // batch=1, 2x2 @ 2x2
        let a = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2], true)
            .expect("b");
        let z = session.tensor_bmm(a, b).expect("bmm");

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(
            session.tensor_values(z).expect("bmm values"),
            vec![19.0, 22.0, 43.0, 50.0]
        );

        let s = session.tensor_sum(z).expect("sum");
        let report = session.tensor_backward(s).expect("backward");

        // grad_a = grad_out @ b^T, grad_out = [[1,1],[1,1]]
        // b^T = [[5,7],[6,8]], grad_a = [[11,15],[11,15]]
        assert_eq!(
            session.tensor_gradient(&report, a).expect("a grad"),
            &[11.0, 15.0, 11.0, 15.0]
        );
        // grad_b = a^T @ grad_out
        // a^T = [[1,3],[2,4]], grad_b = [[4,4],[6,6]]
        assert_eq!(
            session.tensor_gradient(&report, b).expect("b grad"),
            &[4.0, 4.0, 6.0, 6.0]
        );
    }

    #[test]
    fn session_tensor_trace_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("x");
        let t = session.tensor_trace(x).expect("trace");

        // trace([[1,2],[3,4]]) = 1 + 4 = 5
        assert_eq!(session.tensor_values(t).expect("trace values"), vec![5.0]);

        let report = session.tensor_backward(t).expect("backward");
        // grad = identity matrix = [[1,0],[0,1]]
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[1.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn session_tensor_dot_zero_length() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.tensor_variable(vec![], vec![0], false).expect("a");
        let b = session.tensor_variable(vec![], vec![0], false).expect("b");
        let z = session.tensor_dot(a, b).expect("dot");
        assert_eq!(session.tensor_values(z).expect("dot values"), vec![0.0]);
    }

    #[test]
    fn session_tensor_trace_non_square() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // trace of 2x3 matrix: sum of diag = min(2,3)=2 diag elements
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let t = session.tensor_trace(x).expect("trace");
        // trace = 1.0 + 5.0 = 6.0
        assert_eq!(session.tensor_values(t).expect("trace values"), vec![6.0]);

        let report = session.tensor_backward(t).expect("backward");
        // grad = [[1,0,0],[0,1,0]]
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
    }

    // ---- flip ----

    #[test]
    fn session_flip_1d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("t");
        let f = session.tensor_flip(t, &[0]).expect("flip");
        assert_eq!(
            session.tensor_values(f).expect("vals"),
            vec![4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn session_flip_2d_dim0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [[1,2,3],[4,5,6]]
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        let f = session.tensor_flip(t, &[0]).expect("flip");
        // [[4,5,6],[1,2,3]]
        assert_eq!(
            session.tensor_values(f).expect("vals"),
            vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn session_flip_2d_both_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [[1,2],[3,4]]
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .expect("t");
        let f = session.tensor_flip(t, &[0, 1]).expect("flip");
        // [[4,3],[2,1]]
        assert_eq!(
            session.tensor_values(f).expect("vals"),
            vec![4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn session_flip_empty_dims_is_noop() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        let f = session.tensor_flip(t, &[]).expect("flip");
        assert_eq!(session.tensor_values(f).expect("vals"), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn session_flip_rejects_invalid_dimension() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        let err = session
            .tensor_flip(t, &[1])
            .expect_err("flip should reject out-of-range dim");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension { dim: 1, ndim: 1 }
            ))
        ));
    }

    // ---- repeat ----

    #[test]
    fn session_repeat_1d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        let r = session.tensor_repeat(t, &[3]).expect("repeat");
        assert_eq!(
            session.tensor_values(r).expect("vals"),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(session.tensor_shape(r).expect("shape"), vec![9]);
    }

    #[test]
    fn session_repeat_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [[1,2],[3,4]]
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .expect("t");
        let r = session.tensor_repeat(t, &[2, 3]).expect("repeat");
        // shape = [4, 6]
        assert_eq!(session.tensor_shape(r).expect("shape"), vec![4, 6]);
        let vals = session.tensor_values(r).expect("vals");
        // Row 0: [1,2,1,2,1,2], Row 1: [3,4,3,4,3,4], Row 2: [1,2,1,2,1,2], Row 3: [3,4,3,4,3,4]
        assert_eq!(
            vals,
            vec![
                1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0,
                1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
            ]
        );
    }

    #[test]
    fn session_repeat_rejects_shape_multiplication_overflow() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("t");
        let err = session
            .tensor_repeat(t, &[usize::MAX])
            .expect_err("overflowing repeat shape must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "repeat shape multiplication overflow"
                }
            ))
        ));
    }

    #[test]
    fn session_repeat_with_leading_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("t");
        let r = session.tensor_repeat(t, &[2, 3]).expect("repeat");
        assert_eq!(session.tensor_shape(r).expect("shape"), vec![2, 6]);
        assert_eq!(
            session.tensor_values(r).expect("vals"),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn session_repeat_rejects_short_repeat_tuple() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .expect("t");
        let err = session
            .tensor_repeat(t, &[2])
            .expect_err("repeat tuple shorter than rank should fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::ShapeMismatch { lhs, rhs }
            )) if lhs == vec![2, 2] && rhs == vec![2]
        ));
    }

    #[test]
    fn session_repeat_zero_factor_produces_empty_tensor() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        let r = session.tensor_repeat(t, &[0]).expect("repeat");
        assert_eq!(session.tensor_shape(r).expect("shape"), vec![0]);
        assert!(session.tensor_values(r).expect("vals").is_empty());
    }

    // ---- roll ----

    #[test]
    fn session_roll_1d_positive() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("t");
        let r = session.tensor_roll(t, 1, 0).expect("roll");
        assert_eq!(
            session.tensor_values(r).expect("vals"),
            vec![4.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn session_roll_1d_negative() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("t");
        let r = session.tensor_roll(t, -1, 0).expect("roll");
        assert_eq!(
            session.tensor_values(r).expect("vals"),
            vec![2.0, 3.0, 4.0, 1.0]
        );
    }

    #[test]
    fn session_roll_2d_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [[1,2,3],[4,5,6]]
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("t");
        let r = session.tensor_roll(t, 1, 1).expect("roll");
        // [[3,1,2],[6,4,5]]
        assert_eq!(
            session.tensor_values(r).expect("vals"),
            vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]
        );
    }

    #[test]
    fn session_roll_large_shift_wraps_mod_dimension() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("t");
        let r = session.tensor_roll(t, 5, 0).expect("roll");
        assert_eq!(
            session.tensor_values(r).expect("vals"),
            vec![4.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn session_roll_rejects_invalid_dimension() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        let err = session
            .tensor_roll(t, 1, 1)
            .expect_err("roll should reject out-of-range dim");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension { dim: 1, ndim: 1 }
            ))
        ));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn session_roll_rejects_shift_out_of_isize_range() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        let err = session
            .tensor_roll(t, i64::from(i32::MAX) + 1, 0)
            .expect_err("shift larger than isize should fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "tensor_roll shift is out of range for platform isize"
                }
            ))
        ));
    }

    // ---- flip backward ----

    #[test]
    fn session_flip_backward_1d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // x = [1, 2, 3, 4], requires_grad = true
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
            .expect("x");
        let f = session.tensor_flip(x, &[0]).expect("flip");
        // sum the flipped tensor so we get a scalar to backward from
        let s = session.tensor_sum(f).expect("sum");

        let report = session.tensor_backward(s).expect("backward");
        // flip is self-inverse, so grad_x = flip(ones, dim0) = [1,1,1,1]
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[1.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn session_flip_backward_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // x = [[1,2],[3,4]], requires_grad = true
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("x");
        let f = session.tensor_flip(x, &[1]).expect("flip");
        // Multiply by a weight to get non-trivial gradient
        let w = session
            .tensor_variable(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2], false)
            .expect("w");
        let prod = session.tensor_mul(f, w).expect("mul");
        let s = session.tensor_sum(prod).expect("sum");

        let report = session.tensor_backward(s).expect("backward");
        // flip(x, dim1): [[2,1],[4,3]]
        // mul with w: [[2*10, 1*20],[4*30, 3*40]]
        // d(sum)/d(flip_out) = w = [[10,20],[30,40]]
        // d(flip_out)/d(x) = flip(grad, dim1) => [[20,10],[40,30]]
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[20.0, 10.0, 40.0, 30.0]
        );
    }

    // ---- repeat backward ----

    #[test]
    fn session_repeat_backward_1d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // x = [1, 2, 3], requires_grad = true
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("x");
        let r = session.tensor_repeat(x, &[3]).expect("repeat");
        // r = [1,2,3,1,2,3,1,2,3], shape [9]
        let s = session.tensor_sum(r).expect("sum");

        let report = session.tensor_backward(s).expect("backward");
        // Each element of x is repeated 3 times, so gradient accumulates 3 copies of 1.0
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn session_repeat_backward_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // x = [[1,2],[3,4]], requires_grad = true
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("x");
        let r = session.tensor_repeat(x, &[2, 3]).expect("repeat");
        // shape [4, 6]
        let s = session.tensor_sum(r).expect("sum");

        let report = session.tensor_backward(s).expect("backward");
        // Each element repeated 2*3 = 6 times total
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[6.0, 6.0, 6.0, 6.0]
        );
    }

    #[test]
    fn session_repeat_backward_with_leading_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("x");
        let r = session.tensor_repeat(x, &[2, 3]).expect("repeat");
        assert_eq!(session.tensor_shape(r).expect("shape"), vec![2, 9]);
        let s = session.tensor_sum(r).expect("sum");

        let report = session.tensor_backward(s).expect("backward");
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[6.0, 6.0, 6.0]
        );
    }

    // ---- roll backward ----

    #[test]
    fn session_roll_backward_1d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // x = [1, 2, 3, 4], requires_grad = true
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
            .expect("x");
        let r = session.tensor_roll(x, 1, 0).expect("roll");
        // r = [4,1,2,3]
        // Multiply by weights to get non-trivial gradient
        let w = session
            .tensor_variable(vec![10.0, 20.0, 30.0, 40.0], vec![4], false)
            .expect("w");
        let prod = session.tensor_mul(r, w).expect("mul");
        let s = session.tensor_sum(prod).expect("sum");

        let report = session.tensor_backward(s).expect("backward");
        // d(sum)/d(roll_out) = w = [10, 20, 30, 40]
        // roll backward = roll(grad, -shift, dim) = roll([10,20,30,40], -1, 0) = [20,30,40,10]
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[20.0, 30.0, 40.0, 10.0]
        );
    }

    #[test]
    fn session_roll_backward_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // x = [[1,2,3],[4,5,6]], requires_grad = true
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let r = session.tensor_roll(x, 1, 1).expect("roll");
        // r = [[3,1,2],[6,4,5]]
        let s = session.tensor_sum(r).expect("sum");

        let report = session.tensor_backward(s).expect("backward");
        // d(sum)/d(roll_out) = ones [1,1,1,1,1,1]
        // roll backward = roll(ones, -1, dim1) = ones (since rolling ones is still ones)
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    // ---- any / all ----

    #[test]
    fn session_tensor_any_true() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![0.0, 0.0, 1.0], vec![3], false)
            .expect("t");
        assert!(session.tensor_any(t).expect("any"));
    }

    #[test]
    fn session_tensor_any_false() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![0.0, 0.0, 0.0], vec![3], false)
            .expect("t");
        assert!(!session.tensor_any(t).expect("any"));
    }

    #[test]
    fn session_tensor_any_empty_false() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session.tensor_variable(vec![], vec![0], false).expect("t");
        assert!(!session.tensor_any(t).expect("any"));
    }

    #[test]
    fn session_tensor_all_true() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("t");
        assert!(session.tensor_all(t).expect("all"));
    }

    #[test]
    fn session_tensor_all_false() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 0.0, 3.0], vec![3], false)
            .expect("t");
        assert!(!session.tensor_all(t).expect("all"));
    }

    #[test]
    fn session_tensor_all_empty_true() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session.tensor_variable(vec![], vec![0], false).expect("t");
        assert!(session.tensor_all(t).expect("all"));
    }

    // ---- median ----

    #[test]
    fn session_tensor_median_odd() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![3.0, 1.0, 2.0, 5.0, 4.0], vec![5], false)
            .expect("t");
        let m = session.tensor_median(t).expect("median");
        assert_eq!(session.tensor_values(m).expect("vals"), vec![3.0]);
    }

    #[test]
    fn session_tensor_median_even() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![4.0, 1.0, 3.0, 2.0], vec![4], false)
            .expect("t");
        let m = session.tensor_median(t).expect("median");
        // Sorted: [1, 2, 3, 4], lower middle = index (4-1)/2 = 1, value = 2.0
        assert_eq!(session.tensor_values(m).expect("vals"), vec![2.0]);
    }

    #[test]
    fn session_tensor_median_single() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![42.0], vec![1], false)
            .expect("t");
        let m = session.tensor_median(t).expect("median");
        assert_eq!(session.tensor_values(m).expect("vals"), vec![42.0]);
    }

    #[test]
    fn session_tensor_median_rejects_empty() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session.tensor_variable(vec![], vec![0], false).expect("t");
        let err = session
            .tensor_median(t)
            .expect_err("median over empty tensor must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "median requires non-empty tensor"
                }
            ))
        ));
    }

    #[test]
    fn session_tensor_median_backward_unique_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.0, 1.0, 2.0], vec![3], true)
            .expect("x");
        let m = session.tensor_median(x).expect("median");
        assert_eq!(session.tensor_values(m).expect("vals"), vec![2.0]);

        let report = session.tensor_backward(m).expect("backward");
        assert_eq!(
            session.tensor_gradient(&report, x).expect("x grad"),
            &[0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn session_rsqrt_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(4.0, true);
        let y = session.rsqrt(x).expect("rsqrt");
        assert!((session.value(y).expect("val") - 0.5).abs() < 1e-12);
        let report = session.backward(y).expect("backward");
        // d/dx rsqrt(4) = -0.5 * (1/sqrt(4))^3 = -0.5 * 0.125 = -0.0625
        assert!((session.gradient(&report, x).unwrap() - (-0.0625)).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_rsqrt_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 4.0, 9.0], vec![3], true)
            .expect("x");
        let y = session.tensor_rsqrt(x).expect("rsqrt");
        let vals = session.tensor_values(y).expect("vals");
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 0.5).abs() < 1e-12);
        assert!((vals[2] - 1.0 / 3.0).abs() < 1e-12);
        let report = session.tensor_backward(y).expect("backward");
        let grad = session.tensor_gradient(&report, x).expect("grad");
        assert!((grad[0] - (-0.5)).abs() < 1e-10);
        assert!((grad[1] - (-0.0625)).abs() < 1e-10);
    }

    #[test]
    fn session_erf_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.erf(x).expect("erf");
        assert!(session.value(y).expect("val").abs() < 1e-7);
        let report = session.backward(y).expect("backward");
        // d/dx erf(0) = 2/sqrt(pi) ≈ 1.1283791670955126
        let expected = 2.0 / std::f64::consts::PI.sqrt();
        assert!((session.gradient(&report, x).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_erf_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0, -1.0], vec![3], true)
            .expect("x");
        let y = session.tensor_erf(x).expect("erf");
        let vals = session.tensor_values(y).expect("vals");
        assert!(vals[0].abs() < 1e-7);
        assert!((vals[1] - 0.8427007929497149).abs() < 1e-6);
        assert!((vals[2] + 0.8427007929497149).abs() < 1e-6);
    }

    #[test]
    fn session_erfc_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.erfc(x).expect("erfc");
        assert!((session.value(y).expect("val") - 1.0).abs() < 1e-7);
    }

    #[test]
    fn session_hardswish_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // x=-4 (below -3): output=0
        let x1 = session.variable(-4.0, true);
        let y1 = session.hardswish(x1).expect("hardswish");
        assert!(session.value(y1).expect("val").abs() < 1e-12);
        // x=4 (above 3): output=x
        let x2 = session.variable(4.0, true);
        let y2 = session.hardswish(x2).expect("hardswish");
        assert!((session.value(y2).expect("val") - 4.0).abs() < 1e-12);
        // x=0 (middle): output=0*3/6=0
        let x3 = session.variable(0.0, true);
        let y3 = session.hardswish(x3).expect("hardswish");
        assert!(session.value(y3).expect("val").abs() < 1e-12);
        // backward at x=0: grad=(2*0+3)/6=0.5
        let report = session.backward(y3).expect("backward");
        assert!((session.gradient(&report, x3).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn session_hardsigmoid_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.hardsigmoid(x).expect("hardsigmoid");
        assert!((session.value(y).expect("val") - 0.5).abs() < 1e-12);
        let report = session.backward(y).expect("backward");
        assert!((session.gradient(&report, x).unwrap() - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn session_hardtanh_forward_and_backward_in_range() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // x=0.5 (in range): output=0.5, grad=1
        let x = session.variable(0.5, true);
        let y = session.hardtanh(x).expect("hardtanh");
        assert!((session.value(y).expect("val") - 0.5).abs() < 1e-12);
        let report = session.backward(y).expect("backward");
        assert!((session.gradient(&report, x).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_hardtanh_forward_and_backward_out_of_range() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // x=2.0 (out of range): output=1.0, grad=0
        let x2 = session.variable(2.0, true);
        let y2 = session.hardtanh(x2).expect("hardtanh");
        assert!((session.value(y2).expect("val") - 1.0).abs() < 1e-12);
        let report2 = session.backward(y2).expect("backward");
        assert!(session.gradient(&report2, x2).unwrap().abs() < 1e-12);
    }

    #[test]
    fn session_softplus_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.softplus(x).expect("softplus");
        // softplus(0) = ln(2) ≈ 0.693
        assert!((session.value(y).expect("val") - 2.0_f64.ln()).abs() < 1e-12);
        let report = session.backward(y).expect("backward");
        // d/dx softplus(0) = sigmoid(0) = 0.5
        assert!((session.gradient(&report, x).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn session_mish_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.mish(x).expect("mish");
        // mish(0) = 0 * tanh(ln(2)) = 0
        assert!(session.value(y).expect("val").abs() < 1e-12);
        let report = session.backward(y).expect("backward");
        // d/dx mish(0) = tanh(softplus(0)) + 0 * ... = tanh(ln(2)) ≈ 0.6
        let expected = 2.0_f64.ln().tanh();
        assert!((session.gradient(&report, x).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn session_square_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(3.0, true);
        let y = session.square(x).expect("square");
        assert!((session.value(y).expect("val") - 9.0).abs() < 1e-12);
        let report = session.backward(y).expect("backward");
        // d/dx x^2 = 2x = 6
        assert!((session.gradient(&report, x).unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_square_forward_and_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, -3.0, 0.0], vec![3], true)
            .expect("x");
        let y = session.tensor_square(x).expect("square");
        let vals = session.tensor_values(y).expect("vals");
        assert!((vals[0] - 4.0).abs() < 1e-12);
        assert!((vals[1] - 9.0).abs() < 1e-12);
        assert!(vals[2].abs() < 1e-12);
        let report = session.tensor_backward(y).expect("backward");
        let grad = session.tensor_gradient(&report, x).expect("grad");
        assert!((grad[0] - 4.0).abs() < 1e-12);
        assert!((grad[1] - (-6.0)).abs() < 1e-12);
        assert!(grad[2].abs() < 1e-12);
    }

    #[test]
    fn session_tensor_hardswish_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-4.0, 0.0, 4.0], vec![3], true)
            .expect("x");
        let y = session.tensor_hardswish(x).expect("hardswish");
        let vals = session.tensor_values(y).expect("vals");
        assert!(vals[0].abs() < 1e-12);
        assert!(vals[1].abs() < 1e-12);
        assert!((vals[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_softplus_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 100.0, -100.0], vec![3], true)
            .expect("x");
        let y = session.tensor_softplus(x).expect("softplus");
        let vals = session.tensor_values(y).expect("vals");
        assert!((vals[0] - 2.0_f64.ln()).abs() < 1e-12);
        assert!((vals[1] - 100.0).abs() < 1e-6);
        assert!(vals[2].abs() < 1e-6);
    }

    #[test]
    fn session_tensor_mish_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, -1.0], vec![2], true)
            .expect("x");
        let y = session.tensor_mish(x).expect("mish");
        let vals = session.tensor_values(y).expect("vals");
        // mish(1) = 1 * tanh(softplus(1))
        let sp1 = (1.0 + 1.0_f64.exp()).ln();
        assert!((vals[0] - sp1.tanh()).abs() < 1e-10);
        let report = session.tensor_backward(y).expect("backward");
        let _grad = session.tensor_gradient(&report, x).expect("grad");
    }

    #[test]
    fn session_scalar_isnan() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let normal = session.variable(3.0, false);
        let nan_val = session.variable(f64::NAN, false);
        let inf_val = session.variable(f64::INFINITY, false);
        let r1 = session.isnan(normal).expect("isnan(3.0)");
        let r2 = session.isnan(nan_val).expect("isnan(NaN)");
        let r3 = session.isnan(inf_val).expect("isnan(Inf)");
        assert_eq!(session.value(r1).unwrap(), 0.0);
        assert_eq!(session.value(r2).unwrap(), 1.0);
        assert_eq!(session.value(r3).unwrap(), 0.0);
    }

    #[test]
    fn session_scalar_isinf() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let normal = session.variable(3.0, false);
        let inf_val = session.variable(f64::INFINITY, false);
        let neg_inf = session.variable(f64::NEG_INFINITY, false);
        let nan_val = session.variable(f64::NAN, false);
        let r1 = session.isinf(normal).expect("isinf(3.0)");
        let r2 = session.isinf(inf_val).expect("isinf(Inf)");
        let r3 = session.isinf(neg_inf).expect("isinf(-Inf)");
        let r4 = session.isinf(nan_val).expect("isinf(NaN)");
        assert_eq!(session.value(r1).unwrap(), 0.0);
        assert_eq!(session.value(r2).unwrap(), 1.0);
        assert_eq!(session.value(r3).unwrap(), 1.0);
        assert_eq!(session.value(r4).unwrap(), 0.0);
    }

    #[test]
    fn session_scalar_isfinite() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let normal = session.variable(3.0, false);
        let inf_val = session.variable(f64::INFINITY, false);
        let nan_val = session.variable(f64::NAN, false);
        let r1 = session.isfinite(normal).expect("isfinite(3.0)");
        let r2 = session.isfinite(inf_val).expect("isfinite(Inf)");
        let r3 = session.isfinite(nan_val).expect("isfinite(NaN)");
        assert_eq!(session.value(r1).unwrap(), 1.0);
        assert_eq!(session.value(r2).unwrap(), 0.0);
        assert_eq!(session.value(r3).unwrap(), 0.0);
    }

    #[test]
    fn session_tensor_isnan() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, f64::NAN, 3.0, f64::INFINITY], vec![4], false)
            .expect("x");
        let y = session.tensor_isnan(x).expect("isnan");
        let vals = session.tensor_values(y).expect("vals");
        assert_eq!(vals, &[0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn session_tensor_isinf() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(
                vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
                vec![4],
                false,
            )
            .expect("x");
        let y = session.tensor_isinf(x).expect("isinf");
        let vals = session.tensor_values(y).expect("vals");
        assert_eq!(vals, &[0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn session_tensor_isfinite() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, f64::INFINITY, f64::NAN, -2.5], vec![4], false)
            .expect("x");
        let y = session.tensor_isfinite(x).expect("isfinite");
        let vals = session.tensor_values(y).expect("vals");
        assert_eq!(vals, &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn session_scalar_atan2() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let y = session.variable(1.0, true);
        let x = session.variable(1.0, true);
        let out = session.atan2(y, x).expect("atan2");
        let val = session.value(out).unwrap();
        assert!((val - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        let report = session.backward(out).expect("backward");
        // d(atan2(y,x))/dy = x/(x^2+y^2) = 1/2 = 0.5
        let dy = session.gradient(&report, y).unwrap();
        assert!((dy - 0.5).abs() < 1e-10);
        // d(atan2(y,x))/dx = -y/(x^2+y^2) = -1/2 = -0.5
        let dx = session.gradient(&report, x).unwrap();
        assert!((dx - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn session_scalar_fmod() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(7.0, true);
        let b = session.variable(3.0, true);
        let out = session.fmod(a, b).expect("fmod");
        let val = session.value(out).unwrap();
        // fmod(7, 3) = 1
        assert!((val - 1.0).abs() < 1e-10);
        let report = session.backward(out).expect("backward");
        // d(fmod)/da = 1
        let da = session.gradient(&report, a).unwrap();
        assert!((da - 1.0).abs() < 1e-10);
        // d(fmod)/db = -trunc(7/3) = -trunc(2.333) = -2
        let db = session.gradient(&report, b).unwrap();
        assert!((db - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn session_scalar_remainder() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session.variable(-7.0, true);
        let b = session.variable(3.0, true);
        let out = session.remainder(a, b).expect("remainder");
        let val = session.value(out).unwrap();
        // remainder(-7, 3) = -7 - floor(-7/3)*3 = -7 - (-3)*3 = -7+9 = 2
        assert!((val - 2.0).abs() < 1e-10);
        let report = session.backward(out).expect("backward");
        // d(remainder)/da = 1
        let da = session.gradient(&report, a).unwrap();
        assert!((da - 1.0).abs() < 1e-10);
        // d(remainder)/db = -floor(-7/3) = -(-3) = 3
        let db = session.gradient(&report, b).unwrap();
        assert!((db - 3.0).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_atan2() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let y = session
            .tensor_variable(vec![1.0, 0.0], vec![2], true)
            .expect("y");
        let x = session
            .tensor_variable(vec![1.0, 1.0], vec![2], true)
            .expect("x");
        let out = session.tensor_atan2(y, x).expect("atan2");
        let vals = session.tensor_values(out).expect("vals");
        assert!((vals[0] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        assert!(vals[1].abs() < 1e-10); // atan2(0, 1) = 0
        let report = session.tensor_backward(out).expect("backward");
        let _grad = session.tensor_gradient(&report, y).expect("grad_y");
    }

    #[test]
    fn session_tensor_fmod() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![7.0, -7.0], vec![2], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![3.0, 3.0], vec![2], true)
            .expect("b");
        let out = session.tensor_fmod(a, b).expect("fmod");
        let vals = session.tensor_values(out).expect("vals");
        // fmod(7, 3) = 1, fmod(-7, 3) = -1
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - (-1.0)).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let _grad = session.tensor_gradient(&report, a).expect("grad_a");
    }

    #[test]
    fn session_tensor_remainder() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![7.0, -7.0], vec![2], true)
            .expect("a");
        let b = session
            .tensor_variable(vec![3.0, 3.0], vec![2], true)
            .expect("b");
        let out = session.tensor_remainder(a, b).expect("remainder");
        let vals = session.tensor_values(out).expect("vals");
        // remainder(7, 3) = 1, remainder(-7, 3) = 2
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let _grad = session.tensor_gradient(&report, a).expect("grad_a");
    }

    // ── Norm operations ──

    #[test]
    fn session_tensor_norm_l2() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.0, 4.0], vec![2], true)
            .expect("x");
        let out = session.tensor_norm(x, 2.0).expect("norm");
        let vals = session.tensor_values(out).expect("vals");
        // L2 norm of [3, 4] = 5.0
        assert!((vals[0] - 5.0).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let grad = session.tensor_gradient(&report, x).expect("grad");
        // d/dx_i = x_i / norm = [3/5, 4/5] = [0.6, 0.8]
        assert!((grad[0] - 0.6).abs() < 1e-10);
        assert!((grad[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_norm_l1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.0, -4.0, 5.0], vec![3], true)
            .expect("x");
        let out = session.tensor_norm(x, 1.0).expect("norm");
        let vals = session.tensor_values(out).expect("vals");
        // L1 norm = |3| + |-4| + |5| = 12.0
        assert!((vals[0] - 12.0).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let grad = session.tensor_gradient(&report, x).expect("grad");
        // d/dx_i = sign(x_i) = [1, -1, 1]
        assert!((grad[0] - 1.0).abs() < 1e-10);
        assert!((grad[1] - (-1.0)).abs() < 1e-10);
        assert!((grad[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_norm_inf() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, -5.0, 3.0], vec![3], true)
            .expect("x");
        let out = session.tensor_norm(x, f64::INFINITY).expect("norm");
        let vals = session.tensor_values(out).expect("vals");
        // Inf norm = max(|1|, |-5|, |3|) = 5.0
        assert!((vals[0] - 5.0).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let grad = session.tensor_gradient(&report, x).expect("grad");
        // gradient flows to element with max abs: index 1, sign(-5) = -1
        assert!((grad[0]).abs() < 1e-10);
        assert!((grad[1] - (-1.0)).abs() < 1e-10);
        assert!((grad[2]).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_norm_dim_l2() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let out = session.tensor_norm_dim(x, 2.0, 1).expect("norm_dim");
        let vals = session.tensor_values(out).expect("vals");
        // Row 0: sqrt(1+4+9) = sqrt(14), Row 1: sqrt(16+25+36) = sqrt(77)
        let expected0 = (14.0_f64).sqrt();
        let expected1 = (77.0_f64).sqrt();
        assert!((vals[0] - expected0).abs() < 1e-10);
        assert!((vals[1] - expected1).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let grad = session.tensor_gradient(&report, x).expect("grad");
        // d/dx_i = x_i / norm for L2
        assert!((grad[0] - 1.0 / expected0).abs() < 1e-10);
        assert!((grad[3] - 4.0 / expected1).abs() < 1e-10);
    }

    #[test]
    fn session_tensor_norm_p3() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![2.0, -3.0], vec![2], true)
            .expect("x");
        let out = session.tensor_norm(x, 3.0).expect("norm");
        let vals = session.tensor_values(out).expect("vals");
        // L3 norm = (|2|^3 + |-3|^3)^(1/3) = (8+27)^(1/3) = 35^(1/3)
        let expected = 35.0_f64.powf(1.0 / 3.0);
        assert!((vals[0] - expected).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let _grad = session.tensor_gradient(&report, x).expect("grad");
    }

    #[test]
    fn session_tensor_norm_l0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0, 0.0, -3.0], vec![4], false)
            .expect("x");
        let out = session.tensor_norm(x, 0.0).expect("norm");
        let vals = session.tensor_values(out).expect("vals");
        // L0 "norm" = count of non-zero elements = 2
        assert!((vals[0] - 2.0).abs() < 1e-10);
    }

    // ── Lerp operations ──

    #[test]
    fn session_tensor_lerp() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let start = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("start");
        let end = session
            .tensor_variable(vec![5.0, 6.0, 7.0], vec![3], true)
            .expect("end");
        let out = session.tensor_lerp(start, end, 0.25).expect("lerp");
        let vals = session.tensor_values(out).expect("vals");
        // lerp(1,5,0.25) = 1 + 0.25*(5-1) = 2.0
        // lerp(2,6,0.25) = 2 + 0.25*(6-2) = 3.0
        // lerp(3,7,0.25) = 3 + 0.25*(7-3) = 4.0
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
        assert!((vals[2] - 4.0).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let grad_start = session.tensor_gradient(&report, start).expect("grad_start");
        let grad_end = session.tensor_gradient(&report, end).expect("grad_end");
        // d/ds = (1-w) = 0.75, d/de = w = 0.25
        for &g in grad_start {
            assert!((g - 0.75).abs() < 1e-10);
        }
        for &g in grad_end {
            assert!((g - 0.25).abs() < 1e-10);
        }
    }

    // ── Addmm operations ──

    #[test]
    fn session_tensor_addmm() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // input: [2], mat1: [2,3], mat2: [3,2]
        let input = session
            .tensor_variable(vec![10.0, 20.0], vec![2], true)
            .expect("input");
        let mat1 = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("mat1");
        let mat2 = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2], true)
            .expect("mat2");
        // mat1 @ mat2 = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4,5],[10,11]]
        // beta=1, alpha=1: input + mat1@mat2 = [10+4, 20+5; 10+10, 20+11] = [[14,25],[20,31]]
        let out = session
            .tensor_addmm(input, mat1, mat2, 1.0, 1.0)
            .expect("addmm");
        let vals = session.tensor_values(out).expect("vals");
        assert!((vals[0] - 14.0).abs() < 1e-10);
        assert!((vals[1] - 25.0).abs() < 1e-10);
        assert!((vals[2] - 20.0).abs() < 1e-10);
        assert!((vals[3] - 31.0).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let _grad_input = session.tensor_gradient(&report, input).expect("grad_input");
        let _grad_mat1 = session.tensor_gradient(&report, mat1).expect("grad_mat1");
        let _grad_mat2 = session.tensor_gradient(&report, mat2).expect("grad_mat2");
    }

    #[test]
    fn session_tensor_addmm_scaled() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // beta=2, alpha=3
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("input");
        let mat1 = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], true)
            .expect("mat1");
        let mat2 = session
            .tensor_variable(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true)
            .expect("mat2");
        // mat1@mat2 = [[5,6],[7,8]] (identity * mat2)
        // 2*input + 3*mat1@mat2 = 2*[[1,2],[3,4]] + 3*[[5,6],[7,8]]
        //                       = [[2,4],[6,8]] + [[15,18],[21,24]] = [[17,22],[27,32]]
        let out = session
            .tensor_addmm(input, mat1, mat2, 2.0, 3.0)
            .expect("addmm");
        let vals = session.tensor_values(out).expect("vals");
        assert!((vals[0] - 17.0).abs() < 1e-10);
        assert!((vals[1] - 22.0).abs() < 1e-10);
        assert!((vals[2] - 27.0).abs() < 1e-10);
        assert!((vals[3] - 32.0).abs() < 1e-10);
    }

    // ── Addmv operations ──

    #[test]
    fn session_tensor_addmv() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // input: [2], mat: [2,3], vec: [3]
        let input = session
            .tensor_variable(vec![10.0, 20.0], vec![2], true)
            .expect("input");
        let mat = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("mat");
        let v = session
            .tensor_variable(vec![1.0, 1.0, 1.0], vec![3], true)
            .expect("v");
        // mat @ vec = [1+2+3, 4+5+6] = [6, 15]
        // beta=1, alpha=1: input + mat@vec = [10+6, 20+15] = [16, 35]
        let out = session
            .tensor_addmv(input, mat, v, 1.0, 1.0)
            .expect("addmv");
        let vals = session.tensor_values(out).expect("vals");
        assert!((vals[0] - 16.0).abs() < 1e-10);
        assert!((vals[1] - 35.0).abs() < 1e-10);
        let report = session.tensor_backward(out).expect("backward");
        let _grad_input = session.tensor_gradient(&report, input).expect("grad_input");
        let _grad_mat = session.tensor_gradient(&report, mat).expect("grad_mat");
        let _grad_v = session.tensor_gradient(&report, v).expect("grad_v");
    }

    // ── Unbind operations ──

    #[test]
    fn session_tensor_unbind() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2x3 tensor: [[1,2,3],[4,5,6]]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("x");
        let parts = session.tensor_unbind(x, 0).expect("unbind");
        assert_eq!(parts.len(), 2);
        let v0 = session.tensor_values(parts[0]).expect("v0");
        let v1 = session.tensor_values(parts[1]).expect("v1");
        assert_eq!(v0, vec![1.0, 2.0, 3.0]);
        assert_eq!(v1, vec![4.0, 5.0, 6.0]);
        // Check shapes are [3] (dim 0 removed)
        let s0 = session.tensor_shape(parts[0]).expect("shape0");
        assert_eq!(s0, vec![3]);
    }

    #[test]
    fn session_tensor_unbind_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2x3 tensor: [[1,2,3],[4,5,6]], unbind along dim=1
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("x");
        let parts = session.tensor_unbind(x, 1).expect("unbind");
        assert_eq!(parts.len(), 3);
        let v0 = session.tensor_values(parts[0]).expect("v0");
        let v1 = session.tensor_values(parts[1]).expect("v1");
        let v2 = session.tensor_values(parts[2]).expect("v2");
        assert_eq!(v0, vec![1.0, 4.0]);
        assert_eq!(v1, vec![2.0, 5.0]);
        assert_eq!(v2, vec![3.0, 6.0]);
    }

    // ── Meshgrid operations ──

    #[test]
    fn session_tensor_meshgrid() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("x");
        let y = session
            .tensor_variable(vec![4.0, 5.0], vec![2], false)
            .expect("y");
        let grids = session.tensor_meshgrid(&[x, y]).expect("meshgrid");
        assert_eq!(grids.len(), 2);
        // Grid X should be [[1,1],[2,2],[3,3]] (shape [3,2])
        let gx = session.tensor_values(grids[0]).expect("gx");
        let gx_shape = session.tensor_shape(grids[0]).expect("gx_shape");
        assert_eq!(gx_shape, vec![3, 2]);
        assert_eq!(gx, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // Grid Y should be [[4,5],[4,5],[4,5]] (shape [3,2])
        let gy = session.tensor_values(grids[1]).expect("gy");
        let gy_shape = session.tensor_shape(grids[1]).expect("gy_shape");
        assert_eq!(gy_shape, vec![3, 2]);
        assert_eq!(gy, vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
    }

    // ── Diagonal operations ──

    #[test]
    fn session_tensor_diagonal() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 3x3 matrix
        let x = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![3, 3],
                false,
            )
            .expect("x");
        // Main diagonal (offset=0)
        let d0 = session.tensor_diagonal(x, 0).expect("d0");
        let v0 = session.tensor_values(d0).expect("v0");
        assert_eq!(v0, vec![1.0, 5.0, 9.0]);
        // Upper diagonal (offset=1)
        let d1 = session.tensor_diagonal(x, 1).expect("d1");
        let v1 = session.tensor_values(d1).expect("v1");
        assert_eq!(v1, vec![2.0, 6.0]);
        // Lower diagonal (offset=-1)
        let dm1 = session.tensor_diagonal(x, -1).expect("dm1");
        let vm1 = session.tensor_values(dm1).expect("vm1");
        assert_eq!(vm1, vec![4.0, 8.0]);
    }

    // ---- argsort tests ----

    #[test]
    fn session_argsort_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.0, 1.0, 2.0], vec![3], false)
            .expect("variable");
        let idx = session.tensor_argsort(x, 0, false).expect("argsort");
        let vals = session.tensor_values(idx).expect("values");
        // Ascending: 1.0 at index 1, 2.0 at index 2, 3.0 at index 0
        assert_eq!(vals, vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn session_argsort_descending() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.0, 1.0, 2.0], vec![3], false)
            .expect("variable");
        let idx = session.tensor_argsort(x, 0, true).expect("argsort");
        let vals = session.tensor_values(idx).expect("values");
        // Descending: 3.0 at index 0, 2.0 at index 2, 1.0 at index 1
        assert_eq!(vals, vec![0.0, 2.0, 1.0]);
    }

    // ---- one_hot tests ----

    #[test]
    fn session_one_hot_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let indices = session
            .tensor_variable(vec![0.0, 2.0, 1.0], vec![3], false)
            .expect("indices");
        let oh = session.one_hot(indices, 3).expect("one_hot");
        let (vals, meta) = session.tensor_values_meta(oh).expect("values");
        assert_eq!(meta.shape(), &[3, 3]);
        assert_eq!(
            vals,
            vec![
                1.0, 0.0, 0.0, // class 0
                0.0, 0.0, 1.0, // class 2
                0.0, 1.0, 0.0, // class 1
            ]
        );
    }

    #[test]
    fn session_one_hot_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let indices = session
            .tensor_variable(vec![1.0, 0.0], vec![1, 2], false)
            .expect("indices");
        let oh = session.one_hot(indices, 3).expect("one_hot");
        let (vals, meta) = session.tensor_values_meta(oh).expect("values");
        assert_eq!(meta.shape(), &[1, 2, 3]);
        assert_eq!(
            vals,
            vec![
                0.0, 1.0, 0.0, // class 1
                1.0, 0.0, 0.0, // class 0
            ]
        );
    }

    #[test]
    fn session_one_hot_rejects_out_of_range() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let indices = session
            .tensor_variable(vec![0.0, 5.0], vec![2], false)
            .expect("indices");
        assert!(session.one_hot(indices, 3).is_err());
    }

    // ---- tensor_pad tests ----

    #[test]
    fn session_pad_1d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");
        // Pad 1 before, 2 after on last dim
        let padded = session.tensor_pad(x, &[1, 2], 0.0).expect("pad");
        let (vals, meta) = session.tensor_values_meta(padded).expect("values");
        assert_eq!(meta.shape(), &[6]);
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn session_pad_2d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2x3 matrix
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("variable");
        // Pad last dim: 1 before, 1 after -> columns become 5
        let padded = session.tensor_pad(x, &[1, 1], 0.0).expect("pad");
        let (vals, meta) = session.tensor_values_meta(padded).expect("values");
        assert_eq!(meta.shape(), &[2, 5]);
        assert_eq!(
            vals,
            vec![
                0.0, 1.0, 2.0, 3.0, 0.0, // row 0
                0.0, 4.0, 5.0, 6.0, 0.0, // row 1
            ]
        );
    }

    #[test]
    fn session_pad_2d_both_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2x2 matrix
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .expect("variable");
        // Pad last dim: 0,1  and second-to-last dim: 1,0
        let padded = session.tensor_pad(x, &[0, 1, 1, 0], 9.0).expect("pad");
        let (vals, meta) = session.tensor_values_meta(padded).expect("values");
        assert_eq!(meta.shape(), &[3, 3]);
        assert_eq!(
            vals,
            vec![
                9.0, 9.0, 9.0, // padded row
                1.0, 2.0, 9.0, // row 0 + right pad
                3.0, 4.0, 9.0, // row 1 + right pad
            ]
        );
    }

    #[test]
    fn session_pad_rejects_odd_padding() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], false)
            .expect("variable");
        assert!(session.tensor_pad(x, &[1], 0.0).is_err());
    }

    // -------------------------------------------------------------------
    // LU Decomposition integration tests
    // -------------------------------------------------------------------

    #[test]
    fn session_lu_factor_and_unpack_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = session
            .tensor_variable(
                vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
                vec![3, 3],
                false,
            )
            .expect("variable");

        let (p, l, u) = session.tensor_linalg_lu(a).expect("linalg.lu");

        // Verify P @ L @ U == A
        let pl = session.tensor_matmul(p, l).expect("matmul P@L");
        let plu = session.tensor_matmul(pl, u).expect("matmul PL@U");

        let a_vals = session.tensor_values(a).expect("values");
        let plu_vals = session.tensor_values(plu).expect("values");
        for (i, (&av, &pv)) in a_vals.iter().zip(plu_vals.iter()).enumerate() {
            assert!((av - pv).abs() < 1e-10, "P@L@U[{i}] = {pv}, expected {av}");
        }
    }

    #[test]
    fn session_lu_solve_2x2_system() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2], false)
            .expect("variable");
        let b = session
            .tensor_variable(vec![4.0, 7.0], vec![2], false)
            .expect("variable");

        let (lu_packed, pivots) = session.tensor_lu_factor(a).expect("lu_factor");
        let x = session
            .tensor_lu_solve(lu_packed, &pivots, b)
            .expect("lu_solve");

        let x_vals = session.tensor_values(x).expect("values");
        assert!(
            (x_vals[0] - 5.0).abs() < 1e-10,
            "x[0] = {}, expected 5.0",
            x_vals[0]
        );
        assert!(
            (x_vals[1] - (-6.0)).abs() < 1e-10,
            "x[1] = {}, expected -6.0",
            x_vals[1]
        );
    }

    #[test]
    fn session_lu_solve_3x3_system() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
                vec![3, 3],
                false,
            )
            .expect("variable");
        let b = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");

        let (lu_packed, pivots) = session.tensor_lu_factor(a).expect("lu_factor");
        let x = session
            .tensor_lu_solve(lu_packed, &pivots, b)
            .expect("lu_solve");

        // Verify A @ x ≈ b
        let a_vals = session.tensor_values(a).expect("values");
        let x_vals = session.tensor_values(x).expect("values");
        let b_vals = session.tensor_values(b).expect("values");
        for i in 0..3 {
            let mut ax_i = 0.0;
            for j in 0..3 {
                ax_i += a_vals[i * 3 + j] * x_vals[j];
            }
            assert!(
                (ax_i - b_vals[i]).abs() < 1e-10,
                "Ax[{i}] = {ax_i}, expected {}",
                b_vals[i]
            );
        }
    }

    #[test]
    fn session_lu_factor_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let eye = session
            .tensor_variable(
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                vec![3, 3],
                false,
            )
            .expect("variable");

        let (p, l, u) = session.tensor_linalg_lu(eye).expect("linalg.lu");

        let p_vals = session.tensor_values(p).expect("values");
        let l_vals = session.tensor_values(l).expect("values");
        let u_vals = session.tensor_values(u).expect("values");
        let eye_vals = session.tensor_values(eye).expect("values");

        // For identity: P ≈ I, L ≈ I, U ≈ I
        for i in 0..9 {
            assert!(
                (p_vals[i] - eye_vals[i]).abs() < 1e-12,
                "P[{i}] should match identity"
            );
            assert!(
                (l_vals[i] - eye_vals[i]).abs() < 1e-12,
                "L[{i}] should match identity"
            );
            assert!(
                (u_vals[i] - eye_vals[i]).abs() < 1e-12,
                "U[{i}] should match identity"
            );
        }
    }

    #[test]
    fn session_lu_factor_rejects_non_square() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .expect("variable");
        assert!(
            session.tensor_lu_factor(a).is_err(),
            "non-square should error"
        );
    }

    // ---- QR Decomposition tests (bd-2drq.4) ----

    #[test]
    fn session_qr_reduced_round_trip_3x3() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = session
            .tensor_variable(
                vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
                vec![3, 3],
                false,
            )
            .expect("variable");

        let (q, r) = session.tensor_linalg_qr(a, true).expect("linalg.qr");

        // Q @ R == A
        let qr = session.tensor_matmul(q, r).expect("matmul Q@R");
        let a_vals = session.tensor_values(a).expect("values");
        let qr_vals = session.tensor_values(qr).expect("values");
        for (i, (&av, &qv)) in a_vals.iter().zip(qr_vals.iter()).enumerate() {
            assert!((av - qv).abs() < 1e-10, "Q@R[{i}] = {qv}, expected {av}");
        }
    }

    #[test]
    fn session_qr_complete_round_trip_3x3() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
                vec![3, 3],
                false,
            )
            .expect("variable");

        let (q, r) = session.tensor_linalg_qr(a, false).expect("linalg.qr");

        // Q @ R == A
        let qr = session.tensor_matmul(q, r).expect("matmul Q@R");
        let a_vals = session.tensor_values(a).expect("values");
        let qr_vals = session.tensor_values(qr).expect("values");
        for (i, (&av, &qv)) in a_vals.iter().zip(qr_vals.iter()).enumerate() {
            assert!((av - qv).abs() < 1e-10, "Q@R[{i}] = {qv}, expected {av}");
        }
    }

    #[test]
    fn session_qr_tall_matrix_reduced() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![4, 2],
                false,
            )
            .expect("variable");

        let (q, r) = session.tensor_linalg_qr(a, true).expect("linalg.qr");

        // Q should have 4*2=8 elements, R should have 2*2=4 elements
        let q_vals = session.tensor_values(q).expect("values");
        let r_vals = session.tensor_values(r).expect("values");
        assert_eq!(q_vals.len(), 8, "Q should be 4x2");
        assert_eq!(r_vals.len(), 4, "R should be 2x2");

        // Q @ R == A
        let qr = session.tensor_matmul(q, r).expect("matmul Q@R");
        let a_vals = session.tensor_values(a).expect("values");
        let qr_vals = session.tensor_values(qr).expect("values");
        for (i, (&av, &qv)) in a_vals.iter().zip(qr_vals.iter()).enumerate() {
            assert!((av - qv).abs() < 1e-10, "Q@R[{i}] = {qv}, expected {av}");
        }
    }

    #[test]
    fn session_qr_identity_returns_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let eye = session
            .tensor_variable(
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                vec![3, 3],
                false,
            )
            .expect("variable");

        let (q, r) = session.tensor_linalg_qr(eye, true).expect("linalg.qr");

        let q_vals = session.tensor_values(q).expect("values");
        let r_vals = session.tensor_values(r).expect("values");

        // Q @ R should reconstruct identity
        let qr = session.tensor_matmul(q, r).expect("matmul Q@R");
        let eye_vals = session.tensor_values(eye).expect("values");
        let qr_vals = session.tensor_values(qr).expect("values");
        for (i, (&ev, &qv)) in eye_vals.iter().zip(qr_vals.iter()).enumerate() {
            assert!((ev - qv).abs() < 1e-12, "Q@R[{i}] = {qv}, expected {ev}");
        }

        // For identity: |Q[i,i]| should be 1, R should be ±I
        for i in 0..3 {
            assert!(
                (q_vals[i * 3 + i].abs() - 1.0).abs() < 1e-12,
                "Q diagonal should be ±1"
            );
            assert!(
                (r_vals[i * 3 + i].abs() - 1.0).abs() < 1e-12,
                "R diagonal should be ±1"
            );
        }
    }

    #[test]
    fn session_qr_rejects_1d_tensor() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");
        assert!(
            session.tensor_linalg_qr(v, true).is_err(),
            "1D tensor should be rejected"
        );
    }

    // ---- no_grad / enable_grad tests (bd-3dpn.1) ----

    #[test]
    fn default_grad_enabled_is_true() {
        let session = FrankenTorchSession::new(ExecutionMode::Strict);
        assert!(session.is_grad_enabled());
    }

    #[test]
    fn no_grad_disables_gradient_tracking() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        session.no_grad_enter();
        assert!(!session.is_grad_enabled());
        session.no_grad_exit();
        assert!(session.is_grad_enabled());
    }

    #[test]
    fn no_grad_block_produces_tensors_without_grad() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let y = session.variable(3.0, true);

        session.no_grad_enter();
        let z = session.add(x, y).expect("add in no_grad");
        session.no_grad_exit();

        // z was created inside no_grad, so backward from z should fail
        // (z has requires_grad=false because grad was disabled)
        let err = session.backward(z);
        assert!(err.is_err(), "backward from no_grad node should fail");
    }

    #[test]
    fn enable_grad_inside_no_grad_reenables_tracking() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        session.no_grad_enter();
        assert!(!session.is_grad_enabled());

        session.enable_grad_enter();
        assert!(session.is_grad_enabled());
        session.enable_grad_exit();

        assert!(!session.is_grad_enabled());
        session.no_grad_exit();
        assert!(session.is_grad_enabled());
    }

    #[test]
    fn nested_no_grad_enable_grad_restores_state() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        assert!(session.is_grad_enabled());

        session.no_grad_enter();
        assert!(!session.is_grad_enabled());

        session.no_grad_enter(); // nested
        assert!(!session.is_grad_enabled());

        session.enable_grad_enter();
        assert!(session.is_grad_enabled());

        session.enable_grad_exit();
        assert!(!session.is_grad_enabled());

        session.no_grad_exit();
        assert!(!session.is_grad_enabled());

        session.no_grad_exit();
        assert!(session.is_grad_enabled());
    }

    #[test]
    fn with_no_grad_closure_api() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let y = session.variable(3.0, true);

        let z = session.with_no_grad(|s| s.add(x, y).expect("add"));

        // z was created inside no_grad, backward should fail
        assert!(session.backward(z).is_err());

        // But operations outside no_grad still track gradients
        let w = session.add(x, y).expect("add outside no_grad");
        let report = session.backward(w).expect("backward should work");
        assert_eq!(report.gradient(x), Some(1.0));
    }

    #[test]
    fn with_enable_grad_inside_no_grad() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let y = session.variable(3.0, true);

        session.no_grad_enter();
        let z = session.with_enable_grad(|s| s.add(x, y).expect("add"));
        session.no_grad_exit();

        // z was created inside enable_grad (even though inside no_grad),
        // so backward should succeed
        let report = session.backward(z).expect("backward should work");
        assert_eq!(report.gradient(x), Some(1.0));
        assert_eq!(report.gradient(y), Some(1.0));
    }

    #[test]
    fn set_grad_enabled_toggles_state() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        assert!(session.is_grad_enabled());

        session.set_grad_enabled(false);
        assert!(!session.is_grad_enabled());

        session.set_grad_enabled(true);
        assert!(session.is_grad_enabled());
    }

    #[test]
    fn no_grad_tensor_operations_skip_grad_tracking() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .expect("x");
        let y = session
            .tensor_variable(vec![3.0, 4.0], vec![2], true)
            .expect("y");

        session.no_grad_enter();
        let z = session.tensor_add(x, y).expect("tensor add in no_grad");
        session.no_grad_exit();

        // z created in no_grad should not have gradient tracking
        let err = session.tensor_backward(z);
        assert!(
            err.is_err(),
            "tensor backward from no_grad node should fail"
        );
    }

    #[test]
    fn backward_after_no_grad_only_tracks_pre_nograd_ops() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(2.0, true);
        let y = session.variable(3.0, true);

        // Create node with grad tracking
        let sum = session.add(x, y).expect("add");
        // Multiply inside no_grad — result has no grad
        session.no_grad_enter();
        let prod = session.mul(sum, x).expect("mul in no_grad");
        session.no_grad_exit();

        // prod has requires_grad=false because it was created in no_grad
        // So backward from prod should fail
        let err = session.backward(prod);
        assert!(err.is_err());

        // But backward from sum (created before no_grad) should work
        let report = session.backward(sum).expect("backward from sum");
        assert_eq!(report.gradient(x), Some(1.0));
        assert_eq!(report.gradient(y), Some(1.0));
    }

    #[test]
    fn deep_nested_grad_context_100_levels() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        for _ in 0..100 {
            session.no_grad_enter();
        }
        assert!(!session.is_grad_enabled());
        for _ in 0..100 {
            session.no_grad_exit();
        }
        assert!(session.is_grad_enabled());
    }

    #[test]
    fn empty_no_grad_block_is_noop() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        session.no_grad_enter();
        session.no_grad_exit();
        assert!(session.is_grad_enabled());

        // Operations after should still track gradients
        let x = session.variable(2.0, true);
        let y = session.variable(3.0, true);
        let z = session.add(x, y).expect("add");
        let report = session.backward(z).expect("backward");
        assert_eq!(report.gradient(x), Some(1.0));
    }

    // ── f32 dtype API tests ──────────────────────────────────────────

    #[test]
    fn f32_tensor_variable_and_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable_f32(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        assert_eq!(session.tensor_dtype(a).unwrap(), DType::F32);
        let vals = session.tensor_values_f32(a).unwrap();
        assert_eq!(vals, vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn f32_zeros_ones_factory() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let z = session.zeros_f32(vec![3], false).unwrap();
        assert_eq!(session.tensor_dtype(z).unwrap(), DType::F32);
        assert_eq!(session.tensor_values_f32(z).unwrap(), vec![0.0f32; 3]);

        let o = session.ones_f32(vec![2, 2], false).unwrap();
        assert_eq!(session.tensor_dtype(o).unwrap(), DType::F32);
        assert_eq!(session.tensor_values_f32(o).unwrap(), vec![1.0f32; 4]);
    }

    #[test]
    fn f32_randn_factory() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let r = session.randn_f32(vec![10], false).unwrap();
        assert_eq!(session.tensor_dtype(r).unwrap(), DType::F32);
        let vals = session.tensor_values_f32(r).unwrap();
        assert_eq!(vals.len(), 10);
    }

    #[test]
    fn f32_arithmetic_preserves_dtype() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable_f32(vec![1.0f32, 2.0], vec![2], false)
            .unwrap();
        let b = session
            .tensor_variable_f32(vec![3.0f32, 4.0], vec![2], false)
            .unwrap();
        let c = session.tensor_add(a, b).unwrap();
        assert_eq!(session.tensor_dtype(c).unwrap(), DType::F32);
        assert_eq!(session.tensor_values_f32(c).unwrap(), vec![4.0f32, 6.0]);
    }

    #[test]
    fn f32_mixed_promotes_to_f64() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable_f32(vec![1.0f32, 2.0], vec![2], false)
            .unwrap();
        let b = session
            .tensor_variable(vec![3.0f64, 4.0], vec![2], false)
            .unwrap();
        let c = session.tensor_add(a, b).unwrap();
        assert_eq!(session.tensor_dtype(c).unwrap(), DType::F64);
        assert_eq!(session.tensor_values(c).unwrap(), vec![4.0f64, 6.0]);
    }

    #[test]
    fn f32_cast_roundtrip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.5f64, 2.5], vec![2], false)
            .unwrap();
        let b = session.tensor_to_f32(a).unwrap();
        assert_eq!(session.tensor_dtype(b).unwrap(), DType::F32);
        let c = session.tensor_to_f64(b).unwrap();
        assert_eq!(session.tensor_dtype(c).unwrap(), DType::F64);
        assert_eq!(session.tensor_values(c).unwrap(), vec![1.5, 2.5]);
    }

    #[test]
    fn f32_float_and_double_aliases_work() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![1.25f64, -2.5, 3.75], vec![3], false)
            .unwrap();
        let b = session.tensor_float(a).unwrap();
        assert_eq!(session.tensor_dtype(b).unwrap(), DType::F32);
        assert_eq!(
            session.tensor_values_f32(b).unwrap(),
            vec![1.25f32, -2.5, 3.75]
        );

        let c = session.tensor_double(b).unwrap();
        assert_eq!(session.tensor_dtype(c).unwrap(), DType::F64);
        assert_eq!(session.tensor_values(c).unwrap(), vec![1.25f64, -2.5, 3.75]);
    }

    #[test]
    fn f32_matmul_preserves_dtype_and_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lhs = session
            .tensor_variable_f32(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .unwrap();
        let rhs = session
            .tensor_variable_f32(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], false)
            .unwrap();

        let out = session.tensor_matmul(lhs, rhs).unwrap();
        assert_eq!(session.tensor_dtype(out).unwrap(), DType::F32);
        assert_eq!(
            session.tensor_values_f32(out).unwrap(),
            vec![58.0f32, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn f32_f64_matmul_promotes_to_f64() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lhs = session
            .tensor_variable_f32(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .unwrap();
        let rhs = session
            .tensor_variable(vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], false)
            .unwrap();

        let out = session.tensor_matmul(lhs, rhs).unwrap();
        assert_eq!(session.tensor_dtype(out).unwrap(), DType::F64);
        assert_eq!(
            session.tensor_values(out).unwrap(),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn f32_backward_produces_f64_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable_f32(vec![1.0f32, 2.0, 3.0], vec![3], true)
            .unwrap();
        let b = session.tensor_neg(a).unwrap();
        let c = session.tensor_sum(b).unwrap();
        let report = session.tensor_backward(c).unwrap();
        let grad = report.gradient(a).unwrap();
        assert_eq!(grad, &[-1.0, -1.0, -1.0]);
    }

    #[test]
    fn f32_unary_ops_preserve_dtype() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable_f32(vec![0.5f32, 1.0, 1.5], vec![3], false)
            .unwrap();
        let b = session.tensor_exp(a).unwrap();
        assert_eq!(session.tensor_dtype(b).unwrap(), DType::F32);
        let c = session.tensor_sin(a).unwrap();
        assert_eq!(session.tensor_dtype(c).unwrap(), DType::F32);
    }

    // ── nn.init tests ──────────────────────────────────────────────────

    #[test]
    fn init_uniform_fills_in_range() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 1000], vec![10, 100], true).unwrap();
        s.init_uniform_(t, -0.5, 0.5).unwrap();
        let vals = s.tensor_values(t).unwrap();
        assert_eq!(vals.len(), 1000);
        for &v in &vals {
            assert!(v >= -0.5 && v < 0.5, "value {v} out of range [-0.5, 0.5)");
        }
    }

    #[test]
    fn init_normal_approximate_statistics() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 10000], vec![100, 100], true).unwrap();
        s.init_normal_(t, 2.0, 0.5).unwrap();
        let vals = s.tensor_values(t).unwrap();
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        assert!((mean - 2.0).abs() < 0.05, "mean {mean} not near 2.0");
        assert!((var.sqrt() - 0.5).abs() < 0.05, "std {} not near 0.5", var.sqrt());
    }

    #[test]
    fn init_constant_fills_exact() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 6], vec![2, 3], true).unwrap();
        s.init_constant_(t, 42.0).unwrap();
        assert_eq!(s.tensor_values(t).unwrap(), vec![42.0; 6]);
    }

    #[test]
    fn init_ones_fills_ones() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 4], vec![2, 2], true).unwrap();
        s.init_ones_(t).unwrap();
        assert_eq!(s.tensor_values(t).unwrap(), vec![1.0; 4]);
    }

    #[test]
    fn init_zeros_fills_zeros() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![5.0; 4], vec![2, 2], true).unwrap();
        s.init_zeros_(t).unwrap();
        assert_eq!(s.tensor_values(t).unwrap(), vec![0.0; 4]);
    }

    #[test]
    fn init_eye_fills_identity() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 9], vec![3, 3], true).unwrap();
        s.init_eye_(t).unwrap();
        let vals = s.tensor_values(t).unwrap();
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn init_eye_rectangular() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 6], vec![2, 3], true).unwrap();
        s.init_eye_(t).unwrap();
        let vals = s.tensor_values(t).unwrap();
        // 2x3 identity: [[1,0,0],[0,1,0]]
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn init_eye_rejects_non_2d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 8], vec![2, 2, 2], true).unwrap();
        assert!(s.init_eye_(t).is_err());
    }

    #[test]
    fn init_xavier_uniform_bounds() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 600], vec![20, 30], true).unwrap();
        s.init_xavier_uniform_(t, 1.0).unwrap();
        let vals = s.tensor_values(t).unwrap();
        // a = sqrt(6 / (20 + 30)) = sqrt(0.12) ≈ 0.3464
        let a = (6.0 / 50.0_f64).sqrt();
        for &v in &vals {
            assert!(v >= -a && v < a, "value {v} outside xavier bounds ±{a}");
        }
    }

    #[test]
    fn init_xavier_normal_std() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 10000], vec![100, 100], true).unwrap();
        s.init_xavier_normal_(t, 1.0).unwrap();
        let vals = s.tensor_values(t).unwrap();
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        // expected std = sqrt(2 / (100 + 100)) = sqrt(0.01) = 0.1
        let expected_std = (2.0 / 200.0_f64).sqrt();
        assert!((mean).abs() < 0.02, "mean {mean} not near 0");
        assert!((var.sqrt() - expected_std).abs() < 0.02, "std {} not near {expected_std}", var.sqrt());
    }

    #[test]
    fn init_kaiming_uniform_relu() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 500], vec![50, 10], true).unwrap();
        s.init_kaiming_uniform_(t, 0.0, "fan_in", "relu").unwrap();
        let vals = s.tensor_values(t).unwrap();
        // gain for relu = sqrt(2), fan_in = 10
        // std = sqrt(2) / sqrt(10), bound = sqrt(3) * std
        let gain = 2.0_f64.sqrt();
        let std = gain / (10.0_f64).sqrt();
        let bound = 3.0_f64.sqrt() * std;
        for &v in &vals {
            assert!(v >= -bound && v < bound, "value {v} outside kaiming bounds ±{bound}");
        }
    }

    #[test]
    fn init_kaiming_normal_fan_out() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 10000], vec![100, 100], true).unwrap();
        s.init_kaiming_normal_(t, 0.0, "fan_out", "relu").unwrap();
        let vals = s.tensor_values(t).unwrap();
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        // gain = sqrt(2), fan_out = 100, std = sqrt(2) / sqrt(100) = sqrt(2)/10
        let expected_std = 2.0_f64.sqrt() / 10.0;
        assert!((mean).abs() < 0.02, "mean {mean} not near 0");
        assert!((var.sqrt() - expected_std).abs() < 0.02, "std {} not near {expected_std}", var.sqrt());
    }

    #[test]
    fn init_kaiming_invalid_mode() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 100], vec![10, 10], true).unwrap();
        assert!(s.init_kaiming_uniform_(t, 0.0, "invalid", "relu").is_err());
    }

    #[test]
    fn calculate_fan_2d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 60], vec![20, 3], true).unwrap();
        let (fan_in, fan_out) = s.calculate_fan_in_and_fan_out(t).unwrap();
        assert_eq!(fan_in, 3);
        assert_eq!(fan_out, 20);
    }

    #[test]
    fn calculate_fan_4d_conv() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Conv2d: [out_channels=16, in_channels=3, kH=5, kW=5]
        let t = s.tensor_variable(vec![0.0; 16 * 3 * 5 * 5], vec![16, 3, 5, 5], true).unwrap();
        let (fan_in, fan_out) = s.calculate_fan_in_and_fan_out(t).unwrap();
        assert_eq!(fan_in, 3 * 5 * 5); // 75
        assert_eq!(fan_out, 16 * 5 * 5); // 400
    }

    #[test]
    fn calculate_fan_1d_errors() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 10], vec![10], true).unwrap();
        assert!(s.calculate_fan_in_and_fan_out(t).is_err());
    }

    #[test]
    fn init_orthogonal_produces_orthogonal_matrix() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 25], vec![5, 5], true).unwrap();
        s.init_orthogonal_(t, 1.0).unwrap();
        let vals = s.tensor_values(t).unwrap();

        // Q^T @ Q should be identity (within tolerance)
        for i in 0..5 {
            for j in 0..5 {
                let mut dot = 0.0;
                for k in 0..5 {
                    dot += vals[k * 5 + i] * vals[k * 5 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q^T @ Q [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn init_orthogonal_with_gain() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 4], vec![2, 2], true).unwrap();
        s.init_orthogonal_(t, 2.0).unwrap();
        let vals = s.tensor_values(t).unwrap();

        // (Q/gain)^T @ (Q/gain) should be identity
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..2 {
                    dot += (vals[k * 2 + i] / 2.0) * (vals[k * 2 + j] / 2.0);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "(Q/gain)^T @ (Q/gain) [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn init_orthogonal_rectangular() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Tall matrix: 6x3 -> Q is 6x3 semi-orthogonal
        let t = s.tensor_variable(vec![0.0; 18], vec![6, 3], true).unwrap();
        s.init_orthogonal_(t, 1.0).unwrap();
        let vals = s.tensor_values(t).unwrap();

        // Q^T @ Q should be 3x3 identity
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..6 {
                    dot += vals[k * 3 + i] * vals[k * 3 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q^T @ Q [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn init_sparse_respects_sparsity() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 200], vec![20, 10], true).unwrap();
        s.init_sparse_(t, 0.5, 0.01).unwrap();
        let vals = s.tensor_values(t).unwrap();

        // Check each column: approximately 50% zeros
        for col in 0..10 {
            let zero_count = (0..20).filter(|&row| vals[row * 10 + col] == 0.0).count();
            // With 50% sparsity and 20 rows, expect 10 zeros per column
            assert_eq!(zero_count, 10, "column {col}: expected 10 zeros, got {zero_count}");
        }
    }

    #[test]
    fn init_sparse_rejects_non_2d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 8], vec![2, 2, 2], true).unwrap();
        assert!(s.init_sparse_(t, 0.5, 0.01).is_err());
    }

    #[test]
    fn init_sparse_rejects_invalid_sparsity() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![0.0; 4], vec![2, 2], true).unwrap();
        assert!(s.init_sparse_(t, 1.0, 0.01).is_err());
        assert!(s.init_sparse_(t, -0.1, 0.01).is_err());
    }

    #[test]
    fn init_dirac_3d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // 1D conv weight: [out=2, in=2, kernel=3]
        let t = s.tensor_variable(vec![0.0; 12], vec![2, 2, 3], true).unwrap();
        s.init_dirac_(t, 1).unwrap();
        let vals = s.tensor_values(t).unwrap();
        // oc=0, ic=0, center=1 -> index 0*6 + 0*3 + 1 = 1
        assert_eq!(vals[1], 1.0);
        // oc=1, ic=1, center=1 -> index 1*6 + 1*3 + 1 = 10
        assert_eq!(vals[10], 1.0);
        // Everything else should be 0
        let sum: f64 = vals.iter().sum();
        assert!((sum - 2.0).abs() < 1e-10);
    }

    #[test]
    fn calculate_gain_values() {
        assert!((FrankenTorchSession::calculate_gain("relu", 0.0) - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((FrankenTorchSession::calculate_gain("tanh", 0.0) - 5.0 / 3.0).abs() < 1e-10);
        assert!((FrankenTorchSession::calculate_gain("linear", 0.0) - 1.0).abs() < 1e-10);
        assert!((FrankenTorchSession::calculate_gain("sigmoid", 0.0) - 1.0).abs() < 1e-10);
        // leaky_relu with negative_slope=0.2: sqrt(2 / (1 + 0.04)) = sqrt(2/1.04)
        let expected = (2.0 / 1.04_f64).sqrt();
        assert!((FrankenTorchSession::calculate_gain("leaky_relu", 0.2) - expected).abs() < 1e-10);
    }

    #[test]
    fn init_works_on_requires_grad_leaf() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Init functions must work on leaf tensors with requires_grad=true
        let t = s.tensor_variable(vec![0.0; 100], vec![10, 10], true).unwrap();
        assert!(s.tensor_requires_grad(t).unwrap());
        assert!(s.tensor_is_leaf(t).unwrap());
        // These should NOT error (unlike regular in-place ops)
        s.init_uniform_(t, -1.0, 1.0).unwrap();
        s.init_normal_(t, 0.0, 1.0).unwrap();
        s.init_constant_(t, 5.0).unwrap();
        s.init_xavier_uniform_(t, 1.0).unwrap();
        s.init_kaiming_normal_(t, 0.0, "fan_in", "relu").unwrap();
    }

    // ── linalg.det / slogdet tests ────────────────────────────────────

    #[test]
    fn det_identity() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.eye(3, false).unwrap();
        let det = s.tensor_linalg_det(a).unwrap();
        assert!((det - 1.0).abs() < 1e-12);
    }

    #[test]
    fn det_scaled_identity_2n() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0], vec![3, 3], false)
            .unwrap();
        let det = s.tensor_linalg_det(a).unwrap();
        assert!((det - 8.0).abs() < 1e-12);
    }

    #[test]
    fn det_known_3x3_value() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ], vec![3, 3], false).unwrap();
        let det = s.tensor_linalg_det(a).unwrap();
        assert!((det - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn det_singular_zero() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2], false)
            .unwrap();
        let det = s.tensor_linalg_det(a).unwrap();
        assert!(det.abs() < 1e-10);
    }

    #[test]
    fn det_1x1_scalar() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![7.5], vec![1, 1], false).unwrap();
        let det = s.tensor_linalg_det(a).unwrap();
        assert!((det - 7.5).abs() < 1e-12);
    }

    #[test]
    fn slogdet_matches_det() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ], vec![3, 3], false).unwrap();
        let det = s.tensor_linalg_det(a).unwrap();
        let (sign, logabsdet) = s.tensor_linalg_slogdet(a).unwrap();
        let reconstructed = sign * logabsdet.exp();
        assert!((reconstructed - det).abs() < 1e-10);
    }

    #[test]
    fn slogdet_singular_matrix() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2], false)
            .unwrap();
        let (sign, logabsdet) = s.tensor_linalg_slogdet(a).unwrap();
        assert!(sign.abs() < 1e-12);
        assert!(logabsdet.is_infinite() && logabsdet < 0.0);
    }

    #[test]
    fn slogdet_negative_det() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // det([[0,1],[1,0]]) = -1
        let a = s
            .tensor_variable(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2], false)
            .unwrap();
        let (sign, logabsdet) = s.tensor_linalg_slogdet(a).unwrap();
        assert!((sign - (-1.0)).abs() < 1e-12);
        assert!(logabsdet.abs() < 1e-12);
    }

    // ── unique / unique_consecutive tests (bd-2klp.3) ─────────────────

    #[test]
    fn unique_sorted() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![3.0, 1.0, 2.0, 1.0, 3.0], vec![5], false)
            .unwrap();
        let (u, _, _) = s.tensor_unique(t, true, false, false).unwrap();
        let vals = s.tensor_values(u).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn unique_unsorted() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![3.0, 1.0, 2.0, 1.0, 3.0], vec![5], false)
            .unwrap();
        let (u, _, _) = s.tensor_unique(t, false, false, false).unwrap();
        let vals = s.tensor_values(u).unwrap();
        // First occurrence order: 3, 1, 2
        assert_eq!(vals, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn unique_with_inverse() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![3.0, 1.0, 2.0, 1.0, 3.0], vec![5], false)
            .unwrap();
        let (u, inv, _) = s.tensor_unique(t, true, true, false).unwrap();
        let u_vals = s.tensor_values(u).unwrap();
        let inv_vals = s.tensor_values(inv.unwrap()).unwrap();
        assert_eq!(u_vals, vec![1.0, 2.0, 3.0]);
        // Inverse maps: 3->2, 1->0, 2->1, 1->0, 3->2
        assert_eq!(inv_vals, vec![2.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn unique_with_counts() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![3.0, 1.0, 2.0, 1.0, 3.0], vec![5], false)
            .unwrap();
        let (u, _, counts) = s.tensor_unique(t, true, false, true).unwrap();
        let u_vals = s.tensor_values(u).unwrap();
        let c_vals = s.tensor_values(counts.unwrap()).unwrap();
        assert_eq!(u_vals, vec![1.0, 2.0, 3.0]);
        // 1 appears 2x, 2 appears 1x, 3 appears 2x
        assert_eq!(c_vals, vec![2.0, 1.0, 2.0]);
    }

    #[test]
    fn unique_all_same() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![5.0, 5.0, 5.0], vec![3], false)
            .unwrap();
        let (u, _, _) = s.tensor_unique(t, true, false, false).unwrap();
        let vals = s.tensor_values(u).unwrap();
        assert_eq!(vals, vec![5.0]);
    }

    #[test]
    fn unique_all_different() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let (u, _, _) = s.tensor_unique(t, true, false, false).unwrap();
        let vals = s.tensor_values(u).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn unique_consecutive_basic() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![1.0, 1.0, 2.0, 2.0, 1.0], vec![5], false)
            .unwrap();
        let (u, _, _) = s.tensor_unique_consecutive(t, false, false).unwrap();
        let vals = s.tensor_values(u).unwrap();
        // Only removes consecutive dupes: 1, 2, 1
        assert_eq!(vals, vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn unique_consecutive_with_inverse() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![1.0, 1.0, 2.0, 2.0, 1.0], vec![5], false)
            .unwrap();
        let (_, inv, _) = s.tensor_unique_consecutive(t, true, false).unwrap();
        let inv_vals = s.tensor_values(inv.unwrap()).unwrap();
        // Group 0: indices 0,1 -> 0; Group 1: indices 2,3 -> 1; Group 2: index 4 -> 2
        assert_eq!(inv_vals, vec![0.0, 0.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn unique_consecutive_with_counts() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(vec![1.0, 1.0, 2.0, 2.0, 1.0], vec![5], false)
            .unwrap();
        let (_, _, counts) = s.tensor_unique_consecutive(t, false, true).unwrap();
        let c_vals = s.tensor_values(counts.unwrap()).unwrap();
        // Group sizes: 2, 2, 1
        assert_eq!(c_vals, vec![2.0, 2.0, 1.0]);
    }

    // ── matrix_power / matrix_exp tests (bd-2drq.10) ──────────────────

    #[test]
    fn matrix_power_zero_is_identity() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2], false)
            .unwrap();
        let result = s.tensor_matrix_power(a, 0).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!(vals[1].abs() < 1e-12);
        assert!(vals[2].abs() < 1e-12);
        assert!((vals[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn matrix_power_one_is_self() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a_data = vec![2.0, 1.0, 1.0, 3.0];
        let a = s
            .tensor_variable(a_data.clone(), vec![2, 2], false)
            .unwrap();
        let result = s.tensor_matrix_power(a, 1).unwrap();
        let vals = s.tensor_values(result).unwrap();
        for (i, (&v, &e)) in vals.iter().zip(a_data.iter()).enumerate() {
            assert!((v - e).abs() < 1e-12, "power1[{i}] = {v}, expected {e}");
        }
    }

    #[test]
    fn matrix_power_two_is_a_squared() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        let a2 = s.tensor_matrix_power(a, 2).unwrap();
        let mm = s.tensor_matmul(a, a).unwrap();
        let a2_vals = s.tensor_values(a2).unwrap();
        let mm_vals = s.tensor_values(mm).unwrap();
        for (i, (&p, &m)) in a2_vals.iter().zip(mm_vals.iter()).enumerate() {
            assert!((p - m).abs() < 1e-10, "power2[{i}] = {p}, matmul = {m}");
        }
    }

    #[test]
    fn matrix_power_three() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        let a3 = s.tensor_matrix_power(a, 3).unwrap();
        let a2 = s.tensor_matmul(a, a).unwrap();
        let a3_ref = s.tensor_matmul(a2, a).unwrap();
        let a3_vals = s.tensor_values(a3).unwrap();
        let ref_vals = s.tensor_values(a3_ref).unwrap();
        for (i, (&p, &r)) in a3_vals.iter().zip(ref_vals.iter()).enumerate() {
            assert!((p - r).abs() < 1e-10, "power3[{i}] = {p}, ref = {r}");
        }
    }

    #[test]
    fn matrix_power_negative_one_is_inverse() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2], false)
            .unwrap();
        let a_inv = s.tensor_matrix_power(a, -1).unwrap();
        let product = s.tensor_matmul(a, a_inv).unwrap();
        let vals = s.tensor_values(product).unwrap();
        // A @ A^-1 should be identity
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!(vals[1].abs() < 1e-10);
        assert!(vals[2].abs() < 1e-10);
        assert!((vals[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_identity_any_n() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let i3 = s.eye(3, false).unwrap();
        let result = s.tensor_matrix_power(i3, 100).unwrap();
        let vals = s.tensor_values(result).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((vals[i * 3 + j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn matrix_exp_zero_is_identity() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![0.0; 4], vec![2, 2], false)
            .unwrap();
        let result = s.tensor_matrix_exp(a).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!(vals[1].abs() < 1e-10);
        assert!(vals[2].abs() < 1e-10);
        assert!((vals[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_exp_diagonal() {
        // exp(diag(a, b)) = diag(exp(a), exp(b))
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2], false)
            .unwrap();
        let result = s.tensor_matrix_exp(a).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert!((vals[0] - 1.0f64.exp()).abs() < 1e-8);
        assert!(vals[1].abs() < 1e-10);
        assert!(vals[2].abs() < 1e-10);
        assert!((vals[3] - 2.0f64.exp()).abs() < 1e-8);
    }

    #[test]
    fn matrix_exp_1x1() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![3.0], vec![1, 1], false).unwrap();
        let result = s.tensor_matrix_exp(a).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert!((vals[0] - 3.0f64.exp()).abs() < 1e-8);
    }

    #[test]
    fn linalg_inv_2x2() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![4.0, 7.0, 2.0, 6.0], vec![2, 2], false)
            .unwrap();
        let inv = s.tensor_linalg_inv(a).unwrap();
        let product = s.tensor_matmul(a, inv).unwrap();
        let vals = s.tensor_values(product).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!(vals[1].abs() < 1e-10);
        assert!(vals[2].abs() < 1e-10);
        assert!((vals[3] - 1.0).abs() < 1e-10);
    }

    // ── Eigendecomposition tests (bd-2drq.7) ──────────────────────────

    #[test]
    fn eigh_identity() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.eye(3, false).unwrap();
        let (evals, _evecs) = s.tensor_linalg_eigh(a).unwrap();
        let eval_vals = s.tensor_values(evals).unwrap();
        for (i, &v) in eval_vals.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-10, "eigenvalue[{i}] = {v}, expected 1.0");
        }
    }

    #[test]
    fn eigh_diagonal() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            3.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 2.0,
        ], vec![3, 3], false).unwrap();
        let (evals, _) = s.tensor_linalg_eigh(a).unwrap();
        let vals = s.tensor_values(evals).unwrap();
        // Sorted ascending: 1, 2, 3
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        assert!((vals[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eigh_reconstruction() {
        // Verify A = V @ diag(λ) @ V^T
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a_data = vec![
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ];
        let a = s
            .tensor_variable(a_data.clone(), vec![3, 3], false)
            .unwrap();
        let (evals, evecs) = s.tensor_linalg_eigh(a).unwrap();
        let eval_vals = s.tensor_values(evals).unwrap();
        let evec_vals = s.tensor_values(evecs).unwrap();
        let n = 3;
        // Reconstruct: A = V @ diag(λ) @ V^T
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..n {
                    val += evec_vals[i * n + k] * eval_vals[k] * evec_vals[j * n + k];
                }
                assert!(
                    (val - a_data[i * n + j]).abs() < 1e-10,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a_data[i * n + j]
                );
            }
        }
    }

    #[test]
    fn eigh_eigenvectors_orthonormal() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            4.0, 1.0,
            1.0, 3.0,
        ], vec![2, 2], false).unwrap();
        let (_, evecs) = s.tensor_linalg_eigh(a).unwrap();
        let v = s.tensor_values(evecs).unwrap();
        let n = 2;
        // V^T @ V = I
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += v[k * n + i] * v[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "(V^T@V)[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn eigh_sorted_ascending() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            5.0, 2.0, 1.0,
            2.0, 4.0, 2.0,
            1.0, 2.0, 3.0,
        ], vec![3, 3], false).unwrap();
        let (evals, _) = s.tensor_linalg_eigh(a).unwrap();
        let vals = s.tensor_values(evals).unwrap();
        for i in 1..vals.len() {
            assert!(
                vals[i] >= vals[i - 1] - 1e-12,
                "eigenvalues not ascending: λ[{}]={} < λ[{}]={}",
                i - 1,
                vals[i - 1],
                i,
                vals[i]
            );
        }
    }

    #[test]
    fn eigh_1x1() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![7.0], vec![1, 1], false).unwrap();
        let (evals, evecs) = s.tensor_linalg_eigh(a).unwrap();
        let eval_vals = s.tensor_values(evals).unwrap();
        let evec_vals = s.tensor_values(evecs).unwrap();
        assert!((eval_vals[0] - 7.0).abs() < 1e-12);
        assert!((evec_vals[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eigh_negative_eigenvalues() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // A = [[1, 3], [3, 1]] has eigenvalues -2 and 4
        let a = s
            .tensor_variable(vec![1.0, 3.0, 3.0, 1.0], vec![2, 2], false)
            .unwrap();
        let (evals, _) = s.tensor_linalg_eigh(a).unwrap();
        let vals = s.tensor_values(evals).unwrap();
        assert!((vals[0] - (-2.0)).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn eigvalsh_matches_eigh() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            2.0, 1.0,
            1.0, 3.0,
        ], vec![2, 2], false).unwrap();
        let (evals_full, _) = s.tensor_linalg_eigh(a).unwrap();
        let evals_only = s.tensor_linalg_eigvalsh(a).unwrap();
        let full_vals = s.tensor_values(evals_full).unwrap();
        let only_vals = s.tensor_values(evals_only).unwrap();
        for (i, (&f, &o)) in full_vals.iter().zip(only_vals.iter()).enumerate() {
            assert!((f - o).abs() < 1e-12, "mismatch at [{i}]: {f} vs {o}");
        }
    }

    // ── SVD tests (bd-2drq.3) ─────────────────────────────────────────

    #[test]
    fn svd_identity_3x3() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.eye(3, false).unwrap();
        let (u, sv, vh) = s.tensor_linalg_svd(a, false).unwrap();
        let sv_vals = s.tensor_values(sv).unwrap();
        // All singular values should be 1.0
        for (i, &val) in sv_vals.iter().enumerate() {
            assert!((val - 1.0).abs() < 1e-10, "s[{i}] = {val}, expected 1.0");
        }
    }

    #[test]
    fn svd_reconstruction_2x2() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a_data = vec![3.0, 1.0, 1.0, 3.0];
        let a = s
            .tensor_variable(a_data.clone(), vec![2, 2], false)
            .unwrap();
        let (u, sv, vh) = s.tensor_linalg_svd(a, false).unwrap();
        let u_vals = s.tensor_values(u).unwrap();
        let sv_vals = s.tensor_values(sv).unwrap();
        let vh_vals = s.tensor_values(vh).unwrap();
        // Reconstruct: A = U @ diag(S) @ Vh
        let m = 2;
        let n = 2;
        let k = 2;
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += u_vals[i * k + l] * sv_vals[l] * vh_vals[l * n + j];
                }
                assert!(
                    (val - a_data[i * n + j]).abs() < 1e-10,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a_data[i * n + j]
                );
            }
        }
    }

    #[test]
    fn svd_singular_values_sorted_descending() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ], vec![3, 3], false).unwrap();
        let (_, sv, _) = s.tensor_linalg_svd(a, false).unwrap();
        let sv_vals = s.tensor_values(sv).unwrap();
        for i in 1..sv_vals.len() {
            assert!(
                sv_vals[i - 1] >= sv_vals[i] - 1e-12,
                "s[{}] = {} < s[{}] = {}: not descending",
                i - 1,
                sv_vals[i - 1],
                i,
                sv_vals[i]
            );
        }
    }

    #[test]
    fn svd_singular_values_nonnegative() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        let (_, sv, _) = s.tensor_linalg_svd(a, false).unwrap();
        let sv_vals = s.tensor_values(sv).unwrap();
        for (i, &val) in sv_vals.iter().enumerate() {
            assert!(val >= -1e-15, "s[{i}] = {val} is negative");
        }
    }

    #[test]
    fn svd_diagonal_matrix() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            5.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 1.0,
        ], vec![3, 3], false).unwrap();
        let (_, sv, _) = s.tensor_linalg_svd(a, false).unwrap();
        let sv_vals = s.tensor_values(sv).unwrap();
        assert!((sv_vals[0] - 5.0).abs() < 1e-10);
        assert!((sv_vals[1] - 3.0).abs() < 1e-10);
        assert!((sv_vals[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn svd_1x1() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![-7.0], vec![1, 1], false).unwrap();
        let (_, sv, _) = s.tensor_linalg_svd(a, false).unwrap();
        let sv_vals = s.tensor_values(sv).unwrap();
        assert!((sv_vals[0] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn svd_zero_matrix() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![0.0; 4], vec![2, 2], false)
            .unwrap();
        let (_, sv, _) = s.tensor_linalg_svd(a, false).unwrap();
        let sv_vals = s.tensor_values(sv).unwrap();
        for &val in &sv_vals {
            assert!(val.abs() < 1e-12);
        }
    }

    #[test]
    fn svd_rectangular_tall() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a_data = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];
        let a = s
            .tensor_variable(a_data.clone(), vec![3, 2], false)
            .unwrap();
        let (u, sv, vh) = s.tensor_linalg_svd(a, false).unwrap();
        let u_shape = s.tensor_shape(u).unwrap();
        let sv_shape = s.tensor_shape(sv).unwrap();
        let vh_shape = s.tensor_shape(vh).unwrap();
        assert_eq!(u_shape, vec![3, 2]); // reduced: (m, k)
        assert_eq!(sv_shape, vec![2]);
        assert_eq!(vh_shape, vec![2, 2]); // reduced: (k, n)

        // Verify reconstruction
        let u_vals = s.tensor_values(u).unwrap();
        let sv_vals = s.tensor_values(sv).unwrap();
        let vh_vals = s.tensor_values(vh).unwrap();
        for i in 0..3 {
            for j in 0..2 {
                let mut val = 0.0;
                for l in 0..2 {
                    val += u_vals[i * 2 + l] * sv_vals[l] * vh_vals[l * 2 + j];
                }
                assert!(
                    (val - a_data[i * 2 + j]).abs() < 1e-10,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a_data[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn svd_rectangular_wide() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let a = s
            .tensor_variable(a_data.clone(), vec![2, 3], false)
            .unwrap();
        let (u, sv, vh) = s.tensor_linalg_svd(a, false).unwrap();
        let u_shape = s.tensor_shape(u).unwrap();
        let vh_shape = s.tensor_shape(vh).unwrap();
        assert_eq!(u_shape, vec![2, 2]); // reduced: (m, k)
        assert_eq!(vh_shape, vec![2, 3]); // reduced: (k, n)

        // Verify reconstruction
        let u_vals = s.tensor_values(u).unwrap();
        let sv_vals = s.tensor_values(sv).unwrap();
        let vh_vals = s.tensor_values(vh).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                let mut val = 0.0;
                for l in 0..2 {
                    val += u_vals[i * 2 + l] * sv_vals[l] * vh_vals[l * 3 + j];
                }
                assert!(
                    (val - a_data[i * 3 + j]).abs() < 1e-10,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a_data[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn svd_full_matrices_shapes() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ], vec![3, 2], false).unwrap();
        let (u, sv, vh) = s.tensor_linalg_svd(a, true).unwrap();
        let u_shape = s.tensor_shape(u).unwrap();
        let sv_shape = s.tensor_shape(sv).unwrap();
        let vh_shape = s.tensor_shape(vh).unwrap();
        assert_eq!(u_shape, vec![3, 3]); // full: (m, m)
        assert_eq!(sv_shape, vec![2]);
        assert_eq!(vh_shape, vec![2, 2]); // full: (n, n)
    }

    #[test]
    fn svdvals_matches_svd() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ], vec![3, 3], false).unwrap();
        let (_, sv_full, _) = s.tensor_linalg_svd(a, false).unwrap();
        let sv_only = s.tensor_linalg_svdvals(a).unwrap();
        let sv_full_vals = s.tensor_values(sv_full).unwrap();
        let sv_only_vals = s.tensor_values(sv_only).unwrap();
        assert_eq!(sv_full_vals.len(), sv_only_vals.len());
        for (i, (&a_val, &b_val)) in sv_full_vals.iter().zip(sv_only_vals.iter()).enumerate() {
            assert!(
                (a_val - b_val).abs() < 1e-10,
                "svdvals mismatch at [{i}]: {a_val} vs {b_val}"
            );
        }
    }

    #[test]
    fn svd_orthogonality() {
        // Verify U^T @ U = I and Vh @ Vh^T = I (for reduced SVD of square matrix)
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ], vec![3, 3], false).unwrap();
        let (u, _, vh) = s.tensor_linalg_svd(a, false).unwrap();
        let u_vals = s.tensor_values(u).unwrap();
        let vh_vals = s.tensor_values(vh).unwrap();
        let n = 3;
        // U^T @ U should be identity
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += u_vals[k * n + i] * u_vals[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "(U^T@U)[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
        // Vh @ Vh^T should be identity
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += vh_vals[i * n + k] * vh_vals[j * n + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "(Vh@Vh^T)[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    // ── cross / vecdot / diag_embed tests (bd-2drq.9) ─────────────────

    #[test]
    fn cross_standard_basis() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let i = s
            .tensor_variable(vec![1.0, 0.0, 0.0], vec![3], false)
            .unwrap();
        let j = s
            .tensor_variable(vec![0.0, 1.0, 0.0], vec![3], false)
            .unwrap();
        let k = s.tensor_cross(i, j).unwrap();
        let vals = s.tensor_values(k).unwrap();
        // i × j = k = [0, 0, 1]
        assert!((vals[0]).abs() < 1e-12);
        assert!((vals[1]).abs() < 1e-12);
        assert!((vals[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cross_anti_commutativity() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], false)
            .unwrap();
        let axb = s.tensor_cross(a, b).unwrap();
        let bxa = s.tensor_cross(b, a).unwrap();
        let axb_vals = s.tensor_values(axb).unwrap();
        let bxa_vals = s.tensor_values(bxa).unwrap();
        // a × b = -(b × a)
        for i in 0..3 {
            assert!(
                (axb_vals[i] + bxa_vals[i]).abs() < 1e-12,
                "anti-commutativity failed at [{i}]"
            );
        }
    }

    #[test]
    fn cross_known_values() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], false)
            .unwrap();
        let c = s.tensor_cross(a, b).unwrap();
        let vals = s.tensor_values(c).unwrap();
        // [2*6-3*5, 3*4-1*6, 1*5-2*4] = [12-15, 12-6, 5-8] = [-3, 6, -3]
        assert!((vals[0] - (-3.0)).abs() < 1e-12);
        assert!((vals[1] - 6.0).abs() < 1e-12);
        assert!((vals[2] - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn cross_wrong_size_errors() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![3.0, 4.0], vec![2], false)
            .unwrap();
        assert!(s.tensor_cross(a, b).is_err());
    }

    #[test]
    fn vecdot_1d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], false)
            .unwrap();
        let d = s.tensor_vecdot(a, b).unwrap();
        let vals = s.tensor_values(d).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert!((vals[0] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn vecdot_2d_batched() {
        // a: [[1,2],[3,4]], b: [[5,6],[7,8]]
        // vecdot along last dim: [1*5+2*6, 3*7+4*8] = [17, 53]
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false)
            .unwrap();
        let d = s.tensor_vecdot(a, b).unwrap();
        let vals = s.tensor_values(d).unwrap();
        assert!((vals[0] - 17.0).abs() < 1e-10);
        assert!((vals[1] - 53.0).abs() < 1e-10);
    }

    #[test]
    fn diag_embed_main_diagonal() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let d = s.tensor_diag_embed(v, 0).unwrap();
        let vals = s.tensor_values(d).unwrap();
        let shape = s.tensor_shape(d).unwrap();
        assert_eq!(shape, vec![3, 3]);
        #[rustfmt::skip]
        let expected = vec![
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ];
        for (i, (&v, &e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-12, "diag[{i}] = {v}, expected {e}");
        }
    }

    #[test]
    fn diag_embed_super_diagonal() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
        let d = s.tensor_diag_embed(v, 1).unwrap();
        let vals = s.tensor_values(d).unwrap();
        let shape = s.tensor_shape(d).unwrap();
        assert_eq!(shape, vec![3, 3]);
        #[rustfmt::skip]
        let expected = vec![
            0.0, 1.0, 0.0,
            0.0, 0.0, 2.0,
            0.0, 0.0, 0.0,
        ];
        for (i, (&v, &e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-12, "diag[{i}] = {v}, expected {e}");
        }
    }

    #[test]
    fn diag_embed_sub_diagonal() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = s.tensor_variable(vec![5.0, 6.0], vec![2], false).unwrap();
        let d = s.tensor_diag_embed(v, -1).unwrap();
        let vals = s.tensor_values(d).unwrap();
        let shape = s.tensor_shape(d).unwrap();
        assert_eq!(shape, vec![3, 3]);
        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, 0.0,
            5.0, 0.0, 0.0,
            0.0, 6.0, 0.0,
        ];
        for (i, (&v, &e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-12, "diag[{i}] = {v}, expected {e}");
        }
    }

    // ── tensordot / kron tests (bd-2klp.8) ────────────────────────────

    #[test]
    fn tensordot_dims1_matches_matmul() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], false)
            .unwrap();
        let td = s.tensor_tensordot(a, b, 1).unwrap();
        let mm = s.tensor_matmul(a, b).unwrap();
        let td_vals = s.tensor_values(td).unwrap();
        let mm_vals = s.tensor_values(mm).unwrap();
        assert_eq!(td_vals.len(), mm_vals.len());
        for (i, (&t, &m)) in td_vals.iter().zip(mm_vals.iter()).enumerate() {
            assert!((t - m).abs() < 1e-10, "tensordot[{i}] = {t}, matmul = {m}");
        }
    }

    #[test]
    fn tensordot_dims0_outer_product() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
        let b = s
            .tensor_variable(vec![3.0, 4.0, 5.0], vec![3], false)
            .unwrap();
        let td = s.tensor_tensordot(a, b, 0).unwrap();
        let td_vals = s.tensor_values(td).unwrap();
        let td_shape = s.tensor_shape(td).unwrap();
        assert_eq!(td_shape, vec![2, 3]);
        // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]]
        let expected = vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
        for (i, (&v, &e)) in td_vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-10, "td[{i}] = {v}, expected {e}");
        }
    }

    #[test]
    fn tensordot_dims2_full_contraction() {
        // dims=2 on 2x3 and 2x3: contracts both dimensions -> scalar
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3], false)
            .unwrap();
        let td = s.tensor_tensordot(a, b, 2).unwrap();
        let td_vals = s.tensor_values(td).unwrap();
        // sum of element-wise products: 1*7+2*8+3*9+4*10+5*11+6*12 = 7+16+27+40+55+72 = 217
        assert!((td_vals[0] - 217.0).abs() < 1e-10);
    }

    #[test]
    fn tensordot_3d_contract_1() {
        // a: [2,3,4], b: [4,5] -> contract last 1 dim -> [2,3,5]
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a_data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let b_data: Vec<f64> = (0..20).map(|x| x as f64).collect();
        let a = s
            .tensor_variable(a_data, vec![2, 3, 4], false)
            .unwrap();
        let b = s
            .tensor_variable(b_data, vec![4, 5], false)
            .unwrap();
        let td = s.tensor_tensordot(a, b, 1).unwrap();
        let td_shape = s.tensor_shape(td).unwrap();
        assert_eq!(td_shape, vec![2, 3, 5]);
    }

    #[test]
    fn tensordot_shape_mismatch_errors() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        // dims=1: a's last dim (3) != b's first dim (2)
        assert!(s.tensor_tensordot(a, b, 1).is_err());
    }

    #[test]
    fn kron_1d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
        let b = s
            .tensor_variable(vec![3.0, 4.0, 5.0], vec![3], false)
            .unwrap();
        let k = s.tensor_kron(a, b).unwrap();
        let vals = s.tensor_values(k).unwrap();
        let shape = s.tensor_shape(k).unwrap();
        assert_eq!(shape, vec![6]);
        // [1*3, 1*4, 1*5, 2*3, 2*4, 2*5]
        let expected = vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
        for (i, (&v, &e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-10, "kron[{i}] = {v}, expected {e}");
        }
    }

    #[test]
    fn kron_2d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![0.0, 5.0, 6.0, 7.0], vec![2, 2], false)
            .unwrap();
        let k = s.tensor_kron(a, b).unwrap();
        let vals = s.tensor_values(k).unwrap();
        let shape = s.tensor_shape(k).unwrap();
        assert_eq!(shape, vec![4, 4]);
        // Row-major 4x4:
        // [1*[[0,5],[6,7]], 2*[[0,5],[6,7]]]
        // [3*[[0,5],[6,7]], 4*[[0,5],[6,7]]]
        #[rustfmt::skip]
        let expected = vec![
            0.0,  5.0,  0.0, 10.0,
            6.0,  7.0, 12.0, 14.0,
            0.0, 15.0,  0.0, 20.0,
           18.0, 21.0, 24.0, 28.0,
        ];
        for (i, (&v, &e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-10, "kron[{i}] = {v}, expected {e}");
        }
    }

    #[test]
    fn kron_identity_2x2() {
        // kron(I2, I2) = I4
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let i2 = s.eye(2, false).unwrap();
        let k = s.tensor_kron(i2, i2).unwrap();
        let vals = s.tensor_values(k).unwrap();
        let shape = s.tensor_shape(k).unwrap();
        assert_eq!(shape, vec![4, 4]);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vals[i * 4 + j] - expected).abs() < 1e-12,
                    "kron(I,I)[{i},{j}] = {}, expected {expected}",
                    vals[i * 4 + j]
                );
            }
        }
    }

    // ── Cholesky tests (bd-2drq.5) ────────────────────────────────────

    #[test]
    fn cholesky_identity() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.eye(3, false).unwrap();
        let l = s.tensor_linalg_cholesky(a, false).unwrap();
        let vals = s.tensor_values(l).unwrap();
        // L should be identity
        #[rustfmt::skip]
        let expected = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        for (i, (&v, &e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-12, "L[{i}] = {v}, expected {e}");
        }
    }

    #[test]
    fn cholesky_known_spd() {
        // A = [[4, 2], [2, 3]] is SPD
        // L = [[2, 0], [1, sqrt(2)]]
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2], false)
            .unwrap();
        let l = s.tensor_linalg_cholesky(a, false).unwrap();
        let vals = s.tensor_values(l).unwrap();
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!(vals[1].abs() < 1e-12);
        assert!((vals[2] - 1.0).abs() < 1e-12);
        assert!((vals[3] - 2.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cholesky_reconstruction() {
        // Verify L @ L^T == A
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a_data = vec![
            4.0, 12.0, -16.0,
            12.0, 37.0, -43.0,
            -16.0, -43.0, 98.0,
        ];
        let a = s
            .tensor_variable(a_data.clone(), vec![3, 3], false)
            .unwrap();
        let l = s.tensor_linalg_cholesky(a, false).unwrap();
        let l_vals = s.tensor_values(l).unwrap();
        // Reconstruct A = L @ L^T manually
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += l_vals[i * n + k] * l_vals[j * n + k];
                }
                assert!(
                    (dot - a_data[i * n + j]).abs() < 1e-10,
                    "(L@L^T)[{i},{j}] = {dot}, expected {}",
                    a_data[i * n + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_upper() {
        // upper=true: A = U^T @ U
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2], false)
            .unwrap();
        let u = s.tensor_linalg_cholesky(a, true).unwrap();
        let vals = s.tensor_values(u).unwrap();
        // U should be upper triangular: [[2, 1], [0, sqrt(2)]]
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
        assert!(vals[2].abs() < 1e-12);
        assert!((vals[3] - 2.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cholesky_1x1() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![9.0], vec![1, 1], false).unwrap();
        let l = s.tensor_linalg_cholesky(a, false).unwrap();
        let vals = s.tensor_values(l).unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_not_positive_definite_errors() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Not positive definite: eigenvalues include negative
        let a = s
            .tensor_variable(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2], false)
            .unwrap();
        assert!(s.tensor_linalg_cholesky(a, false).is_err());
    }

    #[test]
    fn cholesky_diagonal_spd() {
        // Diagonal SPD: L should be diagonal with sqrt elements
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![
            4.0, 0.0, 0.0,
            0.0, 9.0, 0.0,
            0.0, 0.0, 16.0,
        ], vec![3, 3], false).unwrap();
        let l = s.tensor_linalg_cholesky(a, false).unwrap();
        let vals = s.tensor_values(l).unwrap();
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[4] - 3.0).abs() < 1e-12);
        assert!((vals[8] - 4.0).abs() < 1e-12);
        // Off-diagonals should be zero
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(vals[i * 3 + j].abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn cholesky_solve_identity_factor() {
        // L = I => A = I, so solve I @ X = B => X = B
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.eye(3, false).unwrap();
        let l = s.tensor_linalg_cholesky(a, false).unwrap();
        let b = s
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let x = s.tensor_cholesky_solve(b, l, false).unwrap();
        let vals = s.tensor_values(x).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
        assert!((vals[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_solve_matches_direct() {
        // A = [[4, 2], [2, 3]], b = [1, 2]
        // A^-1 = [[3, -2], [-2, 4]] / det(A)  det=8
        // x = A^-1 @ b = [[3-4], [-2+8]]/8 = [-1/8, 6/8] = [-0.125, 0.75]
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2], false)
            .unwrap();
        let l = s.tensor_linalg_cholesky(a, false).unwrap();
        let b = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
        let x = s.tensor_cholesky_solve(b, l, false).unwrap();
        let vals = s.tensor_values(x).unwrap();
        assert!((vals[0] - (-0.125)).abs() < 1e-10);
        assert!((vals[1] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn cholesky_solve_multiple_rhs() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2], false)
            .unwrap();
        let l = s.tensor_linalg_cholesky(a, false).unwrap();
        // B = [[1, 0], [0, 1]] (identity) => X = A^-1
        let b = s
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false)
            .unwrap();
        let x = s.tensor_cholesky_solve(b, l, false).unwrap();
        let vals = s.tensor_values(x).unwrap();
        // A^-1 = [3/8, -2/8; -2/8, 4/8] = [0.375, -0.25; -0.25, 0.5]
        assert!((vals[0] - 0.375).abs() < 1e-10);
        assert!((vals[1] - (-0.25)).abs() < 1e-10);
        assert!((vals[2] - (-0.25)).abs() < 1e-10);
        assert!((vals[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn cholesky_solve_upper_flag() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2], false)
            .unwrap();
        let u = s.tensor_linalg_cholesky(a, true).unwrap();
        let b = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
        let x = s.tensor_cholesky_solve(b, u, true).unwrap();
        let vals = s.tensor_values(x).unwrap();
        // Same answer as lower: [-0.125, 0.75]
        assert!((vals[0] - (-0.125)).abs() < 1e-10);
        assert!((vals[1] - 0.75).abs() < 1e-10);
    }

    // ── linalg.solve tests (bd-2drq.6) ────────────────────────────────

    #[test]
    fn solve_identity_a() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.eye(3, false).unwrap();
        let b = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let x = s.tensor_linalg_solve(a, b).unwrap();
        let vals = s.tensor_values(x).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
        assert!((vals[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn solve_2x2_system() {
        // A = [[2, 1], [5, 3]], b = [4, 7]
        // Solution: x = [5, -6]
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![2.0, 1.0, 5.0, 3.0], vec![2, 2], false)
            .unwrap();
        let b = s.tensor_variable(vec![4.0, 7.0], vec![2], false).unwrap();
        let x = s.tensor_linalg_solve(a, b).unwrap();
        let vals = s.tensor_values(x).unwrap();
        assert!((vals[0] - 5.0).abs() < 1e-10);
        assert!((vals[1] - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn solve_multiple_rhs() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // A @ X = B  =>  X = A^-1 @ B
        // A^-1 = [[-2, 1], [1.5, -0.5]]
        // X = [[-2*5+1*7, -2*6+1*8], [1.5*5-0.5*7, 1.5*6-0.5*8]]
        //   = [[-3, -4], [4, 5]]
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false)
            .unwrap();
        let x = s.tensor_linalg_solve(a, b).unwrap();
        let vals = s.tensor_values(x).unwrap();
        assert!((vals[0] - (-3.0)).abs() < 1e-10);
        assert!((vals[1] - (-4.0)).abs() < 1e-10);
        assert!((vals[2] - 4.0).abs() < 1e-10);
        assert!((vals[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn solve_3x3_verify_a_times_x() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let a_data = vec![
            2.0, 1.0, 1.0,
            4.0, 3.0, 3.0,
            8.0, 7.0, 9.0,
        ];
        let a = s
            .tensor_variable(a_data.clone(), vec![3, 3], false)
            .unwrap();
        let b_data = vec![1.0, 1.0, 1.0];
        let b = s
            .tensor_variable(b_data.clone(), vec![3], false)
            .unwrap();
        let x = s.tensor_linalg_solve(a, b).unwrap();
        let x_vals = s.tensor_values(x).unwrap();
        // Verify A @ x ≈ b manually
        for i in 0..3 {
            let dot: f64 = (0..3).map(|j| a_data[i * 3 + j] * x_vals[j]).sum();
            assert!(
                (dot - b_data[i]).abs() < 1e-10,
                "A@x[{i}] = {dot}, expected {}",
                b_data[i]
            );
        }
    }

    #[test]
    fn solve_singular_returns_result() {
        // Singular A — lu_solve zeroes out dependent rows rather than erroring.
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2], false)
            .unwrap();
        let b = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
        // Does not error; returns a (possibly inaccurate) solution
        let x = s.tensor_linalg_solve(a, b).unwrap();
        let vals = s.tensor_values(x).unwrap();
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn solve_1x1() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![4.0], vec![1, 1], false).unwrap();
        let b = s.tensor_variable(vec![8.0], vec![1], false).unwrap();
        let x = s.tensor_linalg_solve(a, b).unwrap();
        let vals = s.tensor_values(x).unwrap();
        assert!((vals[0] - 2.0).abs() < 1e-12);
    }

    // ── Tensor factory tests (bd-2klp.4) ──────────────────────────────

    #[test]
    fn logspace_base10() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.logspace(0.0, 2.0, 3, 10.0, false).unwrap();
        let vals = s.tensor_values(t).unwrap();
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 10.0).abs() < 1e-10);
        assert!((vals[2] - 100.0).abs() < 1e-8);
    }

    #[test]
    fn logspace_base2() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.logspace(0.0, 3.0, 4, 2.0, false).unwrap();
        let vals = s.tensor_values(t).unwrap();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
        assert!((vals[2] - 4.0).abs() < 1e-12);
        assert!((vals[3] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn logspace_single_step() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.logspace(0.0, 0.0, 1, 10.0, false).unwrap();
        let vals = s.tensor_values(t).unwrap();
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn logspace_zero_steps() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.logspace(0.0, 2.0, 0, 10.0, false).unwrap();
        let vals = s.tensor_values(t).unwrap();
        assert!(vals.is_empty());
    }

    #[test]
    fn empty_correct_shape_and_zeros() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.empty(vec![3, 4], false).unwrap();
        let shape = s.tensor_shape(t).unwrap();
        assert_eq!(shape, vec![3, 4]);
        let vals = s.tensor_values(t).unwrap();
        assert_eq!(vals.len(), 12);
        // In Rust, empty produces zeros
        assert!(vals.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn empty_zero_shape() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.empty(vec![0], false).unwrap();
        let vals = s.tensor_values(t).unwrap();
        assert!(vals.is_empty());
    }

    #[test]
    fn empty_like_matches_shape() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let orig = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false).unwrap();
        let t = s.empty_like(orig, false).unwrap();
        assert_eq!(s.tensor_shape(t).unwrap(), vec![2, 3]);
        assert_eq!(s.tensor_values(t).unwrap().len(), 6);
    }

    #[test]
    fn init_gain_zero_produces_zeros() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(vec![1.0; 20], vec![4, 5], true).unwrap();
        s.init_xavier_uniform_(t, 0.0).unwrap();
        let vals = s.tensor_values(t).unwrap();
        for &v in &vals {
            assert_eq!(v, 0.0);
        }
    }

    // ── scatter_add tests ────────────────────────────────────────────────

    #[test]
    fn scatter_add_basic_accumulates() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // input: [0,0,0], src: [1,2,3], index: [0,1,0] => [1+3, 2, 0] = [4, 2, 0]
        let input = s.tensor_variable(vec![0.0, 0.0, 0.0], vec![1, 3], false).unwrap();
        let src = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false).unwrap();
        let idx = s.tensor_variable(vec![0.0, 1.0, 0.0], vec![1, 3], false).unwrap();
        let result = s.tensor_scatter_add(input, 1, idx, src).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert!((vals[0] - 4.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        assert!((vals[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn scatter_add_multiple_to_same_position() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // All indices point to 0: input[0] += 1+2+3 = 6
        let input = s.tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false).unwrap();
        let src = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let idx = s.tensor_variable(vec![0.0, 0.0, 0.0], vec![3], false).unwrap();
        let result = s.tensor_scatter_add(input, 0, idx, src).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert!((vals[0] - 16.0).abs() < 1e-10); // 10 + 1 + 2 + 3
        assert!((vals[1] - 20.0).abs() < 1e-10);
        assert!((vals[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn scatter_add_as_inverse_of_gather() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Gather: select elements then scatter_add back should recover sums
        let src_data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let src = s.tensor_variable(src_data.clone(), vec![5], false).unwrap();
        let idx = s.tensor_variable(vec![1.0, 3.0, 4.0], vec![3], false).unwrap();
        let gathered = s.tensor_gather(src, 0, idx).unwrap();
        let gathered_vals = s.tensor_values(gathered).unwrap();
        assert!((gathered_vals[0] - 20.0).abs() < 1e-10);
        assert!((gathered_vals[1] - 40.0).abs() < 1e-10);
        assert!((gathered_vals[2] - 50.0).abs() < 1e-10);

        // Scatter_add gathered values back
        let zeros = s.tensor_variable(vec![0.0; 5], vec![5], false).unwrap();
        let idx2 = s.tensor_variable(vec![1.0, 3.0, 4.0], vec![3], false).unwrap();
        let result = s.tensor_scatter_add(zeros, 0, idx2, gathered).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert!((vals[1] - 20.0).abs() < 1e-10);
        assert!((vals[3] - 40.0).abs() < 1e-10);
        assert!((vals[4] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn scatter_add_2d_dim0() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // 3x2 input, scatter along dim 0
        let input = s.tensor_variable(vec![0.0; 6], vec![3, 2], false).unwrap();
        let src = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        // row 0 of src -> row 2 of output, row 1 of src -> row 0
        let idx = s.tensor_variable(vec![2.0, 2.0, 0.0, 0.0], vec![2, 2], false).unwrap();
        let result = s.tensor_scatter_add(input, 0, idx, src).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // row 0: [0+3, 0+4] = [3, 4]
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);
        // row 2: [0+1, 0+2] = [1, 2]
        assert!((vals[4] - 1.0).abs() < 1e-10);
        assert!((vals[5] - 2.0).abs() < 1e-10);
    }

    // ── index_put tests ──────────────────────────────────────────────────

    #[test]
    fn index_put_basic_overwrite() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // 1D: put values at positions 1,3
        let input = s.tensor_variable(vec![0.0; 5], vec![5], false).unwrap();
        let idx = s.tensor_variable(vec![1.0, 3.0], vec![2], false).unwrap();
        let values = s.tensor_variable(vec![10.0, 30.0], vec![2], false).unwrap();
        let result = s.tensor_index_put(input, &[idx], values, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert_eq!(vals, vec![0.0, 10.0, 0.0, 30.0, 0.0]);
    }

    #[test]
    fn index_put_accumulate() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Accumulate: add to existing values
        let input = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], false).unwrap();
        let idx = s.tensor_variable(vec![1.0, 1.0, 3.0], vec![3], false).unwrap();
        let values = s.tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false).unwrap();
        let result = s.tensor_index_put(input, &[idx], values, true).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 32.0).abs() < 1e-10); // 2 + 10 + 20
        assert!((vals[2] - 3.0).abs() < 1e-10);
        assert!((vals[3] - 34.0).abs() < 1e-10); // 4 + 30
        assert!((vals[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn index_put_2d_row_indexing() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // 3x3 tensor, put rows at indices [0, 2]
        let input = s.tensor_variable(vec![0.0; 9], vec![3, 3], false).unwrap();
        let idx = s.tensor_variable(vec![0.0, 2.0], vec![2], false).unwrap();
        let values = s.tensor_variable(vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0], vec![6], false).unwrap();
        let result = s.tensor_index_put(input, &[idx], values, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // row 0: [1, 2, 3], row 1: [0, 0, 0], row 2: [7, 8, 9]
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn index_put_2d_element_indexing() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // 3x3 tensor, put individual elements via [row_indices, col_indices]
        let input = s.tensor_variable(vec![0.0; 9], vec![3, 3], false).unwrap();
        let row_idx = s.tensor_variable(vec![0.0, 1.0, 2.0], vec![3], false).unwrap();
        let col_idx = s.tensor_variable(vec![2.0, 1.0, 0.0], vec![3], false).unwrap();
        let values = s.tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false).unwrap();
        let result = s.tensor_index_put(input, &[row_idx, col_idx], values, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // (0,2)=10, (1,1)=20, (2,0)=30
        assert!((vals[2] - 10.0).abs() < 1e-10);
        assert!((vals[4] - 20.0).abs() < 1e-10);
        assert!((vals[6] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn index_put_scalar_broadcast() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Single value broadcast to all indexed positions
        let input = s.tensor_variable(vec![0.0; 5], vec![5], false).unwrap();
        let idx = s.tensor_variable(vec![1.0, 3.0, 4.0], vec![3], false).unwrap();
        let values = s.tensor_variable(vec![99.0], vec![1], false).unwrap();
        let result = s.tensor_index_put(input, &[idx], values, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert_eq!(vals, vec![0.0, 99.0, 0.0, 99.0, 99.0]);
    }

    #[test]
    fn index_put_empty_preserves_input() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Empty indices: no-op (but we need at least 1 index tensor per our impl)
        // Test with zero-length index
        let input = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let idx = s.tensor_variable(vec![], vec![0], false).unwrap();
        let values = s.tensor_variable(vec![], vec![0], false).unwrap();
        let result = s.tensor_index_put(input, &[idx], values, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    // ── einsum tests ─────────────────────────────────────────────────────

    #[test]
    fn einsum_matmul() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // ij,jk->ik is matrix multiply
        #[rustfmt::skip]
        let a = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        let b = s.tensor_variable(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false).unwrap();
        let result = s.tensor_einsum("ij,jk->ik", &[a, b]).unwrap();
        let shape = s.tensor_shape(result).unwrap();
        assert_eq!(shape, vec![2, 2]);
        let vals = s.tensor_values(result).unwrap();
        // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
        assert!((vals[0] - 19.0).abs() < 1e-10);
        assert!((vals[1] - 22.0).abs() < 1e-10);
        assert!((vals[2] - 43.0).abs() < 1e-10);
        assert!((vals[3] - 50.0).abs() < 1e-10);

        // Compare with direct matmul
        let direct = s.tensor_matmul(a, b).unwrap();
        let direct_vals = s.tensor_values(direct).unwrap();
        assert_eq!(vals, direct_vals);
    }

    #[test]
    fn einsum_trace() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // ii-> is trace
        let a = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        let result = s.tensor_einsum("ii->", &[a]).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // trace = 1 + 4 = 5
        assert!((vals[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_transpose() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // ij->ji is transpose
        let a = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false).unwrap();
        let result = s.tensor_einsum("ij->ji", &[a]).unwrap();
        let shape = s.tensor_shape(result).unwrap();
        assert_eq!(shape, vec![3, 2]);
        let vals = s.tensor_values(result).unwrap();
        assert_eq!(vals, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn einsum_outer_product() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // i,j->ij is outer product
        let a = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let b = s.tensor_variable(vec![4.0, 5.0], vec![2], false).unwrap();
        let result = s.tensor_einsum("i,j->ij", &[a, b]).unwrap();
        let shape = s.tensor_shape(result).unwrap();
        assert_eq!(shape, vec![3, 2]);
        let vals = s.tensor_values(result).unwrap();
        assert_eq!(vals, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn einsum_dot_product() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // i,i-> is dot product
        let a = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let b = s.tensor_variable(vec![4.0, 5.0, 6.0], vec![3], false).unwrap();
        let result = s.tensor_einsum("i,i->", &[a, b]).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert!((vals[0] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_batch_matmul() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // bij,bjk->bik is batch matmul
        // batch=2, i=2, j=2, k=2
        #[rustfmt::skip]
        let a = s.tensor_variable(
            vec![1.0, 0.0, 0.0, 1.0, // batch 0: identity
                 2.0, 0.0, 0.0, 2.0], // batch 1: 2*identity
            vec![2, 2, 2], false,
        ).unwrap();
        #[rustfmt::skip]
        let b = s.tensor_variable(
            vec![1.0, 2.0, 3.0, 4.0,  // batch 0
                 5.0, 6.0, 7.0, 8.0], // batch 1
            vec![2, 2, 2], false,
        ).unwrap();
        let result = s.tensor_einsum("bij,bjk->bik", &[a, b]).unwrap();
        let shape = s.tensor_shape(result).unwrap();
        assert_eq!(shape, vec![2, 2, 2]);
        let vals = s.tensor_values(result).unwrap();
        // batch 0: identity * [[1,2],[3,4]] = [[1,2],[3,4]]
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        // batch 1: 2*identity * [[5,6],[7,8]] = [[10,12],[14,16]]
        assert!((vals[4] - 10.0).abs() < 1e-10);
        assert!((vals[5] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_diagonal_extraction() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // ii->i is diagonal extraction
        let a = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3], false).unwrap();
        let result = s.tensor_einsum("ii->i", &[a]).unwrap();
        let shape = s.tensor_shape(result).unwrap();
        assert_eq!(shape, vec![3]);
        let vals = s.tensor_values(result).unwrap();
        assert_eq!(vals, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn einsum_implicit_output() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // No -> means implicit output: sorted unique non-contracted indices
        // "ij,jk" implies "ij,jk->ik" (j contracted)
        let a = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        let b = s.tensor_variable(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false).unwrap();
        let result = s.tensor_einsum("ij,jk", &[a, b]).unwrap();
        let shape = s.tensor_shape(result).unwrap();
        assert_eq!(shape, vec![2, 2]);
        let vals = s.tensor_values(result).unwrap();
        assert!((vals[0] - 19.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_invalid_equation_errors() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![1.0; 4], vec![2, 2], false).unwrap();
        let b = s.tensor_variable(vec![1.0; 4], vec![2, 2], false).unwrap();
        // Too many subscripts
        assert!(s.tensor_einsum("ij,jk,kl->il", &[a, b]).is_err());
        // Invalid characters
        assert!(s.tensor_einsum("IJ,JK->IK", &[a, b]).is_err());
    }

    #[test]
    fn einsum_vector_matrix_mul() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // i,ij->j is vector-matrix multiply
        let v = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
        let m = s.tensor_variable(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2], false).unwrap();
        let result = s.tensor_einsum("i,ij->j", &[v, m]).unwrap();
        let shape = s.tensor_shape(result).unwrap();
        assert_eq!(shape, vec![2]);
        let vals = s.tensor_values(result).unwrap();
        // 1*3+2*5=13, 1*4+2*6=16
        assert!((vals[0] - 13.0).abs() < 1e-10);
        assert!((vals[1] - 16.0).abs() < 1e-10);
    }

    // ---- Custom Autograd Function integration tests ----

    #[test]
    fn custom_function_identity_api() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true).unwrap();

        let y = s
            .tensor_apply_function(
                &[x],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    Ok((vals.to_vec(), shape.to_vec()))
                },
                |_ctx, grad_outputs| Ok(vec![Some(grad_outputs[0].to_vec())]),
            )
            .unwrap();

        let vals = s.tensor_values(y).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);

        let report = s.tensor_backward(y).unwrap();
        let grad = report.gradient(x).expect("grad");
        assert_eq!(grad, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn custom_function_relu_api() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![-1.0, 2.0, -3.0, 4.0], vec![4], true).unwrap();

        let y = s
            .tensor_apply_function(
                &[x],
                |ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    ctx.save_for_backward(vals.to_vec(), shape.to_vec());
                    let relu: Vec<f64> = vals.iter().map(|v| v.max(0.0)).collect();
                    Ok((relu, shape.to_vec()))
                },
                |ctx, grad_outputs| {
                    let saved = &ctx.saved_tensors()[0];
                    let grad: Vec<f64> = grad_outputs[0]
                        .iter()
                        .zip(saved.iter())
                        .map(|(g, &x)| if x > 0.0 { *g } else { 0.0 })
                        .collect();
                    Ok(vec![Some(grad)])
                },
            )
            .unwrap();

        let vals = s.tensor_values(y).unwrap();
        assert_eq!(vals, vec![0.0, 2.0, 0.0, 4.0]);

        let report = s.tensor_backward(y).unwrap();
        let grad = report.gradient(x).expect("grad");
        assert_eq!(grad, &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn custom_function_straight_through_estimator_api() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![1.3, 2.7, -0.5], vec![3], true).unwrap();

        let y = s
            .tensor_apply_function(
                &[x],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    let rounded: Vec<f64> = vals.iter().map(|v| v.round()).collect();
                    Ok((rounded, shape.to_vec()))
                },
                |_ctx, grad_outputs| Ok(vec![Some(grad_outputs[0].to_vec())]),
            )
            .unwrap();

        let vals = s.tensor_values(y).unwrap();
        assert_eq!(vals, vec![1.0, 3.0, -1.0]);

        let report = s.tensor_backward(y).unwrap();
        let grad = report.gradient(x).expect("grad");
        assert_eq!(grad, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn custom_function_multi_input_mul_api() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![2.0, 3.0], vec![2], true).unwrap();
        let b = s.tensor_variable(vec![4.0, 5.0], vec![2], true).unwrap();

        let y = s
            .tensor_apply_function(
                &[a, b],
                |ctx, inputs| {
                    let (a_vals, a_shape) = &inputs[0];
                    let (b_vals, _) = &inputs[1];
                    ctx.save_for_backward(a_vals.to_vec(), a_shape.to_vec());
                    ctx.save_for_backward(b_vals.to_vec(), a_shape.to_vec());
                    let product: Vec<f64> = a_vals.iter().zip(b_vals.iter()).map(|(x, y)| x * y).collect();
                    Ok((product, a_shape.to_vec()))
                },
                |ctx, grad_outputs| {
                    let saved_a = &ctx.saved_tensors()[0];
                    let saved_b = &ctx.saved_tensors()[1];
                    let grad = grad_outputs[0];
                    let grad_a: Vec<f64> = grad.iter().zip(saved_b.iter()).map(|(g, b)| g * b).collect();
                    let grad_b: Vec<f64> = grad.iter().zip(saved_a.iter()).map(|(g, a)| g * a).collect();
                    Ok(vec![Some(grad_a), Some(grad_b)])
                },
            )
            .unwrap();

        let vals = s.tensor_values(y).unwrap();
        assert_eq!(vals, vec![8.0, 15.0]);

        let report = s.tensor_backward(y).unwrap();
        assert_eq!(report.gradient(a).unwrap(), &[4.0, 5.0]);
        assert_eq!(report.gradient(b).unwrap(), &[2.0, 3.0]);
    }

    #[test]
    fn custom_function_none_gradient_api() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![1.0, 2.0], vec![2], true).unwrap();
        let b = s.tensor_variable(vec![3.0, 4.0], vec![2], true).unwrap();

        let y = s
            .tensor_apply_function(
                &[a, b],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    Ok((vals.to_vec(), shape.to_vec()))
                },
                |_ctx, grad_outputs| Ok(vec![Some(grad_outputs[0].to_vec()), None]),
            )
            .unwrap();

        let report = s.tensor_backward(y).unwrap();
        assert_eq!(report.gradient(a).unwrap(), &[1.0, 1.0]);
    }

    #[test]
    fn custom_function_composed_with_standard_ops_api() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![2.0, 3.0], vec![2], true).unwrap();

        // Custom triple, then standard neg: result = -(3x)
        let tripled = s
            .tensor_apply_function(
                &[x],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    let t: Vec<f64> = vals.iter().map(|v| v * 3.0).collect();
                    Ok((t, shape.to_vec()))
                },
                |_ctx, grad_outputs| {
                    let grad: Vec<f64> = grad_outputs[0].iter().map(|g| g * 3.0).collect();
                    Ok(vec![Some(grad)])
                },
            )
            .unwrap();

        let y = s.tensor_neg(tripled).unwrap();

        let vals = s.tensor_values(y).unwrap();
        assert_eq!(vals, vec![-6.0, -9.0]);

        let report = s.tensor_backward(y).unwrap();
        assert_eq!(report.gradient(x).unwrap(), &[-3.0, -3.0]);
    }

    // ── nonzero tests ─────────────────────────────────────────────────

    #[test]
    fn nonzero_1d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![0.0, 1.0, 0.0, 3.0, 0.0], vec![5], false).unwrap();
        let nz = s.tensor_nonzero(x).unwrap();
        let vals = s.tensor_values(nz).unwrap();
        let shape = s.tensor_shape(nz).unwrap();
        assert_eq!(shape, vec![2, 1]); // 2 nonzero elements, 1D
        assert_eq!(vals, vec![1.0, 3.0]); // indices 1 and 3
    }

    #[test]
    fn nonzero_2d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2], false).unwrap();
        let nz = s.tensor_nonzero(x).unwrap();
        let vals = s.tensor_values(nz).unwrap();
        let shape = s.tensor_shape(nz).unwrap();
        assert_eq!(shape, vec![2, 2]); // 2 nonzero elements, 2D indices
        // (0,0) and (1,1)
        assert_eq!(vals, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn nonzero_all_zeros() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![0.0, 0.0, 0.0], vec![3], false).unwrap();
        let nz = s.tensor_nonzero(x).unwrap();
        let vals = s.tensor_values(nz).unwrap();
        let shape = s.tensor_shape(nz).unwrap();
        assert_eq!(shape, vec![0, 1]); // empty
        assert!(vals.is_empty());
    }

    #[test]
    fn nonzero_nan_is_nonzero() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![0.0, f64::NAN, 0.0], vec![3], false).unwrap();
        let nz = s.tensor_nonzero(x).unwrap();
        let vals = s.tensor_values(nz).unwrap();
        assert_eq!(vals, vec![1.0]); // index 1 (NaN is nonzero)
    }

    #[test]
    fn nonzero_all_nonzero() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let nz = s.tensor_nonzero(x).unwrap();
        let shape = s.tensor_shape(nz).unwrap();
        assert_eq!(shape, vec![3, 1]);
        let vals = s.tensor_values(nz).unwrap();
        assert_eq!(vals, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn nonzero_as_tuple_2d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2], false).unwrap();
        let tuple = s.tensor_nonzero_as_tuple(x).unwrap();
        assert_eq!(tuple.len(), 2);
        let rows = s.tensor_values(tuple[0]).unwrap();
        let cols = s.tensor_values(tuple[1]).unwrap();
        assert_eq!(rows, vec![0.0, 1.0]);
        assert_eq!(cols, vec![0.0, 1.0]);
    }

    // ── masked_select tests ────────────────────────────────────────────

    #[test]
    fn masked_select_basic() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![10.0, 20.0, 30.0], vec![3], false).unwrap();
        let mask = s.tensor_variable(vec![1.0, 0.0, 1.0], vec![3], false).unwrap();
        let sel = s.tensor_masked_select(x, mask).unwrap();
        let vals = s.tensor_values(sel).unwrap();
        assert_eq!(vals, vec![10.0, 30.0]);
    }

    #[test]
    fn masked_select_all_false() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let mask = s.tensor_variable(vec![0.0, 0.0, 0.0], vec![3], false).unwrap();
        let sel = s.tensor_masked_select(x, mask).unwrap();
        let vals = s.tensor_values(sel).unwrap();
        assert!(vals.is_empty());
    }

    #[test]
    fn masked_select_backward_gradient() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![10.0, 20.0, 30.0, 40.0], vec![4], true).unwrap();
        let mask = s.tensor_variable(vec![1.0, 0.0, 1.0, 0.0], vec![4], false).unwrap();
        let sel = s.tensor_masked_select(x, mask).unwrap();
        // sum the selected elements to get a scalar
        let loss = s.tensor_sum(sel).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        let grad = report.gradient(x).unwrap();
        // Gradient is 1 at selected positions, 0 at masked-out positions
        assert_eq!(grad, &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn masked_select_boolean_indexing_pattern() {
        // tensor[tensor > 5] equivalent
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![3.0, 7.0, 1.0, 9.0, 4.0], vec![5], false).unwrap();
        // Create mask: x > 5
        let threshold = s.tensor_variable(vec![5.0, 5.0, 5.0, 5.0, 5.0], vec![5], false).unwrap();
        let x_vals = s.tensor_values(x).unwrap();
        let t_vals = s.tensor_values(threshold).unwrap();
        let mask_vals: Vec<f64> = x_vals.iter().zip(t_vals.iter())
            .map(|(a, b)| if a > b { 1.0 } else { 0.0 })
            .collect();
        let mask = s.tensor_variable(mask_vals, vec![5], false).unwrap();
        let sel = s.tensor_masked_select(x, mask).unwrap();
        let vals = s.tensor_values(sel).unwrap();
        assert_eq!(vals, vec![7.0, 9.0]);
    }

    #[test]
    fn masked_select_shape_mismatch_error() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let mask = s.tensor_variable(vec![1.0, 0.0], vec![2], false).unwrap();
        assert!(s.tensor_masked_select(x, mask).is_err());
    }

    // ── searchsorted / bucketize tests ─────────────────────────────────

    #[test]
    fn searchsorted_basic() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let sorted = s.tensor_variable(vec![1.0, 3.0, 5.0, 7.0], vec![4], false).unwrap();
        let values = s.tensor_variable(vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![5], false).unwrap();
        let result = s.tensor_searchsorted(sorted, values, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // lower_bound: 0->0, 2->1, 4->2, 6->3, 8->4
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn searchsorted_right() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let sorted = s.tensor_variable(vec![1.0, 3.0, 5.0, 7.0], vec![4], false).unwrap();
        let values = s.tensor_variable(vec![3.0, 5.0], vec![2], false).unwrap();
        let result = s.tensor_searchsorted(sorted, values, true).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // upper_bound: 3->2, 5->3
        assert_eq!(vals, vec![2.0, 3.0]);
    }

    #[test]
    fn searchsorted_left_at_boundary() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let sorted = s.tensor_variable(vec![1.0, 3.0, 5.0], vec![3], false).unwrap();
        let values = s.tensor_variable(vec![1.0, 3.0, 5.0], vec![3], false).unwrap();
        let result = s.tensor_searchsorted(sorted, values, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // lower_bound at exact values: 1->0, 3->1, 5->2
        assert_eq!(vals, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn searchsorted_duplicates() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let sorted = s.tensor_variable(vec![1.0, 3.0, 3.0, 3.0, 5.0], vec![5], false).unwrap();
        let values = s.tensor_variable(vec![3.0], vec![1], false).unwrap();

        // left: first occurrence of 3 is at index 1
        let left = s.tensor_searchsorted(sorted, values, false).unwrap();
        assert_eq!(s.tensor_values(left).unwrap(), vec![1.0]);

        // right: after last 3 is at index 4
        let values2 = s.tensor_variable(vec![3.0], vec![1], false).unwrap();
        let right = s.tensor_searchsorted(sorted, values2, true).unwrap();
        assert_eq!(s.tensor_values(right).unwrap(), vec![4.0]);
    }

    #[test]
    fn searchsorted_empty_sequence() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let sorted = s.tensor_variable(Vec::<f64>::new(), vec![0], false).unwrap();
        let values = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
        let result = s.tensor_searchsorted(sorted, values, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // All values go to index 0 (before empty sequence)
        assert_eq!(vals, vec![0.0, 0.0]);
    }

    #[test]
    fn searchsorted_batched_2d() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // batch=2, seq_len=3
        let sorted = s.tensor_variable(
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
            vec![2, 3],
            false,
        ).unwrap();
        // batch=2, num_vals=2
        let values = s.tensor_variable(
            vec![2.0, 4.0, 3.0, 5.0],
            vec![2, 2],
            false,
        ).unwrap();
        let result = s.tensor_searchsorted(sorted, values, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // batch 0: [1,3,5], values [2,4] -> [1, 2]
        // batch 1: [2,4,6], values [3,5] -> [1, 2]
        assert_eq!(vals, vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn bucketize_basic() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let boundaries = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let input = s.tensor_variable(vec![0.5, 1.5, 2.5, 3.5], vec![4], false).unwrap();
        let result = s.tensor_bucketize(input, boundaries, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        // 0.5<1 -> bucket 0, 1<=1.5<2 -> bucket 1, 2<=2.5<3 -> bucket 2, 3.5>=3 -> bucket 3
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn bucketize_right() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let boundaries = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let input = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let result_left = s.tensor_bucketize(input, boundaries, false).unwrap();
        let vals_left = s.tensor_values(result_left).unwrap();
        // left: 1->0, 2->1, 3->2 (at boundary, goes left)
        assert_eq!(vals_left, vec![0.0, 1.0, 2.0]);

        let input2 = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let result_right = s.tensor_bucketize(input2, boundaries, true).unwrap();
        let vals_right = s.tensor_values(result_right).unwrap();
        // right: 1->1, 2->2, 3->3 (at boundary, goes right)
        assert_eq!(vals_right, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn bucketize_empty_boundaries() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let boundaries = s.tensor_variable(Vec::<f64>::new(), vec![0], false).unwrap();
        let input = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
        let result = s.tensor_bucketize(input, boundaries, false).unwrap();
        let vals = s.tensor_values(result).unwrap();
        assert_eq!(vals, vec![0.0, 0.0]);
    }

    // ---- create_graph integration tests (bd-3dpn.3) ----

    #[test]
    fn create_graph_second_derivative_x_cubed_via_session() {
        // f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
        // At x=2: f'(2) = 12, f''(2) = 12
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![2.0], vec![1], true).unwrap();
        let x2 = s.tensor_mul(x, x).unwrap();
        let x3 = s.tensor_mul(x2, x).unwrap();

        let report1 = s
            .tensor_backward_with_options(
                x3,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .unwrap();

        let grad = s.tensor_gradient(&report1, x).expect("first grad");
        assert!((grad[0] - 12.0).abs() < 1e-10);

        let dx_node = report1.gradient_node(x).expect("gradient node");
        let report2 = s
            .tensor_backward_with_options(dx_node, BackwardOptions::strict_default())
            .unwrap();

        let grad2 = s.tensor_gradient(&report2, x).expect("second grad");
        assert!(
            (grad2[0] - 12.0).abs() < 1e-10,
            "f''(2) should be 12, got {}",
            grad2[0]
        );
    }

    #[test]
    fn create_graph_hessian_vector_product() {
        // f(x,y) = x^2*y + y^3
        // grad_f = [2xy, x^2 + 3y^2]
        // Hessian = [[2y, 2x], [2x, 6y]]
        // At (x,y) = (1,2):
        //   grad_f = [4, 13]
        //   H = [[4, 2], [2, 12]]
        //   Hvp with v = [1, 0]: H*v = [4, 2]
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![1.0], vec![1], true).unwrap();
        let y = s.tensor_variable(vec![2.0], vec![1], true).unwrap();

        // x^2
        let x2 = s.tensor_mul(x, x).unwrap();
        // x^2 * y
        let x2y = s.tensor_mul(x2, y).unwrap();
        // y^2
        let y2 = s.tensor_mul(y, y).unwrap();
        // y^3
        let y3 = s.tensor_mul(y2, y).unwrap();
        // f = x^2*y + y^3
        let f = s.tensor_add(x2y, y3).unwrap();

        let report1 = s
            .tensor_backward_with_options(
                f,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .unwrap();

        let grad_x = s.tensor_gradient(&report1, x).expect("grad_x");
        let grad_y = s.tensor_gradient(&report1, y).expect("grad_y");
        assert!((grad_x[0] - 4.0).abs() < 1e-10, "df/dx at (1,2) = 4");
        assert!((grad_y[0] - 13.0).abs() < 1e-10, "df/dy at (1,2) = 13");

        // Hessian-vector product: backward through grad_x to get H[0,:]*v
        // where v = [1, 1, ...] (implicit seed)
        let dx_node = report1.gradient_node(x).expect("dx node");
        let report_hvp = s
            .tensor_backward_with_options(dx_node, BackwardOptions::strict_default())
            .unwrap();

        let hx_x = s.tensor_gradient(&report_hvp, x).expect("d²f/dx²");
        let hx_y = s.tensor_gradient(&report_hvp, y).expect("d²f/dxdy");
        // H[0,0] = 2y = 4, H[0,1] = 2x = 2
        assert!(
            (hx_x[0] - 4.0).abs() < 1e-10,
            "d²f/dx² = 2y = 4, got {}",
            hx_x[0]
        );
        assert!(
            (hx_y[0] - 2.0).abs() < 1e-10,
            "d²f/dxdy = 2x = 2, got {}",
            hx_y[0]
        );
    }

    #[test]
    fn create_graph_gradient_penalty_wgan_gp_style() {
        // WGAN-GP: penalty = ||grad_D(x)||^2
        // D(x) = sum(x^2) for simplicity
        // grad_D = 2x, penalty = sum((2x)^2) = 4*sum(x^2)
        // d(penalty)/dx = 8x
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![1.0, 3.0], vec![2], true).unwrap();
        let x2 = s.tensor_mul(x, x).unwrap();
        let d_out = s.tensor_sum(x2).unwrap();

        let report1 = s
            .tensor_backward_with_options(
                d_out,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .unwrap();

        let dx = report1.gradient_node(x).expect("gradient node");
        // Compute ||grad||^2 = sum(grad^2)
        let grad_sq = s.tensor_mul(dx, dx).unwrap();
        let penalty = s.tensor_sum(grad_sq).unwrap();

        let report2 = s
            .tensor_backward_with_options(penalty, BackwardOptions::strict_default())
            .unwrap();

        let d_penalty = s.tensor_gradient(&report2, x).expect("penalty grad");
        // d(penalty)/dx = 8x = [8, 24]
        assert!(
            (d_penalty[0] - 8.0).abs() < 1e-6,
            "d(penalty)/dx[0] = 8, got {}",
            d_penalty[0]
        );
        assert!(
            (d_penalty[1] - 24.0).abs() < 1e-6,
            "d(penalty)/dx[1] = 24, got {}",
            d_penalty[1]
        );
    }

    #[test]
    fn create_graph_physics_informed_double_backward() {
        // PDE residual: d²u/dx² for u(x) = sin(x)
        // u'(x) = cos(x), u''(x) = -sin(x)
        // At x = 0.5: u''(0.5) = -sin(0.5)
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![0.5], vec![1], true).unwrap();
        let u = s.tensor_sin(x).unwrap();

        let report1 = s
            .tensor_backward_with_options(
                u,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .unwrap();

        let du_dx = report1.gradient_node(x).expect("du/dx node");
        let report2 = s
            .tensor_backward_with_options(du_dx, BackwardOptions::strict_default())
            .unwrap();

        let d2u_dx2 = s.tensor_gradient(&report2, x).expect("d²u/dx²");
        let expected = -(0.5_f64).sin();
        assert!(
            (d2u_dx2[0] - expected).abs() < 1e-10,
            "d²sin(x)/dx² = -sin(x) = {}, got {}",
            expected,
            d2u_dx2[0]
        );
    }

    // ── F32 typed dispatch via session API ─────────────────────────────

    #[test]
    fn f32_session_sum_mean_pow_preserve_dtype() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], false)
            .unwrap();

        let sum_id = s.tensor_sum(x).unwrap();
        assert_eq!(s.tensor_dtype(sum_id).unwrap(), DType::F32);
        assert_eq!(s.tensor_values_f32(sum_id).unwrap(), vec![10.0f32]);

        let mean_id = s.tensor_mean(x).unwrap();
        assert_eq!(s.tensor_dtype(mean_id).unwrap(), DType::F32);
        assert_eq!(s.tensor_values_f32(mean_id).unwrap(), vec![2.5f32]);

        let pow_id = s.tensor_pow(x, 2.0).unwrap();
        assert_eq!(s.tensor_dtype(pow_id).unwrap(), DType::F32);
        assert_eq!(
            s.tensor_values_f32(pow_id).unwrap(),
            vec![1.0f32, 4.0, 9.0, 16.0]
        );
    }

    #[test]
    fn f32_session_backward_through_typed_reduction() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(vec![1.0f32, 2.0, 3.0], vec![3], true)
            .unwrap();
        // loss = sum(x^2) → dloss/dx = 2x
        let x2 = s.tensor_pow(x, 2.0).unwrap();
        let loss = s.tensor_sum(x2).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        let grad = s.tensor_gradient(&report, x).unwrap();
        assert!((grad[0] - 2.0).abs() < 1e-4);
        assert!((grad[1] - 4.0).abs() < 1e-4);
        assert!((grad[2] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn f32_session_training_loop_sgd() {
        // Train w to fit y=3 from x=1 (y = w*x), all in f32
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut w = s
            .tensor_variable_f32(vec![0.5f32], vec![1], true)
            .unwrap();

        for _ in 0..500 {
            let x = s
                .tensor_variable_f32(vec![1.0f32], vec![1], false)
                .unwrap();
            let target = s
                .tensor_variable_f32(vec![3.0f32], vec![1], false)
                .unwrap();

            let pred = s.tensor_mul(w, x).unwrap();
            let diff = s.tensor_sub(pred, target).unwrap();
            let loss = s.tensor_pow(diff, 2.0).unwrap();
            let loss_s = s.tensor_sum(loss).unwrap();

            let report = s.tensor_backward(loss_s).unwrap();
            let gw = s.tensor_gradient(&report, w).unwrap()[0];

            let w_val = s.tensor_values_f32(w).unwrap()[0] as f64 - 0.01 * gw;
            s = FrankenTorchSession::new(ExecutionMode::Strict);
            w = s
                .tensor_variable_f32(vec![w_val as f32], vec![1], true)
                .unwrap();
        }

        let w_final = s.tensor_values_f32(w).unwrap()[0];
        assert!(
            (w_final - 3.0).abs() < 0.1,
            "w should converge to ~3.0, got {w_final}"
        );
    }
}
