#![forbid(unsafe_code)]

use ft_autograd::{
    AutogradError, BackwardOptions, BackwardReport, ClampOperationEvent, NodeId, OperationEvent,
    PowOperationEvent, Tape, TensorBackwardReport, TensorClampOperationEvent,
    TensorNormalizeDimOperationEvent, TensorNodeId, TensorOperationEvent,
    TensorPowOperationEvent, TensorReductionDimOperationEvent, TensorReductionOperationEvent,
    TensorTape, TensorUnaryOperationEvent, UnaryOperationEvent,
};
use ft_dispatch::{
    ComparisonDispatchDecision, ComparisonOp, dispatch_scalar_comparison,
    dispatch_tensor_comparison_contiguous_f64,
};
use ft_core::{DenseTensor, ExecutionMode};
use ft_runtime::{EvidenceEntry, EvidenceKind, RuntimeContext};

#[derive(Debug, Clone)]
pub struct FrankenTorchSession {
    tape: Tape,
    tensor_tape: TensorTape,
    runtime: RuntimeContext,
}

impl FrankenTorchSession {
    #[must_use]
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            tape: Tape::new(),
            tensor_tape: TensorTape::new(),
            runtime: RuntimeContext::new(mode),
        }
    }

    #[must_use]
    pub fn mode(&self) -> ExecutionMode {
        self.runtime.mode()
    }

    pub fn set_mode(&mut self, mode: ExecutionMode) {
        self.runtime.set_mode(mode);
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

    pub fn zeros(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel = shape.iter().product::<usize>();
        self.tensor_tape.leaf(vec![0.0; numel], shape, requires_grad)
    }

    pub fn ones(
        &mut self,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel = shape.iter().product::<usize>();
        self.tensor_tape.leaf(vec![1.0; numel], shape, requires_grad)
    }

    pub fn full(
        &mut self,
        shape: Vec<usize>,
        fill_value: f64,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel = shape.iter().product::<usize>();
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
        let mut values = Vec::new();
        let mut current = start;
        if step > 0.0 {
            while current < end {
                values.push(current);
                current += step;
            }
        } else if step < 0.0 {
            while current > end {
                values.push(current);
                current += step;
            }
        }
        let n = values.len();
        self.tensor_tape.leaf(values, vec![n], requires_grad)
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

    pub fn tensor_neg(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.neg(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_abs(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.abs(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_exp(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.exp(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_log(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.log(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_relu(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.relu(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sigmoid(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sigmoid(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_tanh(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.tanh(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sin(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sin(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_cos(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.cos(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_tan(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.tan(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_floor(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.floor(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_ceil(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.ceil(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_round(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.round(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_log2(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.log2(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_log10(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.log10(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_log1p(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.log1p(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_expm1(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.expm1(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sign(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sign(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_trunc(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.trunc(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_frac(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.frac(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_asin(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.asin(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_acos(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.acos(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_atan(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.atan(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sinh(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sinh(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_cosh(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.cosh(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_gelu(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.gelu(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_silu(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
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

    pub fn tensor_elu(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.elu(input, self.mode())?;
        self.record_tensor_unary_operation(&event);
        Ok(out)
    }

    pub fn tensor_sqrt(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
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

    pub fn tensor_clamp(
        &mut self,
        input: TensorNodeId,
        min_val: f64,
        max_val: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) =
            self.tensor_tape
                .tensor_clamp(input, min_val, max_val, self.mode())?;
        self.record_tensor_clamp_operation(&event);
        Ok(out)
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

    pub fn tensor_sum(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (out, event) = self.tensor_tape.sum(input, self.mode())?;
        self.record_tensor_reduction_operation(&event);
        Ok(out)
    }

    pub fn tensor_mean(
        &mut self,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
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

    pub fn tensor_values(&self, node: TensorNodeId) -> Result<Vec<f64>, AutogradError> {
        self.tensor_tape.values(node)
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

    #[must_use]
    pub fn evidence(&self) -> &[EvidenceEntry] {
        self.runtime.ledger().entries()
    }

    #[must_use]
    pub fn evidence_len(&self) -> usize {
        self.runtime.ledger().len()
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

    fn record_tensor_reduction_dim_operation(
        &mut self,
        event: &TensorReductionDimOperationEvent,
    ) {
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

    fn record_tensor_normalize_dim_operation(
        &mut self,
        event: &TensorNormalizeDimOperationEvent,
    ) {
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
        let lhs_scalar = ft_core::ScalarTensor::new(lhs_value, ft_core::DType::F64, ft_core::Device::Cpu);
        let rhs_scalar = ft_core::ScalarTensor::new(rhs_value, ft_core::DType::F64, ft_core::Device::Cpu);

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

        let out = self.tensor_tape.leaf(
            outcome.values,
            lhs_meta.shape().to_vec(),
            false,
        )?;
        self.record_comparison_operation(op, &outcome.decision);
        Ok(out)
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
}

pub use ft_autograd::{
    BackwardOptions as DacBackwardOptions, BackwardReport as DacBackwardReport,
    NodeId as DacNodeId, ReentrantPolicy as DacReentrantPolicy,
    TensorBackwardReport as DacTensorBackwardReport, TensorNodeId as DacTensorNodeId,
};

#[cfg(test)]
mod tests {
    use ft_autograd::{BackwardOptions, ReentrantPolicy};
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
            message.contains("incompatible dispatch keyset"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains("AutogradCPU requires CPU backend availability"),
            "missing keyset incompatibility reason: {message}"
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
        assert_eq!(
            report.gradient(t),
            Some(vec![-1.0, -1.0, -1.0].as_slice())
        );
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
        assert_eq!(
            report.gradient(t),
            Some(vec![-1.0, 1.0, 0.0].as_slice())
        );
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
        assert!(
            has_exp_evidence,
            "exp should emit unary dispatch evidence"
        );
    }

    #[test]
    fn session_log_records_evidence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(1.0, true);
        let _log_x = session.log(x).expect("log should succeed");
        let has_log_evidence = session.evidence().iter().any(|entry| {
            entry.kind == EvidenceKind::Dispatch && entry.summary.contains("unary_op=Log")
        });
        assert!(
            has_log_evidence,
            "log should emit unary dispatch evidence"
        );
    }

    #[test]
    fn session_tensor_exp_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![0.0, 1.0, 2.0], vec![3], false)
            .expect("tensor creation should succeed");
        let exp_t = session
            .tensor_exp(t)
            .expect("tensor exp should succeed");
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
        let exp_t = session
            .tensor_exp(t)
            .expect("tensor exp should succeed");
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
        let log_t = session
            .tensor_log(t)
            .expect("tensor log should succeed");
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
        let log_t = session
            .tensor_log(t)
            .expect("tensor log should succeed");
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
        let exp_t = session
            .tensor_exp(t)
            .expect("tensor exp should succeed");
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
        let sum_t = session
            .tensor_sum(t)
            .expect("tensor sum should succeed");
        let values = session.tensor_values(sum_t).expect("values should succeed");
        assert_eq!(values, vec![10.0]);
    }

    #[test]
    fn session_tensor_sum_backward_produces_ones_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("tensor creation should succeed");
        let sum_t = session
            .tensor_sum(t)
            .expect("tensor sum should succeed");
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
        let mean_t = session
            .tensor_mean(t)
            .expect("tensor mean should succeed");
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
        let mean_t = session
            .tensor_mean(t)
            .expect("tensor mean should succeed");
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
        let _mean_t = session
            .tensor_mean(t)
            .expect("tensor mean should succeed");
        let has_evidence = session.evidence().iter().any(|entry| {
            entry.kind == EvidenceKind::Dispatch
                && entry.summary.contains("tensor_reduction_op=Mean")
        });
        assert!(
            has_evidence,
            "mean should emit reduction dispatch evidence"
        );
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
        let product = session
            .tensor_mul(x, y)
            .expect("tensor mul should succeed");
        let loss = session
            .tensor_sum(product)
            .expect("tensor sum should succeed");
        let values = session
            .tensor_values(loss)
            .expect("values should succeed");
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
            session.tensor_gradient(&report, x).expect("x grad should exist"),
            &[0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn session_tensor_sigmoid_returns_expected_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 10.0, -10.0], vec![3], true)
            .expect("tensor variable should succeed");
        let y = session.tensor_sigmoid(x).expect("tensor sigmoid should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert!((values[0] - 0.5).abs() < 1e-10);
        assert!((values[1] - 1.0).abs() < 1e-4);
        assert!(values[2].abs() < 1e-4);
    }

    #[test]
    fn session_tensor_sigmoid_backward_produces_expected_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0], vec![1], true)
            .expect("tensor variable should succeed");
        let y = session.tensor_sigmoid(x).expect("tensor sigmoid should succeed");
        let report = session
            .tensor_backward(y)
            .expect("tensor backward should succeed");
        let grad = session.tensor_gradient(&report, x).expect("x grad should exist");
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
        assert!((values[1] - 1.0).abs() < 1e-6);
        assert!((values[2] + 1.0).abs() < 1e-6);
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
        let grad = session.tensor_gradient(&report, x).expect("x grad should exist");
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
        assert!((session.value(y).expect("value")).abs() < 1e-15);
    }

    #[test]
    fn session_scalar_expm1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.variable(0.0, true);
        let y = session.expm1(x).expect("expm1");
        assert!((session.value(y).expect("value")).abs() < 1e-15);
    }

    #[test]
    fn session_tensor_log2() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![1.0, 4.0], vec![2], false).expect("leaf");
        let y = session.tensor_log2(x).expect("tensor_log2");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_expm1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![0.0, 1.0], vec![2], false).expect("leaf");
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
        let x = session.tensor_variable(vec![-3.0, 2.0, 5.0], vec![3], false).expect("leaf");
        let y = session.tensor_sign(x).expect("tensor_sign");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[-1.0, 1.0, 1.0]);
    }

    #[test]
    fn session_tensor_trunc() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![3.7, -2.3], vec![2], false).expect("leaf");
        let y = session.tensor_trunc(x).expect("tensor_trunc");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[3.0, -2.0]);
    }

    #[test]
    fn session_tensor_frac() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![3.7, -2.3], vec![2], false).expect("leaf");
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
        let x = session.tensor_variable(vec![0.0, 0.5], vec![2], false).expect("leaf");
        let y = session.tensor_asin(x).expect("tensor_asin");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 0.5_f64.asin()).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_atan() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![0.0, 1.0], vec![2], false).expect("leaf");
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
        let x = session.tensor_variable(vec![0.0, 1.0], vec![2], false).expect("leaf");
        let y = session.tensor_sinh(x).expect("tensor_sinh");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 1.0_f64.sinh()).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_cosh() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![0.0, 1.0], vec![2], false).expect("leaf");
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
        let x = session.tensor_variable(vec![0.0, 1.0], vec![2], false).expect("leaf");
        let y = session.tensor_gelu(x).expect("tensor_gelu");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - gelu_val(1.0)).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_silu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![0.0, 1.0], vec![2], false).expect("leaf");
        let y = session.tensor_silu(x).expect("tensor_silu");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        let expected = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((vals[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_leaky_relu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![2.0, -3.0], vec![2], false).expect("leaf");
        let y = session.tensor_leaky_relu(x).expect("tensor_leaky_relu");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals[0], 2.0);
        assert!((vals[1] - (-0.03)).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_elu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![2.0, -1.0], vec![2], false).expect("leaf");
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
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false).expect("leaf");
        let y = session.tensor_prod_dim(x, 0).expect("prod_dim 0");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn session_tensor_prod_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false).expect("leaf");
        let y = session.tensor_prod_dim(x, 1).expect("prod_dim 1");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, &[6.0, 120.0]);
    }

    #[test]
    fn session_tensor_var_dim0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], false).expect("leaf");
        let y = session.tensor_var_dim(x, 0).expect("var_dim 0");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - 4.0).abs() < 1e-12);
        assert!((vals[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_var_dim1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false).expect("leaf");
        let y = session.tensor_var_dim(x, 1).expect("var_dim 1");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_std_dim0() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], false).expect("leaf");
        let y = session.tensor_std_dim(x, 0).expect("std_dim 0");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_prod_dim_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![2.0, 3.0, 4.0], vec![1, 3], true).expect("leaf");
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
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true).expect("leaf");
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
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false).expect("leaf");
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
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false).expect("leaf");
        let y = session.tensor_log_softmax(x, 1).expect("log_softmax");
        let vals = session.tensor_values(y).expect("values");
        // exp of log_softmax should sum to 1
        let sum: f64 = vals.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn session_tensor_softmax_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true).expect("leaf");
        let y = session.tensor_softmax(x, 1).expect("softmax");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // Sum of softmax backward gradients (with all-ones incoming) should be 0
        // because softmax is invariant to constant shift
        let grad_sum: f64 = grads.iter().sum();
        assert!(grad_sum.abs() < 1e-12, "softmax grad sum should be 0, got {grad_sum}");
    }

    #[test]
    fn session_tensor_log_softmax_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], true).expect("leaf");
        let y = session.tensor_log_softmax(x, 1).expect("log_softmax");
        let report = session.tensor_backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // Sum of log_softmax backward gradients should be 0
        let grad_sum: f64 = grads.iter().sum();
        assert!(grad_sum.abs() < 1e-12, "log_softmax grad sum should be 0, got {grad_sum}");
    }
}
