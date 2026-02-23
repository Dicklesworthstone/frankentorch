#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use ft_core::{
    DType, DenseTensor, DenseTensorError, Device, ExecutionMode, ScalarTensor, TensorMeta,
};
use ft_dispatch::{
    BinaryOp, ClampDispatchDecision, DispatchDecision, DispatchError, DispatchKeyError,
    JoinDispatchDecision, JoinOp, NormalizeDimDispatchDecision, NormalizeOp, PowDispatchDecision,
    ReductionDimDispatchDecision, ReductionDispatchDecision, ReductionOp, ScanDimDispatchDecision,
    ScanOp, SortDispatchDecision, TopKDispatchDecision, UnaryDispatchDecision, UnaryOp,
    dispatch_scalar_binary, dispatch_scalar_clamp, dispatch_scalar_pow, dispatch_scalar_unary,
    dispatch_tensor_binary_contiguous_f64, dispatch_tensor_clamp_contiguous_f64,
    dispatch_tensor_join_contiguous_f64, dispatch_tensor_normalize_dim_contiguous_f64,
    dispatch_tensor_pow_contiguous_f64, dispatch_tensor_reduction_contiguous_f64,
    dispatch_tensor_reduction_dim_contiguous_f64, dispatch_tensor_scan_dim_contiguous_f64,
    dispatch_tensor_sort_contiguous_f64, dispatch_tensor_topk_contiguous_f64,
    dispatch_tensor_unary_contiguous_f64,
};
use ft_kernel_cpu::{
    argmax_dim_tensor_contiguous_f64, argmin_dim_tensor_contiguous_f64,
    gather_tensor_contiguous_f64, index_select_tensor_contiguous_f64,
    masked_fill_tensor_contiguous_f64, max_dim_tensor_contiguous_f64,
    min_dim_tensor_contiguous_f64, scatter_tensor_contiguous_f64, where_tensor_contiguous_f64,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorNodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq)]
enum NodeOp {
    Leaf,
    Add {
        lhs: NodeId,
        rhs: NodeId,
    },
    Sub {
        lhs: NodeId,
        rhs: NodeId,
    },
    Div {
        lhs: NodeId,
        rhs: NodeId,
    },
    Mul {
        lhs: NodeId,
        rhs: NodeId,
    },
    Neg {
        input: NodeId,
    },
    Abs {
        input: NodeId,
    },
    Exp {
        input: NodeId,
    },
    Log {
        input: NodeId,
    },
    Relu {
        input: NodeId,
    },
    Sigmoid {
        input: NodeId,
    },
    Tanh {
        input: NodeId,
    },
    Sin {
        input: NodeId,
    },
    Cos {
        input: NodeId,
    },
    Tan {
        input: NodeId,
    },
    Floor {
        input: NodeId,
    },
    Ceil {
        input: NodeId,
    },
    Round {
        input: NodeId,
    },
    Log2 {
        input: NodeId,
    },
    Log10 {
        input: NodeId,
    },
    Log1p {
        input: NodeId,
    },
    Expm1 {
        input: NodeId,
    },
    Sign {
        input: NodeId,
    },
    Trunc {
        input: NodeId,
    },
    Frac {
        input: NodeId,
    },
    Asin {
        input: NodeId,
    },
    Acos {
        input: NodeId,
    },
    Atan {
        input: NodeId,
    },
    Sinh {
        input: NodeId,
    },
    Cosh {
        input: NodeId,
    },
    Gelu {
        input: NodeId,
    },
    Silu {
        input: NodeId,
    },
    LeakyRelu {
        input: NodeId,
    },
    Elu {
        input: NodeId,
    },
    Sqrt {
        input: NodeId,
    },
    Reciprocal {
        input: NodeId,
    },
    Pow {
        input: NodeId,
        exponent: f64,
    },
    Min {
        lhs: NodeId,
        rhs: NodeId,
    },
    Max {
        lhs: NodeId,
        rhs: NodeId,
    },
    Clamp {
        input: NodeId,
        min_val: f64,
        max_val: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct Node {
    tensor: ScalarTensor,
    requires_grad: bool,
    op: NodeOp,
}

#[derive(Debug, Clone, PartialEq)]
enum TensorNodeOp {
    Leaf,
    Add {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Sub {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Div {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Mul {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    MatMul {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Dot {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Outer {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Bmm {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Neg {
        input: TensorNodeId,
    },
    Abs {
        input: TensorNodeId,
    },
    Exp {
        input: TensorNodeId,
    },
    Log {
        input: TensorNodeId,
    },
    Relu {
        input: TensorNodeId,
    },
    Sigmoid {
        input: TensorNodeId,
    },
    Tanh {
        input: TensorNodeId,
    },
    Sin {
        input: TensorNodeId,
    },
    Cos {
        input: TensorNodeId,
    },
    Tan {
        input: TensorNodeId,
    },
    Floor {
        input: TensorNodeId,
    },
    Ceil {
        input: TensorNodeId,
    },
    Round {
        input: TensorNodeId,
    },
    Log2 {
        input: TensorNodeId,
    },
    Log10 {
        input: TensorNodeId,
    },
    Log1p {
        input: TensorNodeId,
    },
    Expm1 {
        input: TensorNodeId,
    },
    Sign {
        input: TensorNodeId,
    },
    Trunc {
        input: TensorNodeId,
    },
    Frac {
        input: TensorNodeId,
    },
    Asin {
        input: TensorNodeId,
    },
    Acos {
        input: TensorNodeId,
    },
    Atan {
        input: TensorNodeId,
    },
    Sinh {
        input: TensorNodeId,
    },
    Cosh {
        input: TensorNodeId,
    },
    Gelu {
        input: TensorNodeId,
    },
    Silu {
        input: TensorNodeId,
    },
    LeakyRelu {
        input: TensorNodeId,
    },
    Elu {
        input: TensorNodeId,
    },
    Sqrt {
        input: TensorNodeId,
    },
    Reciprocal {
        input: TensorNodeId,
    },
    Pow {
        input: TensorNodeId,
        exponent: f64,
    },
    Min {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Max {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Clamp {
        input: TensorNodeId,
        min_val: f64,
        max_val: f64,
    },
    Trace {
        input: TensorNodeId,
        input_shape: Vec<usize>,
    },
    Sum {
        input: TensorNodeId,
        input_numel: usize,
    },
    Mean {
        input: TensorNodeId,
        input_numel: usize,
    },
    SumDim {
        input: TensorNodeId,
        dim: usize,
        input_shape: Vec<usize>,
    },
    MeanDim {
        input: TensorNodeId,
        dim: usize,
        input_shape: Vec<usize>,
    },
    ProdDim {
        input: TensorNodeId,
        dim: usize,
        input_shape: Vec<usize>,
    },
    VarDim {
        input: TensorNodeId,
        dim: usize,
        input_shape: Vec<usize>,
    },
    StdDim {
        input: TensorNodeId,
        dim: usize,
        input_shape: Vec<usize>,
    },
    CumSum {
        input: TensorNodeId,
        dim: usize,
    },
    CumProd {
        input: TensorNodeId,
        dim: usize,
    },
    Where {
        condition: TensorNodeId,
        x: TensorNodeId,
        y: TensorNodeId,
    },
    Sort {
        input: TensorNodeId,
        dim: usize,
        indices: Vec<usize>,
        input_shape: Vec<usize>,
    },
    TopK {
        input: TensorNodeId,
        dim: usize,
        k: usize,
        indices: Vec<usize>,
        input_shape: Vec<usize>,
    },
    Softmax {
        input: TensorNodeId,
        dim: usize,
    },
    LogSoftmax {
        input: TensorNodeId,
        dim: usize,
    },
    Cat {
        inputs: Vec<TensorNodeId>,
        dim: usize,
        input_dim_sizes: Vec<usize>,
    },
    Stack {
        inputs: Vec<TensorNodeId>,
        dim: usize,
    },
    Reshape {
        input: TensorNodeId,
        original_shape: Vec<usize>,
    },
    Squeeze {
        input: TensorNodeId,
        dim: usize,
    },
    Unsqueeze {
        input: TensorNodeId,
        dim: usize,
    },
    Transpose {
        input: TensorNodeId,
        dim0: usize,
        dim1: usize,
    },
    Permute {
        input: TensorNodeId,
        dims: Vec<usize>,
    },
    Narrow {
        input: TensorNodeId,
        dim: usize,
        start: usize,
        original_shape: Vec<usize>,
    },
    Expand {
        input: TensorNodeId,
        original_shape: Vec<usize>,
    },
    Split {
        input: TensorNodeId,
        chunk_index: usize,
        dim: usize,
        start: usize,
        original_shape: Vec<usize>,
    },
    MaxDim {
        input: TensorNodeId,
        dim: usize,
        input_shape: Vec<usize>,
        indices: Vec<f64>,
    },
    MinDim {
        input: TensorNodeId,
        dim: usize,
        input_shape: Vec<usize>,
        indices: Vec<f64>,
    },
    IndexSelect {
        input: TensorNodeId,
        dim: usize,
        indices: Vec<f64>,
        input_shape: Vec<usize>,
    },
    Gather {
        input: TensorNodeId,
        dim: usize,
        index: Vec<f64>,
        index_shape: Vec<usize>,
        input_shape: Vec<usize>,
    },
    Scatter {
        input: TensorNodeId,
        dim: usize,
        index: Vec<f64>,
        index_shape: Vec<usize>,
        input_shape: Vec<usize>,
    },
    Flip {
        input: TensorNodeId,
        dims: Vec<usize>,
    },
    Repeat {
        input: TensorNodeId,
        original_shape: Vec<usize>,
        repeats: Vec<usize>,
    },
    Roll {
        input: TensorNodeId,
        shift: isize,
        dim: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct TensorNode {
    tensor: DenseTensor,
    requires_grad: bool,
    op: TensorNodeOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReentrantPolicy {
    StrictFail,
    HardenedBoundedFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackwardOptions {
    pub max_reentrant_depth: usize,
    pub current_reentrant_depth: usize,
    pub policy: ReentrantPolicy,
}

impl BackwardOptions {
    #[must_use]
    pub const fn strict_default() -> Self {
        Self {
            max_reentrant_depth: 0,
            current_reentrant_depth: 0,
            policy: ReentrantPolicy::StrictFail,
        }
    }

    #[must_use]
    pub const fn hardened_default() -> Self {
        Self {
            max_reentrant_depth: 2,
            current_reentrant_depth: 0,
            policy: ReentrantPolicy::HardenedBoundedFallback,
        }
    }

    #[must_use]
    pub const fn for_mode(mode: ExecutionMode) -> Self {
        match mode {
            ExecutionMode::Strict => Self::strict_default(),
            ExecutionMode::Hardened => Self::hardened_default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerTelemetry {
    pub execution_order: Vec<NodeId>,
    pub queue_pushes: usize,
    pub queue_pops: usize,
    pub max_queue_len: usize,
    pub dependency_snapshot: Vec<usize>,
    pub reentrant_depth: usize,
    pub reentrant_guard_triggered: bool,
    pub hardened_fallback_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReadyTask {
    node: NodeId,
}

impl Ord for ReadyTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.node.0.cmp(&other.node.0)
    }
}

impl PartialOrd for ReadyTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Default)]
struct ReadyQueue {
    heap: BinaryHeap<ReadyTask>,
    pushes: usize,
    pops: usize,
    max_len: usize,
}

impl ReadyQueue {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            pushes: 0,
            pops: 0,
            max_len: 0,
        }
    }

    fn push(&mut self, node: NodeId) {
        self.heap.push(ReadyTask { node });
        self.pushes += 1;
        self.max_len = self.max_len.max(self.heap.len());
    }

    fn pop(&mut self) -> Option<NodeId> {
        let next = self.heap.pop().map(|task| task.node);
        if next.is_some() {
            self.pops += 1;
        }
        next
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorReadyTask {
    node: TensorNodeId,
}

impl Ord for TensorReadyTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.node.0.cmp(&other.node.0)
    }
}

impl PartialOrd for TensorReadyTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Default)]
struct TensorReadyQueue {
    heap: BinaryHeap<TensorReadyTask>,
    pushes: usize,
    pops: usize,
    max_len: usize,
}

impl TensorReadyQueue {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            pushes: 0,
            pops: 0,
            max_len: 0,
        }
    }

    fn push(&mut self, node: TensorNodeId) {
        self.heap.push(TensorReadyTask { node });
        self.pushes += 1;
        self.max_len = self.max_len.max(self.heap.len());
    }

    fn pop(&mut self) -> Option<TensorNodeId> {
        let next = self.heap.pop().map(|task| task.node);
        if next.is_some() {
            self.pops += 1;
        }
        next
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationEvent {
    pub op: BinaryOp,
    pub lhs: NodeId,
    pub rhs: NodeId,
    pub out: NodeId,
    pub decision: DispatchDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnaryOperationEvent {
    pub op: UnaryOp,
    pub input: NodeId,
    pub out: NodeId,
    pub decision: UnaryDispatchDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorOperationEvent {
    pub op: BinaryOp,
    pub lhs: TensorNodeId,
    pub rhs: TensorNodeId,
    pub out: TensorNodeId,
    pub decision: DispatchDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorUnaryOperationEvent {
    pub op: UnaryOp,
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub decision: UnaryDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorReductionOperationEvent {
    pub op: ReductionOp,
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub decision: ReductionDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorReductionDimOperationEvent {
    pub op: ReductionOp,
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub dim: usize,
    pub decision: ReductionDimDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorScanDimOperationEvent {
    pub op: ScanOp,
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub dim: usize,
    pub decision: ScanDimDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorNormalizeDimOperationEvent {
    pub op: NormalizeOp,
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub dim: usize,
    pub decision: NormalizeDimDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorJoinOperationEvent {
    pub op: JoinOp,
    pub inputs: Vec<TensorNodeId>,
    pub out: TensorNodeId,
    pub dim: usize,
    pub decision: JoinDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PowOperationEvent {
    pub input: NodeId,
    pub out: NodeId,
    pub exponent: f64,
    pub decision: PowDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorPowOperationEvent {
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub exponent: f64,
    pub decision: PowDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClampOperationEvent {
    pub input: NodeId,
    pub out: NodeId,
    pub min_val: f64,
    pub max_val: f64,
    pub decision: ClampDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorClampOperationEvent {
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub min_val: f64,
    pub max_val: f64,
    pub decision: ClampDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorSortOperationEvent {
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub dim: usize,
    pub descending: bool,
    pub decision: SortDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorTopKOperationEvent {
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub k: usize,
    pub dim: usize,
    pub decision: TopKDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackwardStep {
    pub node: NodeId,
    pub incoming_grad: f64,
    pub rule: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackwardReport {
    gradients: Vec<Option<f64>>,
    pub steps: Vec<BackwardStep>,
    pub telemetry: SchedulerTelemetry,
}

impl BackwardReport {
    #[must_use]
    pub fn gradient(&self, node: NodeId) -> Option<f64> {
        self.gradients.get(node.0).copied().flatten()
    }

    #[must_use]
    pub fn gradients(&self) -> &[Option<f64>] {
        &self.gradients
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSchedulerTelemetry {
    pub execution_order: Vec<TensorNodeId>,
    pub queue_pushes: usize,
    pub queue_pops: usize,
    pub max_queue_len: usize,
    pub dependency_snapshot: Vec<usize>,
    pub reentrant_depth: usize,
    pub reentrant_guard_triggered: bool,
    pub hardened_fallback_used: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorBackwardStep {
    pub node: TensorNodeId,
    pub incoming_grad_len: usize,
    pub rule: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorBackwardReport {
    gradients: Vec<Option<Vec<f64>>>,
    pub steps: Vec<TensorBackwardStep>,
    pub telemetry: TensorSchedulerTelemetry,
}

impl TensorBackwardReport {
    #[must_use]
    pub fn gradient(&self, node: TensorNodeId) -> Option<&[f64]> {
        self.gradients
            .get(node.0)
            .and_then(|entry| entry.as_deref())
    }

    #[must_use]
    pub fn gradients(&self) -> &[Option<Vec<f64>>] {
        &self.gradients
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AutogradError {
    UnknownNode(NodeId),
    UnknownTensorNode(TensorNodeId),
    RootDoesNotRequireGrad {
        node: NodeId,
    },
    TensorRootDoesNotRequireGrad {
        node: TensorNodeId,
    },
    Dispatch(DispatchError),
    DenseTensor(DenseTensorError),
    ReentrantDepthExceeded {
        current: usize,
        max: usize,
    },
    DependencyUnderflow {
        node: NodeId,
    },
    TensorDependencyUnderflow {
        node: TensorNodeId,
    },
    TensorGradientShapeMismatch {
        node: TensorNodeId,
        expected: usize,
        actual: usize,
    },
    TensorMatMulShapeMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
}

impl fmt::Display for AutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNode(node) => write!(f, "unknown node id {}", node.0),
            Self::UnknownTensorNode(node) => write!(f, "unknown tensor node id {}", node.0),
            Self::RootDoesNotRequireGrad { node } => {
                write!(
                    f,
                    "cannot run backward: root node {} does not require grad",
                    node.0
                )
            }
            Self::TensorRootDoesNotRequireGrad { node } => write!(
                f,
                "cannot run tensor backward: root node {} does not require grad",
                node.0
            ),
            Self::Dispatch(error) => write!(f, "dispatch failure: {error}"),
            Self::DenseTensor(error) => write!(f, "dense tensor failure: {error}"),
            Self::ReentrantDepthExceeded { current, max } => write!(
                f,
                "reentrant backward depth exceeded: current={current} max={max}"
            ),
            Self::DependencyUnderflow { node } => {
                write!(f, "dependency scheduler underflow at node {}", node.0)
            }
            Self::TensorDependencyUnderflow { node } => {
                write!(
                    f,
                    "tensor dependency scheduler underflow at node {}",
                    node.0
                )
            }
            Self::TensorGradientShapeMismatch {
                node,
                expected,
                actual,
            } => write!(
                f,
                "tensor gradient shape mismatch at node {}: expected={expected}, actual={actual}",
                node.0
            ),
            Self::TensorMatMulShapeMismatch { lhs, rhs } => {
                write!(f, "tensor matmul shape mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
        }
    }
}

impl std::error::Error for AutogradError {}

impl From<DenseTensorError> for AutogradError {
    fn from(value: DenseTensorError) -> Self {
        Self::DenseTensor(value)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn leaf(&mut self, value: f64, requires_grad: bool) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: ScalarTensor::new(value, DType::F64, Device::Cpu),
            requires_grad,
            op: NodeOp::Leaf,
        });
        id
    }

    pub fn value(&self, node: NodeId) -> Result<f64, AutogradError> {
        Ok(self.node(node)?.tensor.value())
    }

    pub fn add(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        self.binary(BinaryOp::Add, lhs, rhs, mode)
    }

    pub fn mul(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        self.binary(BinaryOp::Mul, lhs, rhs, mode)
    }

    pub fn sub(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        self.binary(BinaryOp::Sub, lhs, rhs, mode)
    }

    pub fn div(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        self.binary(BinaryOp::Div, lhs, rhs, mode)
    }

    pub fn neg(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Neg, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Neg { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Neg,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn abs(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Abs, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Abs { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Abs,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn exp(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Exp, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Exp { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Exp,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn log(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Log, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Log { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Log,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn relu(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Relu, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Relu { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Relu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sigmoid(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Sigmoid, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Sigmoid { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Sigmoid,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn tanh(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Tanh, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Tanh { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Tanh,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sin(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Sin, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Sin { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Sin,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn cos(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Cos, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Cos { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Cos,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn tan(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Tan, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Tan { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Tan,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn floor(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Floor, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Floor { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Floor,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn ceil(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Ceil, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Ceil { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Ceil,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn round(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Round, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Round { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Round,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn log2(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Log2, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Log2 { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Log2,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn log10(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Log10, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Log10 { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Log10,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn log1p(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Log1p, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Log1p { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Log1p,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn expm1(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Expm1, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Expm1 { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Expm1,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sign(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Sign, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Sign { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Sign,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn trunc(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Trunc, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Trunc { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Trunc,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn frac(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Frac, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Frac { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Frac,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn asin(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Asin, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Asin { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Asin,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn acos(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Acos, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Acos { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Acos,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn atan(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Atan, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Atan { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Atan,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sinh(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Sinh, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Sinh { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Sinh,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn cosh(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Cosh, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Cosh { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Cosh,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn gelu(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Gelu, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Gelu { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Gelu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn silu(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Silu, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Silu { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Silu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn leaky_relu(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::LeakyRelu, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::LeakyRelu { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::LeakyRelu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn elu(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Elu, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Elu { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Elu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sqrt(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Sqrt, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Sqrt { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Sqrt,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn reciprocal(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Reciprocal, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Reciprocal { input },
        });

        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Reciprocal,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn pow(
        &mut self,
        input: NodeId,
        exponent: f64,
        mode: ExecutionMode,
    ) -> Result<(NodeId, PowOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome = dispatch_scalar_pow(mode, &input_node.tensor, exponent, requires_grad)
                .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Pow { input, exponent },
        });

        Ok((
            out,
            PowOperationEvent {
                input,
                out,
                exponent,
                decision: outcome.decision,
            },
        ))
    }

    pub fn min(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad = lhs_node.requires_grad || rhs_node.requires_grad;
            let outcome = dispatch_scalar_binary(
                BinaryOp::Min,
                mode,
                &lhs_node.tensor,
                &rhs_node.tensor,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Min { lhs, rhs },
        });

        Ok((
            out,
            OperationEvent {
                op: BinaryOp::Min,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn max(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad = lhs_node.requires_grad || rhs_node.requires_grad;
            let outcome = dispatch_scalar_binary(
                BinaryOp::Max,
                mode,
                &lhs_node.tensor,
                &rhs_node.tensor,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Max { lhs, rhs },
        });

        Ok((
            out,
            OperationEvent {
                op: BinaryOp::Max,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn clamp(
        &mut self,
        input: NodeId,
        min_val: f64,
        max_val: f64,
        mode: ExecutionMode,
    ) -> Result<(NodeId, ClampOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let outcome =
                dispatch_scalar_clamp(mode, &input_node.tensor, min_val, max_val, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Clamp {
                input,
                min_val,
                max_val,
            },
        });

        Ok((
            out,
            ClampOperationEvent {
                input,
                out,
                min_val,
                max_val,
                decision: outcome.decision,
            },
        ))
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        if matches!(op, BinaryOp::MatMul) {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "matmul is unsupported for scalar tensors",
                },
            )));
        }

        let (requires_grad, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad = lhs_node.requires_grad || rhs_node.requires_grad;
            let outcome =
                dispatch_scalar_binary(op, mode, &lhs_node.tensor, &rhs_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: match op {
                BinaryOp::Add => NodeOp::Add { lhs, rhs },
                BinaryOp::Sub => NodeOp::Sub { lhs, rhs },
                BinaryOp::Div => NodeOp::Div { lhs, rhs },
                BinaryOp::Mul => NodeOp::Mul { lhs, rhs },
                BinaryOp::Min => NodeOp::Min { lhs, rhs },
                BinaryOp::Max => NodeOp::Max { lhs, rhs },
                BinaryOp::MatMul | BinaryOp::Dot | BinaryOp::Outer | BinaryOp::Bmm => {
                    unreachable!(
                        "scalar matmul/dot/outer/bmm should be rejected before node creation"
                    )
                }
            },
        });

        Ok((
            out,
            OperationEvent {
                op,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn backward(&self, root: NodeId) -> Result<BackwardReport, AutogradError> {
        self.backward_with_options(root, BackwardOptions::strict_default())
    }

    pub fn backward_with_options(
        &self,
        root: NodeId,
        options: BackwardOptions,
    ) -> Result<BackwardReport, AutogradError> {
        if root.0 >= self.nodes.len() {
            return Err(AutogradError::UnknownNode(root));
        }
        if !self.nodes[root.0].requires_grad {
            return Err(AutogradError::RootDoesNotRequireGrad { node: root });
        }

        let mut reentrant_guard_triggered = false;
        let mut hardened_fallback_used = false;
        if options.current_reentrant_depth > options.max_reentrant_depth {
            match options.policy {
                ReentrantPolicy::StrictFail => {
                    return Err(AutogradError::ReentrantDepthExceeded {
                        current: options.current_reentrant_depth,
                        max: options.max_reentrant_depth,
                    });
                }
                ReentrantPolicy::HardenedBoundedFallback => {
                    reentrant_guard_triggered = true;
                    hardened_fallback_used = true;
                }
            }
        }

        let reentrant_depth = options
            .current_reentrant_depth
            .min(options.max_reentrant_depth);
        let reachable = self.compute_reachable(root)?;
        let mut pending = self.compute_dependencies(&reachable)?;
        let dependency_snapshot = pending.clone();

        let mut grads = self
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, _)| if reachable[idx] { 0.0 } else { f64::NAN })
            .collect::<Vec<_>>();
        grads[root.0] = 1.0;

        let mut queue = ReadyQueue::with_capacity(self.nodes.len().max(1));
        queue.push(root);

        let mut steps = Vec::with_capacity(self.nodes.len());
        let mut execution_order = Vec::with_capacity(self.nodes.len());

        while let Some(node_id) = queue.pop() {
            let incoming = grads[node_id.0];
            execution_order.push(node_id);

            match self.nodes[node_id.0].op {
                NodeOp::Leaf => {
                    if self.nodes[node_id.0].requires_grad {
                        steps.push(BackwardStep {
                            node: node_id,
                            incoming_grad: incoming,
                            rule: "leaf",
                        });
                    }
                }
                NodeOp::Add { lhs, rhs } => {
                    grads[lhs.0] += incoming;
                    grads[rhs.0] += incoming;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(a+b)/da=1; d(a+b)/db=1",
                    });
                }
                NodeOp::Sub { lhs, rhs } => {
                    grads[lhs.0] += incoming;
                    grads[rhs.0] -= incoming;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(a-b)/da=1; d(a-b)/db=-1",
                    });
                }
                NodeOp::Div { lhs, rhs } => {
                    let lhs_value = self.nodes[lhs.0].tensor.value();
                    let rhs_value = self.nodes[rhs.0].tensor.value();
                    grads[lhs.0] += incoming / rhs_value;
                    grads[rhs.0] -= incoming * lhs_value / (rhs_value * rhs_value);

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(a/b)/da=1/b; d(a/b)/db=-(a/b^2)",
                    });
                }
                NodeOp::Mul { lhs, rhs } => {
                    let lhs_value = self.nodes[lhs.0].tensor.value();
                    let rhs_value = self.nodes[rhs.0].tensor.value();
                    grads[lhs.0] += incoming * rhs_value;
                    grads[rhs.0] += incoming * lhs_value;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(a*b)/da=b; d(a*b)/db=a",
                    });
                }
                NodeOp::Neg { input } => {
                    grads[input.0] -= incoming;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(-x)/dx=-1",
                    });
                }
                NodeOp::Abs { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    let sign = if input_value.is_nan() {
                        f64::NAN
                    } else if input_value > 0.0 {
                        1.0
                    } else if input_value < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    grads[input.0] += incoming * sign;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d|x|/dx=sign(x)",
                    });
                }
                NodeOp::Exp { input } => {
                    let output_value = self.nodes[node_id.0].tensor.value();
                    grads[input.0] += incoming * output_value;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(exp(x))/dx=exp(x)",
                    });
                }
                NodeOp::Log { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    grads[input.0] += incoming / input_value;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(ln(x))/dx=1/x",
                    });
                }
                NodeOp::Relu { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    grads[input.0] += if input_value.is_nan() {
                        f64::NAN
                    } else if input_value > 0.0 {
                        incoming
                    } else {
                        0.0
                    };

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(relu(x))/dx=1 if x>0 else 0",
                    });
                }
                NodeOp::Sigmoid { input } => {
                    let output_value = self.nodes[node_id.0].tensor.value();
                    grads[input.0] += incoming * output_value * (1.0 - output_value);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(sigmoid(x))/dx=sigmoid(x)*(1-sigmoid(x))",
                    });
                }
                NodeOp::Tanh { input } => {
                    let output_value = self.nodes[node_id.0].tensor.value();
                    grads[input.0] += incoming * (1.0 - output_value * output_value);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(tanh(x))/dx=1-tanh(x)^2",
                    });
                }
                NodeOp::Sin { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    grads[input.0] += incoming * input_value.cos();

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(sin(x))/dx=cos(x)",
                    });
                }
                NodeOp::Cos { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    grads[input.0] += incoming * (-input_value.sin());

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(cos(x))/dx=-sin(x)",
                    });
                }
                NodeOp::Tan { input } => {
                    let output_value = self.nodes[node_id.0].tensor.value();
                    grads[input.0] += incoming * (1.0 + output_value * output_value);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(tan(x))/dx=1+tan(x)^2",
                    });
                }
                NodeOp::Floor { input } | NodeOp::Ceil { input } | NodeOp::Round { input } => {
                    // Gradient is zero almost everywhere (step functions)
                    grads[input.0] += 0.0;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(floor|ceil|round)/dx=0",
                    });
                }
                NodeOp::Log2 { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    grads[input.0] += incoming / (input_value * std::f64::consts::LN_2);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(log2(x))/dx=1/(x*ln(2))",
                    });
                }
                NodeOp::Log10 { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    grads[input.0] += incoming / (input_value * std::f64::consts::LN_10);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(log10(x))/dx=1/(x*ln(10))",
                    });
                }
                NodeOp::Log1p { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    grads[input.0] += incoming / (1.0 + input_value);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(log1p(x))/dx=1/(1+x)",
                    });
                }
                NodeOp::Expm1 { input } => {
                    let output_value = self.nodes[node_id.0].tensor.value();
                    // d/dx expm1(x) = exp(x) = expm1(x) + 1
                    grads[input.0] += incoming * (output_value + 1.0);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(expm1(x))/dx=exp(x)=expm1(x)+1",
                    });
                }
                NodeOp::Sign { input } => {
                    // sign is a step function, gradient is 0
                    grads[input.0] += 0.0;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(sign(x))/dx=0",
                    });
                }
                NodeOp::Trunc { input } => {
                    // trunc is a step function, gradient is 0
                    grads[input.0] += 0.0;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(trunc(x))/dx=0",
                    });
                }
                NodeOp::Frac { input } => {
                    // frac(x) = x - floor(x), d/dx = 1 - 0 = 1
                    grads[input.0] += incoming;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(frac(x))/dx=1",
                    });
                }
                NodeOp::Asin { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    // d/dx asin(x) = 1/sqrt(1-x^2)
                    grads[input.0] += incoming / (1.0 - input_value * input_value).sqrt();

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(asin(x))/dx=1/sqrt(1-x^2)",
                    });
                }
                NodeOp::Acos { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    // d/dx acos(x) = -1/sqrt(1-x^2)
                    grads[input.0] += -incoming / (1.0 - input_value * input_value).sqrt();

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(acos(x))/dx=-1/sqrt(1-x^2)",
                    });
                }
                NodeOp::Atan { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    // d/dx atan(x) = 1/(1+x^2)
                    grads[input.0] += incoming / (1.0 + input_value * input_value);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(atan(x))/dx=1/(1+x^2)",
                    });
                }
                NodeOp::Sinh { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    // d/dx sinh(x) = cosh(x)
                    grads[input.0] += incoming * input_value.cosh();

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(sinh(x))/dx=cosh(x)",
                    });
                }
                NodeOp::Cosh { input } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    // d/dx cosh(x) = sinh(x)
                    grads[input.0] += incoming * input_value.sinh();

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(cosh(x))/dx=sinh(x)",
                    });
                }
                NodeOp::Gelu { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    // GELU'(x) via tanh approximation
                    let c = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
                    let k = c * (x + 0.044715 * x * x * x);
                    let t = k.tanh();
                    let dk = c * (1.0 + 0.134145 * x * x);
                    grads[input.0] += incoming * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dk);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(gelu(x))/dx",
                    });
                }
                NodeOp::Silu { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    // SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                    let s = 1.0 / (1.0 + (-x).exp());
                    grads[input.0] += incoming * s * (1.0 + x * (1.0 - s));

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(silu(x))/dx=sigmoid(x)*(1+x*(1-sigmoid(x)))",
                    });
                }
                NodeOp::LeakyRelu { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    grads[input.0] += incoming
                        * if x.is_nan() {
                            f64::NAN
                        } else if x >= 0.0 {
                            1.0
                        } else {
                            0.01
                        };

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(leaky_relu(x))/dx=1|0.01",
                    });
                }
                NodeOp::Elu { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    let output_value = self.nodes[node_id.0].tensor.value();
                    // d/dx elu(x) = 1 if x > 0, else output + alpha (alpha=1.0)
                    grads[input.0] += incoming
                        * if x.is_nan() {
                            f64::NAN
                        } else if x > 0.0 {
                            1.0
                        } else {
                            output_value + 1.0
                        };

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(elu(x))/dx=1|output+alpha",
                    });
                }
                NodeOp::Sqrt { input } => {
                    let output_value = self.nodes[node_id.0].tensor.value();
                    grads[input.0] += incoming * 0.5 / output_value;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(sqrt(x))/dx=0.5/sqrt(x)",
                    });
                }
                NodeOp::Reciprocal { input } => {
                    let output_value = self.nodes[node_id.0].tensor.value();
                    grads[input.0] += incoming * (-output_value * output_value);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(1/x)/dx=-1/x^2",
                    });
                }
                NodeOp::Pow { input, exponent } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    grads[input.0] += if exponent == 0.0 {
                        0.0
                    } else {
                        incoming * exponent * input_value.powf(exponent - 1.0)
                    };

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(x^n)/dx=n*x^(n-1)",
                    });
                }
                NodeOp::Min { lhs, rhs } => {
                    let lhs_val = self.nodes[lhs.0].tensor.value();
                    let rhs_val = self.nodes[rhs.0].tensor.value();
                    if lhs_val.is_nan() || rhs_val.is_nan() {
                        grads[lhs.0] += f64::NAN;
                        grads[rhs.0] += f64::NAN;
                    } else if lhs_val < rhs_val {
                        grads[lhs.0] += incoming;
                    } else if lhs_val > rhs_val {
                        grads[rhs.0] += incoming;
                    } else {
                        grads[lhs.0] += incoming * 0.5;
                        grads[rhs.0] += incoming * 0.5;
                    }

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(min(a,b))/da=1(a<b) or 0.5(a=b); db=1(b<a) or 0.5(a=b)",
                    });
                }
                NodeOp::Max { lhs, rhs } => {
                    let lhs_val = self.nodes[lhs.0].tensor.value();
                    let rhs_val = self.nodes[rhs.0].tensor.value();
                    if lhs_val.is_nan() || rhs_val.is_nan() {
                        grads[lhs.0] += f64::NAN;
                        grads[rhs.0] += f64::NAN;
                    } else if lhs_val > rhs_val {
                        grads[lhs.0] += incoming;
                    } else if lhs_val < rhs_val {
                        grads[rhs.0] += incoming;
                    } else {
                        grads[lhs.0] += incoming * 0.5;
                        grads[rhs.0] += incoming * 0.5;
                    }

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(max(a,b))/da=1(a>b) or 0.5(a=b); db=1(b>a) or 0.5(a=b)",
                    });
                }
                NodeOp::Clamp {
                    input,
                    min_val,
                    max_val,
                } => {
                    let input_value = self.nodes[input.0].tensor.value();
                    grads[input.0] += if input_value.is_nan() {
                        f64::NAN
                    } else if input_value >= min_val && input_value <= max_val {
                        incoming
                    } else {
                        0.0
                    };

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(clamp(x,min,max))/dx=1 if min<=x<=max else 0",
                    });
                }
            }
        }

        let gradients = grads
            .iter()
            .enumerate()
            .map(|(idx, grad)| {
                if self.nodes[idx].requires_grad && reachable[idx] {
                    Some(*grad)
                } else {
                    None
                }
            })
            .collect();

        let telemetry = SchedulerTelemetry {
            execution_order,
            queue_pushes: queue.pushes,
            queue_pops: queue.pops,
            max_queue_len: queue.max_len,
            dependency_snapshot,
            reentrant_depth,
            reentrant_guard_triggered,
            hardened_fallback_used,
        };

        Ok(BackwardReport {
            gradients,
            steps,
            telemetry,
        })
    }

    fn compute_reachable(&self, root: NodeId) -> Result<Vec<bool>, AutogradError> {
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            if node.0 >= self.nodes.len() {
                return Err(AutogradError::UnknownNode(node));
            }
            if reachable[node.0] {
                continue;
            }
            reachable[node.0] = true;

            match self.nodes[node.0].op {
                NodeOp::Leaf => {}
                NodeOp::Add { lhs, rhs }
                | NodeOp::Sub { lhs, rhs }
                | NodeOp::Div { lhs, rhs }
                | NodeOp::Mul { lhs, rhs }
                | NodeOp::Min { lhs, rhs }
                | NodeOp::Max { lhs, rhs } => {
                    stack.push(lhs);
                    stack.push(rhs);
                }
                NodeOp::Neg { input }
                | NodeOp::Abs { input }
                | NodeOp::Exp { input }
                | NodeOp::Log { input }
                | NodeOp::Relu { input }
                | NodeOp::Sigmoid { input }
                | NodeOp::Tanh { input }
                | NodeOp::Sin { input }
                | NodeOp::Cos { input }
                | NodeOp::Tan { input }
                | NodeOp::Floor { input }
                | NodeOp::Ceil { input }
                | NodeOp::Round { input }
                | NodeOp::Log2 { input }
                | NodeOp::Log10 { input }
                | NodeOp::Log1p { input }
                | NodeOp::Expm1 { input }
                | NodeOp::Sign { input }
                | NodeOp::Trunc { input }
                | NodeOp::Frac { input }
                | NodeOp::Asin { input }
                | NodeOp::Acos { input }
                | NodeOp::Atan { input }
                | NodeOp::Sinh { input }
                | NodeOp::Cosh { input }
                | NodeOp::Gelu { input }
                | NodeOp::Silu { input }
                | NodeOp::LeakyRelu { input }
                | NodeOp::Elu { input }
                | NodeOp::Sqrt { input }
                | NodeOp::Reciprocal { input }
                | NodeOp::Pow { input, .. }
                | NodeOp::Clamp { input, .. } => {
                    stack.push(input);
                }
            }
        }

        Ok(reachable)
    }

    fn compute_dependencies(&self, reachable: &[bool]) -> Result<Vec<usize>, AutogradError> {
        if reachable.len() != self.nodes.len() {
            return Err(AutogradError::DependencyUnderflow { node: NodeId(0) });
        }

        let mut pending = vec![0usize; self.nodes.len()];

        for (idx, node) in self.nodes.iter().enumerate() {
            if !reachable[idx] {
                continue;
            }
            match node.op {
                NodeOp::Leaf => {}
                NodeOp::Add { lhs, rhs }
                | NodeOp::Sub { lhs, rhs }
                | NodeOp::Div { lhs, rhs }
                | NodeOp::Mul { lhs, rhs }
                | NodeOp::Min { lhs, rhs }
                | NodeOp::Max { lhs, rhs } => {
                    pending[lhs.0] = pending[lhs.0].saturating_add(1);
                    pending[rhs.0] = pending[rhs.0].saturating_add(1);
                }
                NodeOp::Neg { input }
                | NodeOp::Abs { input }
                | NodeOp::Exp { input }
                | NodeOp::Log { input }
                | NodeOp::Relu { input }
                | NodeOp::Sigmoid { input }
                | NodeOp::Tanh { input }
                | NodeOp::Sin { input }
                | NodeOp::Cos { input }
                | NodeOp::Tan { input }
                | NodeOp::Floor { input }
                | NodeOp::Ceil { input }
                | NodeOp::Round { input }
                | NodeOp::Log2 { input }
                | NodeOp::Log10 { input }
                | NodeOp::Log1p { input }
                | NodeOp::Expm1 { input }
                | NodeOp::Sign { input }
                | NodeOp::Trunc { input }
                | NodeOp::Frac { input }
                | NodeOp::Asin { input }
                | NodeOp::Acos { input }
                | NodeOp::Atan { input }
                | NodeOp::Sinh { input }
                | NodeOp::Cosh { input }
                | NodeOp::Gelu { input }
                | NodeOp::Silu { input }
                | NodeOp::LeakyRelu { input }
                | NodeOp::Elu { input }
                | NodeOp::Sqrt { input }
                | NodeOp::Reciprocal { input }
                | NodeOp::Pow { input, .. }
                | NodeOp::Clamp { input, .. } => {
                    pending[input.0] = pending[input.0].saturating_add(1);
                }
            }
        }

        Ok(pending)
    }

    fn complete_dependency(
        pending: &mut [usize],
        node: NodeId,
        queue: &mut ReadyQueue,
    ) -> Result<(), AutogradError> {
        if pending[node.0] == 0 {
            return Err(AutogradError::DependencyUnderflow { node });
        }
        pending[node.0] -= 1;
        if pending[node.0] == 0 {
            queue.push(node);
        }
        Ok(())
    }

    fn node(&self, id: NodeId) -> Result<&Node, AutogradError> {
        self.nodes.get(id.0).ok_or(AutogradError::UnknownNode(id))
    }
}

#[derive(Debug, Clone, Default)]
pub struct TensorTape {
    nodes: Vec<TensorNode>,
}

impl TensorTape {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn leaf(
        &mut self,
        values: Vec<f64>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let tensor = DenseTensor::from_contiguous_values(values, shape, Device::Cpu)?;
        Ok(self.leaf_tensor(tensor, requires_grad))
    }

    pub fn leaf_tensor(&mut self, tensor: DenseTensor, requires_grad: bool) -> TensorNodeId {
        let id = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor,
            requires_grad,
            op: TensorNodeOp::Leaf,
        });
        id
    }

    pub fn values(&self, node: TensorNodeId) -> Result<Vec<f64>, AutogradError> {
        Ok(self.node(node)?.tensor.contiguous_values()?.to_vec())
    }

    pub fn tensor(&self, node: TensorNodeId) -> Result<&DenseTensor, AutogradError> {
        Ok(&self.node(node)?.tensor)
    }

    pub fn add(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        self.binary(BinaryOp::Add, lhs, rhs, mode)
    }

    pub fn mul(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        self.binary(BinaryOp::Mul, lhs, rhs, mode)
    }

    pub fn sub(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        self.binary(BinaryOp::Sub, lhs, rhs, mode)
    }

    pub fn div(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        self.binary(BinaryOp::Div, lhs, rhs, mode)
    }

    pub fn matmul(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        self.binary(BinaryOp::MatMul, lhs, rhs, mode)
    }

    pub fn dot(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        self.binary(BinaryOp::Dot, lhs, rhs, mode)
    }

    pub fn outer(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        self.binary(BinaryOp::Outer, lhs, rhs, mode)
    }

    pub fn bmm(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        self.binary(BinaryOp::Bmm, lhs, rhs, mode)
    }

    pub fn trace(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorReductionOperationEvent), AutogradError> {
        let (requires_grad, input_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_contiguous_f64(
                ReductionOp::Trace,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(vec![1], output_dtype, output_device),
                vec![outcome.value],
            )?,
            requires_grad,
            op: TensorNodeOp::Trace { input, input_shape },
        });

        Ok((
            out,
            TensorReductionOperationEvent {
                op: ReductionOp::Trace,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn neg(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Neg,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Neg { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Neg,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn abs(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Abs,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Abs { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Abs,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn exp(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Exp,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Exp { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Exp,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn log(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Log,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Log { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Log,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn relu(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Relu,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Relu { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Relu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sigmoid(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Sigmoid,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Sigmoid { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Sigmoid,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn tanh(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Tanh,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Tanh { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Tanh,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sin(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Sin,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Sin { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Sin,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn cos(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Cos,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Cos { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Cos,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn tan(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Tan,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Tan { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Tan,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn floor(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Floor,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Floor { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Floor,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn ceil(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Ceil,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Ceil { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Ceil,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn round(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Round,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Round { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Round,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn log2(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Log2,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Log2 { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Log2,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn log10(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Log10,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Log10 { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Log10,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn log1p(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Log1p,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Log1p { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Log1p,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn expm1(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Expm1,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Expm1 { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Expm1,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sign(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Sign,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Sign { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Sign,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn trunc(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Trunc,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Trunc { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Trunc,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn frac(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Frac,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Frac { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Frac,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn asin(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Asin,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Asin { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Asin,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn acos(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Acos,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Acos { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Acos,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn atan(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Atan,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Atan { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Atan,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sinh(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Sinh,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Sinh { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Sinh,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn cosh(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Cosh,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Cosh { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Cosh,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn gelu(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Gelu,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Gelu { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Gelu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn silu(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Silu,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Silu { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Silu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn leaky_relu(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::LeakyRelu,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::LeakyRelu { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::LeakyRelu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn elu(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Elu,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(input_meta, outcome.values)?,
            requires_grad,
            op: TensorNodeOp::Elu { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Elu,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sqrt(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Sqrt,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Sqrt { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Sqrt,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn reciprocal(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_f64(
                UnaryOp::Reciprocal,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Reciprocal { input },
        });

        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Reciprocal,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn pow(
        &mut self,
        input: TensorNodeId,
        exponent: f64,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorPowOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_pow_contiguous_f64(
                mode,
                input_node.tensor.storage(),
                &meta,
                exponent,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Pow { input, exponent },
        });

        Ok((
            out,
            TensorPowOperationEvent {
                input,
                out,
                exponent,
                decision: outcome.decision,
            },
        ))
    }

    pub fn tensor_min(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad = lhs_node.requires_grad || rhs_node.requires_grad;
            let meta_l = lhs_node.tensor.meta().clone();
            let meta_r = rhs_node.tensor.meta().clone();
            let outcome = dispatch_tensor_binary_contiguous_f64(
                BinaryOp::Min,
                mode,
                lhs_node.tensor.storage(),
                rhs_node.tensor.storage(),
                &meta_l,
                &meta_r,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta_l.shape().to_vec(),
                meta_l.dtype(),
                meta_l.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Min { lhs, rhs },
        });

        Ok((
            out,
            TensorOperationEvent {
                op: BinaryOp::Min,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn tensor_max(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad = lhs_node.requires_grad || rhs_node.requires_grad;
            let meta_l = lhs_node.tensor.meta().clone();
            let meta_r = rhs_node.tensor.meta().clone();
            let outcome = dispatch_tensor_binary_contiguous_f64(
                BinaryOp::Max,
                mode,
                lhs_node.tensor.storage(),
                rhs_node.tensor.storage(),
                &meta_l,
                &meta_r,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta_l.shape().to_vec(),
                meta_l.dtype(),
                meta_l.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Max { lhs, rhs },
        });

        Ok((
            out,
            TensorOperationEvent {
                op: BinaryOp::Max,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn tensor_clamp(
        &mut self,
        input: TensorNodeId,
        min_val: f64,
        max_val: f64,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorClampOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_clamp_contiguous_f64(
                mode,
                input_node.tensor.storage(),
                &meta,
                min_val,
                max_val,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Clamp {
                input,
                min_val,
                max_val,
            },
        });

        Ok((
            out,
            TensorClampOperationEvent {
                input,
                out,
                min_val,
                max_val,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sum(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorReductionOperationEvent), AutogradError> {
        let (requires_grad, input_numel, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_contiguous_f64(
                ReductionOp::Sum,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.numel(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(vec![1], output_dtype, output_device),
                vec![outcome.value],
            )?,
            requires_grad,
            op: TensorNodeOp::Sum { input, input_numel },
        });

        Ok((
            out,
            TensorReductionOperationEvent {
                op: ReductionOp::Sum,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn mean(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorReductionOperationEvent), AutogradError> {
        let (requires_grad, input_numel, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_contiguous_f64(
                ReductionOp::Mean,
                mode,
                input_node.tensor.storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.numel(),
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(vec![1], output_dtype, output_device),
                vec![outcome.value],
            )?,
            requires_grad,
            op: TensorNodeOp::Mean { input, input_numel },
        });

        Ok((
            out,
            TensorReductionOperationEvent {
                op: ReductionOp::Mean,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn sum_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorReductionDimOperationEvent), AutogradError> {
        let (requires_grad, input_shape, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_f64(
                ReductionOp::Sum,
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let input_shape = meta.shape().to_vec();
            let mut out_shape = input_shape.clone();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (
                requires_grad,
                input_shape,
                out_shape,
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::SumDim {
                input,
                dim,
                input_shape,
            },
        });

        Ok((
            out,
            TensorReductionDimOperationEvent {
                op: ReductionOp::Sum,
                input,
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn mean_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorReductionDimOperationEvent), AutogradError> {
        let (requires_grad, input_shape, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_f64(
                ReductionOp::Mean,
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let input_shape = meta.shape().to_vec();
            let mut out_shape = input_shape.clone();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (
                requires_grad,
                input_shape,
                out_shape,
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::MeanDim {
                input,
                dim,
                input_shape,
            },
        });

        Ok((
            out,
            TensorReductionDimOperationEvent {
                op: ReductionOp::Mean,
                input,
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn prod_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorReductionDimOperationEvent), AutogradError> {
        let (requires_grad, input_shape, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_f64(
                ReductionOp::Prod,
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let input_shape = meta.shape().to_vec();
            let mut out_shape = input_shape.clone();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (
                requires_grad,
                input_shape,
                out_shape,
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::ProdDim {
                input,
                dim,
                input_shape,
            },
        });

        Ok((
            out,
            TensorReductionDimOperationEvent {
                op: ReductionOp::Prod,
                input,
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn var_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorReductionDimOperationEvent), AutogradError> {
        let (requires_grad, input_shape, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_f64(
                ReductionOp::Var,
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let input_shape = meta.shape().to_vec();
            let mut out_shape = input_shape.clone();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (
                requires_grad,
                input_shape,
                out_shape,
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::VarDim {
                input,
                dim,
                input_shape,
            },
        });

        Ok((
            out,
            TensorReductionDimOperationEvent {
                op: ReductionOp::Var,
                input,
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn std_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorReductionDimOperationEvent), AutogradError> {
        let (requires_grad, input_shape, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_f64(
                ReductionOp::Std,
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let input_shape = meta.shape().to_vec();
            let mut out_shape = input_shape.clone();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (
                requires_grad,
                input_shape,
                out_shape,
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::StdDim {
                input,
                dim,
                input_shape,
            },
        });

        Ok((
            out,
            TensorReductionDimOperationEvent {
                op: ReductionOp::Std,
                input,
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn cumsum(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorScanDimOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_scan_dim_contiguous_f64(
                ScanOp::CumSum,
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let out_shape = meta.shape().to_vec();
            (
                requires_grad,
                out_shape,
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::CumSum { input, dim },
        });

        Ok((
            out,
            TensorScanDimOperationEvent {
                op: ScanOp::CumSum,
                input,
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn cumprod(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorScanDimOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_scan_dim_contiguous_f64(
                ScanOp::CumProd,
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let out_shape = meta.shape().to_vec();
            (
                requires_grad,
                out_shape,
                meta.dtype(),
                meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::CumProd { input, dim },
        });

        Ok((
            out,
            TensorScanDimOperationEvent {
                op: ScanOp::CumProd,
                input,
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn softmax(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorNormalizeDimOperationEvent), AutogradError> {
        let (requires_grad, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_normalize_dim_contiguous_f64(
                NormalizeOp::Softmax,
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, meta.dtype(), meta.device(), outcome)
        };

        let input_shape = self.nodes[input.0].tensor.meta().shape().to_vec();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(input_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Softmax { input, dim },
        });

        Ok((
            out,
            TensorNormalizeDimOperationEvent {
                op: NormalizeOp::Softmax,
                input,
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn log_softmax(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorNormalizeDimOperationEvent), AutogradError> {
        let (requires_grad, output_dtype, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_normalize_dim_contiguous_f64(
                NormalizeOp::LogSoftmax,
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, meta.dtype(), meta.device(), outcome)
        };

        let input_shape = self.nodes[input.0].tensor.meta().shape().to_vec();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(input_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::LogSoftmax { input, dim },
        });

        Ok((
            out,
            TensorNormalizeDimOperationEvent {
                op: NormalizeOp::LogSoftmax,
                input,
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn argmax(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, output_shape, output_dtype, output_device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta().clone();
            let values = argmax_dim_tensor_contiguous_f64(input_node.tensor.storage(), &meta, dim)
                .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let mut out_shape = meta.shape().to_vec();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (values, out_shape, meta.dtype(), meta.device())
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                values,
            )?,
            requires_grad: false,
            op: TensorNodeOp::Leaf,
        });
        Ok(out)
    }

    pub fn argmin(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, output_shape, output_dtype, output_device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta().clone();
            let values = argmin_dim_tensor_contiguous_f64(input_node.tensor.storage(), &meta, dim)
                .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let mut out_shape = meta.shape().to_vec();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (values, out_shape, meta.dtype(), meta.device())
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                values,
            )?,
            requires_grad: false,
            op: TensorNodeOp::Leaf,
        });
        Ok(out)
    }

    pub fn max_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        let (
            values,
            indices,
            input_shape,
            output_shape,
            output_dtype,
            output_device,
            requires_grad,
        ) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let (values, indices) =
                max_dim_tensor_contiguous_f64(input_node.tensor.storage(), &meta, dim)
                    .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let input_shape = meta.shape().to_vec();
            let mut out_shape = input_shape.clone();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (
                values,
                indices,
                input_shape,
                out_shape,
                meta.dtype(),
                meta.device(),
                requires_grad,
            )
        };

        let indices_clone = indices.clone();
        let out_values = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape.clone(), output_dtype, output_device),
                values,
            )?,
            requires_grad,
            op: TensorNodeOp::MaxDim {
                input,
                dim,
                input_shape,
                indices: indices_clone,
            },
        });

        let out_indices = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                indices,
            )?,
            requires_grad: false,
            op: TensorNodeOp::Leaf,
        });

        Ok((out_values, out_indices))
    }

    pub fn min_dim(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        let (
            values,
            indices,
            input_shape,
            output_shape,
            output_dtype,
            output_device,
            requires_grad,
        ) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let (values, indices) =
                min_dim_tensor_contiguous_f64(input_node.tensor.storage(), &meta, dim)
                    .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let input_shape = meta.shape().to_vec();
            let mut out_shape = input_shape.clone();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (
                values,
                indices,
                input_shape,
                out_shape,
                meta.dtype(),
                meta.device(),
                requires_grad,
            )
        };

        let indices_clone = indices.clone();
        let out_values = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape.clone(), output_dtype, output_device),
                values,
            )?,
            requires_grad,
            op: TensorNodeOp::MinDim {
                input,
                dim,
                input_shape,
                indices: indices_clone,
            },
        });

        let out_indices = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                indices,
            )?,
            requires_grad: false,
            op: TensorNodeOp::Leaf,
        });

        Ok((out_values, out_indices))
    }

    pub fn index_select(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        indices: &[f64],
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, input_shape, output_shape, output_dtype, output_device, requires_grad) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let values = index_select_tensor_contiguous_f64(
                input_node.tensor.storage(),
                &meta,
                dim,
                indices,
            )
            .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let input_shape = meta.shape().to_vec();
            let mut out_shape = input_shape.clone();
            out_shape[dim] = indices.len();
            (
                values,
                input_shape,
                out_shape,
                meta.dtype(),
                meta.device(),
                requires_grad,
            )
        };

        let indices_owned = indices.to_vec();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                values,
            )?,
            requires_grad,
            op: TensorNodeOp::IndexSelect {
                input,
                dim,
                indices: indices_owned,
                input_shape,
            },
        });
        Ok(out)
    }

    pub fn gather(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        index: &[f64],
        index_shape: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, input_shape, output_dtype, output_device, requires_grad) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let idx_meta =
                ft_core::TensorMeta::from_shape(index_shape.clone(), meta.dtype(), meta.device());
            let values = gather_tensor_contiguous_f64(
                input_node.tensor.storage(),
                &meta,
                dim,
                index,
                &idx_meta,
            )
            .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let input_shape = meta.shape().to_vec();
            (
                values,
                input_shape,
                meta.dtype(),
                meta.device(),
                requires_grad,
            )
        };

        let index_owned = index.to_vec();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(index_shape.clone(), output_dtype, output_device),
                values,
            )?,
            requires_grad,
            op: TensorNodeOp::Gather {
                input,
                dim,
                index: index_owned,
                index_shape,
                input_shape,
            },
        });
        Ok(out)
    }

    pub fn scatter(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        index: &[f64],
        index_shape: Vec<usize>,
        src: &[f64],
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, input_shape, output_dtype, output_device, requires_grad) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let idx_meta =
                ft_core::TensorMeta::from_shape(index_shape.clone(), meta.dtype(), meta.device());
            let values = scatter_tensor_contiguous_f64(
                input_node.tensor.storage(),
                &meta,
                dim,
                index,
                &idx_meta,
                src,
            )
            .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let input_shape = meta.shape().to_vec();
            (
                values,
                input_shape,
                meta.dtype(),
                meta.device(),
                requires_grad,
            )
        };

        let index_owned = index.to_vec();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(input_shape.clone(), output_dtype, output_device),
                values,
            )?,
            requires_grad,
            op: TensorNodeOp::Scatter {
                input,
                dim,
                index: index_owned,
                index_shape,
                input_shape,
            },
        });
        Ok(out)
    }

    pub fn masked_fill(
        &mut self,
        input: TensorNodeId,
        mask: &[f64],
        value: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, output_shape, output_dtype, output_device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta().clone();
            let values =
                masked_fill_tensor_contiguous_f64(input_node.tensor.storage(), &meta, mask, value)
                    .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let output_shape = meta.shape().to_vec();
            (values, output_shape, meta.dtype(), meta.device())
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                values,
            )?,
            requires_grad: false,
            op: TensorNodeOp::Leaf,
        });
        Ok(out)
    }

    /// Conditional selection: where(condition, x, y).
    ///
    /// Selects from `x` where condition is non-zero, from `y` otherwise.
    pub fn tensor_where(
        &mut self,
        condition: TensorNodeId,
        x: TensorNodeId,
        y: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, output_shape, output_dtype, output_device, requires_grad) = {
            let cond_node = self.node(condition)?;
            let x_node = self.node(x)?;
            let y_node = self.node(y)?;
            let cond_meta = cond_node.tensor.meta();
            let x_meta = x_node.tensor.meta();
            let y_meta = y_node.tensor.meta();
            if cond_meta.shape() != x_meta.shape() || y_meta.shape() != x_meta.shape() {
                return Err(AutogradError::Dispatch(
                    DispatchKeyError::IncompatibleSet {
                        reason: "where requires condition, x, and y to have the same shape",
                    }
                    .into(),
                ));
            }
            if cond_meta.dtype() != x_meta.dtype() || y_meta.dtype() != x_meta.dtype() {
                return Err(AutogradError::Dispatch(
                    DispatchKeyError::IncompatibleSet {
                        reason: "where requires condition, x, and y to have matching dtypes",
                    }
                    .into(),
                ));
            }
            if cond_meta.device() != x_meta.device() || y_meta.device() != x_meta.device() {
                return Err(AutogradError::Dispatch(
                    DispatchKeyError::IncompatibleSet {
                        reason: "where requires condition, x, and y to be on the same device",
                    }
                    .into(),
                ));
            }
            let requires_grad = x_node.requires_grad || y_node.requires_grad;
            let cond_values = cond_node.tensor.contiguous_values()?;
            let x_values = x_node.tensor.contiguous_values()?;
            let y_values = y_node.tensor.contiguous_values()?;
            let meta = ft_core::TensorMeta::from_shape(
                x_meta.shape().to_vec(),
                x_meta.dtype(),
                x_meta.device(),
            );
            let values = where_tensor_contiguous_f64(cond_values, x_values, y_values, &meta)
                .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let output_shape = meta.shape().to_vec();
            (
                values,
                output_shape,
                meta.dtype(),
                meta.device(),
                requires_grad,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                values,
            )?,
            requires_grad,
            op: TensorNodeOp::Where { condition, x, y },
        });
        Ok(out)
    }

    pub fn sort(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        descending: bool,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, Vec<usize>, TensorSortOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, input_shape, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_sort_contiguous_f64(
                mode,
                input_node.tensor.storage(),
                &meta,
                dim,
                descending,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let out_shape = meta.shape().to_vec();
            let input_shape = meta.shape().to_vec();
            (
                requires_grad,
                out_shape,
                meta.dtype(),
                meta.device(),
                input_shape,
                outcome,
            )
        };

        let indices = outcome.indices.clone();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Sort {
                input,
                dim,
                indices: indices.clone(),
                input_shape,
            },
        });

        Ok((
            out,
            indices,
            TensorSortOperationEvent {
                input,
                out,
                dim,
                descending,
                decision: outcome.decision,
            },
        ))
    }

    pub fn topk(
        &mut self,
        input: TensorNodeId,
        k: usize,
        dim: usize,
        largest: bool,
        sorted: bool,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, Vec<usize>, TensorTopKOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, input_shape, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_topk_contiguous_f64(
                mode,
                input_node.tensor.storage(),
                &meta,
                k,
                dim,
                largest,
                sorted,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let mut out_shape = meta.shape().to_vec();
            out_shape[dim] = k;
            let input_shape = meta.shape().to_vec();
            (
                requires_grad,
                out_shape,
                meta.dtype(),
                meta.device(),
                input_shape,
                outcome,
            )
        };

        let indices = outcome.indices.clone();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::TopK {
                input,
                dim,
                k,
                indices: indices.clone(),
                input_shape,
            },
        });

        Ok((
            out,
            indices,
            TensorTopKOperationEvent {
                input,
                out,
                k,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn cat(
        &mut self,
        inputs: &[TensorNodeId],
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorJoinOperationEvent), AutogradError> {
        if inputs.is_empty() {
            return Err(AutogradError::Dispatch(
                DispatchKeyError::IncompatibleSet {
                    reason: "cat requires at least one input",
                }
                .into(),
            ));
        }
        // Collect input data and metadata
        let mut dispatch_inputs: Vec<(Vec<f64>, ft_core::TensorMeta)> = Vec::new();
        let mut requires_grad = false;
        let mut input_dim_sizes: Vec<usize> = Vec::new();
        for &id in inputs {
            let node = self.node(id)?;
            requires_grad |= node.requires_grad;
            let meta = node.tensor.meta().clone();
            input_dim_sizes.push(meta.shape()[dim]);
            dispatch_inputs.push((node.tensor.contiguous_values()?.to_vec(), meta));
        }

        let refs: Vec<(&[f64], &ft_core::TensorMeta)> = dispatch_inputs
            .iter()
            .map(|(d, m)| (d.as_slice(), m))
            .collect();

        let outcome =
            dispatch_tensor_join_contiguous_f64(JoinOp::Cat, mode, &refs, dim, requires_grad)
                .map_err(AutogradError::Dispatch)?;

        // Compute output shape
        let first_shape = dispatch_inputs[0].1.shape().to_vec();
        let mut out_shape = first_shape.clone();
        out_shape[dim] = input_dim_sizes.iter().sum();
        let output_dtype = dispatch_inputs[0].1.dtype();
        let output_device = dispatch_inputs[0].1.device();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(out_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Cat {
                inputs: inputs.to_vec(),
                dim,
                input_dim_sizes,
            },
        });

        Ok((
            out,
            TensorJoinOperationEvent {
                op: JoinOp::Cat,
                inputs: inputs.to_vec(),
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn stack(
        &mut self,
        inputs: &[TensorNodeId],
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorJoinOperationEvent), AutogradError> {
        if inputs.is_empty() {
            return Err(AutogradError::Dispatch(
                DispatchKeyError::IncompatibleSet {
                    reason: "stack requires at least one input",
                }
                .into(),
            ));
        }
        // Collect input data and metadata
        let mut dispatch_inputs: Vec<(Vec<f64>, ft_core::TensorMeta)> = Vec::new();
        let mut requires_grad = false;
        for &id in inputs {
            let node = self.node(id)?;
            requires_grad |= node.requires_grad;
            let meta = node.tensor.meta().clone();
            dispatch_inputs.push((node.tensor.contiguous_values()?.to_vec(), meta));
        }

        let refs: Vec<(&[f64], &ft_core::TensorMeta)> = dispatch_inputs
            .iter()
            .map(|(d, m)| (d.as_slice(), m))
            .collect();

        let outcome =
            dispatch_tensor_join_contiguous_f64(JoinOp::Stack, mode, &refs, dim, requires_grad)
                .map_err(AutogradError::Dispatch)?;

        // Compute output shape: insert new dim at position
        let first_shape = dispatch_inputs[0].1.shape().to_vec();
        let mut out_shape = first_shape;
        out_shape.insert(dim, inputs.len());
        let output_dtype = dispatch_inputs[0].1.dtype();
        let output_device = dispatch_inputs[0].1.device();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(out_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: TensorNodeOp::Stack {
                inputs: inputs.to_vec(),
                dim,
            },
        });

        Ok((
            out,
            TensorJoinOperationEvent {
                op: JoinOp::Stack,
                inputs: inputs.to_vec(),
                out,
                dim,
                decision: outcome.decision,
            },
        ))
    }

    pub fn reshape(
        &mut self,
        input: TensorNodeId,
        new_shape: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, storage, original_shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let input_numel = meta.numel();
            let new_numel =
                Self::checked_shape_numel(&new_shape, "reshape target shape volume overflow")?;
            if input_numel != new_numel {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::ShapeMismatch {
                        lhs: meta.shape().to_vec(),
                        rhs: new_shape,
                    },
                )));
            }
            (
                input_node.requires_grad,
                input_node.tensor.storage().to_vec(),
                meta.shape().to_vec(),
                meta.dtype(),
                meta.device(),
            )
        };

        let new_meta = ft_core::TensorMeta::from_shape(new_shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, storage)?,
            requires_grad,
            op: TensorNodeOp::Reshape {
                input,
                original_shape,
            },
        });
        Ok(out)
    }

    pub fn squeeze(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, storage, new_shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape();
            if dim >= shape.len() {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::InvalidDimension {
                        dim,
                        ndim: shape.len(),
                    },
                )));
            }
            let mut new_shape = shape.to_vec();
            if new_shape[dim] == 1 {
                new_shape.remove(dim);
                if new_shape.is_empty() {
                    new_shape.push(1);
                }
            }
            (
                input_node.requires_grad,
                input_node.tensor.storage().to_vec(),
                new_shape,
                meta.dtype(),
                meta.device(),
            )
        };

        let new_meta = ft_core::TensorMeta::from_shape(new_shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, storage)?,
            requires_grad,
            op: TensorNodeOp::Squeeze { input, dim },
        });
        Ok(out)
    }

    pub fn unsqueeze(
        &mut self,
        input: TensorNodeId,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, storage, new_shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape();
            if dim > shape.len() {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::InvalidDimension {
                        dim,
                        ndim: shape.len() + 1,
                    },
                )));
            }
            let mut new_shape = shape.to_vec();
            new_shape.insert(dim, 1);
            (
                input_node.requires_grad,
                input_node.tensor.storage().to_vec(),
                new_shape,
                meta.dtype(),
                meta.device(),
            )
        };

        let new_meta = ft_core::TensorMeta::from_shape(new_shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, storage)?,
            requires_grad,
            op: TensorNodeOp::Unsqueeze { input, dim },
        });
        Ok(out)
    }

    pub fn view(
        &mut self,
        input: TensorNodeId,
        new_shape: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        self.reshape(input, new_shape)
    }

    pub fn transpose(
        &mut self,
        input: TensorNodeId,
        dim0: usize,
        dim1: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, new_storage, new_shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape();
            let ndim = shape.len();
            if dim0 >= ndim || dim1 >= ndim {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::InvalidDimension {
                        dim: if dim0 >= ndim { dim0 } else { dim1 },
                        ndim,
                    },
                )));
            }
            if dim0 == dim1 {
                // Identity transpose: same shape, same data
                let storage = input_node.tensor.storage().to_vec();
                (
                    input_node.requires_grad,
                    storage,
                    shape.to_vec(),
                    meta.dtype(),
                    meta.device(),
                )
            } else {
                let mut perm: Vec<usize> = (0..ndim).collect();
                perm.swap(dim0, dim1);
                let storage = input_node.tensor.storage().to_vec();
                let new_storage = Self::permute_data(&storage, shape, &perm)?;
                let mut new_shape = shape.to_vec();
                new_shape.swap(dim0, dim1);
                (
                    input_node.requires_grad,
                    new_storage,
                    new_shape,
                    meta.dtype(),
                    meta.device(),
                )
            }
        };

        let new_meta = ft_core::TensorMeta::from_shape(new_shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, new_storage)?,
            requires_grad,
            op: TensorNodeOp::Transpose { input, dim0, dim1 },
        });
        Ok(out)
    }

    pub fn permute(
        &mut self,
        input: TensorNodeId,
        dims: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, new_storage, new_shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape();
            let ndim = shape.len();

            if dims.len() != ndim {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::ShapeMismatch {
                        lhs: shape.to_vec(),
                        rhs: dims.clone(),
                    },
                )));
            }

            // Validate permutation: each dimension must appear exactly once
            let mut seen = vec![false; ndim];
            for &d in &dims {
                if d >= ndim {
                    return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                        ft_kernel_cpu::KernelError::InvalidDimension { dim: d, ndim },
                    )));
                }
                if seen[d] {
                    return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                        ft_kernel_cpu::KernelError::InvalidDimension { dim: d, ndim },
                    )));
                }
                seen[d] = true;
            }

            let storage = input_node.tensor.storage().to_vec();
            let new_storage = Self::permute_data(&storage, shape, &dims)?;
            let new_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
            (
                input_node.requires_grad,
                new_storage,
                new_shape,
                meta.dtype(),
                meta.device(),
            )
        };

        let new_meta = ft_core::TensorMeta::from_shape(new_shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, new_storage)?,
            requires_grad,
            op: TensorNodeOp::Permute {
                input,
                dims: dims.clone(),
            },
        });
        Ok(out)
    }

    pub fn flatten(
        &mut self,
        input: TensorNodeId,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let new_shape = {
            let input_node = self.node(input)?;
            let shape = input_node.tensor.meta().shape();
            let ndim = shape.len();
            if start_dim >= ndim || end_dim >= ndim || start_dim > end_dim {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::InvalidDimension { dim: end_dim, ndim },
                )));
            }
            let flat_size = Self::checked_shape_numel(
                &shape[start_dim..=end_dim],
                "flatten shape multiplication overflow",
            )?;
            let mut new_shape = Vec::with_capacity(ndim - (end_dim - start_dim));
            new_shape.extend_from_slice(&shape[..start_dim]);
            new_shape.push(flat_size);
            new_shape.extend_from_slice(&shape[end_dim + 1..]);
            new_shape
        };
        self.reshape(input, new_shape)
    }

    pub fn unflatten(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        sizes: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let new_shape = {
            let input_node = self.node(input)?;
            let shape = input_node.tensor.meta().shape();
            let ndim = shape.len();
            if dim >= ndim {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::InvalidDimension { dim, ndim },
                )));
            }
            let expected =
                Self::checked_shape_numel(&sizes, "unflatten sizes multiplication overflow")?;
            if expected != shape[dim] {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::ShapeMismatch {
                        lhs: shape.to_vec(),
                        rhs: sizes.clone(),
                    },
                )));
            }
            let mut new_shape = Vec::with_capacity(ndim - 1 + sizes.len());
            new_shape.extend_from_slice(&shape[..dim]);
            new_shape.extend_from_slice(&sizes);
            new_shape.extend_from_slice(&shape[dim + 1..]);
            new_shape
        };
        self.reshape(input, new_shape)
    }

    pub fn narrow(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result_data, new_shape, original_shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape();
            let original_shape = shape.to_vec();
            let storage = input_node.tensor.contiguous_values()?;

            let result_data =
                ft_kernel_cpu::narrow_tensor_contiguous_f64(storage, meta, dim, start, length)
                    .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;

            let mut new_shape = shape.to_vec();
            new_shape[dim] = length;

            (
                input_node.requires_grad,
                result_data,
                new_shape,
                original_shape,
                meta.dtype(),
                meta.device(),
            )
        };

        let new_meta = ft_core::TensorMeta::from_shape(new_shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, result_data)?,
            requires_grad,
            op: TensorNodeOp::Narrow {
                input,
                dim,
                start,
                original_shape,
            },
        });
        Ok(out)
    }

    pub fn expand(
        &mut self,
        input: TensorNodeId,
        target_shape: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result_data, original_shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let original_shape = meta.shape().to_vec();
            let storage = input_node.tensor.contiguous_values()?;

            let result_data =
                ft_kernel_cpu::expand_tensor_contiguous_f64(storage, meta, &target_shape)
                    .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;

            (
                input_node.requires_grad,
                result_data,
                original_shape,
                meta.dtype(),
                meta.device(),
            )
        };

        let new_meta = ft_core::TensorMeta::from_shape(target_shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, result_data)?,
            requires_grad,
            op: TensorNodeOp::Expand {
                input,
                original_shape,
            },
        });
        Ok(out)
    }

    pub fn split(
        &mut self,
        input: TensorNodeId,
        split_sizes: &[usize],
        dim: usize,
    ) -> Result<Vec<TensorNodeId>, AutogradError> {
        let (requires_grad, storage, shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape().to_vec();
            let ndim = shape.len();
            if dim >= ndim {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::InvalidDimension { dim, ndim },
                )));
            }
            let total: usize = split_sizes.iter().sum();
            if total != shape[dim] {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::ShapeMismatch {
                        lhs: shape.clone(),
                        rhs: split_sizes.to_vec(),
                    },
                )));
            }
            (
                input_node.requires_grad,
                input_node.tensor.contiguous_values()?.to_vec(),
                shape,
                meta.dtype(),
                meta.device(),
            )
        };

        let original_shape = shape.clone();
        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let mut outputs = Vec::with_capacity(split_sizes.len());
        let mut start = 0;

        for (chunk_index, &sz) in split_sizes.iter().enumerate() {
            let chunk_data =
                ft_kernel_cpu::narrow_tensor_contiguous_f64(&storage, &meta, dim, start, sz)
                    .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;

            let mut chunk_shape = original_shape.clone();
            chunk_shape[dim] = sz;
            let chunk_meta = ft_core::TensorMeta::from_shape(chunk_shape, dtype, device);
            let out = TensorNodeId(self.nodes.len());
            self.nodes.push(TensorNode {
                tensor: DenseTensor::from_storage(chunk_meta, chunk_data)?,
                requires_grad,
                op: TensorNodeOp::Split {
                    input,
                    chunk_index,
                    dim,
                    start,
                    original_shape: original_shape.clone(),
                },
            });
            outputs.push(out);
            start += sz;
        }

        Ok(outputs)
    }

    pub fn chunk(
        &mut self,
        input: TensorNodeId,
        chunks: usize,
        dim: usize,
    ) -> Result<Vec<TensorNodeId>, AutogradError> {
        if chunks == 0 {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension {
                    dim: chunks,
                    ndim: 1,
                },
            )));
        }
        let dim_size = {
            let input_node = self.node(input)?;
            let shape = input_node.tensor.meta().shape();
            if dim >= shape.len() {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::InvalidDimension {
                        dim,
                        ndim: shape.len(),
                    },
                )));
            }
            shape[dim]
        };

        let chunk_size = dim_size.div_ceil(chunks);
        let mut split_sizes = Vec::with_capacity(chunks);
        let mut remaining = dim_size;
        while remaining > 0 {
            let sz = remaining.min(chunk_size);
            split_sizes.push(sz);
            remaining -= sz;
        }

        self.split(input, &split_sizes, dim)
    }

    /// Physically rearranges data according to a dimension permutation.
    /// Given source data in contiguous row-major layout with `src_shape`,
    /// produces data in contiguous row-major layout for the permuted shape.
    fn permute_data(
        src: &[f64],
        src_shape: &[usize],
        perm: &[usize],
    ) -> Result<Vec<f64>, AutogradError> {
        let ndim = src_shape.len();
        let numel = Self::checked_shape_numel(src_shape, "permute shape volume overflow")?;
        if numel == 0 {
            return Ok(Vec::new());
        }

        let src_strides = ft_core::contiguous_strides(src_shape);
        let dst_shape: Vec<usize> = perm.iter().map(|&d| src_shape[d]).collect();
        let dst_strides = ft_core::contiguous_strides(&dst_shape);

        let mut dst = vec![0.0; numel];
        let mut coords = vec![0usize; ndim];

        for (flat_src, &val) in src.iter().enumerate().take(numel) {
            // Compute source multi-index from flat source index
            let mut remaining = flat_src;
            for d in 0..ndim {
                coords[d] = remaining / src_strides[d];
                remaining %= src_strides[d];
            }

            // Compute destination flat index using permuted coordinates
            let mut flat_dst = 0;
            for d in 0..ndim {
                flat_dst += coords[perm[d]] * dst_strides[d];
            }

            dst[flat_dst] = val;
        }

        Ok(dst)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn flip(
        &mut self,
        input: TensorNodeId,
        dims: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, storage, shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape().to_vec();
            let ndim = shape.len();
            for &d in &dims {
                if d >= ndim {
                    return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                        ft_kernel_cpu::KernelError::InvalidDimension { dim: d, ndim },
                    )));
                }
            }
            (
                input_node.requires_grad,
                input_node.tensor.contiguous_values()?.to_vec(),
                shape,
                meta.dtype(),
                meta.device(),
            )
        };

        let numel = Self::checked_shape_numel(&shape, "flip shape volume overflow")?;
        let strides = ft_core::contiguous_strides(&shape);
        let ndim = shape.len();
        let mut result = vec![0.0; numel];

        for flat in 0..numel {
            let mut remaining = flat;
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                coords[d] = remaining / strides[d];
                remaining %= strides[d];
            }

            let mut flipped_flat = 0;
            for d in 0..ndim {
                let coord = if dims.contains(&d) {
                    shape[d] - 1 - coords[d]
                } else {
                    coords[d]
                };
                flipped_flat += coord * strides[d];
            }
            result[flipped_flat] = storage[flat];
        }

        let new_meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, result)?,
            requires_grad,
            op: TensorNodeOp::Flip { input, dims },
        });
        Ok(out)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn repeat(
        &mut self,
        input: TensorNodeId,
        repeats: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, storage, original_shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape().to_vec();
            if repeats.len() != shape.len() {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::ShapeMismatch {
                        lhs: shape,
                        rhs: repeats,
                    },
                )));
            }
            (
                input_node.requires_grad,
                input_node.tensor.contiguous_values()?.to_vec(),
                shape,
                meta.dtype(),
                meta.device(),
            )
        };

        let ndim = original_shape.len();
        let mut output_shape = Vec::with_capacity(ndim);
        for (&size, &repeat) in original_shape.iter().zip(repeats.iter()) {
            output_shape.push(Self::checked_mul_usize(
                size,
                repeat,
                "repeat shape multiplication overflow",
            )?);
        }
        let output_numel =
            Self::checked_shape_numel(&output_shape, "repeat output shape volume overflow")?;
        let output_strides = ft_core::contiguous_strides(&output_shape);
        let src_strides = ft_core::contiguous_strides(&original_shape);

        let mut result = vec![0.0; output_numel];
        for flat in 0..output_numel {
            let mut remaining = flat;
            let mut src_flat = 0;
            for d in 0..ndim {
                let coord = remaining / output_strides[d];
                remaining %= output_strides[d];
                let src_coord = coord % original_shape[d];
                src_flat += src_coord * src_strides[d];
            }
            result[flat] = storage[src_flat];
        }

        let new_meta = ft_core::TensorMeta::from_shape(output_shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, result)?,
            requires_grad,
            op: TensorNodeOp::Repeat {
                input,
                original_shape,
                repeats,
            },
        });
        Ok(out)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn roll(
        &mut self,
        input: TensorNodeId,
        shift: isize,
        dim: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, storage, shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape().to_vec();
            let ndim = shape.len();
            if dim >= ndim {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(
                    ft_kernel_cpu::KernelError::InvalidDimension { dim, ndim },
                )));
            }
            (
                input_node.requires_grad,
                input_node.tensor.contiguous_values()?.to_vec(),
                shape,
                meta.dtype(),
                meta.device(),
            )
        };

        let numel = Self::checked_shape_numel(&shape, "roll shape volume overflow")?;
        let strides = ft_core::contiguous_strides(&shape);
        let ndim = shape.len();
        let dim_size = shape[dim];
        let dim_size_i = isize::try_from(dim_size)
            .map_err(|_| Self::shape_overflow_error("roll dimension size exceeds isize range"))?;
        let mut result = vec![0.0; numel];

        if dim_size > 0 {
            for flat in 0..numel {
                let mut remaining = flat;
                let mut coords = vec![0usize; ndim];
                for d in 0..ndim {
                    coords[d] = remaining / strides[d];
                    remaining %= strides[d];
                }

                let old_coord = isize::try_from(coords[dim]).map_err(|_| {
                    Self::shape_overflow_error("roll coordinate exceeds isize range")
                })?;
                let new_coord = ((old_coord + shift) % dim_size_i + dim_size_i) as usize % dim_size;
                coords[dim] = new_coord;

                let mut dst_flat = 0;
                for d in 0..ndim {
                    dst_flat += coords[d] * strides[d];
                }
                result[dst_flat] = storage[flat];
            }
        }

        let new_meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, result)?,
            requires_grad,
            op: TensorNodeOp::Roll { input, shift, dim },
        });
        Ok(out)
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad = lhs_node.requires_grad || rhs_node.requires_grad;
            let lhs_meta = lhs_node.tensor.meta().clone();
            let rhs_meta = rhs_node.tensor.meta().clone();
            let outcome = dispatch_tensor_binary_contiguous_f64(
                op,
                mode,
                lhs_node.tensor.storage(),
                rhs_node.tensor.storage(),
                &lhs_meta,
                &rhs_meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let output_shape = match op {
                BinaryOp::MatMul => {
                    let (m, _, n) = Self::matmul_dims(lhs_meta.shape(), rhs_meta.shape())?;
                    vec![m, n]
                }
                BinaryOp::Dot => vec![1],
                BinaryOp::Outer => {
                    vec![lhs_meta.shape()[0], rhs_meta.shape()[0]]
                }
                BinaryOp::Bmm => {
                    vec![
                        lhs_meta.shape()[0],
                        lhs_meta.shape()[1],
                        rhs_meta.shape()[2],
                    ]
                }
                _ => lhs_meta.shape().to_vec(),
            };
            (
                requires_grad,
                output_shape,
                lhs_meta.dtype(),
                lhs_meta.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.values,
            )?,
            requires_grad,
            op: match op {
                BinaryOp::Add => TensorNodeOp::Add { lhs, rhs },
                BinaryOp::Sub => TensorNodeOp::Sub { lhs, rhs },
                BinaryOp::Div => TensorNodeOp::Div { lhs, rhs },
                BinaryOp::Mul => TensorNodeOp::Mul { lhs, rhs },
                BinaryOp::MatMul => TensorNodeOp::MatMul { lhs, rhs },
                BinaryOp::Dot => TensorNodeOp::Dot { lhs, rhs },
                BinaryOp::Outer => TensorNodeOp::Outer { lhs, rhs },
                BinaryOp::Bmm => TensorNodeOp::Bmm { lhs, rhs },
                BinaryOp::Min => TensorNodeOp::Min { lhs, rhs },
                BinaryOp::Max => TensorNodeOp::Max { lhs, rhs },
            },
        });

        Ok((
            out,
            TensorOperationEvent {
                op,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn backward(&self, root: TensorNodeId) -> Result<TensorBackwardReport, AutogradError> {
        self.backward_with_options(root, BackwardOptions::strict_default())
    }

    #[allow(clippy::needless_range_loop)]
    pub fn backward_with_options(
        &self,
        root: TensorNodeId,
        options: BackwardOptions,
    ) -> Result<TensorBackwardReport, AutogradError> {
        if root.0 >= self.nodes.len() {
            return Err(AutogradError::UnknownTensorNode(root));
        }
        if !self.nodes[root.0].requires_grad {
            return Err(AutogradError::TensorRootDoesNotRequireGrad { node: root });
        }

        let mut reentrant_guard_triggered = false;
        let mut hardened_fallback_used = false;
        if options.current_reentrant_depth > options.max_reentrant_depth {
            match options.policy {
                ReentrantPolicy::StrictFail => {
                    return Err(AutogradError::ReentrantDepthExceeded {
                        current: options.current_reentrant_depth,
                        max: options.max_reentrant_depth,
                    });
                }
                ReentrantPolicy::HardenedBoundedFallback => {
                    reentrant_guard_triggered = true;
                    hardened_fallback_used = true;
                }
            }
        }

        let reentrant_depth = options
            .current_reentrant_depth
            .min(options.max_reentrant_depth);
        let reachable = self.compute_reachable(root)?;
        let mut pending = self.compute_dependencies(&reachable)?;
        let dependency_snapshot = pending.clone();

        let mut grads = self
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, node)| {
                if reachable[idx] {
                    vec![0.0; node.tensor.meta().numel()]
                } else {
                    Vec::new()
                }
            })
            .collect::<Vec<_>>();
        grads[root.0] = vec![1.0; self.nodes[root.0].tensor.meta().numel()];

        let mut queue = TensorReadyQueue::with_capacity(self.nodes.len().max(1));
        queue.push(root);

        let mut steps = Vec::with_capacity(self.nodes.len());
        let mut execution_order = Vec::with_capacity(self.nodes.len());

        while let Some(node_id) = queue.pop() {
            let incoming = grads[node_id.0].clone();
            execution_order.push(node_id);

            match self.nodes[node_id.0].op {
                TensorNodeOp::Leaf => {
                    if self.nodes[node_id.0].requires_grad {
                        steps.push(TensorBackwardStep {
                            node: node_id,
                            incoming_grad_len: incoming.len(),
                            rule: "leaf",
                        });
                    }
                }
                TensorNodeOp::Add { lhs, rhs } => {
                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &incoming)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &incoming)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(a+b)/da=1; d(a+b)/db=1",
                    });
                }
                TensorNodeOp::Sub { lhs, rhs } => {
                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &incoming)?;
                    let rhs_contrib = incoming.iter().map(|value| -*value).collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(a-b)/da=1; d(a-b)/db=-1",
                    });
                }
                TensorNodeOp::Div { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(lhs, lhs_values.len(), incoming.len())?;
                    Self::ensure_tensor_len(rhs, rhs_values.len(), incoming.len())?;

                    let lhs_contrib = incoming
                        .iter()
                        .zip(rhs_values.iter())
                        .map(|(grad, rhs_value)| grad / rhs_value)
                        .collect::<Vec<_>>();
                    let rhs_contrib = incoming
                        .iter()
                        .zip(lhs_values.iter())
                        .zip(rhs_values.iter())
                        .map(|((grad, lhs_value), rhs_value)| {
                            -grad * lhs_value / (rhs_value * rhs_value)
                        })
                        .collect::<Vec<_>>();

                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(a/b)/da=1/b; d(a/b)/db=-(a/b^2)",
                    });
                }
                TensorNodeOp::Mul { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(lhs, lhs_values.len(), incoming.len())?;
                    Self::ensure_tensor_len(rhs, rhs_values.len(), incoming.len())?;

                    let lhs_contrib = incoming
                        .iter()
                        .zip(rhs_values.iter())
                        .map(|(grad, rhs_value)| grad * rhs_value)
                        .collect::<Vec<_>>();
                    let rhs_contrib = incoming
                        .iter()
                        .zip(lhs_values.iter())
                        .map(|(grad, lhs_value)| grad * lhs_value)
                        .collect::<Vec<_>>();

                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(a*b)/da=b; d(a*b)/db=a",
                    });
                }
                TensorNodeOp::MatMul { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values()?;
                    let lhs_shape = self.nodes[lhs.0].tensor.meta().shape();
                    let rhs_shape = self.nodes[rhs.0].tensor.meta().shape();
                    let (m, k, n) = Self::matmul_dims(lhs_shape, rhs_shape)?;

                    Self::ensure_tensor_len(lhs, lhs_values.len(), m.saturating_mul(k))?;
                    Self::ensure_tensor_len(rhs, rhs_values.len(), k.saturating_mul(n))?;
                    Self::ensure_tensor_len(node_id, incoming.len(), m.saturating_mul(n))?;

                    let mut lhs_contrib = vec![0.0; m.saturating_mul(k)];
                    let mut rhs_contrib = vec![0.0; k.saturating_mul(n)];

                    for row in 0..m {
                        for inner in 0..k {
                            let mut acc = 0.0;
                            for col in 0..n {
                                let grad_out = incoming[row * n + col];
                                let rhs_value = rhs_values[inner * n + col];
                                acc += grad_out * rhs_value;
                            }
                            lhs_contrib[row * k + inner] = acc;
                        }
                    }

                    for inner in 0..k {
                        for col in 0..n {
                            let mut acc = 0.0;
                            for row in 0..m {
                                let lhs_value = lhs_values[row * k + inner];
                                let grad_out = incoming[row * n + col];
                                acc += lhs_value * grad_out;
                            }
                            rhs_contrib[inner * n + col] = acc;
                        }
                    }

                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(A@B)/dA=dOut@B^T; d(A@B)/dB=A^T@dOut",
                    });
                }
                TensorNodeOp::Dot { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values()?;
                    let grad_out = incoming[0];

                    let lhs_contrib: Vec<f64> = rhs_values.iter().map(|&v| grad_out * v).collect();
                    let rhs_contrib: Vec<f64> = lhs_values.iter().map(|&v| grad_out * v).collect();

                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(dot(a,b))/da=grad_out*b; d(dot(a,b))/db=grad_out*a",
                    });
                }
                TensorNodeOp::Outer { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values()?;
                    let m = lhs_values.len();
                    let n = rhs_values.len();

                    Self::ensure_tensor_len(node_id, incoming.len(), m.saturating_mul(n))?;

                    let mut lhs_contrib = vec![0.0; m];
                    let mut rhs_contrib = vec![0.0; n];

                    for i in 0..m {
                        let mut acc = 0.0;
                        for j in 0..n {
                            acc += incoming[i * n + j] * rhs_values[j];
                        }
                        lhs_contrib[i] = acc;
                    }

                    for j in 0..n {
                        let mut acc = 0.0;
                        for i in 0..m {
                            acc += incoming[i * n + j] * lhs_values[i];
                        }
                        rhs_contrib[j] = acc;
                    }

                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(outer(a,b))/da=dOut@b; d(outer(a,b))/db=dOut^T@a",
                    });
                }
                TensorNodeOp::Bmm { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values()?;
                    let lhs_shape = self.nodes[lhs.0].tensor.meta().shape();
                    let rhs_shape = self.nodes[rhs.0].tensor.meta().shape();
                    let batch = lhs_shape[0];
                    let m = lhs_shape[1];
                    let k = lhs_shape[2];
                    let n = rhs_shape[2];

                    let lhs_batch_stride = m * k;
                    let rhs_batch_stride = k * n;
                    let out_batch_stride = m * n;

                    let mut lhs_contrib = vec![0.0; batch * lhs_batch_stride];
                    let mut rhs_contrib = vec![0.0; batch * rhs_batch_stride];

                    for b in 0..batch {
                        let lhs_base = b * lhs_batch_stride;
                        let rhs_base = b * rhs_batch_stride;
                        let out_base = b * out_batch_stride;

                        // grad_lhs[b] = grad_out[b] @ rhs[b]^T
                        for row in 0..m {
                            for inner in 0..k {
                                let mut acc = 0.0;
                                for col in 0..n {
                                    acc += incoming[out_base + row * n + col]
                                        * rhs_values[rhs_base + inner * n + col];
                                }
                                lhs_contrib[lhs_base + row * k + inner] = acc;
                            }
                        }

                        // grad_rhs[b] = lhs[b]^T @ grad_out[b]
                        for inner in 0..k {
                            for col in 0..n {
                                let mut acc = 0.0;
                                for row in 0..m {
                                    acc += lhs_values[lhs_base + row * k + inner]
                                        * incoming[out_base + row * n + col];
                                }
                                rhs_contrib[rhs_base + inner * n + col] = acc;
                            }
                        }
                    }

                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(bmm(A,B))/dA=dOut@B^T; d(bmm(A,B))/dB=A^T@dOut (batched)",
                    });
                }
                TensorNodeOp::Neg { input } => {
                    let neg_contrib = incoming.iter().map(|value| -*value).collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &neg_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(-x)/dx=-1",
                    });
                }
                TensorNodeOp::Abs { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;

                    let abs_contrib = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(grad, value)| {
                            let sign = if value.is_nan() {
                                f64::NAN
                            } else if *value > 0.0 {
                                1.0
                            } else if *value < 0.0 {
                                -1.0
                            } else {
                                0.0
                            };
                            grad * sign
                        })
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &abs_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d|x|/dx=sign(x)",
                    });
                }
                TensorNodeOp::Exp { input } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(node_id, output_values.len(), incoming.len())?;

                    let exp_contrib = incoming
                        .iter()
                        .zip(output_values.iter())
                        .map(|(grad, out_val)| grad * out_val)
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &exp_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(exp(x))/dx=exp(x)",
                    });
                }
                TensorNodeOp::Log { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;

                    let log_contrib = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(grad, val)| grad / val)
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &log_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(ln(x))/dx=1/x",
                    });
                }
                TensorNodeOp::Relu { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;

                    let relu_contrib = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(grad, val)| {
                            if val.is_nan() {
                                f64::NAN
                            } else if *val > 0.0 {
                                *grad
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &relu_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(relu(x))/dx=1 if x>0 else 0",
                    });
                }
                TensorNodeOp::Sigmoid { input } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(node_id, output_values.len(), incoming.len())?;

                    let sigmoid_contrib = incoming
                        .iter()
                        .zip(output_values.iter())
                        .map(|(grad, s)| grad * s * (1.0 - s))
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &sigmoid_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(sigmoid(x))/dx=sigmoid(x)*(1-sigmoid(x))",
                    });
                }
                TensorNodeOp::Tanh { input } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(node_id, output_values.len(), incoming.len())?;

                    let tanh_contrib = incoming
                        .iter()
                        .zip(output_values.iter())
                        .map(|(grad, t)| grad * (1.0 - t * t))
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &tanh_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(tanh(x))/dx=1-tanh(x)^2",
                    });
                }
                TensorNodeOp::Sin { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;

                    let sin_contrib = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(grad, x)| grad * x.cos())
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &sin_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(sin(x))/dx=cos(x)",
                    });
                }
                TensorNodeOp::Cos { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;

                    let cos_contrib = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(grad, x)| grad * (-x.sin()))
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &cos_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(cos(x))/dx=-sin(x)",
                    });
                }
                TensorNodeOp::Tan { input } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(node_id, output_values.len(), incoming.len())?;

                    let tan_contrib = incoming
                        .iter()
                        .zip(output_values.iter())
                        .map(|(grad, t)| grad * (1.0 + t * t))
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &tan_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(tan(x))/dx=1+tan(x)^2",
                    });
                }
                TensorNodeOp::Floor { input }
                | TensorNodeOp::Ceil { input }
                | TensorNodeOp::Round { input } => {
                    // Gradient is zero almost everywhere (step functions)
                    let zero_contrib = vec![0.0; incoming.len()];
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &zero_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(floor|ceil|round)/dx=0",
                    });
                }
                TensorNodeOp::Log2 { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g / (x * std::f64::consts::LN_2))
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(log2(x))/dx=1/(x*ln(2))",
                    });
                }
                TensorNodeOp::Log10 { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g / (x * std::f64::consts::LN_10))
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(log10(x))/dx=1/(x*ln(10))",
                    });
                }
                TensorNodeOp::Log1p { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g / (1.0 + x))
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(log1p(x))/dx=1/(1+x)",
                    });
                }
                TensorNodeOp::Expm1 { input } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(node_id, output_values.len(), incoming.len())?;
                    // d/dx expm1(x) = exp(x) = expm1(x) + 1
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(output_values.iter())
                        .map(|(g, y)| g * (y + 1.0))
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(expm1(x))/dx=exp(x)=expm1(x)+1",
                    });
                }
                TensorNodeOp::Sign { input } => {
                    // sign is a step function, gradient is 0
                    let contrib = vec![0.0; incoming.len()];
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(sign(x))/dx=0",
                    });
                }
                TensorNodeOp::Trunc { input } => {
                    // trunc is a step function, gradient is 0
                    let contrib = vec![0.0; incoming.len()];
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(trunc(x))/dx=0",
                    });
                }
                TensorNodeOp::Frac { input } => {
                    // frac(x) = x - floor(x), d/dx = 1
                    let contrib: Vec<f64> = incoming.to_vec();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(frac(x))/dx=1",
                    });
                }
                TensorNodeOp::Asin { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    // d/dx asin(x) = 1/sqrt(1-x^2)
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g / (1.0 - x * x).sqrt())
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(asin(x))/dx=1/sqrt(1-x^2)",
                    });
                }
                TensorNodeOp::Acos { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    // d/dx acos(x) = -1/sqrt(1-x^2)
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| -g / (1.0 - x * x).sqrt())
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(acos(x))/dx=-1/sqrt(1-x^2)",
                    });
                }
                TensorNodeOp::Atan { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    // d/dx atan(x) = 1/(1+x^2)
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g / (1.0 + x * x))
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(atan(x))/dx=1/(1+x^2)",
                    });
                }
                TensorNodeOp::Sinh { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    // d/dx sinh(x) = cosh(x)
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g * x.cosh())
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(sinh(x))/dx=cosh(x)",
                    });
                }
                TensorNodeOp::Cosh { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    // d/dx cosh(x) = sinh(x)
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g * x.sinh())
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(cosh(x))/dx=sinh(x)",
                    });
                }
                TensorNodeOp::Gelu { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let c = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, &x)| {
                            let k = c * (x + 0.044715 * x * x * x);
                            let t = k.tanh();
                            let dk = c * (1.0 + 0.134145 * x * x);
                            g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dk)
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(gelu(x))/dx",
                    });
                }
                TensorNodeOp::Silu { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, &x)| {
                            let s = 1.0 / (1.0 + (-x).exp());
                            g * s * (1.0 + x * (1.0 - s))
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(silu(x))/dx=sigmoid(x)*(1+x*(1-sigmoid(x)))",
                    });
                }
                TensorNodeOp::LeakyRelu { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| {
                            g * if x.is_nan() {
                                f64::NAN
                            } else if *x >= 0.0 {
                                1.0
                            } else {
                                0.01
                            }
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(leaky_relu(x))/dx=1|0.01",
                    });
                }
                TensorNodeOp::Elu { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .zip(output_values.iter())
                        .map(|((g, x), y)| {
                            g * if x.is_nan() {
                                f64::NAN
                            } else if *x > 0.0 {
                                1.0
                            } else {
                                y + 1.0
                            }
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(elu(x))/dx=1|output+alpha",
                    });
                }
                TensorNodeOp::Sqrt { input } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(node_id, output_values.len(), incoming.len())?;

                    let sqrt_contrib = incoming
                        .iter()
                        .zip(output_values.iter())
                        .map(|(grad, s)| grad * 0.5 / s)
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &sqrt_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(sqrt(x))/dx=0.5/sqrt(x)",
                    });
                }
                TensorNodeOp::Reciprocal { input } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(node_id, output_values.len(), incoming.len())?;

                    let recip_contrib = incoming
                        .iter()
                        .zip(output_values.iter())
                        .map(|(grad, r)| grad * (-r * r))
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &recip_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(1/x)/dx=-1/x^2",
                    });
                }
                TensorNodeOp::Pow { input, exponent } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;

                    let pow_contrib = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(grad, x)| {
                            if exponent == 0.0 {
                                0.0
                            } else {
                                grad * exponent * x.powf(exponent - 1.0)
                            }
                        })
                        .collect::<Vec<_>>();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &pow_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(x^n)/dx=n*x^(n-1)",
                    });
                }
                TensorNodeOp::Min { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(lhs, lhs_values.len(), incoming.len())?;

                    let lhs_contrib: Vec<f64> = incoming
                        .iter()
                        .zip(lhs_values.iter().zip(rhs_values.iter()))
                        .map(|(grad, (a, b))| {
                            if a.is_nan() || b.is_nan() {
                                f64::NAN
                            } else if a < b {
                                *grad
                            } else if a == b {
                                *grad * 0.5
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let rhs_contrib: Vec<f64> = incoming
                        .iter()
                        .zip(lhs_values.iter().zip(rhs_values.iter()))
                        .map(|(grad, (a, b))| {
                            if a.is_nan() || b.is_nan() {
                                f64::NAN
                            } else if b < a {
                                *grad
                            } else if a == b {
                                *grad * 0.5
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(min(a,b))/da=1(a<b) or 0.5(a=b); db=1(b<a) or 0.5(a=b)",
                    });
                }
                TensorNodeOp::Max { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(lhs, lhs_values.len(), incoming.len())?;

                    let lhs_contrib: Vec<f64> = incoming
                        .iter()
                        .zip(lhs_values.iter().zip(rhs_values.iter()))
                        .map(|(grad, (a, b))| {
                            if a.is_nan() || b.is_nan() {
                                f64::NAN
                            } else if a > b {
                                *grad
                            } else if a == b {
                                *grad * 0.5
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let rhs_contrib: Vec<f64> = incoming
                        .iter()
                        .zip(lhs_values.iter().zip(rhs_values.iter()))
                        .map(|(grad, (a, b))| {
                            if a.is_nan() || b.is_nan() {
                                f64::NAN
                            } else if b > a {
                                *grad
                            } else if a == b {
                                *grad * 0.5
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(max(a,b))/da=1(a>b) or 0.5(a=b); db=1(b>a) or 0.5(a=b)",
                    });
                }
                TensorNodeOp::Clamp {
                    input,
                    min_val,
                    max_val,
                } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;

                    let clamp_contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(grad, x)| {
                            if x.is_nan() {
                                f64::NAN
                            } else if *x >= min_val && *x <= max_val {
                                *grad
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &clamp_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(clamp(x,min,max))/dx=1 if min<=x<=max else 0",
                    });
                }
                TensorNodeOp::Trace {
                    input,
                    ref input_shape,
                } => {
                    let grad_scalar = incoming[0];
                    let rows = input_shape[0];
                    let cols = input_shape[1];
                    let numel = rows * cols;
                    let diag_len = rows.min(cols);
                    let mut trace_contrib = vec![0.0; numel];
                    for i in 0..diag_len {
                        trace_contrib[i * cols + i] = grad_scalar;
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &trace_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(trace(X))/dX=grad_out*I",
                    });
                }
                TensorNodeOp::Sum { input, input_numel } => {
                    let grad_scalar = incoming[0];
                    let sum_contrib = vec![grad_scalar; input_numel];
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &sum_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(sum(x))/dx_i=1",
                    });
                }
                TensorNodeOp::Mean { input, input_numel } => {
                    let grad_scalar = incoming[0];
                    let scale = if input_numel > 0 {
                        1.0 / input_numel as f64
                    } else {
                        0.0
                    };
                    let mean_contrib = vec![grad_scalar * scale; input_numel];
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &mean_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(mean(x))/dx_i=1/n",
                    });
                }
                TensorNodeOp::SumDim {
                    input,
                    dim,
                    ref input_shape,
                } => {
                    let reduce_size = input_shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "sum_dim backward shape volume overflow",
                    )?;
                    let expected_incoming = Self::checked_mul_usize(
                        outer_size,
                        inner_size,
                        "sum_dim backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_incoming, incoming.len())?;
                    let mut sum_dim_contrib = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let grad_val = incoming[outer * inner_size + inner];
                            for r in 0..reduce_size {
                                sum_dim_contrib
                                    [outer * reduce_size * inner_size + r * inner_size + inner] =
                                    grad_val;
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &sum_dim_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(sum_dim(x))/dx=broadcast_grad_along_dim",
                    });
                }
                TensorNodeOp::MeanDim {
                    input,
                    dim,
                    ref input_shape,
                } => {
                    let reduce_size = input_shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "mean_dim backward shape volume overflow",
                    )?;
                    let expected_incoming = Self::checked_mul_usize(
                        outer_size,
                        inner_size,
                        "mean_dim backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_incoming, incoming.len())?;
                    let scale = if reduce_size > 0 {
                        1.0 / reduce_size as f64
                    } else {
                        0.0
                    };
                    let mut mean_dim_contrib = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let grad_val = incoming[outer * inner_size + inner] * scale;
                            for r in 0..reduce_size {
                                mean_dim_contrib
                                    [outer * reduce_size * inner_size + r * inner_size + inner] =
                                    grad_val;
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(
                        input,
                        &mut grads[input.0],
                        &mean_dim_contrib,
                    )?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(mean_dim(x))/dx=broadcast_grad_along_dim/reduce_size",
                    });
                }
                TensorNodeOp::ProdDim {
                    input,
                    dim,
                    ref input_shape,
                } => {
                    let reduce_size = input_shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "prod_dim backward shape volume overflow",
                    )?;
                    let expected_incoming = Self::checked_mul_usize(
                        outer_size,
                        inner_size,
                        "prod_dim backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_incoming, incoming.len())?;
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    let mut prod_dim_contrib = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let out_idx = outer * inner_size + inner;
                            let grad_val = incoming[out_idx];
                            let prod_val = output_values[out_idx];
                            // Count zeros in this slice
                            let mut zero_count = 0;
                            let mut prod_no_zero = 1.0;
                            for r in 0..reduce_size {
                                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                                let v = input_values[idx];
                                if v == 0.0 {
                                    zero_count += 1;
                                } else {
                                    prod_no_zero *= v;
                                }
                            }
                            for r in 0..reduce_size {
                                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                                let v = input_values[idx];
                                prod_dim_contrib[idx] = if zero_count == 0 {
                                    grad_val * prod_val / v
                                } else if zero_count == 1 && v == 0.0 {
                                    grad_val * prod_no_zero
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(
                        input,
                        &mut grads[input.0],
                        &prod_dim_contrib,
                    )?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(prod_dim(x))/dx_i=prod/x_i",
                    });
                }
                TensorNodeOp::VarDim {
                    input,
                    dim,
                    ref input_shape,
                } => {
                    let reduce_size = input_shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "var_dim backward shape volume overflow",
                    )?;
                    let expected_incoming = Self::checked_mul_usize(
                        outer_size,
                        inner_size,
                        "var_dim backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_incoming, incoming.len())?;
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    let correction = if reduce_size > 1 {
                        (reduce_size - 1) as f64
                    } else {
                        1.0
                    };
                    let mut var_dim_contrib = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let grad_val = incoming[outer * inner_size + inner];
                            // Compute mean along dim
                            let mut sum = 0.0;
                            for r in 0..reduce_size {
                                sum += input_values
                                    [outer * reduce_size * inner_size + r * inner_size + inner];
                            }
                            let mean = sum / reduce_size as f64;
                            for r in 0..reduce_size {
                                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                                let diff = input_values[idx] - mean;
                                var_dim_contrib[idx] = grad_val * 2.0 * diff / correction;
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &var_dim_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(var_dim(x))/dx_i=2*(x_i-mean)/(n-1)",
                    });
                }
                TensorNodeOp::StdDim {
                    input,
                    dim,
                    ref input_shape,
                } => {
                    let reduce_size = input_shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "std_dim backward shape volume overflow",
                    )?;
                    let expected_incoming = Self::checked_mul_usize(
                        outer_size,
                        inner_size,
                        "std_dim backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_incoming, incoming.len())?;
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    let correction = if reduce_size > 1 {
                        (reduce_size - 1) as f64
                    } else {
                        1.0
                    };
                    let mut std_dim_contrib = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let out_idx = outer * inner_size + inner;
                            let grad_val = incoming[out_idx];
                            let std_val = output_values[out_idx];
                            // Compute mean along dim
                            let mut sum = 0.0;
                            for r in 0..reduce_size {
                                sum += input_values
                                    [outer * reduce_size * inner_size + r * inner_size + inner];
                            }
                            let mean = sum / reduce_size as f64;
                            for r in 0..reduce_size {
                                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                                let diff = input_values[idx] - mean;
                                std_dim_contrib[idx] = if std_val != 0.0 {
                                    grad_val * diff / (correction * std_val)
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &std_dim_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(std_dim(x))/dx_i=(x_i-mean)/((n-1)*std)",
                    });
                }
                TensorNodeOp::CumSum { input, dim } => {
                    // Backward of cumsum is reverse cumsum of the incoming gradient
                    let shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let dim_size = shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        shape.as_slice(),
                        dim,
                        "cumsum backward shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, input_numel, incoming.len())?;
                    let mut cumsum_grad = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let mut acc = 0.0;
                            for d in (0..dim_size).rev() {
                                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                                acc += incoming[idx];
                                cumsum_grad[idx] = acc;
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &cumsum_grad)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(cumsum(x))/dx = reverse_cumsum(grad)",
                    });
                }
                TensorNodeOp::CumProd { input, dim } => {
                    // Backward of cumprod uses: grad_input[i] = sum_{j>=i} grad_output[j] * output[j] / input[i]
                    let shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let dim_size = shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        shape.as_slice(),
                        dim,
                        "cumprod backward shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, input_numel, incoming.len())?;
                    let input_values = self.nodes[input.0].tensor.contiguous_values()?;
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    let mut cumprod_grad = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let mut acc = 0.0;
                            for d in (0..dim_size).rev() {
                                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                                acc += incoming[idx] * output_values[idx];
                                let inp = input_values[idx];
                                if inp.abs() > f64::EPSILON {
                                    cumprod_grad[idx] = acc / inp;
                                } else {
                                    // Handle zero input: compute gradient by direct summation
                                    let mut sum = 0.0;
                                    for j in d..dim_size {
                                        let j_idx =
                                            outer * dim_size * inner_size + j * inner_size + inner;
                                        let mut prod = 1.0;
                                        for k in d..=j {
                                            if k != d {
                                                let k_idx = outer * dim_size * inner_size
                                                    + k * inner_size
                                                    + inner;
                                                prod *= input_values[k_idx];
                                            }
                                        }
                                        if d > 0 {
                                            let prev_idx = outer * dim_size * inner_size
                                                + (d - 1) * inner_size
                                                + inner;
                                            prod *= output_values[prev_idx];
                                        }
                                        sum += incoming[j_idx] * prod;
                                    }
                                    cumprod_grad[idx] = sum;
                                }
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &cumprod_grad)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(cumprod(x))/dx = reverse_cumsum(grad*output)/input",
                    });
                }
                TensorNodeOp::Where { condition, x, y } => {
                    // Gradient flows to x where condition is true, to y where condition is false
                    let cond_vals = self.nodes[condition.0].tensor.contiguous_values()?;
                    let numel = incoming.len();
                    Self::ensure_tensor_len(condition, numel, cond_vals.len())?;

                    let mut x_grad = vec![0.0; numel];
                    let mut y_grad = vec![0.0; numel];

                    for i in 0..numel {
                        if cond_vals[i] != 0.0 {
                            x_grad[i] = incoming[i];
                        } else {
                            y_grad[i] = incoming[i];
                        }
                    }

                    Self::accumulate_tensor_gradient(x, &mut grads[x.0], &x_grad)?;
                    Self::accumulate_tensor_gradient(y, &mut grads[y.0], &y_grad)?;

                    Self::complete_dependency(&mut pending, x, &mut queue)?;
                    Self::complete_dependency(&mut pending, y, &mut queue)?;
                    // condition doesn't need gradient (it's a boolean mask)
                    Self::complete_dependency(&mut pending, condition, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(where(c,x,y))/dx = grad*c, d/dy = grad*(1-c)",
                    });
                }
                TensorNodeOp::Sort {
                    input,
                    dim,
                    ref indices,
                    ref input_shape,
                } => {
                    let ndim = input_shape.len();
                    if dim >= ndim {
                        return Err(AutogradError::Dispatch(DispatchError::Kernel(
                            ft_kernel_cpu::KernelError::InvalidDimension { dim, ndim },
                        )));
                    }
                    // Scatter incoming gradient back to original positions
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "sort backward shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, input_numel, incoming.len())?;
                    Self::ensure_tensor_len(node_id, input_numel, indices.len())?;
                    let dim_size = input_shape[dim];
                    let mut grad_input = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            for d in 0..dim_size {
                                let out_idx =
                                    outer * dim_size * inner_size + d * inner_size + inner;
                                let orig_d = indices[out_idx];
                                if orig_d >= dim_size {
                                    return Err(AutogradError::Dispatch(DispatchError::Kernel(
                                        ft_kernel_cpu::KernelError::InvalidDimension {
                                            dim: orig_d,
                                            ndim: dim_size,
                                        },
                                    )));
                                }
                                let in_idx =
                                    outer * dim_size * inner_size + orig_d * inner_size + inner;
                                grad_input[in_idx] += incoming[out_idx];
                            }
                        }
                    }

                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &grad_input)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(sort(x))/dx = scatter(grad, indices)",
                    });
                }
                TensorNodeOp::TopK {
                    input,
                    dim,
                    k,
                    ref indices,
                    ref input_shape,
                } => {
                    let ndim = input_shape.len();
                    if dim >= ndim {
                        return Err(AutogradError::Dispatch(DispatchError::Kernel(
                            ft_kernel_cpu::KernelError::InvalidDimension { dim, ndim },
                        )));
                    }
                    // Scatter incoming gradient back to original positions
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "topk backward shape volume overflow",
                    )?;
                    let input_dim_size = input_shape[dim];
                    if k > input_dim_size {
                        return Err(AutogradError::Dispatch(DispatchError::Kernel(
                            ft_kernel_cpu::KernelError::ShapeMismatch {
                                lhs: vec![k],
                                rhs: vec![input_dim_size],
                            },
                        )));
                    }
                    let expected_out_numel = Self::checked_mul_usize(
                        Self::checked_mul_usize(
                            outer_size,
                            k,
                            "topk backward shape multiplication overflow",
                        )?,
                        inner_size,
                        "topk backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_out_numel, incoming.len())?;
                    Self::ensure_tensor_len(node_id, expected_out_numel, indices.len())?;
                    let mut grad_input = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            for d in 0..k {
                                let out_idx = outer * k * inner_size + d * inner_size + inner;
                                let orig_d = indices[out_idx];
                                if orig_d >= input_dim_size {
                                    return Err(AutogradError::Dispatch(DispatchError::Kernel(
                                        ft_kernel_cpu::KernelError::InvalidDimension {
                                            dim: orig_d,
                                            ndim: input_dim_size,
                                        },
                                    )));
                                }
                                let in_idx = outer * input_dim_size * inner_size
                                    + orig_d * inner_size
                                    + inner;
                                grad_input[in_idx] += incoming[out_idx];
                            }
                        }
                    }

                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &grad_input)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(topk(x))/dx = scatter(grad, indices)",
                    });
                }
                TensorNodeOp::Softmax { input, dim } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    let shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let reduce_size = shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        shape.as_slice(),
                        dim,
                        "softmax backward shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, input_numel, incoming.len())?;
                    Self::ensure_tensor_len(node_id, input_numel, output_values.len())?;
                    let mut softmax_contrib = vec![0.0; input_numel];

                    // grad_input_i = output_i * (grad_i - sum_j(grad_j * output_j))
                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let mut dot = 0.0;
                            for r in 0..reduce_size {
                                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                                dot += incoming[idx] * output_values[idx];
                            }
                            for r in 0..reduce_size {
                                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                                softmax_contrib[idx] = output_values[idx] * (incoming[idx] - dot);
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &softmax_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(softmax(x))/dx_i=s_i*(grad_i-sum(grad*s))",
                    });
                }
                TensorNodeOp::LogSoftmax { input, dim } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values()?;
                    let shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let reduce_size = shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        shape.as_slice(),
                        dim,
                        "log_softmax backward shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, input_numel, incoming.len())?;
                    Self::ensure_tensor_len(node_id, input_numel, output_values.len())?;
                    let mut logsoftmax_contrib = vec![0.0; input_numel];

                    // grad_input_i = grad_i - exp(output_i) * sum_j(grad_j)
                    // where exp(output_i) = softmax_i
                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let mut grad_sum = 0.0;
                            for r in 0..reduce_size {
                                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                                grad_sum += incoming[idx];
                            }
                            for r in 0..reduce_size {
                                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                                let softmax_i = output_values[idx].exp();
                                logsoftmax_contrib[idx] = incoming[idx] - softmax_i * grad_sum;
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(
                        input,
                        &mut grads[input.0],
                        &logsoftmax_contrib,
                    )?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(log_softmax(x))/dx_i=grad_i-softmax_i*sum(grad)",
                    });
                }
                TensorNodeOp::Cat {
                    ref inputs,
                    dim,
                    ref input_dim_sizes,
                } => {
                    // Split gradient along the cat dimension
                    let shape = self.nodes[node_id.0].tensor.meta().shape().to_vec();
                    let (outer_size, inner_size, output_numel) = Self::checked_dim_loop_sizes(
                        shape.as_slice(),
                        dim,
                        "cat backward shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, output_numel, incoming.len())?;

                    let mut offset = 0;
                    for (i, &input_id) in inputs.iter().enumerate() {
                        let cat_size = input_dim_sizes[i];
                        let input_numel = Self::checked_mul_usize(
                            Self::checked_mul_usize(
                                cat_size,
                                outer_size,
                                "cat backward shape multiplication overflow",
                            )?,
                            inner_size,
                            "cat backward shape multiplication overflow",
                        )?;
                        let mut contrib = vec![0.0; input_numel];
                        for outer in 0..outer_size {
                            for r in 0..cat_size {
                                for inner in 0..inner_size {
                                    let grad_idx = outer * shape[dim] * inner_size
                                        + (offset + r) * inner_size
                                        + inner;
                                    let input_idx =
                                        outer * cat_size * inner_size + r * inner_size + inner;
                                    contrib[input_idx] = incoming[grad_idx];
                                }
                            }
                        }
                        Self::accumulate_tensor_gradient(
                            input_id,
                            &mut grads[input_id.0],
                            &contrib,
                        )?;
                        Self::complete_dependency(&mut pending, input_id, &mut queue)?;
                        offset += cat_size;
                    }

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(cat(x...))/dx_i=split_grad_along_dim",
                    });
                }
                TensorNodeOp::Stack { ref inputs, dim } => {
                    // Slice gradient along the stacked dimension
                    let shape = self.nodes[node_id.0].tensor.meta().shape().to_vec();
                    let (outer_size, inner_size, output_numel) = Self::checked_dim_loop_sizes(
                        shape.as_slice(),
                        dim,
                        "stack backward shape volume overflow",
                    )?;
                    let num_inputs = inputs.len();
                    Self::ensure_tensor_len(node_id, output_numel, incoming.len())?;

                    for (i, &input_id) in inputs.iter().enumerate() {
                        let input_numel = Self::checked_mul_usize(
                            outer_size,
                            inner_size,
                            "stack backward shape multiplication overflow",
                        )?;
                        let mut contrib = vec![0.0; input_numel];
                        for outer in 0..outer_size {
                            for inner in 0..inner_size {
                                let grad_idx =
                                    outer * num_inputs * inner_size + i * inner_size + inner;
                                let input_idx = outer * inner_size + inner;
                                contrib[input_idx] = incoming[grad_idx];
                            }
                        }
                        Self::accumulate_tensor_gradient(
                            input_id,
                            &mut grads[input_id.0],
                            &contrib,
                        )?;
                        Self::complete_dependency(&mut pending, input_id, &mut queue)?;
                    }

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(stack(x...))/dx_i=slice_grad_along_dim",
                    });
                }
                TensorNodeOp::Reshape { input, .. }
                | TensorNodeOp::Squeeze { input, .. }
                | TensorNodeOp::Unsqueeze { input, .. } => {
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &incoming)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(shape_op(x))/dx=identity",
                    });
                }
                TensorNodeOp::Transpose { input, dim0, dim1 } => {
                    let output_shape = self.nodes[node_id.0].tensor.meta().shape();
                    let ndim = output_shape.len();
                    let mut inv_perm: Vec<usize> = (0..ndim).collect();
                    inv_perm.swap(dim0, dim1);
                    let permuted_grad = Self::permute_data(&incoming, output_shape, &inv_perm)?;
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &permuted_grad)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(transpose(x))/dx=transpose_inverse(grad)",
                    });
                }
                TensorNodeOp::Permute { input, ref dims } => {
                    let output_shape = self.nodes[node_id.0].tensor.meta().shape();
                    let ndim = dims.len();
                    let mut inv_perm = vec![0usize; ndim];
                    for (i, &d) in dims.iter().enumerate() {
                        inv_perm[d] = i;
                    }
                    let permuted_grad = Self::permute_data(&incoming, output_shape, &inv_perm)?;
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &permuted_grad)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(permute(x))/dx=inverse_permute(grad)",
                    });
                }
                TensorNodeOp::Narrow {
                    input,
                    dim,
                    start,
                    ref original_shape,
                } => {
                    // Backward: place the gradient into a zero tensor at the original
                    // positions, effectively zero-padding back to the original shape.
                    let (outer_size, inner_size, orig_numel) = Self::checked_dim_loop_sizes(
                        original_shape,
                        dim,
                        "narrow backward shape volume overflow",
                    )?;
                    let mut contrib = vec![0.0; orig_numel];
                    let output_shape = self.nodes[node_id.0].tensor.meta().shape();
                    let length = output_shape[dim];
                    let expected_incoming = Self::checked_mul_usize(
                        Self::checked_mul_usize(
                            outer_size,
                            length,
                            "narrow backward shape multiplication overflow",
                        )?,
                        inner_size,
                        "narrow backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_incoming, incoming.len())?;
                    let orig_dim_size = original_shape[dim];

                    for outer in 0..outer_size {
                        for r in 0..length {
                            for inner in 0..inner_size {
                                let grad_idx = outer * length * inner_size + r * inner_size + inner;
                                let orig_idx = outer * orig_dim_size * inner_size
                                    + (start + r) * inner_size
                                    + inner;
                                contrib[orig_idx] = incoming[grad_idx];
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(narrow(x))/dx=zero_pad_grad",
                    });
                }
                TensorNodeOp::Expand {
                    input,
                    ref original_shape,
                } => {
                    // Backward: sum gradient along expanded (broadcast) dimensions.
                    // Dimensions where original_shape[d] == 1 and output > 1 were expanded;
                    // we reduce (sum) the gradient back along those dims.
                    let output_shape = self.nodes[node_id.0].tensor.meta().shape().to_vec();
                    let ndim = output_shape.len();
                    let orig_numel = Self::checked_shape_numel(
                        original_shape,
                        "expand backward input shape volume overflow",
                    )?;
                    let out_numel = Self::checked_shape_numel(
                        output_shape.as_slice(),
                        "expand backward output shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, out_numel, incoming.len())?;
                    let mut contrib = vec![0.0; orig_numel];

                    let grad_strides = ft_core::contiguous_strides(&output_shape);
                    let orig_strides = ft_core::contiguous_strides(original_shape);
                    let mut coords = vec![0usize; ndim];

                    for _ in 0..out_numel {
                        let mut orig_idx = 0usize;
                        let mut grad_idx = 0usize;
                        for d in 0..ndim {
                            grad_idx += coords[d] * grad_strides[d];
                            if original_shape[d] != 1 {
                                orig_idx += coords[d] * orig_strides[d];
                            }
                        }
                        contrib[orig_idx] += incoming[grad_idx];

                        for d in (0..ndim).rev() {
                            coords[d] += 1;
                            if coords[d] < output_shape[d] {
                                break;
                            }
                            coords[d] = 0;
                        }
                    }

                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(expand(x))/dx=sum_broadcast_dims(grad)",
                    });
                }
                TensorNodeOp::Split {
                    input,
                    dim,
                    start,
                    ref original_shape,
                    ..
                } => {
                    // Backward: place chunk gradient back into the original shape
                    // (same as narrow backward).
                    let (outer_size, inner_size, orig_numel) = Self::checked_dim_loop_sizes(
                        original_shape,
                        dim,
                        "split backward shape volume overflow",
                    )?;
                    let mut contrib = vec![0.0; orig_numel];
                    let output_shape = self.nodes[node_id.0].tensor.meta().shape();
                    let length = output_shape[dim];
                    let expected_incoming = Self::checked_mul_usize(
                        Self::checked_mul_usize(
                            outer_size,
                            length,
                            "split backward shape multiplication overflow",
                        )?,
                        inner_size,
                        "split backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_incoming, incoming.len())?;
                    let orig_dim_size = original_shape[dim];

                    for outer in 0..outer_size {
                        for r in 0..length {
                            for inner in 0..inner_size {
                                let grad_idx = outer * length * inner_size + r * inner_size + inner;
                                let orig_idx = outer * orig_dim_size * inner_size
                                    + (start + r) * inner_size
                                    + inner;
                                contrib[orig_idx] = incoming[grad_idx];
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(split(x))/dx=zero_pad_grad",
                    });
                }
                TensorNodeOp::MaxDim {
                    input,
                    dim,
                    ref input_shape,
                    ref indices,
                }
                | TensorNodeOp::MinDim {
                    input,
                    dim,
                    ref input_shape,
                    ref indices,
                } => {
                    let rule = if matches!(self.nodes[node_id.0].op, TensorNodeOp::MaxDim { .. }) {
                        "d(max_dim(x))/dx=scatter_grad_to_max_positions"
                    } else {
                        "d(min_dim(x))/dx=scatter_grad_to_min_positions"
                    };
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "max/min backward shape volume overflow",
                    )?;
                    let reduce_size = input_shape[dim];
                    let expected_out = Self::checked_mul_usize(
                        outer_size,
                        inner_size,
                        "max/min backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_out, incoming.len())?;
                    Self::ensure_tensor_len(node_id, expected_out, indices.len())?;
                    let mut contrib = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let out_idx = outer * inner_size + inner;
                            let selected_f = indices[out_idx];
                            if !selected_f.is_finite()
                                || selected_f < 0.0
                                || selected_f.fract().abs() > f64::EPSILON
                            {
                                return Err(AutogradError::Dispatch(
                                    DispatchKeyError::IncompatibleSet {
                                        reason: "max/min backward received invalid index value",
                                    }
                                    .into(),
                                ));
                            }
                            let selected_r = selected_f as usize;
                            if selected_r >= reduce_size {
                                return Err(AutogradError::Dispatch(
                                    DispatchKeyError::IncompatibleSet {
                                        reason:
                                            "max/min backward received out-of-bounds index value",
                                    }
                                    .into(),
                                ));
                            }
                            let in_idx =
                                outer * reduce_size * inner_size + selected_r * inner_size + inner;
                            contrib[in_idx] = incoming[out_idx];
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule,
                    });
                }
                TensorNodeOp::IndexSelect {
                    input,
                    dim,
                    ref indices,
                    ref input_shape,
                } => {
                    // Backward: scatter_add the gradient back to original positions.
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "index_select backward shape volume overflow",
                    )?;
                    let dim_size = input_shape[dim];
                    let dim_size_i = isize::try_from(dim_size).map_err(|_| {
                        Self::shape_overflow_error(
                            "index_select backward dimension size exceeds isize range",
                        )
                    })?;
                    let num_indices = indices.len();
                    let expected_incoming = Self::checked_mul_usize(
                        Self::checked_mul_usize(
                            outer_size,
                            num_indices,
                            "index_select backward shape multiplication overflow",
                        )?,
                        inner_size,
                        "index_select backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_incoming, incoming.len())?;
                    let mut contrib = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for (r, &idx_f) in indices.iter().enumerate() {
                            if !idx_f.is_finite() || idx_f.fract().abs() > f64::EPSILON {
                                return Err(AutogradError::Dispatch(
                                    DispatchKeyError::IncompatibleSet {
                                        reason: "index_select backward received invalid index value",
                                    }
                                    .into(),
                                ));
                            }
                            let mut idx_i = idx_f as isize;
                            if idx_i < 0 {
                                idx_i += dim_size_i;
                            }
                            if idx_i < 0 || idx_i >= dim_size_i {
                                return Err(AutogradError::Dispatch(
                                    DispatchKeyError::IncompatibleSet {
                                        reason:
                                            "index_select backward received out-of-bounds index value",
                                    }
                                    .into(),
                                ));
                            }
                            let idx = idx_i as usize;
                            for inner in 0..inner_size {
                                let grad_pos =
                                    outer * num_indices * inner_size + r * inner_size + inner;
                                let orig_pos =
                                    outer * dim_size * inner_size + idx * inner_size + inner;
                                contrib[orig_pos] += incoming[grad_pos];
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(index_select(x))/dx=scatter_add_grad",
                    });
                }
                TensorNodeOp::Gather {
                    input,
                    dim,
                    ref index,
                    ref index_shape,
                    ref input_shape,
                } => {
                    // Backward: scatter gradient using the same index to original positions.
                    let input_numel = Self::checked_shape_numel(
                        input_shape,
                        "gather backward input shape volume overflow",
                    )?;
                    let dim_size = input_shape[dim];
                    let dim_size_i = isize::try_from(dim_size).map_err(|_| {
                        Self::shape_overflow_error(
                            "gather backward dimension size exceeds isize range",
                        )
                    })?;
                    let idx_dim_size = index_shape[dim];
                    let (outer_size, inner_size, index_numel) = Self::checked_dim_loop_sizes(
                        index_shape,
                        dim,
                        "gather backward index shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, index_numel, incoming.len())?;
                    Self::ensure_tensor_len(node_id, index_numel, index.len())?;
                    let mut contrib = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for r in 0..idx_dim_size {
                            for inner in 0..inner_size {
                                let idx_pos =
                                    outer * idx_dim_size * inner_size + r * inner_size + inner;
                                let selected_f = index[idx_pos];
                                if !selected_f.is_finite()
                                    || selected_f.fract().abs() > f64::EPSILON
                                {
                                    return Err(AutogradError::Dispatch(
                                        DispatchKeyError::IncompatibleSet {
                                            reason: "gather backward received invalid index value",
                                        }
                                        .into(),
                                    ));
                                }
                                let mut selected_i = selected_f as isize;
                                if selected_i < 0 {
                                    selected_i += dim_size_i;
                                }
                                if selected_i < 0 || selected_i >= dim_size_i {
                                    return Err(AutogradError::Dispatch(
                                        DispatchKeyError::IncompatibleSet {
                                            reason:
                                                "gather backward received out-of-bounds index value",
                                        }
                                        .into(),
                                    ));
                                }
                                let selected = selected_i as usize;
                                let orig_pos =
                                    outer * dim_size * inner_size + selected * inner_size + inner;
                                contrib[orig_pos] += incoming[idx_pos];
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(gather(x))/dx=scatter_add_grad",
                    });
                }
                TensorNodeOp::Scatter {
                    input,
                    dim,
                    ref index,
                    ref index_shape,
                    ref input_shape,
                } => {
                    // Backward: gradient passes through for non-overwritten positions;
                    // overwritten positions get zero gradient w.r.t. input.
                    let mut contrib = incoming.to_vec();
                    let input_numel = Self::checked_shape_numel(
                        input_shape,
                        "scatter backward input shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, input_numel, incoming.len())?;
                    let dim_size = input_shape[dim];
                    let dim_size_i = isize::try_from(dim_size).map_err(|_| {
                        Self::shape_overflow_error(
                            "scatter backward dimension size exceeds isize range",
                        )
                    })?;
                    let idx_dim_size = index_shape[dim];
                    let (outer_size, inner_size, index_numel) = Self::checked_dim_loop_sizes(
                        index_shape,
                        dim,
                        "scatter backward index shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, index_numel, index.len())?;

                    for outer in 0..outer_size {
                        for r in 0..idx_dim_size {
                            for inner in 0..inner_size {
                                let idx_pos =
                                    outer * idx_dim_size * inner_size + r * inner_size + inner;
                                let selected_f = index[idx_pos];
                                if !selected_f.is_finite()
                                    || selected_f.fract().abs() > f64::EPSILON
                                {
                                    return Err(AutogradError::Dispatch(
                                        DispatchKeyError::IncompatibleSet {
                                            reason: "scatter backward received invalid index value",
                                        }
                                        .into(),
                                    ));
                                }
                                let mut selected_i = selected_f as isize;
                                if selected_i < 0 {
                                    selected_i += dim_size_i;
                                }
                                if selected_i < 0 || selected_i >= dim_size_i {
                                    return Err(AutogradError::Dispatch(
                                        DispatchKeyError::IncompatibleSet {
                                            reason:
                                                "scatter backward received out-of-bounds index value",
                                        }
                                        .into(),
                                    ));
                                }
                                let selected = selected_i as usize;
                                let orig_pos =
                                    outer * dim_size * inner_size + selected * inner_size + inner;
                                contrib[orig_pos] = 0.0;
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(scatter(x))/dx=mask_overwritten_to_zero",
                    });
                }
                TensorNodeOp::Flip { input, ref dims } => {
                    // flip is self-inverse: grad_input = flip(grad_out, dims)
                    let output_shape = self.nodes[node_id.0].tensor.meta().shape();
                    let strides = ft_core::contiguous_strides(output_shape);
                    let ndim = output_shape.len();
                    let numel = incoming.len();
                    let mut contrib = vec![0.0; numel];

                    for flat in 0..numel {
                        let mut remaining = flat;
                        let mut coords = vec![0usize; ndim];
                        for d in 0..ndim {
                            coords[d] = remaining / strides[d];
                            remaining %= strides[d];
                        }
                        let mut src_flat = 0;
                        for d in 0..ndim {
                            let coord = if dims.contains(&d) {
                                output_shape[d] - 1 - coords[d]
                            } else {
                                coords[d]
                            };
                            src_flat += coord * strides[d];
                        }
                        contrib[src_flat] = incoming[flat];
                    }

                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(flip(x,dims))/dx=flip(grad,dims)",
                    });
                }
                TensorNodeOp::Repeat {
                    input,
                    ref original_shape,
                    ref repeats,
                } => {
                    // Sum gradients over the repeated tiles
                    let ndim = original_shape.len();
                    let input_numel = Self::checked_shape_numel(
                        original_shape,
                        "repeat backward input shape volume overflow",
                    )?;
                    let mut output_shape = Vec::with_capacity(ndim);
                    for (&size, &repeat) in original_shape.iter().zip(repeats.iter()) {
                        output_shape.push(Self::checked_mul_usize(
                            size,
                            repeat,
                            "repeat backward shape multiplication overflow",
                        )?);
                    }
                    let output_numel = Self::checked_shape_numel(
                        &output_shape,
                        "repeat backward output shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, output_numel, incoming.len())?;
                    let output_strides = ft_core::contiguous_strides(&output_shape);
                    let input_strides = ft_core::contiguous_strides(original_shape);

                    let mut contrib = vec![0.0; input_numel];
                    for flat in 0..output_numel {
                        let mut remaining = flat;
                        let mut src_flat = 0;
                        for d in 0..ndim {
                            let coord = remaining / output_strides[d];
                            remaining %= output_strides[d];
                            let src_coord = coord % original_shape[d];
                            src_flat += src_coord * input_strides[d];
                        }
                        contrib[src_flat] += incoming[flat];
                    }

                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(repeat(x))/dx=sum_over_tiles(grad)",
                    });
                }
                TensorNodeOp::Roll { input, shift, dim } => {
                    // Inverse roll: roll(grad_out, -shift, dim)
                    let output_shape = self.nodes[node_id.0].tensor.meta().shape();
                    let strides = ft_core::contiguous_strides(output_shape);
                    let ndim = output_shape.len();
                    let dim_size = output_shape[dim];
                    let dim_size_i = isize::try_from(dim_size).map_err(|_| {
                        Self::shape_overflow_error(
                            "roll backward dimension size exceeds isize range",
                        )
                    })?;
                    let numel = incoming.len();
                    let mut contrib = vec![0.0; numel];

                    if dim_size > 0 {
                        for flat in 0..numel {
                            let mut remaining = flat;
                            let mut coords = vec![0usize; ndim];
                            for d in 0..ndim {
                                coords[d] = remaining / strides[d];
                                remaining %= strides[d];
                            }
                            let old_coord = isize::try_from(coords[dim]).map_err(|_| {
                                Self::shape_overflow_error(
                                    "roll backward coordinate exceeds isize range",
                                )
                            })?;
                            let new_coord =
                                ((old_coord - shift) % dim_size_i + dim_size_i) as usize % dim_size;
                            coords[dim] = new_coord;
                            let mut dst_flat = 0;
                            for d in 0..ndim {
                                dst_flat += coords[d] * strides[d];
                            }
                            contrib[dst_flat] = incoming[flat];
                        }
                    }

                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(roll(x,s,d))/dx=roll(grad,-s,d)",
                    });
                }
            }
        }

        let gradients = grads
            .iter()
            .enumerate()
            .map(|(idx, grad)| {
                if self.nodes[idx].requires_grad && reachable[idx] {
                    Some(grad.clone())
                } else {
                    None
                }
            })
            .collect();

        let telemetry = TensorSchedulerTelemetry {
            execution_order,
            queue_pushes: queue.pushes,
            queue_pops: queue.pops,
            max_queue_len: queue.max_len,
            dependency_snapshot,
            reentrant_depth,
            reentrant_guard_triggered,
            hardened_fallback_used,
        };

        Ok(TensorBackwardReport {
            gradients,
            steps,
            telemetry,
        })
    }

    fn compute_reachable(&self, root: TensorNodeId) -> Result<Vec<bool>, AutogradError> {
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            if node.0 >= self.nodes.len() {
                return Err(AutogradError::UnknownTensorNode(node));
            }
            if reachable[node.0] {
                continue;
            }
            reachable[node.0] = true;

            match self.nodes[node.0].op {
                TensorNodeOp::Leaf => {}
                TensorNodeOp::Add { lhs, rhs }
                | TensorNodeOp::Sub { lhs, rhs }
                | TensorNodeOp::Div { lhs, rhs }
                | TensorNodeOp::Mul { lhs, rhs }
                | TensorNodeOp::MatMul { lhs, rhs }
                | TensorNodeOp::Dot { lhs, rhs }
                | TensorNodeOp::Outer { lhs, rhs }
                | TensorNodeOp::Bmm { lhs, rhs }
                | TensorNodeOp::Min { lhs, rhs }
                | TensorNodeOp::Max { lhs, rhs } => {
                    stack.push(lhs);
                    stack.push(rhs);
                }
                TensorNodeOp::Neg { input }
                | TensorNodeOp::Abs { input }
                | TensorNodeOp::Exp { input }
                | TensorNodeOp::Log { input }
                | TensorNodeOp::Relu { input }
                | TensorNodeOp::Sigmoid { input }
                | TensorNodeOp::Tanh { input }
                | TensorNodeOp::Sin { input }
                | TensorNodeOp::Cos { input }
                | TensorNodeOp::Tan { input }
                | TensorNodeOp::Floor { input }
                | TensorNodeOp::Ceil { input }
                | TensorNodeOp::Round { input }
                | TensorNodeOp::Log2 { input }
                | TensorNodeOp::Log10 { input }
                | TensorNodeOp::Log1p { input }
                | TensorNodeOp::Expm1 { input }
                | TensorNodeOp::Sign { input }
                | TensorNodeOp::Trunc { input }
                | TensorNodeOp::Frac { input }
                | TensorNodeOp::Asin { input }
                | TensorNodeOp::Acos { input }
                | TensorNodeOp::Atan { input }
                | TensorNodeOp::Sinh { input }
                | TensorNodeOp::Cosh { input }
                | TensorNodeOp::Gelu { input }
                | TensorNodeOp::Silu { input }
                | TensorNodeOp::LeakyRelu { input }
                | TensorNodeOp::Elu { input }
                | TensorNodeOp::Sqrt { input }
                | TensorNodeOp::Reciprocal { input }
                | TensorNodeOp::Pow { input, .. }
                | TensorNodeOp::Clamp { input, .. }
                | TensorNodeOp::Trace { input, .. }
                | TensorNodeOp::Sum { input, .. }
                | TensorNodeOp::Mean { input, .. }
                | TensorNodeOp::SumDim { input, .. }
                | TensorNodeOp::MeanDim { input, .. }
                | TensorNodeOp::ProdDim { input, .. }
                | TensorNodeOp::VarDim { input, .. }
                | TensorNodeOp::StdDim { input, .. }
                | TensorNodeOp::CumSum { input, .. }
                | TensorNodeOp::CumProd { input, .. }
                | TensorNodeOp::Softmax { input, .. }
                | TensorNodeOp::LogSoftmax { input, .. }
                | TensorNodeOp::Reshape { input, .. }
                | TensorNodeOp::Squeeze { input, .. }
                | TensorNodeOp::Unsqueeze { input, .. }
                | TensorNodeOp::Transpose { input, .. }
                | TensorNodeOp::Permute { input, .. }
                | TensorNodeOp::Narrow { input, .. }
                | TensorNodeOp::Expand { input, .. }
                | TensorNodeOp::Split { input, .. }
                | TensorNodeOp::MaxDim { input, .. }
                | TensorNodeOp::MinDim { input, .. }
                | TensorNodeOp::IndexSelect { input, .. }
                | TensorNodeOp::Gather { input, .. }
                | TensorNodeOp::Scatter { input, .. }
                | TensorNodeOp::Sort { input, .. }
                | TensorNodeOp::TopK { input, .. }
                | TensorNodeOp::Flip { input, .. }
                | TensorNodeOp::Repeat { input, .. }
                | TensorNodeOp::Roll { input, .. } => {
                    stack.push(input);
                }
                TensorNodeOp::Cat { ref inputs, .. } | TensorNodeOp::Stack { ref inputs, .. } => {
                    for &id in inputs {
                        stack.push(id);
                    }
                }
                TensorNodeOp::Where { condition, x, y } => {
                    stack.push(condition);
                    stack.push(x);
                    stack.push(y);
                }
            }
        }

        Ok(reachable)
    }

    fn compute_dependencies(&self, reachable: &[bool]) -> Result<Vec<usize>, AutogradError> {
        if reachable.len() != self.nodes.len() {
            return Err(AutogradError::TensorDependencyUnderflow {
                node: TensorNodeId(0),
            });
        }

        let mut pending = vec![0usize; self.nodes.len()];
        for (idx, node) in self.nodes.iter().enumerate() {
            if !reachable[idx] {
                continue;
            }

            match node.op {
                TensorNodeOp::Leaf => {}
                TensorNodeOp::Add { lhs, rhs }
                | TensorNodeOp::Sub { lhs, rhs }
                | TensorNodeOp::Div { lhs, rhs }
                | TensorNodeOp::Mul { lhs, rhs }
                | TensorNodeOp::MatMul { lhs, rhs }
                | TensorNodeOp::Dot { lhs, rhs }
                | TensorNodeOp::Outer { lhs, rhs }
                | TensorNodeOp::Bmm { lhs, rhs }
                | TensorNodeOp::Min { lhs, rhs }
                | TensorNodeOp::Max { lhs, rhs } => {
                    pending[lhs.0] = pending[lhs.0].saturating_add(1);
                    pending[rhs.0] = pending[rhs.0].saturating_add(1);
                }
                TensorNodeOp::Neg { input }
                | TensorNodeOp::Abs { input }
                | TensorNodeOp::Exp { input }
                | TensorNodeOp::Log { input }
                | TensorNodeOp::Relu { input }
                | TensorNodeOp::Sigmoid { input }
                | TensorNodeOp::Tanh { input }
                | TensorNodeOp::Sin { input }
                | TensorNodeOp::Cos { input }
                | TensorNodeOp::Tan { input }
                | TensorNodeOp::Floor { input }
                | TensorNodeOp::Ceil { input }
                | TensorNodeOp::Round { input }
                | TensorNodeOp::Log2 { input }
                | TensorNodeOp::Log10 { input }
                | TensorNodeOp::Log1p { input }
                | TensorNodeOp::Expm1 { input }
                | TensorNodeOp::Sign { input }
                | TensorNodeOp::Trunc { input }
                | TensorNodeOp::Frac { input }
                | TensorNodeOp::Asin { input }
                | TensorNodeOp::Acos { input }
                | TensorNodeOp::Atan { input }
                | TensorNodeOp::Sinh { input }
                | TensorNodeOp::Cosh { input }
                | TensorNodeOp::Gelu { input }
                | TensorNodeOp::Silu { input }
                | TensorNodeOp::LeakyRelu { input }
                | TensorNodeOp::Elu { input }
                | TensorNodeOp::Sqrt { input }
                | TensorNodeOp::Reciprocal { input }
                | TensorNodeOp::Pow { input, .. }
                | TensorNodeOp::Clamp { input, .. }
                | TensorNodeOp::Trace { input, .. }
                | TensorNodeOp::Sum { input, .. }
                | TensorNodeOp::Mean { input, .. }
                | TensorNodeOp::SumDim { input, .. }
                | TensorNodeOp::MeanDim { input, .. }
                | TensorNodeOp::ProdDim { input, .. }
                | TensorNodeOp::VarDim { input, .. }
                | TensorNodeOp::StdDim { input, .. }
                | TensorNodeOp::CumSum { input, .. }
                | TensorNodeOp::CumProd { input, .. }
                | TensorNodeOp::Softmax { input, .. }
                | TensorNodeOp::LogSoftmax { input, .. }
                | TensorNodeOp::Reshape { input, .. }
                | TensorNodeOp::Squeeze { input, .. }
                | TensorNodeOp::Unsqueeze { input, .. }
                | TensorNodeOp::Transpose { input, .. }
                | TensorNodeOp::Permute { input, .. }
                | TensorNodeOp::Narrow { input, .. }
                | TensorNodeOp::Expand { input, .. }
                | TensorNodeOp::Split { input, .. }
                | TensorNodeOp::MaxDim { input, .. }
                | TensorNodeOp::MinDim { input, .. }
                | TensorNodeOp::IndexSelect { input, .. }
                | TensorNodeOp::Gather { input, .. }
                | TensorNodeOp::Scatter { input, .. }
                | TensorNodeOp::Sort { input, .. }
                | TensorNodeOp::TopK { input, .. }
                | TensorNodeOp::Flip { input, .. }
                | TensorNodeOp::Repeat { input, .. }
                | TensorNodeOp::Roll { input, .. } => {
                    pending[input.0] = pending[input.0].saturating_add(1);
                }
                TensorNodeOp::Cat { ref inputs, .. } | TensorNodeOp::Stack { ref inputs, .. } => {
                    for &id in inputs {
                        pending[id.0] = pending[id.0].saturating_add(1);
                    }
                }
                TensorNodeOp::Where { condition, x, y } => {
                    pending[condition.0] = pending[condition.0].saturating_add(1);
                    pending[x.0] = pending[x.0].saturating_add(1);
                    pending[y.0] = pending[y.0].saturating_add(1);
                }
            }
        }

        Ok(pending)
    }

    fn complete_dependency(
        pending: &mut [usize],
        node: TensorNodeId,
        queue: &mut TensorReadyQueue,
    ) -> Result<(), AutogradError> {
        if pending[node.0] == 0 {
            return Err(AutogradError::TensorDependencyUnderflow { node });
        }
        pending[node.0] -= 1;
        if pending[node.0] == 0 {
            queue.push(node);
        }
        Ok(())
    }

    fn ensure_tensor_len(
        node: TensorNodeId,
        expected: usize,
        actual: usize,
    ) -> Result<(), AutogradError> {
        if expected != actual {
            return Err(AutogradError::TensorGradientShapeMismatch {
                node,
                expected,
                actual,
            });
        }
        Ok(())
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
                return Err(Self::shape_overflow_error(overflow_reason));
            };
            product = next;
        }
        Ok(product)
    }

    fn checked_dim_loop_sizes(
        shape: &[usize],
        dim: usize,
        overflow_reason: &'static str,
    ) -> Result<(usize, usize, usize), AutogradError> {
        let outer_size = Self::checked_shape_numel(&shape[..dim], overflow_reason)?;
        let inner_size = Self::checked_shape_numel(&shape[dim + 1..], overflow_reason)?;
        let total_size = Self::checked_shape_numel(shape, overflow_reason)?;
        Ok((outer_size, inner_size, total_size))
    }

    fn checked_mul_usize(
        lhs: usize,
        rhs: usize,
        overflow_reason: &'static str,
    ) -> Result<usize, AutogradError> {
        lhs.checked_mul(rhs)
            .ok_or_else(|| Self::shape_overflow_error(overflow_reason))
    }

    fn shape_overflow_error(reason: &'static str) -> AutogradError {
        AutogradError::Dispatch(DispatchError::Key(DispatchKeyError::IncompatibleSet {
            reason,
        }))
    }

    fn matmul_dims(lhs: &[usize], rhs: &[usize]) -> Result<(usize, usize, usize), AutogradError> {
        if lhs.len() != 2 || rhs.len() != 2 {
            return Err(AutogradError::TensorMatMulShapeMismatch {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
        }

        let m = lhs[0];
        let k = lhs[1];
        let rhs_k = rhs[0];
        let n = rhs[1];
        if k != rhs_k {
            return Err(AutogradError::TensorMatMulShapeMismatch {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
        }
        Ok((m, k, n))
    }

    fn accumulate_tensor_gradient(
        node: TensorNodeId,
        target: &mut [f64],
        contribution: &[f64],
    ) -> Result<(), AutogradError> {
        Self::ensure_tensor_len(node, target.len(), contribution.len())?;
        for (target_value, value) in target.iter_mut().zip(contribution.iter()) {
            *target_value += value;
        }
        Ok(())
    }

    fn node(&self, id: TensorNodeId) -> Result<&TensorNode, AutogradError> {
        self.nodes
            .get(id.0)
            .ok_or(AutogradError::UnknownTensorNode(id))
    }

    fn node_mut(&mut self, id: TensorNodeId) -> Result<&mut TensorNode, AutogradError> {
        self.nodes
            .get_mut(id.0)
            .ok_or(AutogradError::UnknownTensorNode(id))
    }

    /// Return the tensor metadata for a node.
    pub fn tensor_meta(&self, id: TensorNodeId) -> Result<&TensorMeta, AutogradError> {
        Ok(self.node(id)?.tensor.meta())
    }

    /// Replace the storage values of a tensor node in-place (version is bumped).
    pub fn update_tensor_values(
        &mut self,
        id: TensorNodeId,
        new_values: Vec<f64>,
    ) -> Result<(), AutogradError> {
        let node = self.node_mut(id)?;
        node.tensor
            .update_contiguous_values(&new_values)
            .map_err(AutogradError::DenseTensor)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ft_core::{DType, DenseTensor, DenseTensorError, Device, ExecutionMode, TensorMeta};
    use ft_dispatch::DispatchError;
    use proptest::prelude::*;

    use super::{
        AutogradError, BackwardOptions, NodeId, ReentrantPolicy, SchedulerTelemetry, Tape,
        TensorNode, TensorNodeId, TensorNodeOp, TensorTape,
    };

    fn as_u64(value: usize) -> u64 {
        u64::try_from(value).unwrap_or(u64::MAX)
    }

    fn det_seed(parts: &[u64]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in parts {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn output_digest(telemetry: &SchedulerTelemetry) -> u64 {
        let mut parts = Vec::with_capacity(telemetry.execution_order.len() + 6);
        parts.extend(telemetry.execution_order.iter().map(|node| as_u64(node.0)));
        parts.push(as_u64(telemetry.queue_pushes));
        parts.push(as_u64(telemetry.queue_pops));
        parts.push(as_u64(telemetry.max_queue_len));
        parts.push(as_u64(telemetry.reentrant_depth));
        parts.push(u64::from(telemetry.reentrant_guard_triggered));
        parts.push(u64::from(telemetry.hardened_fallback_used));
        det_seed(parts.as_slice())
    }

    fn build_scheduler_property_log(
        test_id: &str,
        mode: ExecutionMode,
        seed: u64,
        telemetry: &SchedulerTelemetry,
        reason_code: &str,
    ) -> BTreeMap<String, String> {
        let mode_label = match mode {
            ExecutionMode::Strict => "strict",
            ExecutionMode::Hardened => "hardened",
        };
        let input_digest = det_seed(
            [
                seed,
                as_u64(telemetry.execution_order.len()),
                as_u64(telemetry.dependency_snapshot.len()),
            ]
            .as_slice(),
        );
        let mut log = BTreeMap::new();
        log.insert("ts_utc".to_string(), "1970-01-01T00:00:00Z".to_string());
        log.insert("suite_id".to_string(), "ft_autograd_property".to_string());
        log.insert("test_id".to_string(), test_id.to_string());
        log.insert("packet_id".to_string(), "FT-P2C-004".to_string());
        log.insert(
            "fixture_id".to_string(),
            "ft_autograd_property_generated".to_string(),
        );
        log.insert(
            "scenario_id".to_string(),
            format!("autograd_scheduler_property/{mode_label}:{test_id}"),
        );
        log.insert("mode".to_string(), mode_label.to_string());
        log.insert("seed".to_string(), seed.to_string());
        log.insert(
            "input_digest".to_string(),
            format!("det64:{input_digest:016x}"),
        );
        log.insert(
            "output_digest".to_string(),
            format!("det64:{:016x}", output_digest(telemetry)),
        );
        log.insert(
            "env_fingerprint".to_string(),
            "det64:ft-autograd-test".to_string(),
        );
        log.insert(
            "artifact_refs".to_string(),
            "artifacts/phase2c/FT-P2C-004/fixture_manifest.json".to_string(),
        );
        log.insert(
            "replay_command".to_string(),
            "cargo test -p ft-autograd -- --nocapture".to_string(),
        );
        log.insert("duration_ms".to_string(), "0".to_string());
        log.insert("outcome".to_string(), "pass".to_string());
        log.insert("reason_code".to_string(), reason_code.to_string());
        log.insert(
            "execution_order".to_string(),
            telemetry
                .execution_order
                .iter()
                .map(|node| node.0.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
        log.insert(
            "queue_pushes".to_string(),
            telemetry.queue_pushes.to_string(),
        );
        log.insert("queue_pops".to_string(), telemetry.queue_pops.to_string());
        log.insert(
            "max_queue_len".to_string(),
            telemetry.max_queue_len.to_string(),
        );
        log.insert(
            "dependency_snapshot".to_string(),
            telemetry
                .dependency_snapshot
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
        log.insert(
            "reentrant_depth".to_string(),
            telemetry.reentrant_depth.to_string(),
        );
        log.insert(
            "reentrant_guard_triggered".to_string(),
            telemetry.reentrant_guard_triggered.to_string(),
        );
        log.insert(
            "hardened_fallback_used".to_string(),
            telemetry.hardened_fallback_used.to_string(),
        );
        log
    }

    fn assert_scheduler_log_contract(log: &BTreeMap<String, String>) {
        for key in [
            "ts_utc",
            "suite_id",
            "test_id",
            "packet_id",
            "fixture_id",
            "scenario_id",
            "mode",
            "seed",
            "input_digest",
            "output_digest",
            "env_fingerprint",
            "artifact_refs",
            "replay_command",
            "duration_ms",
            "outcome",
            "reason_code",
            "execution_order",
            "queue_pushes",
            "queue_pops",
            "max_queue_len",
            "dependency_snapshot",
            "reentrant_depth",
            "reentrant_guard_triggered",
            "hardened_fallback_used",
        ] {
            assert!(
                log.contains_key(key),
                "property log missing required key '{key}'"
            );
        }
    }

    #[test]
    fn backward_options_for_mode_strict_matches_default() {
        let options = BackwardOptions::for_mode(ExecutionMode::Strict);
        assert_eq!(options, BackwardOptions::strict_default());
        assert_eq!(options.policy, ReentrantPolicy::StrictFail);
        assert_eq!(options.max_reentrant_depth, 0);
        assert_eq!(options.current_reentrant_depth, 0);
    }

    #[test]
    fn backward_options_for_mode_hardened_matches_default() {
        let options = BackwardOptions::for_mode(ExecutionMode::Hardened);
        assert_eq!(options, BackwardOptions::hardened_default());
        assert_eq!(options.policy, ReentrantPolicy::HardenedBoundedFallback);
        assert_eq!(options.max_reentrant_depth, 2);
        assert_eq!(options.current_reentrant_depth, 0);
    }

    #[test]
    fn add_backward_matches_expected_gradient() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (z, _) = tape
            .add(x, y, ExecutionMode::Strict)
            .expect("add should succeed");

        let report = tape.backward(z).expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(1.0));
        assert_eq!(report.gradient(y), Some(1.0));
        assert_eq!(report.telemetry.execution_order, vec![z, y, x]);
    }

    #[test]
    fn mul_backward_matches_expected_gradient() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (z, _) = tape
            .mul(x, y, ExecutionMode::Strict)
            .expect("mul should succeed");

        let report = tape.backward(z).expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(3.0));
        assert_eq!(report.gradient(y), Some(2.0));
    }

    #[test]
    fn sub_backward_matches_expected_gradient() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (z, _) = tape
            .sub(x, y, ExecutionMode::Strict)
            .expect("sub should succeed");

        let report = tape.backward(z).expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(1.0));
        assert_eq!(report.gradient(y), Some(-1.0));
    }

    #[test]
    fn div_backward_matches_expected_gradient() {
        let mut tape = Tape::new();
        let x = tape.leaf(6.0, true);
        let y = tape.leaf(3.0, true);
        let (z, _) = tape
            .div(x, y, ExecutionMode::Strict)
            .expect("div should succeed");

        let report = tape.backward(z).expect("backward should succeed");
        let x_grad = report.gradient(x).expect("x grad should exist");
        let y_grad = report.gradient(y).expect("y grad should exist");

        assert!((x_grad - (1.0 / 3.0)).abs() <= 1e-12);
        assert!((y_grad - (-2.0 / 3.0)).abs() <= 1e-12);
    }

    #[test]
    fn tensor_add_forward_backward_matches_expected_gradients() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("lhs leaf should succeed");
        let y = tape
            .leaf(vec![4.0, 5.0, 6.0], vec![3], true)
            .expect("rhs leaf should succeed");
        let (z, event) = tape
            .add(x, y, ExecutionMode::Strict)
            .expect("tensor add should succeed");

        assert_eq!(
            event.decision.kernel,
            "autograd_cpu::add_tensor_contiguous_f64"
        );
        assert_eq!(
            tape.values(z).expect("tensor values should resolve"),
            vec![5.0, 7.0, 9.0]
        );

        let report = tape.backward(z).expect("tensor backward should succeed");
        assert_eq!(
            report.gradient(x).expect("x grad should exist"),
            &[1.0, 1.0, 1.0]
        );
        assert_eq!(
            report.gradient(y).expect("y grad should exist"),
            &[1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn tensor_matmul_forward_backward_matches_expected_gradients() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("lhs leaf should succeed");
        let y = tape
            .leaf(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true)
            .expect("rhs leaf should succeed");
        let (z, event) = tape
            .matmul(x, y, ExecutionMode::Strict)
            .expect("tensor matmul should succeed");

        assert_eq!(
            event.decision.kernel,
            "autograd_cpu::matmul_tensor_contiguous_f64"
        );
        assert_eq!(
            tape.values(z).expect("tensor values should resolve"),
            vec![19.0, 22.0, 43.0, 50.0]
        );

        let report = tape.backward(z).expect("tensor backward should succeed");
        assert_eq!(
            report.gradient(x).expect("x grad should exist"),
            &[11.0, 15.0, 11.0, 15.0]
        );
        assert_eq!(
            report.gradient(y).expect("y grad should exist"),
            &[4.0, 4.0, 6.0, 6.0]
        );
    }

    #[test]
    fn tensor_dispatch_rejects_non_contiguous_layout_end_to_end() -> Result<(), String> {
        let mut tape = TensorTape::new();
        let lhs_meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![4, 1], 0, DType::F64, Device::Cpu)
                .expect("non-contiguous meta should validate");
        let lhs = DenseTensor::from_storage(lhs_meta, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("lhs should build");
        let rhs = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu),
            vec![5.0, 6.0, 7.0, 8.0],
        )
        .expect("rhs should build");

        let lhs_node = tape.leaf_tensor(lhs, true);
        let rhs_node = tape.leaf_tensor(rhs, true);
        let err = tape
            .add(lhs_node, rhs_node, ExecutionMode::Strict)
            .expect_err("non-contiguous layout should fail closed");
        let error = match err {
            AutogradError::Dispatch(DispatchError::Kernel(error)) => error,
            other => return Err(format!("expected kernel dispatch error, got {other:?}")),
        };
        assert!(
            error
                .to_string()
                .contains("unsupported non-contiguous layout on lhs")
        );
        Ok(())
    }

    #[test]
    fn tensor_values_reject_non_contiguous_layout() {
        let mut tape = TensorTape::new();
        let meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![4, 1], 0, DType::F64, Device::Cpu)
                .expect("non-contiguous meta should validate");
        let tensor = DenseTensor::from_storage(meta, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("tensor should build");

        let node = tape.leaf_tensor(tensor, true);
        let err = tape
            .values(node)
            .expect_err("non-contiguous values should fail closed");
        assert!(matches!(
            err,
            AutogradError::DenseTensor(DenseTensorError::UnsupportedLayout)
        ));
    }

    #[test]
    fn tensor_backward_mul_rejects_non_contiguous_operand_layout() {
        let mut tape = TensorTape::new();
        let lhs_meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![4, 1], 0, DType::F64, Device::Cpu)
                .expect("non-contiguous meta should validate");
        let lhs_tensor = DenseTensor::from_storage(lhs_meta, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("lhs should build");
        let rhs_tensor = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu),
            vec![2.0, 2.0, 2.0, 2.0],
        )
        .expect("rhs should build");
        let out_tensor = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu),
            vec![2.0, 4.0, 6.0, 8.0],
        )
        .expect("out should build");

        let lhs = tape.leaf_tensor(lhs_tensor, true);
        let rhs = tape.leaf_tensor(rhs_tensor, true);
        let out = TensorNodeId(tape.nodes.len());
        tape.nodes.push(TensorNode {
            tensor: out_tensor,
            requires_grad: true,
            op: TensorNodeOp::Mul { lhs, rhs },
        });

        let err = tape
            .backward(out)
            .expect_err("non-contiguous backward operand should fail closed");
        assert!(matches!(
            err,
            AutogradError::DenseTensor(DenseTensorError::UnsupportedLayout)
        ));
    }

    #[test]
    fn tensor_backward_div_rejects_non_contiguous_operand_layout() {
        let mut tape = TensorTape::new();
        let lhs_meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![4, 1], 0, DType::F64, Device::Cpu)
                .expect("non-contiguous meta should validate");
        let lhs_tensor = DenseTensor::from_storage(lhs_meta, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("lhs should build");
        let rhs_tensor = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu),
            vec![2.0, 2.0, 2.0, 2.0],
        )
        .expect("rhs should build");
        let out_tensor = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu),
            vec![0.5, 1.0, 1.5, 2.0],
        )
        .expect("out should build");

        let lhs = tape.leaf_tensor(lhs_tensor, true);
        let rhs = tape.leaf_tensor(rhs_tensor, true);
        let out = TensorNodeId(tape.nodes.len());
        tape.nodes.push(TensorNode {
            tensor: out_tensor,
            requires_grad: true,
            op: TensorNodeOp::Div { lhs, rhs },
        });

        let err = tape
            .backward(out)
            .expect_err("non-contiguous backward operand should fail closed");
        assert!(matches!(
            err,
            AutogradError::DenseTensor(DenseTensorError::UnsupportedLayout)
        ));
    }

    #[test]
    fn tensor_backward_sort_rejects_out_of_bounds_scatter_index() {
        let mut tape = TensorTape::new();
        let input = tape
            .leaf(vec![3.0, 1.0, 2.0], vec![3], true)
            .expect("input leaf should build");
        let (sorted, _, _) = tape
            .sort(input, 0, false, ExecutionMode::Strict)
            .expect("sort should succeed");
        assert!(matches!(tape.nodes[sorted.0].op, TensorNodeOp::Sort { .. }));
        if let TensorNodeOp::Sort { indices, .. } = &mut tape.nodes[sorted.0].op {
            indices[0] = 3;
        }

        let err = tape
            .backward(sorted)
            .expect_err("out-of-bounds sort index should fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(DispatchError::Kernel(
                ft_kernel_cpu::KernelError::InvalidDimension { dim: 3, ndim: 3 }
            ))
        ));
    }

    #[test]
    fn tensor_backward_sort_rejects_incoming_gradient_len_mismatch() {
        let mut tape = TensorTape::new();
        let input = tape
            .leaf(vec![3.0, 1.0, 2.0], vec![3], true)
            .expect("input leaf should build");
        let (sorted, _, _) = tape
            .sort(input, 0, false, ExecutionMode::Strict)
            .expect("sort should succeed");
        tape.nodes[sorted.0].tensor = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu),
            vec![1.0, 2.0],
        )
        .expect("replacement tensor should build");

        let err = tape
            .backward(sorted)
            .expect_err("incoming gradient shape mismatch should fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorGradientShapeMismatch { node, expected: 3, actual: 2 }
            if node == sorted
        ));
    }

    #[test]
    fn tensor_backward_topk_rejects_malformed_indices_len() {
        let mut tape = TensorTape::new();
        let input = tape
            .leaf(vec![1.0, 5.0, 2.0, 4.0], vec![4], true)
            .expect("input leaf should build");
        let (topk, _, _) = tape
            .topk(input, 2, 0, true, true, ExecutionMode::Strict)
            .expect("topk should succeed");
        assert!(matches!(tape.nodes[topk.0].op, TensorNodeOp::TopK { .. }));
        if let TensorNodeOp::TopK { indices, .. } = &mut tape.nodes[topk.0].op {
            indices.pop();
        }

        let err = tape
            .backward(topk)
            .expect_err("malformed topk indices should fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorGradientShapeMismatch { node, expected: 2, actual: 1 }
            if node == topk
        ));
    }

    #[test]
    fn tensor_dependency_snapshot_preserves_initial_pending_counts() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![2.0], vec![1], true)
            .expect("x leaf should build");
        let y = tape
            .leaf(vec![3.0], vec![1], true)
            .expect("y leaf should build");
        let (sum, _) = tape
            .add(x, y, ExecutionMode::Strict)
            .expect("sum should build");
        let (out, _) = tape
            .mul(sum, x, ExecutionMode::Strict)
            .expect("out should build");

        let report = tape.backward(out).expect("backward should succeed");
        assert_eq!(report.telemetry.dependency_snapshot, vec![2, 1, 1, 0]);
    }

    #[test]
    fn tensor_add_with_offset_view_input_returns_fresh_contiguous_output() {
        let mut tape = TensorTape::new();
        let lhs_meta =
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(2);
        let lhs = DenseTensor::from_storage(lhs_meta, vec![0.0, 0.0, 1.0, 2.0, 3.0])
            .expect("lhs offset view should build");
        let rhs = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu),
            vec![10.0, 20.0, 30.0],
        )
        .expect("rhs should build");

        let lhs_node = tape.leaf_tensor(lhs, true);
        let rhs_node = tape.leaf_tensor(rhs, true);
        let (out_node, _) = tape
            .add(lhs_node, rhs_node, ExecutionMode::Strict)
            .expect("offset view add should succeed");

        let out = tape.tensor(out_node).expect("output tensor should resolve");
        assert_eq!(out.meta().storage_offset(), 0);
        assert!(out.meta().is_contiguous());
        assert_eq!(
            out.dispatch_values().expect("output values"),
            &[11.0, 22.0, 33.0]
        );
    }

    #[test]
    fn dependency_scheduler_waits_for_all_children() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let z = tape.leaf(4.0, true);
        let (xy, _) = tape
            .mul(x, y, ExecutionMode::Strict)
            .expect("mul should succeed");
        let (xz, _) = tape
            .mul(x, z, ExecutionMode::Strict)
            .expect("mul should succeed");
        let (out, _) = tape
            .add(xy, xz, ExecutionMode::Strict)
            .expect("add should succeed");

        let report = tape.backward(out).expect("backward should succeed");
        let order = report.telemetry.execution_order;
        let x_pos = order
            .iter()
            .position(|node| *node == x)
            .expect("x should be scheduled");
        let xy_pos = order
            .iter()
            .position(|node| *node == xy)
            .expect("xy should be scheduled");
        let xz_pos = order
            .iter()
            .position(|node| *node == xz)
            .expect("xz should be scheduled");

        assert!(x_pos > xy_pos);
        assert!(x_pos > xz_pos);
    }

    #[test]
    fn dependency_snapshot_preserves_initial_pending_counts() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (sum, _) = tape
            .add(x, y, ExecutionMode::Strict)
            .expect("sum should succeed");
        let (out, _) = tape
            .mul(sum, x, ExecutionMode::Strict)
            .expect("out should succeed");

        let report = tape.backward(out).expect("backward should succeed");
        assert_eq!(report.telemetry.dependency_snapshot, vec![2, 1, 1, 0]);
    }

    #[test]
    fn composite_graph_gradient_is_deterministic() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (sum, _) = tape
            .add(x, y, ExecutionMode::Strict)
            .expect("add should succeed");
        let (out, _) = tape
            .mul(sum, x, ExecutionMode::Strict)
            .expect("mul should succeed");

        let report = tape.backward(out).expect("backward should succeed");
        assert_eq!(report.gradient(x), Some(7.0));
        assert_eq!(report.gradient(y), Some(2.0));

        let report_2 = tape.backward(out).expect("backward should be repeatable");
        assert_eq!(report.gradients(), report_2.gradients());
        assert_eq!(
            report.telemetry.execution_order,
            report_2.telemetry.execution_order
        );
    }

    #[test]
    fn strict_mode_reentrant_depth_overflow_fails() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (z, _) = tape
            .add(x, y, ExecutionMode::Strict)
            .expect("add should succeed");

        let err = tape
            .backward_with_options(
                z,
                BackwardOptions {
                    max_reentrant_depth: 1,
                    current_reentrant_depth: 2,
                    policy: ReentrantPolicy::StrictFail,
                },
            )
            .expect_err("strict overflow should fail");

        assert!(
            err.to_string()
                .contains("reentrant backward depth exceeded")
        );
    }

    #[test]
    fn hardened_mode_reentrant_depth_overflow_fallbacks() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (z, _) = tape
            .add(x, y, ExecutionMode::Hardened)
            .expect("add should succeed");

        let report = tape
            .backward_with_options(
                z,
                BackwardOptions {
                    max_reentrant_depth: 1,
                    current_reentrant_depth: 2,
                    policy: ReentrantPolicy::HardenedBoundedFallback,
                },
            )
            .expect("hardened overflow should fallback");

        assert!(report.telemetry.reentrant_guard_triggered);
        assert!(report.telemetry.hardened_fallback_used);
    }

    #[test]
    fn unknown_node_returns_error() {
        let tape = Tape::new();
        let err = tape
            .backward(NodeId(99))
            .expect_err("expected unknown node");
        let msg = err.to_string();
        assert!(msg.contains("unknown node"));
    }

    #[test]
    fn backward_rejects_root_without_requires_grad() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, false);
        let y = tape.leaf(3.0, false);
        let (root, _) = tape
            .add(x, y, ExecutionMode::Strict)
            .expect("add should succeed");

        let err = tape
            .backward(root)
            .expect_err("root without requires_grad must fail closed");
        assert!(matches!(
            err,
            AutogradError::RootDoesNotRequireGrad { node } if node == root
        ));
    }

    #[test]
    fn tensor_backward_rejects_root_without_requires_grad() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0], vec![2], false)
            .expect("x leaf should succeed");
        let y = tape
            .leaf(vec![3.0, 4.0], vec![2], false)
            .expect("y leaf should succeed");
        let (root, _) = tape
            .add(x, y, ExecutionMode::Strict)
            .expect("add should succeed");

        let err = tape
            .backward(root)
            .expect_err("tensor root without requires_grad must fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorRootDoesNotRequireGrad { node } if node == root
        ));
    }

    #[test]
    fn dependency_underflow_is_fail_closed() {
        let mut pending = vec![0usize];
        let mut queue = super::ReadyQueue::default();
        let err = Tape::complete_dependency(&mut pending, NodeId(0), &mut queue)
            .expect_err("underflow should fail closed");
        assert!(matches!(
            err,
            AutogradError::DependencyUnderflow { node } if node == NodeId(0)
        ));
    }

    #[test]
    fn tensor_dependency_compute_underflow_is_fail_closed() {
        let tape = TensorTape::new();
        let err = tape
            .compute_dependencies(&[true])
            .expect_err("reachable mismatch must fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorDependencyUnderflow { node } if node == TensorNodeId(0)
        ));
    }

    #[test]
    fn tensor_dependency_complete_underflow_is_fail_closed() {
        let mut pending = vec![0usize];
        let mut queue = super::TensorReadyQueue::default();
        let err = TensorTape::complete_dependency(&mut pending, TensorNodeId(0), &mut queue)
            .expect_err("underflow should fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorDependencyUnderflow { node } if node == TensorNodeId(0)
        ));
    }

    #[test]
    fn tensor_ensure_len_mismatch_is_fail_closed() {
        let err = TensorTape::ensure_tensor_len(TensorNodeId(3), 2, 1)
            .expect_err("shape mismatch must fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorGradientShapeMismatch {
                node,
                expected: 2,
                actual: 1
            } if node == TensorNodeId(3)
        ));
    }

    #[test]
    fn tensor_accumulate_gradient_mismatch_is_fail_closed() {
        let mut target = vec![0.0, 0.0];
        let err = TensorTape::accumulate_tensor_gradient(TensorNodeId(1), &mut target, &[1.0])
            .expect_err("shape mismatch must fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorGradientShapeMismatch {
                node,
                expected: 2,
                actual: 1
            } if node == TensorNodeId(1)
        ));
        assert_eq!(target, vec![0.0, 0.0]);
    }

    proptest! {
        #[test]
        fn prop_scheduler_replay_is_deterministic(
            x_in in -32i16..32i16,
            y_in in -32i16..32i16,
        ) {
            let x = f64::from(x_in);
            let y = f64::from(y_in);
            let mut tape = Tape::new();
            let lhs = tape.leaf(x, true);
            let rhs = tape.leaf(y, true);
            let (sum, _) = tape
                .add(lhs, rhs, ExecutionMode::Strict)
                .expect("add should succeed");
            let (out, _) = tape
                .mul(sum, lhs, ExecutionMode::Strict)
                .expect("mul should succeed");

            let first = tape.backward(out).expect("backward should succeed");
            let second = tape.backward(out).expect("backward should succeed");

            prop_assert_eq!(first.gradients(), second.gradients());
            prop_assert_eq!(
                &first.telemetry.execution_order,
                &second.telemetry.execution_order
            );

            let seed = det_seed(&[
                u64::from(x_in.unsigned_abs()),
                u64::from(y_in.unsigned_abs()),
                as_u64(first.telemetry.execution_order.len()),
            ]);
            let log = build_scheduler_property_log(
                "prop_scheduler_replay_is_deterministic",
                ExecutionMode::Strict,
                seed,
                &first.telemetry,
                "scheduler_replay_stable",
            );
            assert_scheduler_log_contract(&log);
        }

        #[test]
        fn prop_shared_parent_waits_for_all_children(
            x_in in 1i16..16i16,
            y_in in 1i16..16i16,
            z_in in 1i16..16i16,
        ) {
            let x = f64::from(x_in);
            let y = f64::from(y_in);
            let z = f64::from(z_in);

            let mut tape = Tape::new();
            let parent = tape.leaf(x, true);
            let lhs = tape.leaf(y, true);
            let rhs = tape.leaf(z, true);
            let (left_branch, _) = tape
                .mul(parent, lhs, ExecutionMode::Strict)
                .expect("mul should succeed");
            let (right_branch, _) = tape
                .mul(parent, rhs, ExecutionMode::Strict)
                .expect("mul should succeed");
            let (root, _) = tape
                .add(left_branch, right_branch, ExecutionMode::Strict)
                .expect("add should succeed");

            let report = tape.backward(root).expect("backward should succeed");
            let order = &report.telemetry.execution_order;
            let parent_pos = order.iter().position(|node| *node == parent).expect("parent should be scheduled");
            let left_pos = order.iter().position(|node| *node == left_branch).expect("left branch should be scheduled");
            let right_pos = order.iter().position(|node| *node == right_branch).expect("right branch should be scheduled");

            prop_assert!(parent_pos > left_pos);
            prop_assert!(parent_pos > right_pos);

            let seed = det_seed(&[
                u64::from(x_in.unsigned_abs()),
                u64::from(y_in.unsigned_abs()),
                u64::from(z_in.unsigned_abs()),
                as_u64(order.len()),
            ]);
            let log = build_scheduler_property_log(
                "prop_shared_parent_waits_for_all_children",
                ExecutionMode::Strict,
                seed,
                &report.telemetry,
                "dependency_scheduler_waits_for_all_children",
            );
            assert_scheduler_log_contract(&log);
        }

        #[test]
        fn prop_strict_reentrant_overflow_is_fail_closed(
            x_in in 1i16..16i16,
            y_in in 1i16..16i16,
        ) {
            let x = f64::from(x_in);
            let y = f64::from(y_in);
            let mut tape = Tape::new();
            let lhs = tape.leaf(x, true);
            let rhs = tape.leaf(y, true);
            let (root, _) = tape
                .add(lhs, rhs, ExecutionMode::Strict)
                .expect("add should succeed");

            let overflow = tape.backward_with_options(
                root,
                BackwardOptions {
                    max_reentrant_depth: 1,
                    current_reentrant_depth: 2,
                    policy: ReentrantPolicy::StrictFail,
                },
            );
            assert!(matches!(
                overflow,
                Err(AutogradError::ReentrantDepthExceeded { .. })
            ));
        }

        #[test]
        fn prop_hardened_reentrant_overflow_is_explicitly_flagged(
            x_in in 1i16..16i16,
            y_in in 1i16..16i16,
        ) {
            let x = f64::from(x_in);
            let y = f64::from(y_in);
            let mut tape = Tape::new();
            let lhs = tape.leaf(x, true);
            let rhs = tape.leaf(y, true);
            let (root, _) = tape
                .add(lhs, rhs, ExecutionMode::Hardened)
                .expect("add should succeed");

            let report = tape
                .backward_with_options(
                    root,
                    BackwardOptions {
                        max_reentrant_depth: 1,
                        current_reentrant_depth: 2,
                        policy: ReentrantPolicy::HardenedBoundedFallback,
                    },
                )
                .expect("hardened fallback should succeed");
            prop_assert!(report.telemetry.reentrant_guard_triggered);
            prop_assert!(report.telemetry.hardened_fallback_used);
            prop_assert_eq!(report.telemetry.reentrant_depth, 1);

            let seed = det_seed(&[
                u64::from(x_in.unsigned_abs()),
                u64::from(y_in.unsigned_abs()),
                as_u64(report.telemetry.reentrant_depth),
            ]);
            let log = build_scheduler_property_log(
                "prop_hardened_reentrant_overflow_is_explicitly_flagged",
                ExecutionMode::Hardened,
                seed,
                &report.telemetry,
                "hardened_reentrant_guard_triggered",
            );
            assert_scheduler_log_contract(&log);
        }

        #[test]
        fn prop_scheduler_telemetry_is_self_consistent(
            x_in in -16i16..16i16,
            y_in in -16i16..16i16,
        ) {
            let x = f64::from(x_in);
            let y = f64::from(y_in);
            let mut tape = Tape::new();
            let lhs = tape.leaf(x, true);
            let rhs = tape.leaf(y, true);
            let (sum, _) = tape
                .add(lhs, rhs, ExecutionMode::Strict)
                .expect("add should succeed");
            let (root, _) = tape
                .mul(sum, lhs, ExecutionMode::Strict)
                .expect("mul should succeed");
            let report = tape.backward(root).expect("backward should succeed");

            prop_assert!(report.telemetry.queue_pushes >= report.telemetry.queue_pops);
            prop_assert!(report.telemetry.max_queue_len >= 1);
            prop_assert_eq!(report.telemetry.dependency_snapshot.len(), tape.node_count());

            let seed = det_seed(&[
                u64::from(x_in.unsigned_abs()),
                u64::from(y_in.unsigned_abs()),
                as_u64(report.telemetry.queue_pushes),
                as_u64(report.telemetry.queue_pops),
            ]);
            let log = build_scheduler_property_log(
                "prop_scheduler_telemetry_is_self_consistent",
                ExecutionMode::Strict,
                seed,
                &report.telemetry,
                "scheduler_telemetry_contract_ok",
            );
            assert_scheduler_log_contract(&log);
        }
    }

    #[test]
    fn tensor_transpose_2d_swaps_shape_and_data() {
        let mut tape = TensorTape::new();
        // [[1, 2, 3], [4, 5, 6]] shape [2, 3]
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf should succeed");
        let y = tape.transpose(x, 0, 1).expect("transpose should succeed");

        let values = tape.values(y).expect("values should resolve");
        // Transposed: [[1, 4], [2, 5], [3, 6]] shape [3, 2]
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(tape.nodes[y.0].tensor.meta().shape(), &[3, 2]);
    }

    #[test]
    fn tensor_transpose_identity_same_dim() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf should succeed");
        let y = tape
            .transpose(x, 0, 0)
            .expect("identity transpose should succeed");

        let values = tape.values(y).expect("values should resolve");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tape.nodes[y.0].tensor.meta().shape(), &[2, 3]);
    }

    #[test]
    fn tensor_transpose_3d() {
        let mut tape = TensorTape::new();
        // shape [2, 3, 4], 24 elements
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let x = tape
            .leaf(data, vec![2, 3, 4], true)
            .expect("leaf should succeed");
        let y = tape.transpose(x, 0, 2).expect("transpose should succeed");

        assert_eq!(tape.nodes[y.0].tensor.meta().shape(), &[4, 3, 2]);
        let values = tape.values(y).expect("values should resolve");
        assert_eq!(values.len(), 24);
        // Element at original [0,0,0]=0 should be at transposed [0,0,0]=0
        assert_eq!(values[0], 0.0);
        // Element at original [0,0,1]=1 should be at transposed [1,0,0]
        // In shape [4,3,2], flat index for [1,0,0] = 1*6 + 0*2 + 0 = 6
        assert_eq!(values[6], 1.0);
        // Element at original [1,0,0]=12 should be at transposed [0,0,1]
        // In shape [4,3,2], flat index for [0,0,1] = 0*6 + 0*2 + 1 = 1
        assert_eq!(values[1], 12.0);
    }

    #[test]
    fn tensor_transpose_invalid_dim_fails() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf should succeed");
        let err = tape
            .transpose(x, 0, 2)
            .expect_err("out-of-bounds dim should fail");
        assert!(matches!(err, AutogradError::Dispatch(_)));
    }

    #[test]
    fn tensor_transpose_backward_propagates_gradient() {
        let mut tape = TensorTape::new();
        // x = [[1, 2, 3], [4, 5, 6]] shape [2, 3]
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf should succeed");
        let y = tape.transpose(x, 0, 1).expect("transpose should succeed");
        // Sum the transposed tensor to get a scalar for backward
        let (z, _) = tape
            .sum(y, ExecutionMode::Strict)
            .expect("sum should succeed");

        let report = tape.backward(z).expect("backward should succeed");
        let grad = report.gradient(x).expect("x grad should exist");
        // d(sum(transpose(x)))/dx = all ones (sum gradient broadcast, transpose inverse)
        assert_eq!(grad, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn tensor_transpose_backward_preserves_element_gradients() {
        let mut tape = TensorTape::new();
        // x = [[1, 2], [3, 4]] shape [2, 2]
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("leaf should succeed");
        let y = tape.transpose(x, 0, 1).expect("transpose should succeed");
        // y = [[1, 3], [2, 4]] — multiply by a known tensor
        let w = tape
            .leaf(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2], false)
            .expect("weights leaf should succeed");
        let (product, _) = tape
            .mul(y, w, ExecutionMode::Strict)
            .expect("mul should succeed");
        let (z, _) = tape
            .sum(product, ExecutionMode::Strict)
            .expect("sum should succeed");

        let report = tape.backward(z).expect("backward should succeed");
        let grad = report.gradient(x).expect("x grad should exist");
        // y = transpose(x), product = y * w, z = sum(product)
        // dz/dy = w = [10, 20, 30, 40] in shape [2, 2]
        // dz/dx = inverse_transpose(dz/dy) = transpose([10, 20, 30, 40]) = [10, 30, 20, 40]
        assert_eq!(grad, &[10.0, 30.0, 20.0, 40.0]);
    }

    #[test]
    fn tensor_permute_3d_reorders_data() {
        let mut tape = TensorTape::new();
        // shape [2, 3, 4]
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let x = tape
            .leaf(data, vec![2, 3, 4], true)
            .expect("leaf should succeed");
        let y = tape
            .permute(x, vec![2, 0, 1])
            .expect("permute should succeed");

        assert_eq!(tape.nodes[y.0].tensor.meta().shape(), &[4, 2, 3]);
        let values = tape.values(y).expect("values should resolve");
        assert_eq!(values.len(), 24);
        // Element at original [0,0,0]=0 -> permuted coords [0,0,0] -> same flat index 0
        assert_eq!(values[0], 0.0);
        // Element at original [0,0,1]=1 -> permuted: dim2->dim0, dim0->dim1, dim1->dim2
        // new coords [1, 0, 0] -> flat in [4,2,3]: 1*6 + 0*3 + 0 = 6
        assert_eq!(values[6], 1.0);
    }

    #[test]
    fn tensor_permute_identity_no_change() {
        let mut tape = TensorTape::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = tape
            .leaf(data.clone(), vec![2, 3], true)
            .expect("leaf should succeed");
        let y = tape
            .permute(x, vec![0, 1])
            .expect("identity permute should succeed");

        assert_eq!(tape.values(y).expect("values"), data);
        assert_eq!(tape.nodes[y.0].tensor.meta().shape(), &[2, 3]);
    }

    #[test]
    fn tensor_permute_invalid_dim_fails() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf should succeed");
        let err = tape
            .permute(x, vec![0, 5])
            .expect_err("out-of-bounds dim should fail");
        assert!(matches!(err, AutogradError::Dispatch(_)));
    }

    #[test]
    fn tensor_permute_duplicate_dim_fails() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf should succeed");
        let err = tape
            .permute(x, vec![0, 0])
            .expect_err("duplicate dims should fail");
        assert!(matches!(err, AutogradError::Dispatch(_)));
    }

    #[test]
    fn tensor_permute_wrong_ndim_fails() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf should succeed");
        let err = tape
            .permute(x, vec![0, 1, 2])
            .expect_err("wrong number of dims should fail");
        assert!(matches!(err, AutogradError::Dispatch(_)));
    }

    #[test]
    fn tensor_permute_backward_propagates_gradient() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf should succeed");
        let y = tape.permute(x, vec![1, 0]).expect("permute should succeed");
        let (z, _) = tape
            .sum(y, ExecutionMode::Strict)
            .expect("sum should succeed");

        let report = tape.backward(z).expect("backward should succeed");
        let grad = report.gradient(x).expect("x grad should exist");
        assert_eq!(grad, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn tensor_permute_backward_preserves_element_gradients() {
        let mut tape = TensorTape::new();
        // x shape [2, 3]
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf should succeed");
        // permute to [3, 2]
        let y = tape.permute(x, vec![1, 0]).expect("permute should succeed");
        // multiply by weights
        let w = tape
            .leaf(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2], false)
            .expect("weights should succeed");
        let (product, _) = tape
            .mul(y, w, ExecutionMode::Strict)
            .expect("mul should succeed");
        let (z, _) = tape
            .sum(product, ExecutionMode::Strict)
            .expect("sum should succeed");

        let report = tape.backward(z).expect("backward should succeed");
        let grad = report.gradient(x).expect("x grad should exist");
        // permute([1,0]) on shape [2,3] produces shape [3,2]
        // grad of y w.r.t. product is w = [10, 20, 30, 40, 50, 60] in shape [3, 2]
        // inverse permute [1,0] on shape [3,2] -> shape [2,3]
        // This transposes back: [10, 30, 50, 20, 40, 60]
        assert_eq!(grad, &[10.0, 30.0, 50.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn tensor_double_transpose_is_identity() {
        let mut tape = TensorTape::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = tape
            .leaf(data.clone(), vec![2, 3], true)
            .expect("leaf should succeed");
        let y = tape
            .transpose(x, 0, 1)
            .expect("first transpose should succeed");
        let z = tape
            .transpose(y, 0, 1)
            .expect("second transpose should succeed");

        assert_eq!(tape.values(z).expect("values"), data);
        assert_eq!(tape.nodes[z.0].tensor.meta().shape(), &[2, 3]);
    }

    #[test]
    fn tensor_transpose_is_permute_special_case() {
        let mut tape = TensorTape::new();
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let x1 = tape
            .leaf(data.clone(), vec![2, 3, 4], true)
            .expect("x1 leaf");
        let y1 = tape.transpose(x1, 0, 2).expect("transpose(0,2)");

        let mut tape2 = TensorTape::new();
        let x2 = tape2.leaf(data, vec![2, 3, 4], true).expect("x2 leaf");
        let y2 = tape2.permute(x2, vec![2, 1, 0]).expect("permute [2,1,0]");

        assert_eq!(
            tape.values(y1).expect("transpose values"),
            tape2.values(y2).expect("permute values")
        );
    }

    // ── sin/cos/tan scalar tests ─────────────────────────────────────

    #[test]
    fn scalar_sin_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(std::f64::consts::FRAC_PI_2, true);
        let (y, _) = tape
            .sin(x, ExecutionMode::Strict)
            .expect("sin should succeed");
        let val = tape.value(y).expect("value");
        assert!(
            (val - 1.0).abs() < 1e-12,
            "sin(pi/2) should be 1.0, got {val}"
        );
    }

    #[test]
    fn scalar_sin_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(std::f64::consts::FRAC_PI_4, true);
        let (y, _) = tape
            .sin(x, ExecutionMode::Strict)
            .expect("sin should succeed");
        let report = tape.backward(y).expect("backward should succeed");
        // d/dx sin(x) = cos(x)
        let expected = std::f64::consts::FRAC_PI_4.cos();
        let grad = report.gradient(x).expect("gradient should exist");
        assert!(
            (grad - expected).abs() < 1e-12,
            "sin grad should be cos(pi/4)={expected}, got {grad}"
        );
    }

    #[test]
    fn scalar_cos_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape
            .cos(x, ExecutionMode::Strict)
            .expect("cos should succeed");
        let val = tape.value(y).expect("value");
        assert!((val - 1.0).abs() < 1e-12, "cos(0) should be 1.0, got {val}");
    }

    #[test]
    fn scalar_cos_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(std::f64::consts::FRAC_PI_4, true);
        let (y, _) = tape
            .cos(x, ExecutionMode::Strict)
            .expect("cos should succeed");
        let report = tape.backward(y).expect("backward should succeed");
        // d/dx cos(x) = -sin(x)
        let expected = -std::f64::consts::FRAC_PI_4.sin();
        let grad = report.gradient(x).expect("gradient should exist");
        assert!(
            (grad - expected).abs() < 1e-12,
            "cos grad should be -sin(pi/4)={expected}, got {grad}"
        );
    }

    #[test]
    fn scalar_tan_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(std::f64::consts::FRAC_PI_4, true);
        let (y, _) = tape
            .tan(x, ExecutionMode::Strict)
            .expect("tan should succeed");
        let val = tape.value(y).expect("value");
        assert!(
            (val - 1.0).abs() < 1e-12,
            "tan(pi/4) should be 1.0, got {val}"
        );
    }

    #[test]
    fn scalar_tan_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(std::f64::consts::FRAC_PI_4, true);
        let (y, _) = tape
            .tan(x, ExecutionMode::Strict)
            .expect("tan should succeed");
        let report = tape.backward(y).expect("backward should succeed");
        // d/dx tan(x) = 1 + tan(x)^2 = sec(x)^2
        let tan_val = std::f64::consts::FRAC_PI_4.tan();
        let expected = 1.0 + tan_val * tan_val;
        let grad = report.gradient(x).expect("gradient should exist");
        assert!(
            (grad - expected).abs() < 1e-12,
            "tan grad should be 1+tan^2={expected}, got {grad}"
        );
    }

    // ── sin/cos/tan tensor tests ─────────────────────────────────────

    #[test]
    fn tensor_sin_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(
                vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
                vec![3],
                true,
            )
            .expect("leaf");
        let (y, _) = tape
            .sin(x, ExecutionMode::Strict)
            .expect("sin should succeed");
        let vals = tape.values(y).expect("values");
        assert!((vals[0] - 0.0).abs() < 1e-12, "sin(0) should be 0");
        assert!((vals[1] - 1.0).abs() < 1e-12, "sin(pi/2) should be 1");
        assert!(vals[2].abs() < 1e-12, "sin(pi) should be ~0");
    }

    #[test]
    fn tensor_sin_backward() {
        let mut tape = TensorTape::new();
        let vals = vec![0.0, 1.0, 2.0];
        let x = tape.leaf(vals.clone(), vec![3], true).expect("leaf");
        let (y, _) = tape.sin(x, ExecutionMode::Strict).expect("sin");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("gradient");
        // d/dx sin(x) = cos(x)
        for (i, &v) in vals.iter().enumerate() {
            let expected = v.cos();
            assert!(
                (grads[i] - expected).abs() < 1e-12,
                "sin grad[{i}] should be {expected}, got {}",
                grads[i]
            );
        }
    }

    #[test]
    fn tensor_cos_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(
                vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
                vec![3],
                true,
            )
            .expect("leaf");
        let (y, _) = tape
            .cos(x, ExecutionMode::Strict)
            .expect("cos should succeed");
        let vals = tape.values(y).expect("values");
        assert!((vals[0] - 1.0).abs() < 1e-12, "cos(0) should be 1");
        assert!(vals[1].abs() < 1e-12, "cos(pi/2) should be ~0");
        assert!((vals[2] - (-1.0)).abs() < 1e-12, "cos(pi) should be -1");
    }

    #[test]
    fn tensor_cos_backward() {
        let mut tape = TensorTape::new();
        let vals = vec![0.0, 1.0, 2.0];
        let x = tape.leaf(vals.clone(), vec![3], true).expect("leaf");
        let (y, _) = tape.cos(x, ExecutionMode::Strict).expect("cos");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("gradient");
        // d/dx cos(x) = -sin(x)
        for (i, &v) in vals.iter().enumerate() {
            let expected = -v.sin();
            assert!(
                (grads[i] - expected).abs() < 1e-12,
                "cos grad[{i}] should be {expected}, got {}",
                grads[i]
            );
        }
    }

    #[test]
    fn tensor_tan_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![0.0, std::f64::consts::FRAC_PI_4], vec![2], true)
            .expect("leaf");
        let (y, _) = tape
            .tan(x, ExecutionMode::Strict)
            .expect("tan should succeed");
        let vals = tape.values(y).expect("values");
        assert!(vals[0].abs() < 1e-12, "tan(0) should be 0");
        assert!((vals[1] - 1.0).abs() < 1e-12, "tan(pi/4) should be 1");
    }

    #[test]
    fn tensor_tan_backward() {
        let mut tape = TensorTape::new();
        let vals = vec![0.0, 0.5, 1.0];
        let x = tape.leaf(vals.clone(), vec![3], true).expect("leaf");
        let (y, _) = tape.tan(x, ExecutionMode::Strict).expect("tan");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("gradient");
        // d/dx tan(x) = 1 + tan(x)^2
        for (i, &v) in vals.iter().enumerate() {
            let tan_v = v.tan();
            let expected = 1.0 + tan_v * tan_v;
            assert!(
                (grads[i] - expected).abs() < 1e-12,
                "tan grad[{i}] should be {expected}, got {}",
                grads[i]
            );
        }
    }

    #[test]
    fn scalar_sin_cos_identity() {
        // sin^2(x) + cos^2(x) = 1
        let mut tape = Tape::new();
        let x = tape.leaf(1.23, true);
        let (s, _) = tape.sin(x, ExecutionMode::Strict).expect("sin");
        let (c, _) = tape.cos(x, ExecutionMode::Strict).expect("cos");
        let (s2, _) = tape.mul(s, s, ExecutionMode::Strict).expect("s*s");
        let (c2, _) = tape.mul(c, c, ExecutionMode::Strict).expect("c*c");
        let (sum, _) = tape.add(s2, c2, ExecutionMode::Strict).expect("s2+c2");
        let val = tape.value(sum).expect("value");
        assert!(
            (val - 1.0).abs() < 1e-12,
            "sin^2+cos^2 should be 1, got {val}"
        );
    }

    #[test]
    fn scalar_sin_zero() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.sin(x, ExecutionMode::Strict).expect("sin");
        assert!(
            (tape.value(y).expect("value")).abs() < 1e-12,
            "sin(0) should be 0"
        );
    }

    // ── floor/ceil/round scalar tests ────────────────────────────────

    #[test]
    fn scalar_floor_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.7, true);
        let (y, _) = tape.floor(x, ExecutionMode::Strict).expect("floor");
        assert_eq!(tape.value(y).expect("value"), 2.0);
    }

    #[test]
    fn scalar_ceil_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.3, true);
        let (y, _) = tape.ceil(x, ExecutionMode::Strict).expect("ceil");
        assert_eq!(tape.value(y).expect("value"), 3.0);
    }

    #[test]
    fn scalar_round_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.5, true);
        let (y, _) = tape.round(x, ExecutionMode::Strict).expect("round");
        // f64::round() uses round-half-away-from-zero: 2.5 -> 3.0
        assert_eq!(tape.value(y).expect("value"), 3.0);
    }

    #[test]
    fn scalar_round_forward_down() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.3, true);
        let (y, _) = tape.round(x, ExecutionMode::Strict).expect("round");
        assert_eq!(tape.value(y).expect("value"), 2.0);
    }

    #[test]
    fn scalar_floor_backward_zero_grad() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.7, true);
        let (y, _) = tape.floor(x, ExecutionMode::Strict).expect("floor");
        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(x), Some(0.0));
    }

    #[test]
    fn scalar_ceil_backward_zero_grad() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.3, true);
        let (y, _) = tape.ceil(x, ExecutionMode::Strict).expect("ceil");
        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(x), Some(0.0));
    }

    #[test]
    fn scalar_round_backward_zero_grad() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.7, true);
        let (y, _) = tape.round(x, ExecutionMode::Strict).expect("round");
        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(x), Some(0.0));
    }

    #[test]
    fn scalar_floor_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-2.3, true);
        let (y, _) = tape.floor(x, ExecutionMode::Strict).expect("floor");
        assert_eq!(tape.value(y).expect("value"), -3.0);
    }

    #[test]
    fn scalar_ceil_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-2.7, true);
        let (y, _) = tape.ceil(x, ExecutionMode::Strict).expect("ceil");
        assert_eq!(tape.value(y).expect("value"), -2.0);
    }

    // ── floor/ceil/round tensor tests ────────────────────────────────

    #[test]
    fn tensor_floor_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.1, 2.7, -0.3, -1.9], vec![4], true)
            .expect("leaf");
        let (y, _) = tape.floor(x, ExecutionMode::Strict).expect("floor");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, vec![1.0, 2.0, -1.0, -2.0]);
    }

    #[test]
    fn tensor_ceil_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.1, 2.7, -0.3, -1.9], vec![4], true)
            .expect("leaf");
        let (y, _) = tape.ceil(x, ExecutionMode::Strict).expect("ceil");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, vec![2.0, 3.0, 0.0, -1.0]);
    }

    #[test]
    fn tensor_round_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.1, 2.7, -0.3, -1.9], vec![4], true)
            .expect("leaf");
        let (y, _) = tape.round(x, ExecutionMode::Strict).expect("round");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, vec![1.0, 3.0, 0.0, -2.0]);
    }

    #[test]
    fn tensor_floor_backward_zero_grad() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.5, 2.5], vec![2], true).expect("leaf");
        let (y, _) = tape.floor(x, ExecutionMode::Strict).expect("floor");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("gradient");
        assert_eq!(grads, &[0.0, 0.0]);
    }

    #[test]
    fn tensor_ceil_backward_zero_grad() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.5, 2.5], vec![2], true).expect("leaf");
        let (y, _) = tape.ceil(x, ExecutionMode::Strict).expect("ceil");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("gradient");
        assert_eq!(grads, &[0.0, 0.0]);
    }

    #[test]
    fn tensor_round_backward_zero_grad() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.5, 2.5], vec![2], true).expect("leaf");
        let (y, _) = tape.round(x, ExecutionMode::Strict).expect("round");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("gradient");
        assert_eq!(grads, &[0.0, 0.0]);
    }

    // ── log2/log10/log1p/expm1 scalar tests ──────────────────────────

    #[test]
    fn scalar_log2_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(8.0, true);
        let (y, _) = tape.log2(x, ExecutionMode::Strict).expect("log2");
        assert!((tape.value(y).expect("value") - 3.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_log2_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(4.0, true);
        let (y, _) = tape.log2(x, ExecutionMode::Strict).expect("log2");
        let report = tape.backward(y).expect("backward");
        // d/dx log2(x) = 1/(x * ln(2))
        let expected = 1.0 / (4.0 * std::f64::consts::LN_2);
        let grad = report.gradient(x).expect("grad");
        assert!((grad - expected).abs() < 1e-12);
    }

    #[test]
    fn scalar_log10_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1000.0, true);
        let (y, _) = tape.log10(x, ExecutionMode::Strict).expect("log10");
        assert!((tape.value(y).expect("value") - 3.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_log10_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(10.0, true);
        let (y, _) = tape.log10(x, ExecutionMode::Strict).expect("log10");
        let report = tape.backward(y).expect("backward");
        let expected = 1.0 / (10.0 * std::f64::consts::LN_10);
        let grad = report.gradient(x).expect("grad");
        assert!((grad - expected).abs() < 1e-12);
    }

    #[test]
    fn scalar_log1p_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.log1p(x, ExecutionMode::Strict).expect("log1p");
        assert!(
            (tape.value(y).expect("value")).abs() < 1e-12,
            "log1p(0) = 0"
        );
    }

    #[test]
    fn scalar_log1p_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.log1p(x, ExecutionMode::Strict).expect("log1p");
        let report = tape.backward(y).expect("backward");
        // d/dx log1p(x) = 1/(1+x) = 0.5
        let grad = report.gradient(x).expect("grad");
        assert!((grad - 0.5).abs() < 1e-12);
    }

    #[test]
    fn scalar_expm1_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.expm1(x, ExecutionMode::Strict).expect("expm1");
        assert!(
            (tape.value(y).expect("value")).abs() < 1e-12,
            "expm1(0) = 0"
        );
    }

    #[test]
    fn scalar_expm1_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.expm1(x, ExecutionMode::Strict).expect("expm1");
        let report = tape.backward(y).expect("backward");
        // d/dx expm1(x) = exp(x)
        let expected = std::f64::consts::E;
        let grad = report.gradient(x).expect("grad");
        assert!((grad - expected).abs() < 1e-12);
    }

    // ── log2/log10/log1p/expm1 tensor tests ──────────────────────────

    #[test]
    fn tensor_log2_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 4.0, 8.0], vec![4], true)
            .expect("leaf");
        let (y, _) = tape.log2(x, ExecutionMode::Strict).expect("log2");
        let vals = tape.values(y).expect("values");
        assert!((vals[0]).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
        assert!((vals[2] - 2.0).abs() < 1e-12);
        assert!((vals[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn tensor_log2_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![2.0, 4.0], vec![2], true).expect("leaf");
        let (y, _) = tape.log2(x, ExecutionMode::Strict).expect("log2");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert!((grads[0] - 1.0 / (2.0 * std::f64::consts::LN_2)).abs() < 1e-12);
        assert!((grads[1] - 1.0 / (4.0 * std::f64::consts::LN_2)).abs() < 1e-12);
    }

    #[test]
    fn tensor_log1p_forward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.log1p(x, ExecutionMode::Strict).expect("log1p");
        let vals = tape.values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn tensor_log1p_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.log1p(x, ExecutionMode::Strict).expect("log1p");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // d/dx log1p(x) = 1/(1+x)
        assert!((grads[0] - 1.0).abs() < 1e-12); // 1/(1+0) = 1
        assert!((grads[1] - 0.5).abs() < 1e-12); // 1/(1+1) = 0.5
    }

    #[test]
    fn tensor_expm1_forward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.expm1(x, ExecutionMode::Strict).expect("expm1");
        let vals = tape.values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - (std::f64::consts::E - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn tensor_expm1_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.expm1(x, ExecutionMode::Strict).expect("expm1");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // d/dx expm1(x) = exp(x)
        assert!((grads[0] - 1.0).abs() < 1e-12); // exp(0) = 1
        assert!((grads[1] - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn scalar_log1p_expm1_roundtrip() {
        // log1p(expm1(x)) = x for moderate x
        let mut tape = Tape::new();
        let x = tape.leaf(0.5, true);
        let (em, _) = tape.expm1(x, ExecutionMode::Strict).expect("expm1");
        let (y, _) = tape.log1p(em, ExecutionMode::Strict).expect("log1p");
        let val = tape.value(y).expect("value");
        assert!(
            (val - 0.5).abs() < 1e-12,
            "log1p(expm1(x)) should be x, got {val}"
        );
    }

    // ── sign/trunc/frac scalar tests ──────────────────────────

    #[test]
    fn scalar_sign_forward_positive() {
        let mut tape = Tape::new();
        let x = tape.leaf(3.5, true);
        let (y, _) = tape.sign(x, ExecutionMode::Strict).expect("sign");
        assert_eq!(tape.value(y).expect("value"), 1.0);
    }

    #[test]
    fn scalar_sign_forward_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-2.0, true);
        let (y, _) = tape.sign(x, ExecutionMode::Strict).expect("sign");
        assert_eq!(tape.value(y).expect("value"), -1.0);
    }

    #[test]
    fn scalar_sign_forward_zero() {
        // Rust signum(+0.0) = 1.0 (not 0.0 like PyTorch)
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.sign(x, ExecutionMode::Strict).expect("sign");
        assert_eq!(tape.value(y).expect("value"), 1.0);
    }

    #[test]
    fn scalar_sign_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(5.0, true);
        let (y, _) = tape.sign(x, ExecutionMode::Strict).expect("sign");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        assert_eq!(grad, 0.0);
    }

    #[test]
    fn scalar_trunc_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(3.7, true);
        let (y, _) = tape.trunc(x, ExecutionMode::Strict).expect("trunc");
        assert_eq!(tape.value(y).expect("value"), 3.0);
    }

    #[test]
    fn scalar_trunc_forward_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-3.7, true);
        let (y, _) = tape.trunc(x, ExecutionMode::Strict).expect("trunc");
        assert_eq!(tape.value(y).expect("value"), -3.0);
    }

    #[test]
    fn scalar_trunc_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(3.7, true);
        let (y, _) = tape.trunc(x, ExecutionMode::Strict).expect("trunc");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        assert_eq!(grad, 0.0);
    }

    #[test]
    fn scalar_frac_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(3.7, true);
        let (y, _) = tape.frac(x, ExecutionMode::Strict).expect("frac");
        let val = tape.value(y).expect("value");
        assert!((val - 0.7).abs() < 1e-12);
    }

    #[test]
    fn scalar_frac_forward_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-3.7, true);
        let (y, _) = tape.frac(x, ExecutionMode::Strict).expect("frac");
        let val = tape.value(y).expect("value");
        // Rust fract() returns -0.7 for -3.7 (preserves sign)
        assert!((val - (-0.7)).abs() < 1e-12);
    }

    #[test]
    fn scalar_frac_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(3.7, true);
        let (y, _) = tape.frac(x, ExecutionMode::Strict).expect("frac");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        assert_eq!(grad, 1.0);
    }

    #[test]
    fn scalar_trunc_frac_identity() {
        // trunc(x) + frac(x) = x
        let mut tape = Tape::new();
        let x = tape.leaf(3.7, true);
        let (t, _) = tape.trunc(x, ExecutionMode::Strict).expect("trunc");
        let (f, _) = tape.frac(x, ExecutionMode::Strict).expect("frac");
        let (y, _) = tape.add(t, f, ExecutionMode::Strict).expect("add");
        let val = tape.value(y).expect("value");
        assert!(
            (val - 3.7).abs() < 1e-12,
            "trunc(x)+frac(x) should be x, got {val}"
        );
    }

    // ── sign/trunc/frac tensor tests ──────────────────────────

    #[test]
    fn tensor_sign_forward() {
        let mut tape = TensorTape::new();
        // Rust signum(+0.0) = 1.0
        let x = tape
            .leaf(vec![-3.0, 0.0, 5.0, -1.0], vec![4], true)
            .expect("leaf");
        let (y, _) = tape.sign(x, ExecutionMode::Strict).expect("sign");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[-1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn tensor_sign_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![-3.0, 5.0], vec![2], true).expect("leaf");
        let (y, _) = tape.sign(x, ExecutionMode::Strict).expect("sign");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert_eq!(grads, &[0.0, 0.0]);
    }

    #[test]
    fn tensor_trunc_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![3.7, -2.3, 0.9], vec![3], true)
            .expect("leaf");
        let (y, _) = tape.trunc(x, ExecutionMode::Strict).expect("trunc");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[3.0, -2.0, 0.0]);
    }

    #[test]
    fn tensor_trunc_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![3.7, -2.3], vec![2], true).expect("leaf");
        let (y, _) = tape.trunc(x, ExecutionMode::Strict).expect("trunc");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert_eq!(grads, &[0.0, 0.0]);
    }

    #[test]
    fn tensor_frac_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![3.7, -2.3, 5.0], vec![3], true)
            .expect("leaf");
        let (y, _) = tape.frac(x, ExecutionMode::Strict).expect("frac");
        let vals = tape.values(y).expect("values");
        assert!((vals[0] - 0.7).abs() < 1e-12);
        assert!((vals[1] - (-0.3)).abs() < 1e-12);
        assert!(vals[2].abs() < 1e-12);
    }

    #[test]
    fn tensor_frac_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![3.7, -2.3], vec![2], true).expect("leaf");
        let (y, _) = tape.frac(x, ExecutionMode::Strict).expect("frac");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert_eq!(grads, &[1.0, 1.0]);
    }

    // ── asin/acos/atan scalar tests ──────────────────────────

    #[test]
    fn scalar_asin_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.5, true);
        let (y, _) = tape.asin(x, ExecutionMode::Strict).expect("asin");
        let val = tape.value(y).expect("value");
        assert!((val - 0.5_f64.asin()).abs() < 1e-12);
    }

    #[test]
    fn scalar_asin_forward_zero() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.asin(x, ExecutionMode::Strict).expect("asin");
        assert_eq!(tape.value(y).expect("value"), 0.0);
    }

    #[test]
    fn scalar_asin_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.5, true);
        let (y, _) = tape.asin(x, ExecutionMode::Strict).expect("asin");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // d/dx asin(x) = 1/sqrt(1-x^2)
        let expected = 1.0 / (1.0 - 0.25_f64).sqrt();
        assert!((grad - expected).abs() < 1e-12);
    }

    #[test]
    fn scalar_acos_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.5, true);
        let (y, _) = tape.acos(x, ExecutionMode::Strict).expect("acos");
        let val = tape.value(y).expect("value");
        assert!((val - 0.5_f64.acos()).abs() < 1e-12);
    }

    #[test]
    fn scalar_acos_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.5, true);
        let (y, _) = tape.acos(x, ExecutionMode::Strict).expect("acos");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // d/dx acos(x) = -1/sqrt(1-x^2)
        let expected = -1.0 / (1.0 - 0.25_f64).sqrt();
        assert!((grad - expected).abs() < 1e-12);
    }

    #[test]
    fn scalar_atan_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.atan(x, ExecutionMode::Strict).expect("atan");
        let val = tape.value(y).expect("value");
        assert!((val - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn scalar_atan_forward_zero() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.atan(x, ExecutionMode::Strict).expect("atan");
        assert_eq!(tape.value(y).expect("value"), 0.0);
    }

    #[test]
    fn scalar_atan_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.atan(x, ExecutionMode::Strict).expect("atan");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // d/dx atan(x) = 1/(1+x^2) = 1/2
        assert!((grad - 0.5).abs() < 1e-12);
    }

    #[test]
    fn scalar_asin_acos_sum_is_pi_over_2() {
        // asin(x) + acos(x) = pi/2
        let mut tape = Tape::new();
        let x = tape.leaf(0.3, true);
        let (a, _) = tape.asin(x, ExecutionMode::Strict).expect("asin");
        let (b, _) = tape.acos(x, ExecutionMode::Strict).expect("acos");
        let (y, _) = tape.add(a, b, ExecutionMode::Strict).expect("add");
        let val = tape.value(y).expect("value");
        assert!((val - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    // ── asin/acos/atan tensor tests ──────────────────────────

    #[test]
    fn tensor_asin_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![0.0, 0.5, -0.5], vec![3], true)
            .expect("leaf");
        let (y, _) = tape.asin(x, ExecutionMode::Strict).expect("asin");
        let vals = tape.values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 0.5_f64.asin()).abs() < 1e-12);
        assert!((vals[2] - (-0.5_f64).asin()).abs() < 1e-12);
    }

    #[test]
    fn tensor_asin_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 0.5], vec![2], true).expect("leaf");
        let (y, _) = tape.asin(x, ExecutionMode::Strict).expect("asin");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert!((grads[0] - 1.0).abs() < 1e-12); // 1/sqrt(1-0) = 1
        assert!((grads[1] - 1.0 / (0.75_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn tensor_acos_forward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.acos(x, ExecutionMode::Strict).expect("acos");
        let vals = tape.values(y).expect("values");
        assert!((vals[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!(vals[1].abs() < 1e-12);
    }

    #[test]
    fn tensor_acos_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 0.5], vec![2], true).expect("leaf");
        let (y, _) = tape.acos(x, ExecutionMode::Strict).expect("acos");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // d/dx acos(x) = -1/sqrt(1-x^2)
        assert!((grads[0] - (-1.0)).abs() < 1e-12); // -1/sqrt(1-0) = -1
        assert!((grads[1] - (-1.0 / (0.75_f64).sqrt())).abs() < 1e-12);
    }

    #[test]
    fn tensor_atan_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![0.0, 1.0, -1.0], vec![3], true)
            .expect("leaf");
        let (y, _) = tape.atan(x, ExecutionMode::Strict).expect("atan");
        let vals = tape.values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((vals[2] + std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn tensor_atan_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.atan(x, ExecutionMode::Strict).expect("atan");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert!((grads[0] - 1.0).abs() < 1e-12); // 1/(1+0) = 1
        assert!((grads[1] - 0.5).abs() < 1e-12); // 1/(1+1) = 0.5
    }

    // ── sinh/cosh scalar tests ──────────────────────────

    #[test]
    fn scalar_sinh_forward_zero() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.sinh(x, ExecutionMode::Strict).expect("sinh");
        assert_eq!(tape.value(y).expect("value"), 0.0);
    }

    #[test]
    fn scalar_sinh_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.sinh(x, ExecutionMode::Strict).expect("sinh");
        let val = tape.value(y).expect("value");
        assert!((val - 1.0_f64.sinh()).abs() < 1e-12);
    }

    #[test]
    fn scalar_sinh_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.sinh(x, ExecutionMode::Strict).expect("sinh");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // d/dx sinh(x) = cosh(x)
        assert!((grad - 1.0_f64.cosh()).abs() < 1e-12);
    }

    #[test]
    fn scalar_cosh_forward_zero() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.cosh(x, ExecutionMode::Strict).expect("cosh");
        assert_eq!(tape.value(y).expect("value"), 1.0);
    }

    #[test]
    fn scalar_cosh_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.cosh(x, ExecutionMode::Strict).expect("cosh");
        let val = tape.value(y).expect("value");
        assert!((val - 1.0_f64.cosh()).abs() < 1e-12);
    }

    #[test]
    fn scalar_cosh_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.cosh(x, ExecutionMode::Strict).expect("cosh");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // d/dx cosh(x) = sinh(x)
        assert!((grad - 1.0_f64.sinh()).abs() < 1e-12);
    }

    #[test]
    fn scalar_cosh_squared_minus_sinh_squared_is_one() {
        // cosh^2(x) - sinh^2(x) = 1
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let (s, _) = tape.sinh(x, ExecutionMode::Strict).expect("sinh");
        let (c, _) = tape.cosh(x, ExecutionMode::Strict).expect("cosh");
        let (s2, _) = tape.mul(s, s, ExecutionMode::Strict).expect("s*s");
        let (c2, _) = tape.mul(c, c, ExecutionMode::Strict).expect("c*c");
        let (y, _) = tape.sub(c2, s2, ExecutionMode::Strict).expect("c2-s2");
        let val = tape.value(y).expect("value");
        assert!(
            (val - 1.0).abs() < 1e-10,
            "cosh^2-sinh^2 should be 1, got {val}"
        );
    }

    // ── sinh/cosh tensor tests ──────────────────────────

    #[test]
    fn tensor_sinh_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![0.0, 1.0, -1.0], vec![3], true)
            .expect("leaf");
        let (y, _) = tape.sinh(x, ExecutionMode::Strict).expect("sinh");
        let vals = tape.values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - 1.0_f64.sinh()).abs() < 1e-12);
        assert!((vals[2] - (-1.0_f64).sinh()).abs() < 1e-12);
    }

    #[test]
    fn tensor_sinh_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.sinh(x, ExecutionMode::Strict).expect("sinh");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert!((grads[0] - 1.0).abs() < 1e-12); // cosh(0) = 1
        assert!((grads[1] - 1.0_f64.cosh()).abs() < 1e-12);
    }

    #[test]
    fn tensor_cosh_forward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.cosh(x, ExecutionMode::Strict).expect("cosh");
        let vals = tape.values(y).expect("values");
        assert!((vals[0] - 1.0).abs() < 1e-12); // cosh(0) = 1
        assert!((vals[1] - 1.0_f64.cosh()).abs() < 1e-12);
    }

    #[test]
    fn tensor_cosh_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.cosh(x, ExecutionMode::Strict).expect("cosh");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert!(grads[0].abs() < 1e-12); // sinh(0) = 0
        assert!((grads[1] - 1.0_f64.sinh()).abs() < 1e-12);
    }

    // ── gelu/silu/leaky_relu/elu scalar tests ──────────────────────────

    fn gelu_expected(x: f64) -> f64 {
        let c = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
        let k = c * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + k.tanh())
    }

    fn silu_expected(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn scalar_gelu_forward_zero() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.gelu(x, ExecutionMode::Strict).expect("gelu");
        assert_eq!(tape.value(y).expect("value"), 0.0);
    }

    #[test]
    fn scalar_gelu_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.gelu(x, ExecutionMode::Strict).expect("gelu");
        let val = tape.value(y).expect("value");
        assert!((val - gelu_expected(1.0)).abs() < 1e-12);
    }

    #[test]
    fn scalar_gelu_forward_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-1.0, true);
        let (y, _) = tape.gelu(x, ExecutionMode::Strict).expect("gelu");
        let val = tape.value(y).expect("value");
        assert!((val - gelu_expected(-1.0)).abs() < 1e-12);
    }

    #[test]
    fn scalar_gelu_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.gelu(x, ExecutionMode::Strict).expect("gelu");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // Numerical gradient check
        let eps = 1e-6;
        let numerical = (gelu_expected(1.0 + eps) - gelu_expected(1.0 - eps)) / (2.0 * eps);
        assert!(
            (grad - numerical).abs() < 1e-5,
            "gelu grad {grad} vs numerical {numerical}"
        );
    }

    #[test]
    fn scalar_silu_forward_zero() {
        let mut tape = Tape::new();
        let x = tape.leaf(0.0, true);
        let (y, _) = tape.silu(x, ExecutionMode::Strict).expect("silu");
        assert_eq!(tape.value(y).expect("value"), 0.0);
    }

    #[test]
    fn scalar_silu_forward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.silu(x, ExecutionMode::Strict).expect("silu");
        let val = tape.value(y).expect("value");
        assert!((val - silu_expected(1.0)).abs() < 1e-12);
    }

    #[test]
    fn scalar_silu_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(1.0, true);
        let (y, _) = tape.silu(x, ExecutionMode::Strict).expect("silu");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // Numerical gradient check
        let eps = 1e-6;
        let numerical = (silu_expected(1.0 + eps) - silu_expected(1.0 - eps)) / (2.0 * eps);
        assert!(
            (grad - numerical).abs() < 1e-5,
            "silu grad {grad} vs numerical {numerical}"
        );
    }

    #[test]
    fn scalar_leaky_relu_forward_positive() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let (y, _) = tape
            .leaky_relu(x, ExecutionMode::Strict)
            .expect("leaky_relu");
        assert_eq!(tape.value(y).expect("value"), 2.0);
    }

    #[test]
    fn scalar_leaky_relu_forward_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-3.0, true);
        let (y, _) = tape
            .leaky_relu(x, ExecutionMode::Strict)
            .expect("leaky_relu");
        let val = tape.value(y).expect("value");
        assert!((val - (-0.03)).abs() < 1e-12);
    }

    #[test]
    fn scalar_leaky_relu_backward_positive() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let (y, _) = tape
            .leaky_relu(x, ExecutionMode::Strict)
            .expect("leaky_relu");
        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(x).expect("grad"), 1.0);
    }

    #[test]
    fn scalar_leaky_relu_backward_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-3.0, true);
        let (y, _) = tape
            .leaky_relu(x, ExecutionMode::Strict)
            .expect("leaky_relu");
        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(x).expect("grad"), 0.01);
    }

    #[test]
    fn scalar_elu_forward_positive() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let (y, _) = tape.elu(x, ExecutionMode::Strict).expect("elu");
        assert_eq!(tape.value(y).expect("value"), 2.0);
    }

    #[test]
    fn scalar_elu_forward_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-1.0, true);
        let (y, _) = tape.elu(x, ExecutionMode::Strict).expect("elu");
        let val = tape.value(y).expect("value");
        // elu(-1) = exp(-1) - 1
        assert!((val - ((-1.0_f64).exp() - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn scalar_elu_backward_positive() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let (y, _) = tape.elu(x, ExecutionMode::Strict).expect("elu");
        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(x).expect("grad"), 1.0);
    }

    #[test]
    fn scalar_elu_backward_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-1.0, true);
        let (y, _) = tape.elu(x, ExecutionMode::Strict).expect("elu");
        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("grad");
        // d/dx elu(x) = exp(x) for x < 0 (alpha=1.0)
        assert!((grad - (-1.0_f64).exp()).abs() < 1e-12);
    }

    // ── gelu/silu/leaky_relu/elu tensor tests ──────────────────────────

    #[test]
    fn tensor_gelu_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![0.0, 1.0, -1.0], vec![3], true)
            .expect("leaf");
        let (y, _) = tape.gelu(x, ExecutionMode::Strict).expect("gelu");
        let vals = tape.values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - gelu_expected(1.0)).abs() < 1e-12);
        assert!((vals[2] - gelu_expected(-1.0)).abs() < 1e-12);
    }

    #[test]
    fn tensor_gelu_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.gelu(x, ExecutionMode::Strict).expect("gelu");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        let eps = 1e-6;
        let num0 = (gelu_expected(eps) - gelu_expected(-eps)) / (2.0 * eps);
        let num1 = (gelu_expected(1.0 + eps) - gelu_expected(1.0 - eps)) / (2.0 * eps);
        assert!((grads[0] - num0).abs() < 1e-5);
        assert!((grads[1] - num1).abs() < 1e-5);
    }

    #[test]
    fn tensor_silu_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![0.0, 1.0, -1.0], vec![3], true)
            .expect("leaf");
        let (y, _) = tape.silu(x, ExecutionMode::Strict).expect("silu");
        let vals = tape.values(y).expect("values");
        assert!(vals[0].abs() < 1e-12);
        assert!((vals[1] - silu_expected(1.0)).abs() < 1e-12);
        assert!((vals[2] - silu_expected(-1.0)).abs() < 1e-12);
    }

    #[test]
    fn tensor_silu_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![0.0, 1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.silu(x, ExecutionMode::Strict).expect("silu");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        let eps = 1e-6;
        let num0 = (silu_expected(eps) - silu_expected(-eps)) / (2.0 * eps);
        let num1 = (silu_expected(1.0 + eps) - silu_expected(1.0 - eps)) / (2.0 * eps);
        assert!((grads[0] - num0).abs() < 1e-5);
        assert!((grads[1] - num1).abs() < 1e-5);
    }

    #[test]
    fn tensor_leaky_relu_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![2.0, -3.0, 0.0], vec![3], true)
            .expect("leaf");
        let (y, _) = tape
            .leaky_relu(x, ExecutionMode::Strict)
            .expect("leaky_relu");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals[0], 2.0);
        assert!((vals[1] - (-0.03)).abs() < 1e-12);
        assert_eq!(vals[2], 0.0);
    }

    #[test]
    fn tensor_leaky_relu_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![2.0, -3.0], vec![2], true).expect("leaf");
        let (y, _) = tape
            .leaky_relu(x, ExecutionMode::Strict)
            .expect("leaky_relu");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert_eq!(grads[0], 1.0);
        assert_eq!(grads[1], 0.01);
    }

    #[test]
    fn tensor_elu_forward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![2.0, -1.0, 0.0], vec![3], true)
            .expect("leaf");
        let (y, _) = tape.elu(x, ExecutionMode::Strict).expect("elu");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals[0], 2.0);
        assert!((vals[1] - ((-1.0_f64).exp() - 1.0)).abs() < 1e-12);
        assert_eq!(vals[2], 0.0);
    }

    #[test]
    fn tensor_elu_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![2.0, -1.0], vec![2], true).expect("leaf");
        let (y, _) = tape.elu(x, ExecutionMode::Strict).expect("elu");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        assert_eq!(grads[0], 1.0);
        assert!((grads[1] - (-1.0_f64).exp()).abs() < 1e-12);
    }

    // ── prod_dim/var_dim/std_dim tensor tests ──────────────────────────

    #[test]
    fn tensor_prod_dim_forward_dim0() {
        let mut tape = TensorTape::new();
        // shape [2,3]: [[1,2,3],[4,5,6]]
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf");
        let (y, _) = tape
            .prod_dim(x, 0, ExecutionMode::Strict)
            .expect("prod_dim 0");
        let vals = tape.values(y).expect("values");
        // [1*4, 2*5, 3*6] = [4, 10, 18]
        assert_eq!(vals, &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn tensor_prod_dim_forward_dim1() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf");
        let (y, _) = tape
            .prod_dim(x, 1, ExecutionMode::Strict)
            .expect("prod_dim 1");
        let vals = tape.values(y).expect("values");
        // [1*2*3, 4*5*6] = [6, 120]
        assert_eq!(vals, &[6.0, 120.0]);
    }

    #[test]
    fn tensor_prod_dim_backward() {
        let mut tape = TensorTape::new();
        // shape [2,3]: [[2,3,4],[5,6,7]]
        let x = tape
            .leaf(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 3], true)
            .expect("leaf");
        let (y, _) = tape
            .prod_dim(x, 1, ExecutionMode::Strict)
            .expect("prod_dim 1");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // prod([2,3,4])=24, grad_i = 24/x_i -> [12, 8, 6]
        // prod([5,6,7])=210, grad_i = 210/x_i -> [42, 35, 30]
        assert!((grads[0] - 12.0).abs() < 1e-10);
        assert!((grads[1] - 8.0).abs() < 1e-10);
        assert!((grads[2] - 6.0).abs() < 1e-10);
        assert!((grads[3] - 42.0).abs() < 1e-10);
        assert!((grads[4] - 35.0).abs() < 1e-10);
        assert!((grads[5] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_prod_dim_backward_with_zero() {
        let mut tape = TensorTape::new();
        // shape [1,3]: [[2, 0, 4]]
        let x = tape
            .leaf(vec![2.0, 0.0, 4.0], vec![1, 3], true)
            .expect("leaf");
        let (y, _) = tape
            .prod_dim(x, 1, ExecutionMode::Strict)
            .expect("prod_dim 1");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // prod=0, zero_count=1, prod_no_zero=8
        // grad for x=2: 0 (since prod=0, x!=0)
        // grad for x=0: prod_no_zero=8
        // grad for x=4: 0
        assert_eq!(grads[0], 0.0);
        assert!((grads[1] - 8.0).abs() < 1e-10);
        assert_eq!(grads[2], 0.0);
    }

    #[test]
    fn tensor_var_dim_forward_dim0() {
        let mut tape = TensorTape::new();
        // shape [3,2]: [[1,2],[3,4],[5,6]]
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], true)
            .expect("leaf");
        let (y, _) = tape
            .var_dim(x, 0, ExecutionMode::Strict)
            .expect("var_dim 0");
        let vals = tape.values(y).expect("values");
        // col0: [1,3,5] mean=3, var=(4+0+4)/2=4
        // col1: [2,4,6] mean=4, var=(4+0+4)/2=4
        assert!((vals[0] - 4.0).abs() < 1e-12);
        assert!((vals[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn tensor_var_dim_forward_dim1() {
        let mut tape = TensorTape::new();
        // shape [2,3]: [[1,2,3],[4,5,6]]
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf");
        let (y, _) = tape
            .var_dim(x, 1, ExecutionMode::Strict)
            .expect("var_dim 1");
        let vals = tape.values(y).expect("values");
        // row0: [1,2,3] mean=2, var=(1+0+1)/2=1
        // row1: [4,5,6] mean=5, var=(1+0+1)/2=1
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tensor_var_dim_backward() {
        let mut tape = TensorTape::new();
        // shape [1,3]: [[1,2,3]] mean=2, correction=2
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let (y, _) = tape
            .var_dim(x, 1, ExecutionMode::Strict)
            .expect("var_dim 1");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // d(var)/dx_i = 2*(x_i - mean)/(n-1) = 2*(x_i - 2)/2 = x_i - 2
        // grad[0] = 1-2 = -1, grad[1] = 2-2 = 0, grad[2] = 3-2 = 1
        assert!((grads[0] - (-1.0)).abs() < 1e-12);
        assert!(grads[1].abs() < 1e-12);
        assert!((grads[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tensor_std_dim_forward_dim0() {
        let mut tape = TensorTape::new();
        // shape [3,2]: [[1,2],[3,4],[5,6]]
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], true)
            .expect("leaf");
        let (y, _) = tape
            .std_dim(x, 0, ExecutionMode::Strict)
            .expect("std_dim 0");
        let vals = tape.values(y).expect("values");
        // sqrt(4) = 2
        assert!((vals[0] - 2.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn tensor_std_dim_backward() {
        let mut tape = TensorTape::new();
        // shape [1,3]: [[1,2,3]] mean=2, std=1, correction=2
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let (y, _) = tape
            .std_dim(x, 1, ExecutionMode::Strict)
            .expect("std_dim 1");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // d(std)/dx_i = (x_i - mean) / ((n-1) * std) = (x_i - 2) / (2 * 1)
        // grad[0] = -1/2 = -0.5, grad[1] = 0, grad[2] = 1/2 = 0.5
        assert!((grads[0] - (-0.5)).abs() < 1e-12);
        assert!(grads[1].abs() < 1e-12);
        assert!((grads[2] - 0.5).abs() < 1e-12);
    }

    // ── softmax/log_softmax tensor tests ──────────────────────────

    #[test]
    fn tensor_softmax_forward_sums_to_one() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("leaf");
        let (y, _) = tape.softmax(x, 1, ExecutionMode::Strict).expect("softmax");
        let vals = tape.values(y).expect("values");
        let row0_sum: f64 = vals[0..3].iter().sum();
        let row1_sum: f64 = vals[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-12);
        assert!((row1_sum - 1.0).abs() < 1e-12);
        assert!(vals.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn tensor_softmax_forward_preserves_order() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 3.0, 2.0], vec![1, 3], true)
            .expect("leaf");
        let (y, _) = tape.softmax(x, 1, ExecutionMode::Strict).expect("softmax");
        let vals = tape.values(y).expect("values");
        assert!(vals[1] > vals[2]);
        assert!(vals[2] > vals[0]);
    }

    #[test]
    fn tensor_softmax_backward_grad_sum_is_zero() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let (y, _) = tape.softmax(x, 1, ExecutionMode::Strict).expect("softmax");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        // softmax is shift-invariant, so grad sum should be 0
        let grad_sum: f64 = grads.iter().sum();
        assert!(
            grad_sum.abs() < 1e-12,
            "softmax grad sum should be 0, got {grad_sum}"
        );
    }

    #[test]
    fn tensor_softmax_backward_dim0() {
        let mut tape = TensorTape::new();
        // shape [3,1]: softmax along dim 0
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0], vec![3, 1], true)
            .expect("leaf");
        let (y, _) = tape.softmax(x, 0, ExecutionMode::Strict).expect("softmax");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        let grad_sum: f64 = grads.iter().sum();
        assert!(grad_sum.abs() < 1e-12);
    }

    #[test]
    fn tensor_log_softmax_forward_consistent() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let (y, _) = tape
            .log_softmax(x, 1, ExecutionMode::Strict)
            .expect("log_softmax");
        let vals = tape.values(y).expect("values");
        // exp of log_softmax should sum to 1
        let sum: f64 = vals.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // All values should be negative (log of probability < 1)
        assert!(vals.iter().all(|&v| v <= 0.0));
    }

    #[test]
    fn tensor_log_softmax_backward_grad_sum_is_zero() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let (y, _) = tape
            .log_softmax(x, 1, ExecutionMode::Strict)
            .expect("log_softmax");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad");
        let grad_sum: f64 = grads.iter().sum();
        assert!(
            grad_sum.abs() < 1e-12,
            "log_softmax grad sum should be 0, got {grad_sum}"
        );
    }

    #[test]
    fn tensor_log_softmax_equals_log_of_softmax() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let (ls, _) = tape
            .log_softmax(x, 1, ExecutionMode::Strict)
            .expect("log_softmax");
        let ls_vals = tape.values(ls).expect("values");

        let mut tape2 = TensorTape::new();
        let x2 = tape2
            .leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true)
            .expect("leaf");
        let (sm, _) = tape2
            .softmax(x2, 1, ExecutionMode::Strict)
            .expect("softmax");
        let sm_vals = tape2.values(sm).expect("values");

        for i in 0..3 {
            assert!((ls_vals[i] - sm_vals[i].ln()).abs() < 1e-12);
        }
    }

    // ── cat/stack tensor tests ──────────────────────────

    #[test]
    fn tensor_cat_dim0_forward() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("a");
        let b = tape.leaf(vec![7.0, 8.0, 9.0], vec![1, 3], true).expect("b");
        let (y, _) = tape.cat(&[a, b], 0, ExecutionMode::Strict).expect("cat 0");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[3, 3]);
    }

    #[test]
    fn tensor_cat_dim1_forward() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("a");
        let b = tape
            .leaf(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3], true)
            .expect("b");
        let (y, _) = tape.cat(&[a, b], 1, ExecutionMode::Strict).expect("cat 1");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[2, 5]);
    }

    #[test]
    fn tensor_cat_backward() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0], vec![1, 2], true).expect("a");
        let b = tape.leaf(vec![3.0, 4.0, 5.0], vec![1, 3], true).expect("b");
        let (y, _) = tape.cat(&[a, b], 1, ExecutionMode::Strict).expect("cat 1");
        let report = tape.backward(y).expect("backward");
        let grads_a = report.gradient(a).expect("grad a");
        let grads_b = report.gradient(b).expect("grad b");
        // Backward of cat splits the gradient: all ones incoming
        assert_eq!(grads_a, &[1.0, 1.0]);
        assert_eq!(grads_b, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn tensor_stack_dim0_forward() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("a");
        let b = tape.leaf(vec![4.0, 5.0, 6.0], vec![3], true).expect("b");
        let (y, _) = tape
            .stack(&[a, b], 0, ExecutionMode::Strict)
            .expect("stack 0");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[2, 3]);
    }

    #[test]
    fn tensor_stack_dim1_forward() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("a");
        let b = tape
            .leaf(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true)
            .expect("b");
        let (y, _) = tape
            .stack(&[a, b], 1, ExecutionMode::Strict)
            .expect("stack 1");
        let vals = tape.values(y).expect("values");
        // shape [2,2,2]: [[[1,2],[5,6]],[[3,4],[7,8]]]
        assert_eq!(vals, &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[2, 2, 2]);
    }

    #[test]
    fn tensor_stack_backward() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("a");
        let b = tape.leaf(vec![4.0, 5.0, 6.0], vec![3], true).expect("b");
        let (y, _) = tape
            .stack(&[a, b], 0, ExecutionMode::Strict)
            .expect("stack 0");
        let report = tape.backward(y).expect("backward");
        let grads_a = report.gradient(a).expect("grad a");
        let grads_b = report.gradient(b).expect("grad b");
        // Backward of stack slices the gradient: all ones incoming
        assert_eq!(grads_a, &[1.0, 1.0, 1.0]);
        assert_eq!(grads_b, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn tensor_cat_three_inputs() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0], vec![1, 1], true).expect("a");
        let b = tape.leaf(vec![2.0], vec![1, 1], true).expect("b");
        let c = tape.leaf(vec![3.0], vec![1, 1], true).expect("c");
        let (y, _) = tape.cat(&[a, b, c], 1, ExecutionMode::Strict).expect("cat");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0]);
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[1, 3]);
    }

    // ---- reshape tests ----

    #[test]
    fn tensor_reshape_2d_to_1d() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let y = tape.reshape(x, vec![6]).expect("reshape");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[6]);
    }

    #[test]
    fn tensor_reshape_1d_to_2d() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], true)
            .expect("x");
        let y = tape.reshape(x, vec![2, 3]).expect("reshape");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[2, 3]);
    }

    #[test]
    fn tensor_reshape_preserves_data_3d() {
        let mut tape = TensorTape::new();
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let x = tape.leaf(data.clone(), vec![2, 3, 4], true).expect("x");
        let y = tape.reshape(x, vec![4, 6]).expect("reshape");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, data);
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[4, 6]);
    }

    #[test]
    fn tensor_reshape_mismatched_numel_fails() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("x");
        assert!(tape.reshape(x, vec![2]).is_err());
    }

    #[test]
    fn tensor_reshape_backward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let y = tape.reshape(x, vec![6]).expect("reshape");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    // ---- view tests ----

    #[test]
    fn tensor_view_same_as_reshape() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("x");
        let y = tape.view(x, vec![4]).expect("view");
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0]);
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[4]);
    }

    #[test]
    fn tensor_view_backward() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
            .expect("x");
        let y = tape.view(x, vec![2, 2]).expect("view");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 1.0, 1.0, 1.0]);
    }

    // ---- squeeze tests ----

    #[test]
    fn tensor_squeeze_removes_dim_of_size_1() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true).expect("x");
        let y = tape.squeeze(x, 0).expect("squeeze");
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[3]);
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn tensor_squeeze_noop_when_dim_not_1() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true).expect("x");
        let y = tape.squeeze(x, 1).expect("squeeze");
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[1, 3]);
    }

    #[test]
    fn tensor_squeeze_invalid_dim_fails() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        assert!(tape.squeeze(x, 5).is_err());
    }

    #[test]
    fn tensor_squeeze_middle_dim() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3], true)
            .expect("x");
        let y = tape.squeeze(x, 1).expect("squeeze");
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[2, 3]);
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn tensor_squeeze_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![1, 3], true).expect("x");
        let y = tape.squeeze(x, 0).expect("squeeze");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 1.0, 1.0]);
    }

    // ---- unsqueeze tests ----

    #[test]
    fn tensor_unsqueeze_adds_dim_at_start() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("x");
        let y = tape.unsqueeze(x, 0).expect("unsqueeze");
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[1, 3]);
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn tensor_unsqueeze_adds_dim_at_end() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("x");
        let y = tape.unsqueeze(x, 1).expect("unsqueeze");
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[3, 1]);
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn tensor_unsqueeze_invalid_dim_fails() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        assert!(tape.unsqueeze(x, 3).is_err());
    }

    #[test]
    fn tensor_unsqueeze_middle_dim() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("x");
        let y = tape.unsqueeze(x, 1).expect("unsqueeze");
        let shape = tape.node(y).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[2, 1, 3]);
        let vals = tape.values(y).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn tensor_unsqueeze_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("x");
        let y = tape.unsqueeze(x, 0).expect("unsqueeze");
        let report = tape.backward(y).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn tensor_reshape_then_squeeze_roundtrip() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("x");
        let y = tape.unsqueeze(x, 0).expect("unsqueeze");
        let z = tape.squeeze(y, 0).expect("squeeze");
        let shape = tape.node(z).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[3]);
        let vals = tape.values(z).expect("values");
        assert_eq!(vals, &[1.0, 2.0, 3.0]);
        let report = tape.backward(z).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn tensor_argmax_returns_correct_indices() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], false)
            .expect("x");
        let out = tape.argmax(x, 1).expect("argmax");
        let vals = tape.values(out).expect("values");
        assert_eq!(vals, &[1.0, 2.0]);
        let shape = tape.node(out).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[2]);
    }

    #[test]
    fn tensor_argmin_returns_correct_indices() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], false)
            .expect("x");
        let out = tape.argmin(x, 1).expect("argmin");
        let vals = tape.values(out).expect("values");
        assert_eq!(vals, &[0.0, 1.0]);
        let shape = tape.node(out).unwrap().tensor.meta().shape();
        assert_eq!(shape, &[2]);
    }

    #[test]
    fn tensor_argmax_not_requires_grad() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let out = tape.argmax(x, 1).expect("argmax");
        assert!(!tape.node(out).unwrap().requires_grad);
    }

    #[test]
    fn tensor_max_dim_returns_values_and_indices() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values_out, indices_out) = tape.max_dim(x, 1).expect("max_dim");
        let values = tape.values(values_out).expect("values");
        let indices = tape.values(indices_out).expect("indices");
        assert_eq!(values, &[5.0, 6.0]);
        assert_eq!(indices, &[1.0, 2.0]);
    }

    #[test]
    fn tensor_max_dim_backward_scatters_gradient() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values_out, _indices_out) = tape.max_dim(x, 1).expect("max_dim");
        let s = tape.sum(values_out, ExecutionMode::Strict).expect("sum");
        let report = tape.backward(s.0).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn tensor_min_dim_returns_values_and_indices() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values_out, indices_out) = tape.min_dim(x, 1).expect("min_dim");
        let values = tape.values(values_out).expect("values");
        let indices = tape.values(indices_out).expect("indices");
        assert_eq!(values, &[1.0, 2.0]);
        assert_eq!(indices, &[0.0, 1.0]);
    }

    #[test]
    fn tensor_min_dim_backward_scatters_gradient() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values_out, _indices_out) = tape.min_dim(x, 1).expect("min_dim");
        let s = tape.sum(values_out, ExecutionMode::Strict).expect("sum");
        let report = tape.backward(s.0).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn tensor_max_dim_along_dim0() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values_out, indices_out) = tape.max_dim(x, 0).expect("max_dim 0");
        let values = tape.values(values_out).expect("values");
        let indices = tape.values(indices_out).expect("indices");
        assert_eq!(values, &[4.0, 5.0, 6.0]);
        assert_eq!(indices, &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn tensor_max_dim_backward_along_dim0() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3], true)
            .expect("x");
        let (values_out, _indices_out) = tape.max_dim(x, 0).expect("max_dim 0");
        let s = tape.sum(values_out, ExecutionMode::Strict).expect("sum");
        let report = tape.backward(s.0).expect("backward");
        let grads = report.gradient(x).expect("grad x");
        assert_eq!(grads, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    // ---- Error path tests for unknown TensorNodeId (bd-3rai) ----

    #[test]
    fn tensor_values_rejects_unknown_node_id() {
        let tape = TensorTape::new();
        let bogus = TensorNodeId(9999);
        let err = tape
            .values(bogus)
            .expect_err("values with unknown node id must fail");
        assert!(matches!(err, AutogradError::UnknownTensorNode(id) if id == bogus));
    }

    #[test]
    fn tensor_accessor_rejects_unknown_node_id() {
        let tape = TensorTape::new();
        let bogus = TensorNodeId(42);
        let err = tape
            .tensor(bogus)
            .expect_err("tensor with unknown node id must fail");
        assert!(matches!(err, AutogradError::UnknownTensorNode(id) if id == bogus));
    }

    #[test]
    fn tensor_meta_rejects_unknown_node_id() {
        let tape = TensorTape::new();
        let bogus = TensorNodeId(100);
        let err = tape
            .tensor_meta(bogus)
            .expect_err("tensor_meta with unknown node id must fail");
        assert!(matches!(err, AutogradError::UnknownTensorNode(id) if id == bogus));
    }

    #[test]
    fn backward_rejects_unknown_root_node_id() {
        let tape = TensorTape::new();
        let bogus = TensorNodeId(999);
        let err = tape
            .backward(bogus)
            .expect_err("backward with unknown root must fail");
        assert!(matches!(err, AutogradError::UnknownTensorNode(id) if id == bogus));
    }

    #[test]
    fn update_tensor_values_rejects_unknown_node_id() {
        let mut tape = TensorTape::new();
        let bogus = TensorNodeId(50);
        let err = tape
            .update_tensor_values(bogus, vec![1.0, 2.0])
            .expect_err("update_tensor_values with unknown node must fail");
        assert!(matches!(err, AutogradError::UnknownTensorNode(id) if id == bogus));
    }
}
