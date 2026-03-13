#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::sync::Arc;

use ft_core::{
    DType, DenseTensor, DenseTensorError, Device, ExecutionMode, ScalarTensor, TensorMeta,
    TensorStorage,
};
use ft_dispatch::{
    AddmmDispatchDecision, BinaryOp, ClampDispatchDecision, DispatchDecision, DispatchError,
    DispatchKeyError, JoinDispatchDecision, JoinOp, LerpDispatchDecision, NormDispatchDecision,
    NormalizeDimDispatchDecision, NormalizeOp, PowDispatchDecision, ReductionDimDispatchDecision,
    ReductionDispatchDecision, ReductionOp, ScanDimDispatchDecision, ScanOp, SortDispatchDecision,
    TopKDispatchDecision, UnaryDispatchDecision, UnaryOp, dispatch_scalar_binary,
    dispatch_scalar_clamp, dispatch_scalar_pow, dispatch_scalar_unary,
    dispatch_tensor_addmm_contiguous_typed, dispatch_tensor_addmv_contiguous_typed,
    dispatch_tensor_binary_contiguous_typed, dispatch_tensor_clamp_contiguous_typed,
    dispatch_tensor_join_contiguous_typed, dispatch_tensor_lerp_contiguous_typed,
    dispatch_tensor_norm_contiguous_typed, dispatch_tensor_norm_dim_contiguous_typed,
    dispatch_tensor_normalize_dim_contiguous_typed, dispatch_tensor_pow_contiguous_typed,
    dispatch_tensor_reduction_contiguous_typed, dispatch_tensor_reduction_dim_contiguous_typed,
    dispatch_tensor_scan_dim_contiguous_typed, dispatch_tensor_sort_contiguous_typed,
    dispatch_tensor_topk_contiguous_typed, dispatch_tensor_unary_contiguous_typed,
};
use ft_kernel_cpu::{
    argmax_dim_tensor_contiguous_f64, argmin_dim_tensor_contiguous_f64,
    gather_tensor_contiguous_f64, index_select_tensor_contiguous_f64,
    index_put_tensor_contiguous_f64, masked_fill_tensor_contiguous_f64,
    max_dim_tensor_contiguous_f64, min_dim_tensor_contiguous_f64,
    scatter_add_tensor_contiguous_f64, scatter_tensor_contiguous_f64,
    where_tensor_contiguous_f64,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorNodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorHookHandle {
    node: TensorNodeId,
    id: u64,
}

impl TensorHookHandle {
    #[must_use]
    pub const fn node(self) -> TensorNodeId {
        self.node
    }

    #[must_use]
    pub const fn id(self) -> u64 {
        self.id
    }
}

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
    Rsqrt {
        input: NodeId,
    },
    Erf {
        input: NodeId,
    },
    Erfc {
        input: NodeId,
    },
    Hardswish {
        input: NodeId,
    },
    Hardsigmoid {
        input: NodeId,
    },
    Hardtanh {
        input: NodeId,
    },
    Softplus {
        input: NodeId,
    },
    Mish {
        input: NodeId,
    },
    Square {
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
    Atan2 {
        lhs: NodeId,
        rhs: NodeId,
    },
    Fmod {
        lhs: NodeId,
        rhs: NodeId,
    },
    Remainder {
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
    Rsqrt {
        input: TensorNodeId,
    },
    Erf {
        input: TensorNodeId,
    },
    Erfc {
        input: TensorNodeId,
    },
    Hardswish {
        input: TensorNodeId,
    },
    Hardsigmoid {
        input: TensorNodeId,
    },
    Hardtanh {
        input: TensorNodeId,
    },
    Softplus {
        input: TensorNodeId,
    },
    Mish {
        input: TensorNodeId,
    },
    Square {
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
    Atan2 {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Fmod {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Remainder {
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
    Norm {
        input: TensorNodeId,
        p: f64,
        input_numel: usize,
    },
    NormDim {
        input: TensorNodeId,
        p: f64,
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
    View {
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
    ScatterAdd {
        input: TensorNodeId,
        dim: usize,
        index: Vec<f64>,
        index_shape: Vec<usize>,
        input_shape: Vec<usize>,
    },
    IndexPut {
        input: TensorNodeId,
        indices: Vec<Vec<f64>>,
        input_shape: Vec<usize>,
        accumulate: bool,
        suffix_size: usize,
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
    Pad {
        input: TensorNodeId,
        padding: Vec<usize>,
        original_shape: Vec<usize>,
    },
    Lerp {
        start: TensorNodeId,
        end: TensorNodeId,
        weight: f64,
    },
    Addmm {
        input: TensorNodeId,
        mat1: TensorNodeId,
        mat2: TensorNodeId,
        beta: f64,
        alpha: f64,
    },
    Addmv {
        input: TensorNodeId,
        mat: TensorNodeId,
        vec: TensorNodeId,
        beta: f64,
        alpha: f64,
    },
    CastF32 {
        input: TensorNodeId,
    },
    CastF64 {
        input: TensorNodeId,
    },
    CustomFunction {
        inputs: Vec<TensorNodeId>,
        function_id: usize,
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
    pub retain_graph: bool,
    pub create_graph: bool,
}

impl BackwardOptions {
    #[must_use]
    pub const fn strict_default() -> Self {
        Self {
            max_reentrant_depth: 0,
            current_reentrant_depth: 0,
            policy: ReentrantPolicy::StrictFail,
            retain_graph: false,
            create_graph: false,
        }
    }

    #[must_use]
    pub const fn hardened_default() -> Self {
        Self {
            max_reentrant_depth: 2,
            current_reentrant_depth: 0,
            policy: ReentrantPolicy::HardenedBoundedFallback,
            retain_graph: false,
            create_graph: false,
        }
    }

    #[must_use]
    pub const fn with_retain_graph(mut self, retain: bool) -> Self {
        self.retain_graph = retain;
        self
    }

    #[must_use]
    pub const fn with_create_graph(mut self, create_graph: bool) -> Self {
        self.create_graph = create_graph;
        self
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
pub struct TensorNormOperationEvent {
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub p: f64,
    pub decision: NormDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorNormDimOperationEvent {
    pub input: TensorNodeId,
    pub out: TensorNodeId,
    pub p: f64,
    pub dim: usize,
    pub decision: NormDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorLerpOperationEvent {
    pub start: TensorNodeId,
    pub end: TensorNodeId,
    pub out: TensorNodeId,
    pub weight: f64,
    pub decision: LerpDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorAddmmOperationEvent {
    pub input: TensorNodeId,
    pub mat1: TensorNodeId,
    pub mat2: TensorNodeId,
    pub out: TensorNodeId,
    pub beta: f64,
    pub alpha: f64,
    pub decision: AddmmDispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorAddmvOperationEvent {
    pub input: TensorNodeId,
    pub mat: TensorNodeId,
    pub vec: TensorNodeId,
    pub out: TensorNodeId,
    pub beta: f64,
    pub alpha: f64,
    pub decision: AddmmDispatchDecision,
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
    gradient_nodes: Vec<Option<TensorNodeId>>,
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

    /// When `create_graph=True` was used, returns the gradient as a tensor
    /// node ID that can itself be differentiated.
    #[must_use]
    pub fn gradient_node(&self, node: TensorNodeId) -> Option<TensorNodeId> {
        self.gradient_nodes
            .get(node.0)
            .and_then(|entry| *entry)
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
    TensorRequiresGradNonLeaf {
        node: TensorNodeId,
    },
    TensorRequiresGradNonFloating {
        node: TensorNodeId,
        dtype: DType,
    },
    TensorMatMulShapeMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    GraphConsumed,
    TensorGraphConsumed,
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
            Self::TensorRequiresGradNonLeaf { node } => write!(
                f,
                "requires_grad_ can only be called on leaf tensors (node {})",
                node.0
            ),
            Self::TensorRequiresGradNonFloating { node, dtype } => write!(
                f,
                "requires_grad_ expects floating-point dtype at node {} but found {dtype:?}",
                node.0
            ),
            Self::TensorMatMulShapeMismatch { lhs, rhs } => {
                write!(f, "tensor matmul shape mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
            Self::GraphConsumed => {
                write!(
                    f,
                    "cannot run backward: graph already consumed by a previous backward pass"
                )
            }
            Self::TensorGraphConsumed => {
                write!(
                    f,
                    "cannot run tensor backward: graph already consumed by a previous backward pass"
                )
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

type TensorGradHook =
    dyn Fn(&[f64]) -> Result<Option<Vec<f64>>, AutogradError> + Send + Sync + 'static;

#[derive(Clone)]
struct TensorHookRegistration {
    id: u64,
    callback: Arc<TensorGradHook>,
}

impl fmt::Debug for TensorHookRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorHookRegistration")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// Context passed to custom autograd functions for saving tensors during forward
/// and retrieving them during backward.
#[derive(Debug, Clone)]
pub struct FunctionCtx {
    saved_tensors: Vec<Vec<f64>>,
    saved_shapes: Vec<Vec<usize>>,
    needs_input_grad: Vec<bool>,
}

impl FunctionCtx {
    fn new(needs_input_grad: Vec<bool>) -> Self {
        Self {
            saved_tensors: Vec::new(),
            saved_shapes: Vec::new(),
            needs_input_grad,
        }
    }

    /// Save tensor data for use in the backward pass.
    pub fn save_for_backward(&mut self, values: Vec<f64>, shape: Vec<usize>) {
        self.saved_tensors.push(values);
        self.saved_shapes.push(shape);
    }

    /// Retrieve saved tensors during backward.
    #[must_use]
    pub fn saved_tensors(&self) -> &[Vec<f64>] {
        &self.saved_tensors
    }

    /// Retrieve saved tensor shapes during backward.
    #[must_use]
    pub fn saved_shapes(&self) -> &[Vec<usize>] {
        &self.saved_shapes
    }

    /// Check which inputs require gradient computation.
    #[must_use]
    pub fn needs_input_grad(&self) -> &[bool] {
        &self.needs_input_grad
    }
}

type AutogradFunctionBackward = dyn Fn(&FunctionCtx, &[&[f64]]) -> Result<Vec<Option<Vec<f64>>>, AutogradError>
    + Send
    + Sync
    + 'static;

#[derive(Clone)]
struct CustomFunctionRecord {
    ctx: FunctionCtx,
    backward_fn: Arc<AutogradFunctionBackward>,
    input_numel: Vec<usize>,
}

impl fmt::Debug for CustomFunctionRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomFunctionRecord")
            .field("ctx", &self.ctx)
            .field("input_numel", &self.input_numel)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct Tape {
    nodes: Vec<Node>,
    consumed: bool,
    consumed_boundary: usize,
    grad_enabled: bool,
}

impl Default for Tape {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            consumed: false,
            consumed_boundary: 0,
            grad_enabled: true,
        }
    }
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

    #[must_use]
    pub fn is_grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    pub fn set_grad_enabled(&mut self, enabled: bool) {
        self.grad_enabled = enabled;
    }

    pub fn leaf(&mut self, value: f64, requires_grad: bool) -> NodeId {
        let effective_requires_grad = requires_grad && self.grad_enabled;
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: ScalarTensor::new(value, DType::F64, Device::Cpu),
            requires_grad: effective_requires_grad,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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

    pub fn rsqrt(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Rsqrt, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Rsqrt { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Rsqrt,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn erf(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Erf, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Erf { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Erf,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn erfc(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Erfc, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Erfc { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Erfc,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn hardswish(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Hardswish, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Hardswish { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Hardswish,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn hardsigmoid(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let outcome = dispatch_scalar_unary(
                UnaryOp::Hardsigmoid,
                mode,
                &input_node.tensor,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Hardsigmoid { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Hardsigmoid,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn hardtanh(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Hardtanh, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Hardtanh { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Hardtanh,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn softplus(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Softplus, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Softplus { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Softplus,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn mish(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Mish, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Mish { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Mish,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn square(
        &mut self,
        input: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, UnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let outcome =
                dispatch_scalar_unary(UnaryOp::Square, mode, &input_node.tensor, requires_grad)
                    .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let out = NodeId(self.nodes.len());
        self.nodes.push(Node {
            tensor: outcome.tensor,
            requires_grad,
            op: NodeOp::Square { input },
        });
        Ok((
            out,
            UnaryOperationEvent {
                op: UnaryOp::Square,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
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
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
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

    pub fn atan2(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
            let outcome = dispatch_scalar_binary(
                BinaryOp::Atan2,
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
            op: NodeOp::Atan2 { lhs, rhs },
        });

        Ok((
            out,
            OperationEvent {
                op: BinaryOp::Atan2,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn fmod(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
            let outcome = dispatch_scalar_binary(
                BinaryOp::Fmod,
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
            op: NodeOp::Fmod { lhs, rhs },
        });

        Ok((
            out,
            OperationEvent {
                op: BinaryOp::Fmod,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn remainder(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
            let outcome = dispatch_scalar_binary(
                BinaryOp::Remainder,
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
            op: NodeOp::Remainder { lhs, rhs },
        });

        Ok((
            out,
            OperationEvent {
                op: BinaryOp::Remainder,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
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
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
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
                BinaryOp::Atan2 => NodeOp::Atan2 { lhs, rhs },
                BinaryOp::Fmod => NodeOp::Fmod { lhs, rhs },
                BinaryOp::Remainder => NodeOp::Remainder { lhs, rhs },
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

    pub fn backward(&mut self, root: NodeId) -> Result<BackwardReport, AutogradError> {
        self.backward_with_options(root, BackwardOptions::strict_default())
    }

    pub fn backward_with_options(
        &mut self,
        root: NodeId,
        options: BackwardOptions,
    ) -> Result<BackwardReport, AutogradError> {
        if self.consumed && root.0 < self.consumed_boundary {
            return Err(AutogradError::GraphConsumed);
        }
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
                NodeOp::Rsqrt { input } => {
                    let output_value = self.nodes[node_id.0].tensor.value();
                    // d/dx rsqrt(x) = -0.5 * x^(-3/2) = -0.5 * rsqrt(x)^3
                    grads[input.0] +=
                        incoming * (-0.5 * output_value * output_value * output_value);

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(rsqrt(x))/dx=-0.5*rsqrt(x)^3",
                    });
                }
                NodeOp::Erf { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    // d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2)
                    let coeff = 2.0 / std::f64::consts::PI.sqrt();
                    grads[input.0] += incoming * coeff * (-x * x).exp();

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(erf(x))/dx=(2/sqrt(pi))*exp(-x^2)",
                    });
                }
                NodeOp::Erfc { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    // d/dx erfc(x) = -(2/sqrt(pi)) * exp(-x^2)
                    let coeff = 2.0 / std::f64::consts::PI.sqrt();
                    grads[input.0] += incoming * (-coeff) * (-x * x).exp();

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(erfc(x))/dx=-(2/sqrt(pi))*exp(-x^2)",
                    });
                }
                NodeOp::Hardswish { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    let grad = if x <= -3.0 {
                        0.0
                    } else if x >= 3.0 {
                        1.0
                    } else {
                        (2.0 * x + 3.0) / 6.0
                    };
                    grads[input.0] += incoming * grad;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(hardswish(x))/dx=(2x+3)/6|0|1",
                    });
                }
                NodeOp::Hardsigmoid { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    let grad = if x <= -3.0 || x >= 3.0 {
                        0.0
                    } else {
                        1.0 / 6.0
                    };
                    grads[input.0] += incoming * grad;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(hardsigmoid(x))/dx=1/6|0",
                    });
                }
                NodeOp::Hardtanh { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    let grad = if (-1.0..=1.0).contains(&x) { 1.0 } else { 0.0 };
                    grads[input.0] += incoming * grad;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(hardtanh(x))/dx=1|0",
                    });
                }
                NodeOp::Softplus { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    // d/dx softplus(x) = sigmoid(x) = 1/(1+exp(-x))
                    let grad = 1.0 / (1.0 + (-x).exp());
                    grads[input.0] += incoming * grad;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(softplus(x))/dx=sigmoid(x)",
                    });
                }
                NodeOp::Mish { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    // d/dx mish(x) = tanh(sp) + x * sig * (1 - tanh(sp)^2)
                    // where sp = softplus(x), sig = sigmoid(x)
                    let sp = if x > 20.0 {
                        x
                    } else if x < -20.0 {
                        0.0
                    } else {
                        (1.0 + x.exp()).ln()
                    };
                    let tsp = sp.tanh();
                    let sig = 1.0 / (1.0 + (-x).exp());
                    let grad = tsp + x * sig * (1.0 - tsp * tsp);
                    grads[input.0] += incoming * grad;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(mish(x))/dx=tanh(sp)+x*sig*(1-tanh(sp)^2)",
                    });
                }
                NodeOp::Square { input } => {
                    let x = self.nodes[input.0].tensor.value();
                    grads[input.0] += incoming * 2.0 * x;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(x^2)/dx=2x",
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
                NodeOp::Atan2 { lhs, rhs } => {
                    let y = self.nodes[lhs.0].tensor.value();
                    let x = self.nodes[rhs.0].tensor.value();
                    let denom = x * x + y * y;
                    grads[lhs.0] += incoming * x / denom;
                    grads[rhs.0] += incoming * (-y) / denom;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(atan2(y,x))/dy=x/(x^2+y^2); dx=-y/(x^2+y^2)",
                    });
                }
                NodeOp::Fmod { lhs, rhs } => {
                    let a = self.nodes[lhs.0].tensor.value();
                    let b = self.nodes[rhs.0].tensor.value();
                    grads[lhs.0] += incoming;
                    grads[rhs.0] += incoming * (-(a / b).trunc());

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(fmod(a,b))/da=1; db=-trunc(a/b)",
                    });
                }
                NodeOp::Remainder { lhs, rhs } => {
                    let a = self.nodes[lhs.0].tensor.value();
                    let b = self.nodes[rhs.0].tensor.value();
                    grads[lhs.0] += incoming;
                    grads[rhs.0] += incoming * (-(a / b).floor());

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(BackwardStep {
                        node: node_id,
                        incoming_grad: incoming,
                        rule: "d(remainder(a,b))/da=1; db=-floor(a/b)",
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

        if !options.retain_graph {
            self.consumed = true;
            self.consumed_boundary = self.nodes.len();
        }

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
                | NodeOp::Max { lhs, rhs }
                | NodeOp::Atan2 { lhs, rhs }
                | NodeOp::Fmod { lhs, rhs }
                | NodeOp::Remainder { lhs, rhs } => {
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
                | NodeOp::Rsqrt { input }
                | NodeOp::Erf { input }
                | NodeOp::Erfc { input }
                | NodeOp::Hardswish { input }
                | NodeOp::Hardsigmoid { input }
                | NodeOp::Hardtanh { input }
                | NodeOp::Softplus { input }
                | NodeOp::Mish { input }
                | NodeOp::Square { input }
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
                | NodeOp::Max { lhs, rhs }
                | NodeOp::Atan2 { lhs, rhs }
                | NodeOp::Fmod { lhs, rhs }
                | NodeOp::Remainder { lhs, rhs } => {
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
                | NodeOp::Rsqrt { input }
                | NodeOp::Erf { input }
                | NodeOp::Erfc { input }
                | NodeOp::Hardswish { input }
                | NodeOp::Hardsigmoid { input }
                | NodeOp::Hardtanh { input }
                | NodeOp::Softplus { input }
                | NodeOp::Mish { input }
                | NodeOp::Square { input }
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

#[derive(Debug, Clone)]
pub struct TensorTape {
    nodes: Vec<TensorNode>,
    persistent_grads: BTreeMap<usize, Vec<f64>>,
    tensor_hooks: BTreeMap<usize, Vec<TensorHookRegistration>>,
    next_tensor_hook_id: u64,
    consumed: bool,
    consumed_boundary: usize,
    grad_enabled: bool,
    custom_functions: BTreeMap<usize, CustomFunctionRecord>,
    next_custom_function_id: usize,
}

impl Default for TensorTape {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            persistent_grads: BTreeMap::new(),
            tensor_hooks: BTreeMap::new(),
            next_tensor_hook_id: 1,
            consumed: false,
            consumed_boundary: 0,
            grad_enabled: true,
            custom_functions: BTreeMap::new(),
            next_custom_function_id: 0,
        }
    }
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

    #[must_use]
    pub fn is_grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    pub fn set_grad_enabled(&mut self, enabled: bool) {
        self.grad_enabled = enabled;
    }

    pub fn tensor_requires_grad(&self, id: TensorNodeId) -> Result<bool, AutogradError> {
        Ok(self.node(id)?.requires_grad)
    }

    pub fn tensor_is_leaf(&self, id: TensorNodeId) -> Result<bool, AutogradError> {
        Ok(matches!(self.node(id)?.op, TensorNodeOp::Leaf))
    }

    pub fn tensor_grad_fn(&self, id: TensorNodeId) -> Result<Option<String>, AutogradError> {
        let node = self.node(id)?;
        if matches!(node.op, TensorNodeOp::Leaf) {
            return Ok(None);
        }
        let debug = format!("{:?}", node.op);
        let label = debug
            .split([' ', '{'])
            .next()
            .unwrap_or(debug.as_str())
            .to_string();
        Ok(Some(label))
    }

    pub fn set_tensor_requires_grad(
        &mut self,
        id: TensorNodeId,
        requires_grad: bool,
    ) -> Result<(), AutogradError> {
        let node = self.node_mut(id)?;
        if !matches!(node.op, TensorNodeOp::Leaf) {
            return Err(AutogradError::TensorRequiresGradNonLeaf { node: id });
        }
        if requires_grad && !node.tensor.meta().dtype().is_floating_point() {
            return Err(AutogradError::TensorRequiresGradNonFloating {
                node: id,
                dtype: node.tensor.meta().dtype(),
            });
        }
        node.requires_grad = requires_grad;
        if !requires_grad {
            self.persistent_grads.remove(&id.0);
        }
        Ok(())
    }

    pub fn detach_tensor_in_place(&mut self, id: TensorNodeId) -> Result<(), AutogradError> {
        let node = self.node_mut(id)?;
        node.requires_grad = false;
        node.op = TensorNodeOp::Leaf;
        self.persistent_grads.remove(&id.0);
        Ok(())
    }

    pub fn register_tensor_hook<F>(
        &mut self,
        id: TensorNodeId,
        hook: F,
    ) -> Result<TensorHookHandle, AutogradError>
    where
        F: Fn(&[f64]) -> Result<Option<Vec<f64>>, AutogradError> + Send + Sync + 'static,
    {
        self.node(id)?;
        let hook_id = self.next_tensor_hook_id;
        self.next_tensor_hook_id = self.next_tensor_hook_id.wrapping_add(1);
        self.tensor_hooks
            .entry(id.0)
            .or_default()
            .push(TensorHookRegistration {
                id: hook_id,
                callback: Arc::new(hook),
            });
        Ok(TensorHookHandle {
            node: id,
            id: hook_id,
        })
    }

    pub fn remove_tensor_hook(&mut self, handle: TensorHookHandle) -> Result<bool, AutogradError> {
        self.node(handle.node)?;
        let mut removed = false;
        let mut clear_bucket = false;
        if let Some(entries) = self.tensor_hooks.get_mut(&handle.node.0) {
            let before = entries.len();
            entries.retain(|entry| entry.id != handle.id);
            removed = entries.len() != before;
            clear_bucket = entries.is_empty();
        }
        if clear_bucket {
            self.tensor_hooks.remove(&handle.node.0);
        }
        Ok(removed)
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
        let effective_requires_grad = requires_grad && self.grad_enabled;
        let id = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor,
            requires_grad: effective_requires_grad,
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

    pub fn tensor_accumulated_gradient(
        &self,
        node: TensorNodeId,
    ) -> Result<Option<&[f64]>, AutogradError> {
        self.node(node)?;
        Ok(self.persistent_grads.get(&node.0).map(Vec::as_slice))
    }

    pub fn tensor_accumulated_gradient_values(
        &self,
        node: TensorNodeId,
    ) -> Result<Option<Vec<f64>>, AutogradError> {
        Ok(self
            .tensor_accumulated_gradient(node)?
            .map(|gradient| gradient.to_vec()))
    }

    pub fn zero_tensor_accumulated_gradient(
        &mut self,
        node: TensorNodeId,
    ) -> Result<(), AutogradError> {
        self.node(node)?;
        if let Some(grad) = self.persistent_grads.get_mut(&node.0) {
            grad.fill(0.0);
        }
        Ok(())
    }

    pub fn set_tensor_accumulated_gradient(
        &mut self,
        node: TensorNodeId,
        gradient: Vec<f64>,
    ) -> Result<(), AutogradError> {
        let expected = self.node(node)?.tensor.meta().numel();
        Self::ensure_tensor_len(node, expected, gradient.len())?;
        self.persistent_grads.insert(node.0, gradient);
        Ok(())
    }

    pub fn leaf_f32(
        &mut self,
        values: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let tensor = DenseTensor::from_contiguous_values_f32(values, shape, Device::Cpu)?;
        Ok(self.leaf_tensor(tensor, requires_grad))
    }

    pub fn values_f32(&self, node: TensorNodeId) -> Result<Vec<f32>, AutogradError> {
        Ok(self.node(node)?.tensor.contiguous_values_f32()?.to_vec())
    }

    pub fn dtype(&self, node: TensorNodeId) -> Result<DType, AutogradError> {
        Ok(self.node(node)?.tensor.meta().dtype())
    }

    pub fn to_f32(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let input_node = self.node(input)?;
        let input_dtype = input_node.tensor.meta().dtype();
        if input_dtype == DType::F32 {
            return Ok(input);
        }
        if input_dtype != DType::F64 {
            return Err(AutogradError::DenseTensor(
                ft_core::DenseTensorError::UnsupportedDType(input_dtype),
            ));
        }
        let requires_grad = input_node.requires_grad && self.grad_enabled;
        let meta = input_node.tensor.meta().clone();
        let f64_values = input_node.tensor.contiguous_values()?;
        let f32_values: Vec<f32> = f64_values.iter().map(|&v| v as f32).collect();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                TensorMeta::from_shape(meta.shape().to_vec(), DType::F32, meta.device()),
                TensorStorage::F32(Arc::new(f32_values)),
            )?,
            requires_grad,
            op: TensorNodeOp::CastF32 { input },
        });

        Ok(out)
    }

    pub fn to_f64(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let input_node = self.node(input)?;
        let input_dtype = input_node.tensor.meta().dtype();
        if input_dtype == DType::F64 {
            return Ok(input);
        }
        if input_dtype != DType::F32 {
            return Err(AutogradError::DenseTensor(
                ft_core::DenseTensorError::UnsupportedDType(input_dtype),
            ));
        }
        let requires_grad = input_node.requires_grad && self.grad_enabled;
        let meta = input_node.tensor.meta().clone();
        let f32_values = input_node.tensor.contiguous_values_f32()?;
        let f64_values: Vec<f64> = f32_values.iter().map(|&v| v as f64).collect();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                TensorMeta::from_shape(meta.shape().to_vec(), DType::F64, meta.device()),
                TensorStorage::F64(Arc::new(f64_values)),
            )?,
            requires_grad,
            op: TensorNodeOp::CastF64 { input },
        });

        Ok(out)
    }

    /// Cast a tensor to the given dtype. Currently supports F32↔F64 casts.
    /// Returns the input unchanged if already the target dtype.
    /// Returns an error for non-floating-point target dtypes.
    pub fn to_dtype(
        &mut self,
        input: TensorNodeId,
        dtype: DType,
    ) -> Result<TensorNodeId, AutogradError> {
        match dtype {
            DType::F32 => self.to_f32(input),
            DType::F64 => self.to_f64(input),
            other => Err(AutogradError::DenseTensor(
                ft_core::DenseTensorError::UnsupportedDType(other),
            )),
        }
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
        let (requires_grad, input_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_contiguous_typed(
                ReductionOp::Trace,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(vec![1], output_dtype, output_device),
                outcome.storage,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Neg,
                mode,
                input_node.tensor.typed_storage(),
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
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Abs,
                mode,
                input_node.tensor.typed_storage(),
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
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Exp,
                mode,
                input_node.tensor.typed_storage(),
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
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Log,
                mode,
                input_node.tensor.typed_storage(),
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
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Relu,
                mode,
                input_node.tensor.typed_storage(),
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
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Sigmoid,
                mode,
                input_node.tensor.typed_storage(),
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
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Tanh,
                mode,
                input_node.tensor.typed_storage(),
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
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Sin,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Cos,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Tan,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Floor,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Ceil,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Round,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };

        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Log2,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Log10,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Log1p,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Expm1,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Sign,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Trunc,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Frac,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Asin,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Acos,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Atan,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Sinh,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Cosh,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Gelu,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Silu,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::LeakyRelu,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Elu,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
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

    pub fn rsqrt(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Rsqrt,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
            requires_grad,
            op: TensorNodeOp::Rsqrt { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Rsqrt,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn erf(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Erf,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
            requires_grad,
            op: TensorNodeOp::Erf { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Erf,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn erfc(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Erfc,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
            requires_grad,
            op: TensorNodeOp::Erfc { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Erfc,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn hardswish(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Hardswish,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
            requires_grad,
            op: TensorNodeOp::Hardswish { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Hardswish,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn hardsigmoid(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Hardsigmoid,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
            requires_grad,
            op: TensorNodeOp::Hardsigmoid { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Hardsigmoid,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn hardtanh(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Hardtanh,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
            requires_grad,
            op: TensorNodeOp::Hardtanh { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Hardtanh,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn softplus(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Softplus,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
            requires_grad,
            op: TensorNodeOp::Softplus { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Softplus,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn mish(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Mish,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
            requires_grad,
            op: TensorNodeOp::Mish { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Mish,
                input,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn square(
        &mut self,
        input: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorUnaryOperationEvent), AutogradError> {
        let (requires_grad, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Square,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, outcome)
        };
        let input_meta = self.nodes[input.0].tensor.meta().clone();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(input_meta, outcome.storage)?,
            requires_grad,
            op: TensorNodeOp::Square { input },
        });
        Ok((
            out,
            TensorUnaryOperationEvent {
                op: UnaryOp::Square,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Sqrt,
                mode,
                input_node.tensor.typed_storage(),
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
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_unary_contiguous_typed(
                UnaryOp::Reciprocal,
                mode,
                input_node.tensor.typed_storage(),
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
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_pow_contiguous_typed(
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                exponent,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
            let meta_l = lhs_node.tensor.meta().clone();
            let meta_r = rhs_node.tensor.meta().clone();
            let outcome = dispatch_tensor_binary_contiguous_typed(
                BinaryOp::Min,
                mode,
                lhs_node.tensor.typed_storage(),
                rhs_node.tensor.typed_storage(),
                &meta_l,
                &meta_r,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let result_dtype = outcome.storage.dtype();
            (
                requires_grad,
                meta_l.shape().to_vec(),
                result_dtype,
                meta_l.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
            let meta_l = lhs_node.tensor.meta().clone();
            let meta_r = rhs_node.tensor.meta().clone();
            let outcome = dispatch_tensor_binary_contiguous_typed(
                BinaryOp::Max,
                mode,
                lhs_node.tensor.typed_storage(),
                rhs_node.tensor.typed_storage(),
                &meta_l,
                &meta_r,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let result_dtype = outcome.storage.dtype();
            (
                requires_grad,
                meta_l.shape().to_vec(),
                result_dtype,
                meta_l.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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

    pub fn tensor_atan2(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
            let meta_l = lhs_node.tensor.meta().clone();
            let meta_r = rhs_node.tensor.meta().clone();
            let outcome = dispatch_tensor_binary_contiguous_typed(
                BinaryOp::Atan2,
                mode,
                lhs_node.tensor.typed_storage(),
                rhs_node.tensor.typed_storage(),
                &meta_l,
                &meta_r,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let result_dtype = outcome.storage.dtype();
            (
                requires_grad,
                meta_l.shape().to_vec(),
                result_dtype,
                meta_l.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
            )?,
            requires_grad,
            op: TensorNodeOp::Atan2 { lhs, rhs },
        });

        Ok((
            out,
            TensorOperationEvent {
                op: BinaryOp::Atan2,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn tensor_fmod(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
            let meta_l = lhs_node.tensor.meta().clone();
            let meta_r = rhs_node.tensor.meta().clone();
            let outcome = dispatch_tensor_binary_contiguous_typed(
                BinaryOp::Fmod,
                mode,
                lhs_node.tensor.typed_storage(),
                rhs_node.tensor.typed_storage(),
                &meta_l,
                &meta_r,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let result_dtype = outcome.storage.dtype();
            (
                requires_grad,
                meta_l.shape().to_vec(),
                result_dtype,
                meta_l.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
            )?,
            requires_grad,
            op: TensorNodeOp::Fmod { lhs, rhs },
        });

        Ok((
            out,
            TensorOperationEvent {
                op: BinaryOp::Fmod,
                lhs,
                rhs,
                out,
                decision: outcome.decision,
            },
        ))
    }

    pub fn tensor_remainder(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_dtype, output_device, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
            let meta_l = lhs_node.tensor.meta().clone();
            let meta_r = rhs_node.tensor.meta().clone();
            let outcome = dispatch_tensor_binary_contiguous_typed(
                BinaryOp::Remainder,
                mode,
                lhs_node.tensor.typed_storage(),
                rhs_node.tensor.typed_storage(),
                &meta_l,
                &meta_r,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let result_dtype = outcome.storage.dtype();
            (
                requires_grad,
                meta_l.shape().to_vec(),
                result_dtype,
                meta_l.device(),
                outcome,
            )
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
            )?,
            requires_grad,
            op: TensorNodeOp::Remainder { lhs, rhs },
        });

        Ok((
            out,
            TensorOperationEvent {
                op: BinaryOp::Remainder,
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
        let (requires_grad, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_clamp_contiguous_typed(
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                min_val,
                max_val,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, input_numel, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_contiguous_typed(
                ReductionOp::Sum,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.numel(),
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(vec![1], output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, input_numel, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_contiguous_typed(
                ReductionOp::Mean,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.numel(),
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(vec![1], output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, input_shape, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_typed(
                ReductionOp::Sum,
                mode,
                input_node.tensor.typed_storage(),
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
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, input_shape, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_typed(
                ReductionOp::Mean,
                mode,
                input_node.tensor.typed_storage(),
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
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, input_shape, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_typed(
                ReductionOp::Prod,
                mode,
                input_node.tensor.typed_storage(),
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
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, input_shape, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_typed(
                ReductionOp::Var,
                mode,
                input_node.tensor.typed_storage(),
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
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, input_shape, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_reduction_dim_contiguous_typed(
                ReductionOp::Std,
                mode,
                input_node.tensor.typed_storage(),
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
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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

    pub fn norm(
        &mut self,
        input: TensorNodeId,
        p: f64,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorNormOperationEvent), AutogradError> {
        let (requires_grad, input_numel, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_norm_contiguous_typed(
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                p,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.numel(),
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(vec![1], output_dtype, output_device),
                outcome.storage,
            )?,
            requires_grad,
            op: TensorNodeOp::Norm {
                input,
                p,
                input_numel,
            },
        });

        Ok((
            out,
            TensorNormOperationEvent {
                input,
                out,
                p,
                decision: outcome.decision,
            },
        ))
    }

    pub fn norm_dim(
        &mut self,
        input: TensorNodeId,
        p: f64,
        dim: usize,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorNormDimOperationEvent), AutogradError> {
        let (requires_grad, input_shape, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_norm_dim_contiguous_typed(
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                p,
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
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
            )?,
            requires_grad,
            op: TensorNodeOp::NormDim {
                input,
                p,
                dim,
                input_shape,
            },
        });

        Ok((
            out,
            TensorNormDimOperationEvent {
                input,
                out,
                p,
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
        let (requires_grad, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_scan_dim_contiguous_typed(
                ScanOp::CumSum,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let out_shape = meta.shape().to_vec();
            (requires_grad, out_shape, meta.device(), outcome)
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_scan_dim_contiguous_typed(
                ScanOp::CumProd,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            let out_shape = meta.shape().to_vec();
            (requires_grad, out_shape, meta.device(), outcome)
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_normalize_dim_contiguous_typed(
                NormalizeOp::Softmax,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, meta.device(), outcome)
        };

        let output_dtype = outcome.storage.dtype();
        let input_shape = self.nodes[input.0].tensor.meta().shape().to_vec();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(input_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, output_device, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_normalize_dim_contiguous_typed(
                NormalizeOp::LogSoftmax,
                mode,
                input_node.tensor.typed_storage(),
                &meta,
                dim,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (requires_grad, meta.device(), outcome)
        };

        let output_dtype = outcome.storage.dtype();
        let input_shape = self.nodes[input.0].tensor.meta().shape().to_vec();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(input_shape, output_dtype, output_device),
                outcome.storage,
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
            let values = argmax_dim_tensor_contiguous_f64(
                &input_node.tensor.contiguous_values_as_f64()?,
                &meta,
                dim,
            )
            .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let mut out_shape = meta.shape().to_vec();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (values, out_shape, DType::F64, meta.device())
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
            let values = argmin_dim_tensor_contiguous_f64(
                &input_node.tensor.contiguous_values_as_f64()?,
                &meta,
                dim,
            )
            .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let mut out_shape = meta.shape().to_vec();
            out_shape.remove(dim);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            (values, out_shape, DType::F64, meta.device())
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let (values, indices) = max_dim_tensor_contiguous_f64(
                &input_node.tensor.contiguous_values_as_f64()?,
                &meta,
                dim,
            )
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
                DType::F64,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let (values, indices) = min_dim_tensor_contiguous_f64(
                &input_node.tensor.contiguous_values_as_f64()?,
                &meta,
                dim,
            )
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
                DType::F64,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let values = index_select_tensor_contiguous_f64(
                &input_node.tensor.contiguous_values_as_f64()?,
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
                DType::F64,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let idx_meta =
                ft_core::TensorMeta::from_shape(index_shape.clone(), meta.dtype(), meta.device());
            let values = gather_tensor_contiguous_f64(
                &input_node.tensor.contiguous_values_as_f64()?,
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
                DType::F64,
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
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let idx_meta =
                ft_core::TensorMeta::from_shape(index_shape.clone(), meta.dtype(), meta.device());
            let values = scatter_tensor_contiguous_f64(
                &input_node.tensor.contiguous_values_as_f64()?,
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
                DType::F64,
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

    pub fn scatter_add(
        &mut self,
        input: TensorNodeId,
        dim: usize,
        index: &[f64],
        index_shape: Vec<usize>,
        src: &[f64],
    ) -> Result<TensorNodeId, AutogradError> {
        let (values, input_shape, output_dtype, output_device, requires_grad) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let idx_meta =
                ft_core::TensorMeta::from_shape(index_shape.clone(), meta.dtype(), meta.device());
            let values = scatter_add_tensor_contiguous_f64(
                &input_node.tensor.contiguous_values_as_f64()?,
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
                DType::F64,
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
            op: TensorNodeOp::ScatterAdd {
                input,
                dim,
                index: index_owned,
                index_shape,
                input_shape,
            },
        });
        Ok(out)
    }

    pub fn index_put(
        &mut self,
        input: TensorNodeId,
        indices: &[Vec<f64>],
        values: &[f64],
        accumulate: bool,
    ) -> Result<TensorNodeId, AutogradError> {
        let (output_values, input_shape, output_dtype, output_device, requires_grad, suffix_size) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let input_data = input_node.tensor.contiguous_values_as_f64()?;
            let output_values = index_put_tensor_contiguous_f64(
                &input_data,
                &meta,
                indices,
                values,
                accumulate,
            )
            .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let shape = meta.shape();
            let num_indexed = indices.len();
            let suffix: usize = shape[num_indexed..].iter().product();
            let input_shape = shape.to_vec();
            (
                output_values,
                input_shape,
                DType::F64,
                meta.device(),
                requires_grad,
                suffix,
            )
        };

        let indices_owned: Vec<Vec<f64>> = indices.iter().map(|v| v.clone()).collect();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(
                    input_shape.clone(),
                    output_dtype,
                    output_device,
                ),
                output_values,
            )?,
            requires_grad,
            op: TensorNodeOp::IndexPut {
                input,
                indices: indices_owned,
                input_shape,
                accumulate,
                suffix_size,
            },
        });
        Ok(out)
    }

    /// Apply a custom autograd function.
    ///
    /// The `forward_fn` receives a `&mut FunctionCtx` and the input tensor data
    /// (values and shapes) and must return output tensor values and shape.
    ///
    /// The `backward_fn` receives the saved context and incoming gradient(s)
    /// and must return one `Option<Vec<f64>>` per input (None for inputs that
    /// don't need gradient).
    pub fn apply_function<F, B>(
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
        let mut needs_input_grad = Vec::with_capacity(inputs.len());
        let mut input_data: Vec<(Vec<f64>, Vec<usize>)> = Vec::with_capacity(inputs.len());
        let mut input_numels: Vec<usize> = Vec::with_capacity(inputs.len());
        let mut any_requires_grad = false;
        let mut output_dtype = DType::F64;
        let mut output_device = Device::Cpu;

        for &input_id in inputs {
            let node = self.node(input_id)?;
            let rg = node.requires_grad && self.grad_enabled;
            needs_input_grad.push(rg);
            if rg {
                any_requires_grad = true;
            }
            let vals = node.tensor.contiguous_values_as_f64()?;
            let shape = node.tensor.meta().shape().to_vec();
            input_numels.push(vals.len());
            output_dtype = node.tensor.meta().dtype();
            output_device = node.tensor.meta().device();
            input_data.push((vals, shape));
        }

        let mut ctx = FunctionCtx::new(needs_input_grad);

        let refs: Vec<(&[f64], &[usize])> = input_data
            .iter()
            .map(|(v, s)| (v.as_slice(), s.as_slice()))
            .collect();
        let (output_values, output_shape) = forward_fn(&mut ctx, &refs)?;

        let function_id = self.next_custom_function_id;
        self.next_custom_function_id += 1;

        self.custom_functions.insert(
            function_id,
            CustomFunctionRecord {
                ctx,
                backward_fn: Arc::new(backward_fn),
                input_numel: input_numels,
            },
        );

        let inputs_owned = inputs.to_vec();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                output_values,
            )?,
            requires_grad: any_requires_grad,
            op: TensorNodeOp::CustomFunction {
                inputs: inputs_owned,
                function_id,
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
            let values = masked_fill_tensor_contiguous_f64(
                &input_node.tensor.contiguous_values_as_f64()?,
                &meta,
                mask,
                value,
            )
            .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let output_shape = meta.shape().to_vec();
            (values, output_shape, DType::F64, meta.device())
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
            let requires_grad = (x_node.requires_grad || y_node.requires_grad) && self.grad_enabled;
            let cond_values = cond_node.tensor.contiguous_values_as_f64()?;
            let x_values = x_node.tensor.contiguous_values_as_f64()?;
            let y_values = y_node.tensor.contiguous_values_as_f64()?;
            let meta = ft_core::TensorMeta::from_shape(
                x_meta.shape().to_vec(),
                DType::F64,
                x_meta.device(),
            );
            let values = where_tensor_contiguous_f64(&cond_values, &x_values, &y_values, &meta)
                .map_err(|e| AutogradError::Dispatch(e.into()))?;
            let output_shape = meta.shape().to_vec();
            (
                values,
                output_shape,
                DType::F64,
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
        let (requires_grad, output_shape, output_device, input_shape, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_sort_contiguous_typed(
                mode,
                input_node.tensor.typed_storage(),
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
                meta.device(),
                input_shape,
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let indices = outcome.indices.clone();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let (requires_grad, output_shape, output_device, input_shape, outcome) = {
            let input_node = self.node(input)?;
            let requires_grad = input_node.requires_grad && self.grad_enabled;
            let meta = input_node.tensor.meta().clone();
            let outcome = dispatch_tensor_topk_contiguous_typed(
                mode,
                input_node.tensor.typed_storage(),
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
                meta.device(),
                input_shape,
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let indices = outcome.indices.clone();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
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
        let mut dispatch_inputs: Vec<(TensorStorage, ft_core::TensorMeta)> = Vec::new();
        let mut requires_grad = false;
        let mut input_dim_sizes: Vec<usize> = Vec::new();
        for &id in inputs {
            let node = self.node(id)?;
            requires_grad |= node.requires_grad;
            let meta = node.tensor.meta().clone();
            input_dim_sizes.push(meta.shape()[dim]);
            dispatch_inputs.push((node.tensor.typed_storage().clone(), meta));
        }

        let refs: Vec<(&TensorStorage, &ft_core::TensorMeta)> = dispatch_inputs
            .iter()
            .map(|(s, m)| (s, m))
            .collect();

        let outcome =
            dispatch_tensor_join_contiguous_typed(JoinOp::Cat, mode, &refs, dim, requires_grad)
                .map_err(AutogradError::Dispatch)?;

        // Compute output shape
        let first_shape = dispatch_inputs[0].1.shape().to_vec();
        let mut out_shape = first_shape.clone();
        out_shape[dim] = input_dim_sizes.iter().sum();
        let output_dtype = outcome.storage.dtype();
        let output_device = dispatch_inputs[0].1.device();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(out_shape, output_dtype, output_device),
                outcome.storage,
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
        let mut dispatch_inputs: Vec<(TensorStorage, ft_core::TensorMeta)> = Vec::new();
        let mut requires_grad = false;
        for &id in inputs {
            let node = self.node(id)?;
            requires_grad |= node.requires_grad;
            let meta = node.tensor.meta().clone();
            dispatch_inputs.push((node.tensor.typed_storage().clone(), meta));
        }

        let refs: Vec<(&TensorStorage, &ft_core::TensorMeta)> = dispatch_inputs
            .iter()
            .map(|(s, m)| (s, m))
            .collect();

        let outcome =
            dispatch_tensor_join_contiguous_typed(JoinOp::Stack, mode, &refs, dim, requires_grad)
                .map_err(AutogradError::Dispatch)?;

        // Compute output shape: insert new dim at position
        let first_shape = dispatch_inputs[0].1.shape().to_vec();
        let mut out_shape = first_shape;
        out_shape.insert(dim, inputs.len());
        let output_dtype = outcome.storage.dtype();
        let output_device = dispatch_inputs[0].1.device();

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(out_shape, output_dtype, output_device),
                outcome.storage,
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
                input_node.tensor.contiguous_values_as_f64()?,
                meta.shape().to_vec(),
                DType::F64,
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
                input_node.tensor.contiguous_values_as_f64()?,
                new_shape,
                DType::F64,
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
                input_node.tensor.contiguous_values_as_f64()?,
                new_shape,
                DType::F64,
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

    /// Zero-copy reshape that shares storage with the input tensor.
    /// Only works for contiguous tensors. Use reshape() for non-contiguous.
    pub fn view(
        &mut self,
        input: TensorNodeId,
        new_shape: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, view_tensor, original_shape) = {
            let input_node = self.node(input)?;
            let original_shape = input_node.tensor.meta().shape().to_vec();
            let view_tensor = input_node.tensor.view(new_shape)?;
            (input_node.requires_grad, view_tensor, original_shape)
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: view_tensor,
            requires_grad,
            op: TensorNodeOp::View {
                input,
                original_shape,
            },
        });
        Ok(out)
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
                let storage = input_node.tensor.contiguous_values_as_f64()?;
                (
                    input_node.requires_grad,
                    storage,
                    shape.to_vec(),
                    DType::F64,
                    meta.device(),
                )
            } else {
                let mut perm: Vec<usize> = (0..ndim).collect();
                perm.swap(dim0, dim1);
                let storage = input_node.tensor.contiguous_values_as_f64()?;
                let new_storage = Self::permute_data(&storage, shape, &perm)?;
                let mut new_shape = shape.to_vec();
                new_shape.swap(dim0, dim1);
                (
                    input_node.requires_grad,
                    new_storage,
                    new_shape,
                    DType::F64,
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

            let storage = input_node.tensor.contiguous_values_as_f64()?;
            let new_storage = Self::permute_data(&storage, shape, &dims)?;
            let new_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
            (
                input_node.requires_grad,
                new_storage,
                new_shape,
                DType::F64,
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
            let storage = input_node.tensor.contiguous_values_as_f64()?;

            let result_data =
                ft_kernel_cpu::narrow_tensor_contiguous_f64(&storage, meta, dim, start, length)
                    .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;

            let mut new_shape = shape.to_vec();
            new_shape[dim] = length;

            (
                input_node.requires_grad,
                result_data,
                new_shape,
                original_shape,
                DType::F64,
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
            let storage = input_node.tensor.contiguous_values_as_f64()?;

            let result_data =
                ft_kernel_cpu::expand_tensor_contiguous_f64(&storage, meta, &target_shape)
                    .map_err(|e| AutogradError::Dispatch(ft_dispatch::DispatchError::Kernel(e)))?;

            (
                input_node.requires_grad,
                result_data,
                original_shape,
                DType::F64,
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
                input_node.tensor.contiguous_values_as_f64()?,
                shape,
                DType::F64,
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
                input_node.tensor.contiguous_values_as_f64()?,
                shape,
                DType::F64,
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
        let (requires_grad, storage, original_shape, repeat_shape, dtype, device) = {
            let input_node = self.node(input)?;
            let meta = input_node.tensor.meta();
            let shape = meta.shape().to_vec();
            let repeat_shape = Self::normalize_repeat_shape(&shape, &repeats)?;
            (
                input_node.requires_grad,
                input_node.tensor.contiguous_values_as_f64()?,
                shape,
                repeat_shape,
                DType::F64,
                meta.device(),
            )
        };

        let ndim = repeats.len();
        let mut output_shape = Vec::with_capacity(ndim);
        for (&size, &repeat) in repeat_shape.iter().zip(repeats.iter()) {
            output_shape.push(Self::checked_mul_usize(
                size,
                repeat,
                "repeat shape multiplication overflow",
            )?);
        }
        let output_numel =
            Self::checked_shape_numel(&output_shape, "repeat output shape volume overflow")?;
        let output_strides = ft_core::contiguous_strides(&output_shape);
        let src_strides = ft_core::contiguous_strides(&repeat_shape);

        let mut result = vec![0.0; output_numel];
        for flat in 0..output_numel {
            let mut remaining = flat;
            let mut src_flat = 0;
            for d in 0..ndim {
                let coord = remaining / output_strides[d];
                remaining %= output_strides[d];
                let src_coord = coord % repeat_shape[d];
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
                input_node.tensor.contiguous_values_as_f64()?,
                shape,
                DType::F64,
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
                let shifted_coord = (old_coord + shift).rem_euclid(dim_size_i);
                let new_coord = usize::try_from(shifted_coord).map_err(|_| {
                    Self::shape_overflow_error("roll coordinate conversion overflow")
                })?;
                coords[dim] = new_coord % dim_size;

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

    /// Pad a tensor with a constant value.
    ///
    /// `padding` is pairs `[left_N, right_N, ..., left_1, right_1]` applied to
    /// the innermost dimensions first (PyTorch convention).
    pub fn pad(
        &mut self,
        input: TensorNodeId,
        padding: &[usize],
        value: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        if !padding.len().is_multiple_of(2) {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "padding must have even number of elements",
                },
            )));
        }

        let (requires_grad, storage, shape, dtype, device) = {
            let node = self.node(input)?;
            (
                node.requires_grad,
                node.tensor.contiguous_values_as_f64()?.to_vec(),
                node.tensor.meta().shape().to_vec(),
                node.tensor.meta().dtype(),
                node.tensor.meta().device(),
            )
        };

        let ndim = shape.len();
        let num_pad_dims = padding.len() / 2;
        if num_pad_dims > ndim {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "padding specifies more dimensions than input has",
                },
            )));
        }

        // Build output shape
        let mut out_shape = shape.clone();
        for i in 0..num_pad_dims {
            let dim = ndim - 1 - i;
            out_shape[dim] = shape[dim] + padding[i * 2] + padding[i * 2 + 1];
        }

        let out_numel: usize = out_shape.iter().product();
        let mut output = vec![value; out_numel];

        // Compute strides
        let in_strides = ft_core::contiguous_strides(&shape);
        let out_strides = ft_core::contiguous_strides(&out_shape);

        // Build per-dimension pad_before offsets
        let mut pad_before = vec![0usize; ndim];
        for i in 0..num_pad_dims {
            let dim = ndim - 1 - i;
            pad_before[dim] = padding[i * 2];
        }

        // Copy input values into padded output
        let in_numel: usize = shape.iter().product();
        let mut coords = vec![0usize; ndim];
        for (flat_in, &val) in storage.iter().enumerate().take(in_numel) {
            let mut rem = flat_in;
            for d in 0..ndim {
                coords[d] = rem / in_strides[d];
                rem %= in_strides[d];
            }
            let mut flat_out = 0;
            for d in 0..ndim {
                flat_out += (coords[d] + pad_before[d]) * out_strides[d];
            }
            output[flat_out] = val;
        }

        let original_shape = shape;
        let new_meta = ft_core::TensorMeta::from_shape(out_shape.clone(), dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(new_meta, output)?,
            requires_grad,
            op: TensorNodeOp::Pad {
                input,
                padding: padding.to_vec(),
                original_shape,
            },
        });
        Ok(out)
    }

    pub fn lerp(
        &mut self,
        start: TensorNodeId,
        end: TensorNodeId,
        weight: f64,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorLerpOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_device, outcome) = {
            let start_node = self.node(start)?;
            let end_node = self.node(end)?;
            let requires_grad =
                (start_node.requires_grad || end_node.requires_grad) && self.grad_enabled;
            let meta = start_node.tensor.meta().clone();
            let outcome = dispatch_tensor_lerp_contiguous_typed(
                mode,
                start_node.tensor.typed_storage(),
                end_node.tensor.typed_storage(),
                weight,
                &meta,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                meta.shape().to_vec(),
                meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
            )?,
            requires_grad,
            op: TensorNodeOp::Lerp { start, end, weight },
        });

        Ok((
            out,
            TensorLerpOperationEvent {
                start,
                end,
                out,
                weight,
                decision: outcome.decision,
            },
        ))
    }

    pub fn addmm(
        &mut self,
        input: TensorNodeId,
        mat1: TensorNodeId,
        mat2: TensorNodeId,
        beta: f64,
        alpha: f64,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorAddmmOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let mat1_node = self.node(mat1)?;
            let mat2_node = self.node(mat2)?;
            let requires_grad =
                (input_node.requires_grad || mat1_node.requires_grad || mat2_node.requires_grad)
                    && self.grad_enabled;
            let input_meta = input_node.tensor.meta().clone();
            let mat1_meta = mat1_node.tensor.meta().clone();
            let mat2_meta = mat2_node.tensor.meta().clone();
            let m = mat1_meta.shape()[0];
            let n = mat2_meta.shape()[1];
            let outcome = dispatch_tensor_addmm_contiguous_typed(
                mode,
                input_node.tensor.typed_storage(),
                mat1_node.tensor.typed_storage(),
                mat2_node.tensor.typed_storage(),
                &input_meta,
                &mat1_meta,
                &mat2_meta,
                beta,
                alpha,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                vec![m, n],
                mat1_meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
            )?,
            requires_grad,
            op: TensorNodeOp::Addmm {
                input,
                mat1,
                mat2,
                beta,
                alpha,
            },
        });

        Ok((
            out,
            TensorAddmmOperationEvent {
                input,
                mat1,
                mat2,
                out,
                beta,
                alpha,
                decision: outcome.decision,
            },
        ))
    }

    pub fn addmv(
        &mut self,
        input: TensorNodeId,
        mat: TensorNodeId,
        vec_input: TensorNodeId,
        beta: f64,
        alpha: f64,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorAddmvOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_device, outcome) = {
            let input_node = self.node(input)?;
            let mat_node = self.node(mat)?;
            let vec_node = self.node(vec_input)?;
            let requires_grad =
                (input_node.requires_grad || mat_node.requires_grad || vec_node.requires_grad)
                    && self.grad_enabled;
            let input_meta = input_node.tensor.meta().clone();
            let mat_meta = mat_node.tensor.meta().clone();
            let vec_meta = vec_node.tensor.meta().clone();
            let m = mat_meta.shape()[0];
            let outcome = dispatch_tensor_addmv_contiguous_typed(
                mode,
                input_node.tensor.typed_storage(),
                mat_node.tensor.typed_storage(),
                vec_node.tensor.typed_storage(),
                &input_meta,
                &mat_meta,
                &vec_meta,
                beta,
                alpha,
                requires_grad,
            )
            .map_err(AutogradError::Dispatch)?;
            (
                requires_grad,
                vec![m],
                mat_meta.device(),
                outcome,
            )
        };

        let output_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, output_dtype, output_device),
                outcome.storage,
            )?,
            requires_grad,
            op: TensorNodeOp::Addmv {
                input,
                mat,
                vec: vec_input,
                beta,
                alpha,
            },
        });

        Ok((
            out,
            TensorAddmvOperationEvent {
                input,
                mat,
                vec: vec_input,
                out,
                beta,
                alpha,
                decision: outcome.decision,
            },
        ))
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        let (requires_grad, output_shape, output_device, outcome) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let requires_grad =
                (lhs_node.requires_grad || rhs_node.requires_grad) && self.grad_enabled;
            let lhs_meta = lhs_node.tensor.meta().clone();
            let rhs_meta = rhs_node.tensor.meta().clone();
            let outcome = dispatch_tensor_binary_contiguous_typed(
                op,
                mode,
                lhs_node.tensor.typed_storage(),
                rhs_node.tensor.typed_storage(),
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
            (requires_grad, output_shape, lhs_meta.device(), outcome)
        };

        let result_dtype = outcome.storage.dtype();
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_typed_storage(
                ft_core::TensorMeta::from_shape(output_shape, result_dtype, output_device),
                outcome.storage,
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
                BinaryOp::Atan2 => TensorNodeOp::Atan2 { lhs, rhs },
                BinaryOp::Fmod => TensorNodeOp::Fmod { lhs, rhs },
                BinaryOp::Remainder => TensorNodeOp::Remainder { lhs, rhs },
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

    pub fn backward(&mut self, root: TensorNodeId) -> Result<TensorBackwardReport, AutogradError> {
        self.backward_with_options(root, BackwardOptions::strict_default())
    }

    #[allow(clippy::needless_range_loop)]
    pub fn backward_with_options(
        &mut self,
        root: TensorNodeId,
        options: BackwardOptions,
    ) -> Result<TensorBackwardReport, AutogradError> {
        if options.create_graph {
            return self.backward_create_graph(root, options);
        }
        if self.consumed && root.0 < self.consumed_boundary {
            return Err(AutogradError::TensorGraphConsumed);
        }
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
            let incoming = self.apply_tensor_hooks(node_id, &grads[node_id.0])?;
            grads[node_id.0] = incoming.clone();
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
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
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
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
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
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
                    let lhs_shape = self.nodes[lhs.0].tensor.meta().shape();
                    let rhs_shape = self.nodes[rhs.0].tensor.meta().shape();
                    let (m, k, n) = Self::matmul_dims(lhs_shape, rhs_shape)?;
                    let lhs_numel = Self::checked_mul_usize(
                        m,
                        k,
                        "matmul backward lhs shape multiplication overflow",
                    )?;
                    let rhs_numel = Self::checked_mul_usize(
                        k,
                        n,
                        "matmul backward rhs shape multiplication overflow",
                    )?;
                    let out_numel = Self::checked_mul_usize(
                        m,
                        n,
                        "matmul backward output shape multiplication overflow",
                    )?;

                    Self::ensure_tensor_len(lhs, lhs_values.len(), lhs_numel)?;
                    Self::ensure_tensor_len(rhs, rhs_values.len(), rhs_numel)?;
                    Self::ensure_tensor_len(node_id, incoming.len(), out_numel)?;

                    let mut lhs_contrib = vec![0.0; lhs_numel];
                    let mut rhs_contrib = vec![0.0; rhs_numel];

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
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(node_id, 1, incoming.len())?;
                    Self::ensure_tensor_len(lhs, lhs_values.len(), rhs_values.len())?;
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
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
                    let m = lhs_values.len();
                    let n = rhs_values.len();
                    let out_numel = Self::checked_mul_usize(
                        m,
                        n,
                        "outer backward shape multiplication overflow",
                    )?;

                    Self::ensure_tensor_len(node_id, incoming.len(), out_numel)?;

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
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
                    let lhs_shape = self.nodes[lhs.0].tensor.meta().shape();
                    let rhs_shape = self.nodes[rhs.0].tensor.meta().shape();
                    if lhs_shape.len() != 3 || rhs_shape.len() != 3 {
                        return Err(AutogradError::Dispatch(
                            DispatchKeyError::IncompatibleSet {
                                reason: "bmm backward expected rank-3 operands",
                            }
                            .into(),
                        ));
                    }
                    let batch = lhs_shape[0];
                    let m = lhs_shape[1];
                    let k = lhs_shape[2];
                    if rhs_shape[0] != batch || rhs_shape[1] != k {
                        return Err(AutogradError::Dispatch(
                            DispatchKeyError::IncompatibleSet {
                                reason: "bmm backward received incompatible operand shapes",
                            }
                            .into(),
                        ));
                    }
                    let n = rhs_shape[2];
                    let lhs_batch_stride =
                        Self::checked_mul_usize(m, k, "bmm backward lhs batch stride overflow")?;
                    let rhs_batch_stride =
                        Self::checked_mul_usize(k, n, "bmm backward rhs batch stride overflow")?;
                    let out_batch_stride =
                        Self::checked_mul_usize(m, n, "bmm backward output batch stride overflow")?;
                    let lhs_numel = Self::checked_mul_usize(
                        batch,
                        lhs_batch_stride,
                        "bmm backward lhs shape multiplication overflow",
                    )?;
                    let rhs_numel = Self::checked_mul_usize(
                        batch,
                        rhs_batch_stride,
                        "bmm backward rhs shape multiplication overflow",
                    )?;
                    let out_numel = Self::checked_mul_usize(
                        batch,
                        out_batch_stride,
                        "bmm backward output shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(lhs, lhs_values.len(), lhs_numel)?;
                    Self::ensure_tensor_len(rhs, rhs_values.len(), rhs_numel)?;
                    Self::ensure_tensor_len(node_id, incoming.len(), out_numel)?;

                    let mut lhs_contrib = vec![0.0; lhs_numel];
                    let mut rhs_contrib = vec![0.0; rhs_numel];

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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                TensorNodeOp::Rsqrt { input } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(node_id, output_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(output_values.iter())
                        .map(|(g, y)| g * (-0.5 * y * y * y))
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(rsqrt(x))/dx=-0.5*rsqrt(x)^3",
                    });
                }
                TensorNodeOp::Erf { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let coeff = 2.0 / std::f64::consts::PI.sqrt();
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g * coeff * (-x * x).exp())
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(erf(x))/dx=(2/sqrt(pi))*exp(-x^2)",
                    });
                }
                TensorNodeOp::Erfc { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let coeff = 2.0 / std::f64::consts::PI.sqrt();
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g * (-coeff) * (-x * x).exp())
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(erfc(x))/dx=-(2/sqrt(pi))*exp(-x^2)",
                    });
                }
                TensorNodeOp::Hardswish { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| {
                            g * if *x <= -3.0 {
                                0.0
                            } else if *x >= 3.0 {
                                1.0
                            } else {
                                (2.0 * x + 3.0) / 6.0
                            }
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(hardswish(x))/dx=(2x+3)/6|0|1",
                    });
                }
                TensorNodeOp::Hardsigmoid { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| {
                            g * if *x <= -3.0 || *x >= 3.0 {
                                0.0
                            } else {
                                1.0 / 6.0
                            }
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(hardsigmoid(x))/dx=1/6|0",
                    });
                }
                TensorNodeOp::Hardtanh { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g * if *x < -1.0 || *x > 1.0 { 0.0 } else { 1.0 })
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(hardtanh(x))/dx=1|0",
                    });
                }
                TensorNodeOp::Softplus { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g * (1.0 / (1.0 + (-x).exp())))
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(softplus(x))/dx=sigmoid(x)",
                    });
                }
                TensorNodeOp::Mish { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| {
                            let sp = if *x > 20.0 {
                                *x
                            } else if *x < -20.0 {
                                0.0
                            } else {
                                (1.0 + x.exp()).ln()
                            };
                            let tsp = sp.tanh();
                            let sig = 1.0 / (1.0 + (-x).exp());
                            g * (tsp + x * sig * (1.0 - tsp * tsp))
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(mish(x))/dx=tanh(sp)+x*sig*(1-tanh(sp)^2)",
                    });
                }
                TensorNodeOp::Square { input } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(input, input_values.len(), incoming.len())?;
                    let contrib: Vec<f64> = incoming
                        .iter()
                        .zip(input_values.iter())
                        .map(|(g, x)| g * 2.0 * x)
                        .collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(x^2)/dx=2x",
                    });
                }
                TensorNodeOp::Sqrt { input } => {
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
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
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
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
                TensorNodeOp::Atan2 { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(lhs, lhs_values.len(), incoming.len())?;

                    let lhs_contrib: Vec<f64> = incoming
                        .iter()
                        .zip(lhs_values.iter().zip(rhs_values.iter()))
                        .map(|(g, (y, x))| {
                            let denom = x * x + y * y;
                            g * x / denom
                        })
                        .collect();
                    let rhs_contrib: Vec<f64> = incoming
                        .iter()
                        .zip(lhs_values.iter().zip(rhs_values.iter()))
                        .map(|(g, (y, x))| {
                            let denom = x * x + y * y;
                            g * (-y) / denom
                        })
                        .collect();
                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(atan2(y,x))/dy=x/(x^2+y^2); dx=-y/(x^2+y^2)",
                    });
                }
                TensorNodeOp::Fmod { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(lhs, lhs_values.len(), incoming.len())?;

                    let lhs_contrib: Vec<f64> = incoming.to_vec();
                    let rhs_contrib: Vec<f64> = incoming
                        .iter()
                        .zip(lhs_values.iter().zip(rhs_values.iter()))
                        .map(|(g, (a, b))| g * (-(a / b).trunc()))
                        .collect();
                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(fmod(a,b))/da=1; db=-trunc(a/b)",
                    });
                }
                TensorNodeOp::Remainder { lhs, rhs } => {
                    let lhs_values = self.nodes[lhs.0].tensor.contiguous_values_as_f64()?;
                    let rhs_values = self.nodes[rhs.0].tensor.contiguous_values_as_f64()?;
                    Self::ensure_tensor_len(lhs, lhs_values.len(), incoming.len())?;

                    let lhs_contrib: Vec<f64> = incoming.to_vec();
                    let rhs_contrib: Vec<f64> = incoming
                        .iter()
                        .zip(lhs_values.iter().zip(rhs_values.iter()))
                        .map(|(g, (a, b))| g * (-(a / b).floor()))
                        .collect();
                    Self::accumulate_tensor_gradient(lhs, &mut grads[lhs.0], &lhs_contrib)?;
                    Self::accumulate_tensor_gradient(rhs, &mut grads[rhs.0], &rhs_contrib)?;

                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(remainder(a,b))/da=1; db=-floor(a/b)",
                    });
                }
                TensorNodeOp::Clamp {
                    input,
                    min_val,
                    max_val,
                } => {
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                TensorNodeOp::Norm {
                    input,
                    p,
                    input_numel,
                } => {
                    let grad_scalar = incoming[0];
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    let norm_val = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?[0];
                    let mut norm_contrib = vec![0.0; input_numel];

                    if p == 2.0 {
                        // d/dx_i = x_i / norm
                        if norm_val != 0.0 {
                            for i in 0..input_numel {
                                norm_contrib[i] = grad_scalar * input_values[i] / norm_val;
                            }
                        }
                    } else if p == 1.0 {
                        // d/dx_i = sign(x_i)
                        for i in 0..input_numel {
                            norm_contrib[i] = grad_scalar * input_values[i].signum();
                        }
                    } else if p.is_infinite() {
                        // Gradient flows to the element(s) achieving the extremum
                        // For inf-norm: the element with max |x_i|
                        // For -inf-norm: the element with min |x_i|
                        if norm_val != 0.0 {
                            for i in 0..input_numel {
                                if input_values[i].abs() == norm_val {
                                    norm_contrib[i] = grad_scalar * input_values[i].signum();
                                }
                            }
                        }
                    } else if p != 0.0 && norm_val != 0.0 {
                        // General p-norm: d/dx_i = sign(x_i) * |x_i|^(p-1) / norm^(p-1)
                        let norm_pow = norm_val.powf(p - 1.0);
                        for i in 0..input_numel {
                            norm_contrib[i] = grad_scalar
                                * input_values[i].signum()
                                * input_values[i].abs().powf(p - 1.0)
                                / norm_pow;
                        }
                    }
                    // p == 0: gradient is zero (non-differentiable)

                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &norm_contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(norm_p(x))/dx_i=sign(x_i)*|x_i|^(p-1)/norm^(p-1)",
                    });
                }
                TensorNodeOp::NormDim {
                    input,
                    p,
                    dim,
                    ref input_shape,
                } => {
                    let reduce_size = input_shape[dim];
                    let (outer_size, inner_size, input_numel) = Self::checked_dim_loop_sizes(
                        input_shape,
                        dim,
                        "norm_dim backward shape volume overflow",
                    )?;
                    let expected_incoming = Self::checked_mul_usize(
                        outer_size,
                        inner_size,
                        "norm_dim backward shape multiplication overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, expected_incoming, incoming.len())?;
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
                    let mut norm_dim_contrib = vec![0.0; input_numel];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let out_idx = outer * inner_size + inner;
                            let grad_val = incoming[out_idx];
                            let norm_val = output_values[out_idx];

                            if p == 2.0 {
                                if norm_val != 0.0 {
                                    for r in 0..reduce_size {
                                        let idx = outer * reduce_size * inner_size
                                            + r * inner_size
                                            + inner;
                                        norm_dim_contrib[idx] =
                                            grad_val * input_values[idx] / norm_val;
                                    }
                                }
                            } else if p == 1.0 {
                                for r in 0..reduce_size {
                                    let idx =
                                        outer * reduce_size * inner_size + r * inner_size + inner;
                                    norm_dim_contrib[idx] = grad_val * input_values[idx].signum();
                                }
                            } else if p.is_infinite() {
                                if norm_val != 0.0 {
                                    for r in 0..reduce_size {
                                        let idx = outer * reduce_size * inner_size
                                            + r * inner_size
                                            + inner;
                                        if input_values[idx].abs() == norm_val {
                                            norm_dim_contrib[idx] =
                                                grad_val * input_values[idx].signum();
                                        }
                                    }
                                }
                            } else if p != 0.0 && norm_val != 0.0 {
                                let norm_pow = norm_val.powf(p - 1.0);
                                for r in 0..reduce_size {
                                    let idx =
                                        outer * reduce_size * inner_size + r * inner_size + inner;
                                    norm_dim_contrib[idx] = grad_val
                                        * input_values[idx].signum()
                                        * input_values[idx].abs().powf(p - 1.0)
                                        / norm_pow;
                                }
                            }
                        }
                    }
                    Self::accumulate_tensor_gradient(
                        input,
                        &mut grads[input.0],
                        &norm_dim_contrib,
                    )?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(norm_dim_p(x))/dx_i=sign(x_i)*|x_i|^(p-1)/norm^(p-1)",
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
                    let input_values = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let cond_vals = self.nodes[condition.0].tensor.contiguous_values_as_f64()?;
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
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                    let output_values = self.nodes[node_id.0].tensor.contiguous_values_as_f64()?;
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
                | TensorNodeOp::View { input, .. }
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
                                || selected_f > usize::MAX as f64
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
                            let idx = Self::normalize_wrapped_index_float(
                                idx_f,
                                dim_size,
                                "index_select backward received invalid index value",
                                "index_select backward received out-of-bounds index value",
                                "index_select backward index conversion overflow",
                            )?;
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
                                let selected = Self::normalize_wrapped_index_float(
                                    selected_f,
                                    dim_size,
                                    "gather backward received invalid index value",
                                    "gather backward received out-of-bounds index value",
                                    "gather backward index conversion overflow",
                                )?;
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
                                let selected = Self::normalize_wrapped_index_float(
                                    selected_f,
                                    dim_size,
                                    "scatter backward received invalid index value",
                                    "scatter backward received out-of-bounds index value",
                                    "scatter backward index conversion overflow",
                                )?;
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
                TensorNodeOp::ScatterAdd {
                    input,
                    dim: _,
                    ref index,
                    ref index_shape,
                    ref input_shape,
                } => {
                    // scatter_add adds src to positions, so gradient for input is
                    // just the incoming gradient unchanged (the original values are
                    // preserved, only additions are made).
                    let input_numel = Self::checked_shape_numel(
                        input_shape,
                        "scatter_add backward input shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, input_numel, incoming.len())?;
                    let _ = index;
                    let _ = index_shape;

                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &incoming)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(scatter_add(x))/dx=passthrough",
                    });
                }
                TensorNodeOp::IndexPut {
                    input,
                    ref indices,
                    ref input_shape,
                    accumulate,
                    suffix_size,
                } => {
                    // For non-accumulate: zero out positions that were overwritten
                    // For accumulate: gradient passes through unchanged
                    let input_numel = Self::checked_shape_numel(
                        input_shape,
                        "index_put backward input shape volume overflow",
                    )?;
                    Self::ensure_tensor_len(node_id, input_numel, incoming.len())?;

                    if accumulate {
                        Self::accumulate_tensor_gradient(input, &mut grads[input.0], &incoming)?;
                    } else {
                        let mut contrib = incoming.to_vec();
                        let n_indices = indices[0].len();
                        let num_indexed = indices.len();

                        let mut indexed_strides = vec![0usize; num_indexed];
                        for d in 0..num_indexed {
                            indexed_strides[d] = input_shape[d + 1..].iter().product();
                        }

                        for i in 0..n_indices {
                            let mut base = 0usize;
                            for d in 0..num_indexed {
                                let idx = Self::normalize_wrapped_index_float(
                                    indices[d][i],
                                    input_shape[d],
                                    "index_put backward received invalid index value",
                                    "index_put backward received out-of-bounds index value",
                                    "index_put backward index conversion overflow",
                                )?;
                                base += idx * indexed_strides[d];
                            }
                            for s in 0..suffix_size {
                                contrib[base + s] = 0.0;
                            }
                        }
                        Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    }
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(index_put(x))/dx",
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
                    let repeat_shape = Self::normalize_repeat_shape(original_shape, repeats)?;
                    let ndim = repeats.len();
                    let input_numel = Self::checked_shape_numel(
                        &repeat_shape,
                        "repeat backward input shape volume overflow",
                    )?;
                    let mut output_shape = Vec::with_capacity(ndim);
                    for (&size, &repeat) in repeat_shape.iter().zip(repeats.iter()) {
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
                    let input_strides = ft_core::contiguous_strides(&repeat_shape);

                    let mut contrib = vec![0.0; input_numel];
                    for flat in 0..output_numel {
                        let mut remaining = flat;
                        let mut src_flat = 0;
                        for d in 0..ndim {
                            let coord = remaining / output_strides[d];
                            remaining %= output_strides[d];
                            let src_coord = coord % repeat_shape[d];
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
                            let shifted_coord = (old_coord - shift).rem_euclid(dim_size_i);
                            let new_coord = usize::try_from(shifted_coord).map_err(|_| {
                                Self::shape_overflow_error(
                                    "roll backward coordinate conversion overflow",
                                )
                            })?;
                            coords[dim] = new_coord % dim_size;
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
                TensorNodeOp::Pad {
                    input,
                    ref padding,
                    ref original_shape,
                } => {
                    // Backward for constant pad: extract the unpadded region from
                    // the incoming gradient, discarding gradient on padded elements.
                    let ndim = original_shape.len();
                    let num_pad_dims = padding.len() / 2;
                    let in_numel: usize = original_shape.iter().product();
                    let in_strides = ft_core::contiguous_strides(original_shape);

                    let out_shape = self.nodes[node_id.0].tensor.meta().shape();
                    let out_strides = ft_core::contiguous_strides(out_shape);

                    let mut pad_before = vec![0usize; ndim];
                    for i in 0..num_pad_dims {
                        let dim = ndim - 1 - i;
                        pad_before[dim] = padding[i * 2];
                    }

                    let mut contrib = vec![0.0; in_numel];
                    let mut coords = vec![0usize; ndim];
                    for flat_in in 0..in_numel {
                        let mut rem = flat_in;
                        for d in 0..ndim {
                            coords[d] = rem / in_strides[d];
                            rem %= in_strides[d];
                        }
                        let mut flat_out = 0;
                        for d in 0..ndim {
                            flat_out += (coords[d] + pad_before[d]) * out_strides[d];
                        }
                        contrib[flat_in] = incoming[flat_out];
                    }

                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &contrib)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(pad(x))/dx=unpad(grad)",
                    });
                }
                TensorNodeOp::Lerp { start, end, weight } => {
                    // lerp(s, e, w) = s + w*(e - s) = (1-w)*s + w*e
                    // d/ds = (1-w) * grad_out, d/de = w * grad_out
                    let start_contrib: Vec<f64> =
                        incoming.iter().map(|&g| g * (1.0 - weight)).collect();
                    let end_contrib: Vec<f64> = incoming.iter().map(|&g| g * weight).collect();
                    Self::accumulate_tensor_gradient(start, &mut grads[start.0], &start_contrib)?;
                    Self::accumulate_tensor_gradient(end, &mut grads[end.0], &end_contrib)?;
                    Self::complete_dependency(&mut pending, start, &mut queue)?;
                    Self::complete_dependency(&mut pending, end, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(lerp(s,e,w))/ds=(1-w)*grad, d/de=w*grad",
                    });
                }
                TensorNodeOp::Addmm {
                    input,
                    mat1,
                    mat2,
                    beta,
                    alpha,
                } => {
                    // addmm(input, mat1, mat2, beta, alpha) = beta*input + alpha*(mat1 @ mat2)
                    // d/d(input) = beta * grad_out
                    // d/d(mat1) = alpha * grad_out @ mat2^T
                    // d/d(mat2) = alpha * mat1^T @ grad_out
                    let mat1_vals = self.nodes[mat1.0].tensor.contiguous_values_as_f64()?;
                    let mat2_vals = self.nodes[mat2.0].tensor.contiguous_values_as_f64()?;
                    let mat1_shape = self.nodes[mat1.0].tensor.meta().shape().to_vec();
                    let mat2_shape = self.nodes[mat2.0].tensor.meta().shape().to_vec();
                    let m = mat1_shape[0];
                    let k = mat1_shape[1];
                    let n = mat2_shape[1];

                    // d/d(input): beta * grad_out
                    // input could be 1-D [n] or 2-D [m,n]
                    let input_shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let input_numel = Self::checked_shape_numel(
                        &input_shape,
                        "addmm backward input shape overflow",
                    )?;
                    let mut input_contrib = vec![0.0; input_numel];
                    if input_shape.len() == 1 {
                        // Broadcast: sum grad_out along rows, scale by beta
                        for row in 0..m {
                            for col in 0..n {
                                input_contrib[col] += incoming[row * n + col] * beta;
                            }
                        }
                    } else {
                        for i in 0..incoming.len() {
                            input_contrib[i] = incoming[i] * beta;
                        }
                    }
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &input_contrib)?;

                    // d/d(mat1): alpha * grad_out @ mat2^T => [m,n] @ [n,k] = [m,k]
                    let mat1_numel = Self::checked_mul_usize(m, k, "addmm mat1 grad overflow")?;
                    let mut mat1_contrib = vec![0.0; mat1_numel];
                    for row in 0..m {
                        for col in 0..k {
                            let mut acc = 0.0;
                            for inner in 0..n {
                                acc += incoming[row * n + inner] * mat2_vals[col + inner * n];
                            }
                            mat1_contrib[row * k + col] = alpha * acc;
                        }
                    }
                    Self::accumulate_tensor_gradient(mat1, &mut grads[mat1.0], &mat1_contrib)?;

                    // d/d(mat2): alpha * mat1^T @ grad_out => [k,m] @ [m,n] = [k,n]
                    let mat2_numel = Self::checked_mul_usize(k, n, "addmm mat2 grad overflow")?;
                    let mut mat2_contrib = vec![0.0; mat2_numel];
                    for row in 0..k {
                        for col in 0..n {
                            let mut acc = 0.0;
                            for inner in 0..m {
                                acc += mat1_vals[inner * k + row] * incoming[inner * n + col];
                            }
                            mat2_contrib[row * n + col] = alpha * acc;
                        }
                    }
                    Self::accumulate_tensor_gradient(mat2, &mut grads[mat2.0], &mat2_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    Self::complete_dependency(&mut pending, mat1, &mut queue)?;
                    Self::complete_dependency(&mut pending, mat2, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(addmm)/d_input=beta*grad, d/d_mat1=alpha*grad@mat2^T, d/d_mat2=alpha*mat1^T@grad",
                    });
                }
                TensorNodeOp::Addmv {
                    input,
                    mat,
                    vec: vec_id,
                    beta,
                    alpha,
                } => {
                    // addmv(input, mat, vec, beta, alpha) = beta*input + alpha*(mat @ vec)
                    // d/d(input) = beta * grad_out  (shape: [m])
                    // d/d(mat) = alpha * grad_out (outer) vec^T  (shape: [m,k])
                    // d/d(vec) = alpha * mat^T @ grad_out  (shape: [k])
                    let mat_vals = self.nodes[mat.0].tensor.contiguous_values_as_f64()?;
                    let vec_vals = self.nodes[vec_id.0].tensor.contiguous_values_as_f64()?;
                    let mat_shape = self.nodes[mat.0].tensor.meta().shape().to_vec();
                    let m = mat_shape[0];
                    let k = mat_shape[1];

                    // d/d(input): beta * grad_out
                    let input_contrib: Vec<f64> = incoming.iter().map(|&g| g * beta).collect();
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &input_contrib)?;

                    // d/d(mat): alpha * outer(grad_out, vec)
                    let mat_numel = Self::checked_mul_usize(m, k, "addmv mat grad overflow")?;
                    let mut mat_contrib = vec![0.0; mat_numel];
                    for row in 0..m {
                        for col in 0..k {
                            mat_contrib[row * k + col] = alpha * incoming[row] * vec_vals[col];
                        }
                    }
                    Self::accumulate_tensor_gradient(mat, &mut grads[mat.0], &mat_contrib)?;

                    // d/d(vec): alpha * mat^T @ grad_out
                    let mut vec_contrib = vec![0.0; k];
                    for col in 0..k {
                        let mut acc = 0.0;
                        for row in 0..m {
                            acc += mat_vals[row * k + col] * incoming[row];
                        }
                        vec_contrib[col] = alpha * acc;
                    }
                    Self::accumulate_tensor_gradient(vec_id, &mut grads[vec_id.0], &vec_contrib)?;

                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    Self::complete_dependency(&mut pending, mat, &mut queue)?;
                    Self::complete_dependency(&mut pending, vec_id, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(addmv)/d_input=beta*grad, d/d_mat=alpha*outer(grad,vec), d/d_vec=alpha*mat^T@grad",
                    });
                }
                TensorNodeOp::CastF32 { input } | TensorNodeOp::CastF64 { input } => {
                    // Cast is identity for gradients — gradient passes through unchanged.
                    // Backward always operates in f64 so no conversion needed.
                    Self::accumulate_tensor_gradient(input, &mut grads[input.0], &incoming)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "d(cast)/d_input=grad (identity)",
                    });
                }
                TensorNodeOp::CustomFunction {
                    ref inputs,
                    function_id,
                } => {
                    let record = self
                        .custom_functions
                        .get(&function_id)
                        .ok_or(AutogradError::UnknownTensorNode(node_id))?;
                    let grad_outputs: Vec<&[f64]> = vec![incoming.as_slice()];
                    let input_grads = (record.backward_fn)(&record.ctx, &grad_outputs)?;

                    if input_grads.len() != inputs.len() {
                        return Err(AutogradError::TensorGradientShapeMismatch {
                            node: node_id,
                            expected: inputs.len(),
                            actual: input_grads.len(),
                        });
                    }

                    let inputs_snapshot = inputs.clone();
                    for (i, maybe_grad) in input_grads.into_iter().enumerate() {
                        let input_id = inputs_snapshot[i];
                        if let Some(grad) = maybe_grad {
                            Self::accumulate_tensor_gradient(
                                input_id,
                                &mut grads[input_id.0],
                                &grad,
                            )?;
                        }
                        Self::complete_dependency(&mut pending, input_id, &mut queue)?;
                    }

                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: incoming.len(),
                        rule: "custom autograd function backward",
                    });
                }
            }
        }

        let gradients: Vec<Option<Vec<f64>>> = grads
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

        self.accumulate_persistent_gradients(&gradients)?;

        if !options.retain_graph {
            self.consumed = true;
            self.consumed_boundary = self.nodes.len();
        }

        Ok(TensorBackwardReport {
            gradients,
            gradient_nodes: vec![None; self.nodes.len()],
            steps,
            telemetry,
        })
    }

    /// Backward pass that records gradient computation as new graph nodes,
    /// enabling higher-order derivatives (second backward through gradients).
    #[allow(clippy::needless_range_loop, clippy::too_many_lines)]
    fn backward_create_graph(
        &mut self,
        root: TensorNodeId,
        options: BackwardOptions,
    ) -> Result<TensorBackwardReport, AutogradError> {
        // create_graph implies retain_graph
        if self.consumed && root.0 < self.consumed_boundary {
            return Err(AutogradError::TensorGraphConsumed);
        }
        if root.0 >= self.nodes.len() {
            return Err(AutogradError::UnknownTensorNode(root));
        }
        if !self.nodes[root.0].requires_grad {
            return Err(AutogradError::TensorRootDoesNotRequireGrad { node: root });
        }

        // Snapshot the node count at start; only original nodes participate in backward
        let orig_node_count = self.nodes.len();

        let reachable = self.compute_reachable(root)?;
        let mut pending = self.compute_dependencies(&reachable)?;

        // Gradient node IDs: each entry is a TensorNodeId representing the gradient
        let mut grad_nodes: Vec<Option<TensorNodeId>> = vec![None; orig_node_count];

        // Initial gradient: ones_like(root) with requires_grad=true
        let root_shape = self.nodes[root.0].tensor.meta().shape().to_vec();
        let root_numel: usize = root_shape.iter().product();
        let root_grad = self.leaf(vec![1.0; root_numel], root_shape, true)?;
        grad_nodes[root.0] = Some(root_grad);

        let mut queue = TensorReadyQueue::with_capacity(orig_node_count.max(1));
        queue.push(root);

        let mut steps = Vec::with_capacity(orig_node_count);
        let mut execution_order = Vec::with_capacity(orig_node_count);

        while let Some(node_id) = queue.pop() {
            if node_id.0 >= orig_node_count {
                continue;
            }
            execution_order.push(node_id);
            let incoming_id = match grad_nodes[node_id.0] {
                Some(id) => id,
                None => continue,
            };

            // Clone the op to avoid borrowing self.nodes during the match
            let op = self.nodes[node_id.0].op.clone();

            match op {
                TensorNodeOp::Leaf => {
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "leaf (create_graph)",
                    });
                }
                TensorNodeOp::Add { lhs, rhs } => {
                    // d(a+b)/da = 1, d(a+b)/db = 1
                    self.cg_accumulate(lhs, &mut grad_nodes, incoming_id)?;
                    self.cg_accumulate(rhs, &mut grad_nodes, incoming_id)?;
                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(a+b)/da=grad,d/db=grad (cg)",
                    });
                }
                TensorNodeOp::Sub { lhs, rhs } => {
                    // d(a-b)/da = 1, d(a-b)/db = -1
                    let neg_incoming = self.cg_neg(incoming_id)?;
                    self.cg_accumulate(lhs, &mut grad_nodes, incoming_id)?;
                    self.cg_accumulate(rhs, &mut grad_nodes, neg_incoming)?;
                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(a-b)/da=grad,d/db=-grad (cg)",
                    });
                }
                TensorNodeOp::Mul { lhs, rhs } => {
                    // d(a*b)/da = b*grad, d(a*b)/db = a*grad
                    let grad_lhs = self.cg_mul(incoming_id, rhs)?;
                    let grad_rhs = self.cg_mul(incoming_id, lhs)?;
                    self.cg_accumulate(lhs, &mut grad_nodes, grad_lhs)?;
                    self.cg_accumulate(rhs, &mut grad_nodes, grad_rhs)?;
                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(a*b)/da=b*grad,d/db=a*grad (cg)",
                    });
                }
                TensorNodeOp::Div { lhs, rhs } => {
                    // d(a/b)/da = grad/b, d(a/b)/db = -a*grad/b^2
                    let grad_lhs = self.cg_div(incoming_id, rhs)?;
                    let neg_in = self.cg_neg(incoming_id)?;
                    let neg_a_grad = self.cg_mul(neg_in, lhs)?;
                    let b_sq = self.cg_mul(rhs, rhs)?;
                    let grad_rhs = self.cg_div(neg_a_grad, b_sq)?;
                    self.cg_accumulate(lhs, &mut grad_nodes, grad_lhs)?;
                    self.cg_accumulate(rhs, &mut grad_nodes, grad_rhs)?;
                    Self::complete_dependency(&mut pending, lhs, &mut queue)?;
                    Self::complete_dependency(&mut pending, rhs, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(a/b) (cg)",
                    });
                }
                TensorNodeOp::Neg { input } => {
                    let grad_in = self.cg_neg(incoming_id)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(-x)/dx=-grad (cg)",
                    });
                }
                TensorNodeOp::Pow { input, exponent } => {
                    // d(x^n)/dx = n * x^(n-1) * grad
                    let exp_shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let exp_numel: usize = exp_shape.iter().product();
                    let exp_node = self.leaf(vec![exponent; exp_numel], exp_shape.clone(), false)?;
                    let exp_m1_node =
                        self.leaf(vec![exponent - 1.0; exp_numel], exp_shape, false)?;
                    let x_pow = self.cg_pow(input, exp_m1_node)?;
                    let n_x_pow = self.cg_mul(exp_node, x_pow)?;
                    let grad_in = self.cg_mul(incoming_id, n_x_pow)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(x^n)/dx=n*x^(n-1)*grad (cg)",
                    });
                }
                TensorNodeOp::Exp { input } => {
                    // d(exp(x))/dx = exp(x) * grad
                    let grad_in = self.cg_mul(incoming_id, node_id)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(exp(x))/dx=exp(x)*grad (cg)",
                    });
                }
                TensorNodeOp::Log { input } => {
                    // d(ln(x))/dx = grad / x
                    let grad_in = self.cg_div(incoming_id, input)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(ln(x))/dx=grad/x (cg)",
                    });
                }
                TensorNodeOp::Sin { input } => {
                    // d(sin(x))/dx = cos(x) * grad
                    let cos_x = self.cg_cos(input)?;
                    let grad_in = self.cg_mul(incoming_id, cos_x)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(sin(x))/dx=cos(x)*grad (cg)",
                    });
                }
                TensorNodeOp::Cos { input } => {
                    // d(cos(x))/dx = -sin(x) * grad
                    let sin_x = self.cg_sin(input)?;
                    let neg_sin = self.cg_neg(sin_x)?;
                    let grad_in = self.cg_mul(incoming_id, neg_sin)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(cos(x))/dx=-sin(x)*grad (cg)",
                    });
                }
                TensorNodeOp::Sqrt { input } => {
                    // d(sqrt(x))/dx = 0.5/sqrt(x) * grad = grad / (2*sqrt(x))
                    let shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let numel: usize = shape.iter().product();
                    let two = self.leaf(vec![2.0; numel], shape, false)?;
                    let two_sqrt = self.cg_mul(two, node_id)?;
                    let grad_in = self.cg_div(incoming_id, two_sqrt)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(sqrt(x))/dx=grad/(2*sqrt(x)) (cg)",
                    });
                }
                TensorNodeOp::Reciprocal { input } => {
                    // d(1/x)/dx = -1/x^2 * grad = -grad * out^2  where out=1/x
                    let out_sq = self.cg_mul(node_id, node_id)?;
                    let neg_out_sq = self.cg_neg(out_sq)?;
                    let grad_in = self.cg_mul(incoming_id, neg_out_sq)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(1/x)/dx=-grad*out^2 (cg)",
                    });
                }
                TensorNodeOp::Square { input } => {
                    // d(x^2)/dx = 2x * grad
                    let shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let numel: usize = shape.iter().product();
                    let two = self.leaf(vec![2.0; numel], shape, false)?;
                    let two_x = self.cg_mul(two, input)?;
                    let grad_in = self.cg_mul(incoming_id, two_x)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(x^2)/dx=2x*grad (cg)",
                    });
                }
                TensorNodeOp::Sum { input, .. } => {
                    // d(sum(x))/dx = expand(grad) to input shape
                    // We need the gradient to expand from [1] to input_shape.
                    // The incoming grad is scalar-shaped. We expand it via a
                    // Sum backward node so the standard backward can handle it.
                    let input_shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let input_numel: usize = input_shape.iter().product();
                    let incoming_val = self.nodes[incoming_id.0]
                        .tensor
                        .contiguous_values_as_f64()?[0];
                    let expanded_data = vec![incoming_val; input_numel];
                    let grad_in = self.cg_expand(incoming_id, expanded_data, input_shape)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(sum(x))/dx=expand(grad) (cg)",
                    });
                }
                TensorNodeOp::Relu { input } => {
                    // d(relu(x))/dx = (x > 0) * grad
                    let vals = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    let shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let mask: Vec<f64> = vals.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
                    let mask_node = self.leaf(mask, shape, false)?;
                    let grad_in = self.cg_mul(incoming_id, mask_node)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(relu(x))/dx=(x>0)*grad (cg)",
                    });
                }
                TensorNodeOp::Sigmoid { input } => {
                    // d(sigmoid(x))/dx = sigmoid(x)*(1-sigmoid(x))*grad
                    let shape = self.nodes[node_id.0].tensor.meta().shape().to_vec();
                    let numel: usize = shape.iter().product();
                    let ones = self.leaf(vec![1.0; numel], shape, false)?;
                    let one_minus_sig = self.cg_sub(ones, node_id)?;
                    let sig_deriv = self.cg_mul(node_id, one_minus_sig)?;
                    let grad_in = self.cg_mul(incoming_id, sig_deriv)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(sigmoid(x))/dx=sig*(1-sig)*grad (cg)",
                    });
                }
                TensorNodeOp::Tanh { input } => {
                    // d(tanh(x))/dx = (1-tanh(x)^2)*grad
                    let shape = self.nodes[node_id.0].tensor.meta().shape().to_vec();
                    let numel: usize = shape.iter().product();
                    let ones = self.leaf(vec![1.0; numel], shape, false)?;
                    let tanh_sq = self.cg_mul(node_id, node_id)?;
                    let one_minus_sq = self.cg_sub(ones, tanh_sq)?;
                    let grad_in = self.cg_mul(incoming_id, one_minus_sq)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(tanh(x))/dx=(1-tanh^2)*grad (cg)",
                    });
                }
                TensorNodeOp::Abs { input } => {
                    // d(|x|)/dx = sign(x) * grad
                    let vals = self.nodes[input.0].tensor.contiguous_values_as_f64()?;
                    let shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let sign: Vec<f64> = vals.iter().map(|&v| v.signum()).collect();
                    let sign_node = self.leaf(sign, shape, false)?;
                    let grad_in = self.cg_mul(incoming_id, sign_node)?;
                    self.cg_accumulate(input, &mut grad_nodes, grad_in)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(|x|)/dx=sign(x)*grad (cg)",
                    });
                }
                TensorNodeOp::Expand { input, .. } => {
                    // d(expand(x))/dx = sum(grad)
                    let incoming_vals = self.nodes[incoming_id.0]
                        .tensor
                        .contiguous_values_as_f64()?
                        .to_vec();
                    let sum_val: f64 = incoming_vals.iter().sum();
                    let input_shape = self.nodes[input.0].tensor.meta().shape().to_vec();
                    let sum_node = self.leaf(vec![sum_val], input_shape, false)?;
                    self.cg_accumulate(input, &mut grad_nodes, sum_node)?;
                    Self::complete_dependency(&mut pending, input, &mut queue)?;
                    steps.push(TensorBackwardStep {
                        node: node_id,
                        incoming_grad_len: self.nodes[incoming_id.0].tensor.meta().numel(),
                        rule: "d(expand(x))/dx=sum(grad) (cg)",
                    });
                }
                // For unsupported ops, fall back to non-differentiable gradient
                _ => {
                    return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                        ft_dispatch::DispatchKeyError::IncompatibleSet {
                            reason: "create_graph not yet supported for this operation",
                        },
                    )));
                }
            }
        }

        // Extract gradient values and node IDs for leaves
        let num_original = orig_node_count;
        let gradients = (0..num_original)
            .map(|idx| {
                if self.nodes[idx].requires_grad && reachable[idx] {
                    grad_nodes[idx].map(|gid| {
                        self.nodes[gid.0]
                            .tensor
                            .contiguous_values_as_f64()
                            .unwrap_or_default()
                            .to_vec()
                    })
                } else {
                    None
                }
            })
            .collect();

        let mut gradient_node_results = vec![None; num_original];
        for (idx, gn) in grad_nodes.into_iter().enumerate() {
            if idx < num_original && self.nodes[idx].requires_grad && reachable[idx] {
                gradient_node_results[idx] = gn;
            }
        }

        // Persist leaf gradients
        for (idx, grad_opt) in gradient_node_results.iter().enumerate() {
            if let Some(gid) = grad_opt {
                if self.nodes[idx].op == TensorNodeOp::Leaf && self.nodes[idx].requires_grad {
                    let vals = self.nodes[gid.0]
                        .tensor
                        .contiguous_values_as_f64()
                        .unwrap_or_default()
                        .to_vec();
                    self.persistent_grads
                        .entry(idx)
                        .and_modify(|existing: &mut Vec<f64>| {
                            for (e, v) in existing.iter_mut().zip(vals.iter()) {
                                *e += v;
                            }
                        })
                        .or_insert(vals);
                }
            }
        }

        let telemetry = TensorSchedulerTelemetry {
            execution_order,
            queue_pushes: 0,
            queue_pops: 0,
            max_queue_len: 0,
            dependency_snapshot: vec![],
            reentrant_depth: options.current_reentrant_depth,
            reentrant_guard_triggered: false,
            hardened_fallback_used: false,
        };

        Ok(TensorBackwardReport {
            gradients,
            gradient_nodes: gradient_node_results,
            steps,
            telemetry,
        })
    }

    /// Helper: accumulate a gradient node contribution for create_graph.
    fn cg_accumulate(
        &mut self,
        target: TensorNodeId,
        grad_nodes: &mut [Option<TensorNodeId>],
        contribution: TensorNodeId,
    ) -> Result<(), AutogradError> {
        match grad_nodes.get(target.0).copied().flatten() {
            None => {
                if target.0 < grad_nodes.len() {
                    grad_nodes[target.0] = Some(contribution);
                }
            }
            Some(existing) => {
                let sum = self.cg_add(existing, contribution)?;
                grad_nodes[target.0] = Some(sum);
            }
        }
        Ok(())
    }

    /// Helper: elementwise add for create_graph backward.
    fn cg_add(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result, shape, dtype, device) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let lhs_data = lhs_node.tensor.contiguous_values_as_f64()?;
            let rhs_data = rhs_node.tensor.contiguous_values_as_f64()?;
            let shape = lhs_node.tensor.meta().shape().to_vec();
            let dtype = lhs_node.tensor.meta().dtype();
            let device = lhs_node.tensor.meta().device();
            let rg = lhs_node.requires_grad || rhs_node.requires_grad;

            // Handle broadcast: scalar grad + tensor
            let result = if lhs_data.len() == rhs_data.len() {
                lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| a + b).collect()
            } else if rhs_data.len() == 1 {
                lhs_data.iter().map(|&a| a + rhs_data[0]).collect()
            } else if lhs_data.len() == 1 {
                rhs_data.iter().map(|&b| lhs_data[0] + b).collect()
            } else {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "create_graph: shape mismatch in gradient add",
                    },
                )));
            };
            (rg, result, shape, dtype, device)
        };

        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(meta, result)?,
            requires_grad,
            op: TensorNodeOp::Add { lhs, rhs },
        });
        Ok(out)
    }

    /// Helper: elementwise sub for create_graph backward.
    fn cg_sub(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result, shape, dtype, device) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let lhs_data = lhs_node.tensor.contiguous_values_as_f64()?;
            let rhs_data = rhs_node.tensor.contiguous_values_as_f64()?;
            let shape = lhs_node.tensor.meta().shape().to_vec();
            let dtype = lhs_node.tensor.meta().dtype();
            let device = lhs_node.tensor.meta().device();
            let rg = lhs_node.requires_grad || rhs_node.requires_grad;
            let result = lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| a - b).collect();
            (rg, result, shape, dtype, device)
        };

        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(meta, result)?,
            requires_grad,
            op: TensorNodeOp::Sub { lhs, rhs },
        });
        Ok(out)
    }

    /// Helper: elementwise mul for create_graph backward.
    fn cg_mul(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result, shape, dtype, device) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let lhs_data = lhs_node.tensor.contiguous_values_as_f64()?;
            let rhs_data = rhs_node.tensor.contiguous_values_as_f64()?;
            let lhs_shape = lhs_node.tensor.meta().shape().to_vec();
            let rhs_shape = rhs_node.tensor.meta().shape().to_vec();
            let dtype = lhs_node.tensor.meta().dtype();
            let device = lhs_node.tensor.meta().device();
            let rg = lhs_node.requires_grad || rhs_node.requires_grad;

            // Handle broadcast: scalar * tensor
            let (result, shape) = if lhs_data.len() == rhs_data.len() {
                (lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| a * b).collect(), lhs_shape)
            } else if rhs_data.len() == 1 {
                (lhs_data.iter().map(|&a| a * rhs_data[0]).collect(), lhs_shape)
            } else if lhs_data.len() == 1 {
                (rhs_data.iter().map(|&b| lhs_data[0] * b).collect(), rhs_shape)
            } else {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "create_graph: shape mismatch in gradient mul",
                    },
                )));
            };
            (rg, result, shape, dtype, device)
        };

        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(meta, result)?,
            requires_grad,
            op: TensorNodeOp::Mul { lhs, rhs },
        });
        Ok(out)
    }

    /// Helper: elementwise div for create_graph backward.
    fn cg_div(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result, shape, dtype, device) = {
            let lhs_node = self.node(lhs)?;
            let rhs_node = self.node(rhs)?;
            let lhs_data = lhs_node.tensor.contiguous_values_as_f64()?;
            let rhs_data = rhs_node.tensor.contiguous_values_as_f64()?;
            let shape = lhs_node.tensor.meta().shape().to_vec();
            let dtype = lhs_node.tensor.meta().dtype();
            let device = lhs_node.tensor.meta().device();
            let rg = lhs_node.requires_grad || rhs_node.requires_grad;
            let result = lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| a / b).collect();
            (rg, result, shape, dtype, device)
        };

        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(meta, result)?,
            requires_grad,
            op: TensorNodeOp::Div { lhs, rhs },
        });
        Ok(out)
    }

    /// Helper: elementwise neg for create_graph backward.
    fn cg_neg(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result, shape, dtype, device) = {
            let node = self.node(input)?;
            let data = node.tensor.contiguous_values_as_f64()?;
            let shape = node.tensor.meta().shape().to_vec();
            let dtype = node.tensor.meta().dtype();
            let device = node.tensor.meta().device();
            let result = data.iter().map(|&v| -v).collect();
            (node.requires_grad, result, shape, dtype, device)
        };

        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(meta, result)?,
            requires_grad,
            op: TensorNodeOp::Neg { input },
        });
        Ok(out)
    }

    /// Helper: elementwise sin for create_graph backward.
    fn cg_sin(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result, shape, dtype, device) = {
            let node = self.node(input)?;
            let data = node.tensor.contiguous_values_as_f64()?;
            let shape = node.tensor.meta().shape().to_vec();
            let dtype = node.tensor.meta().dtype();
            let device = node.tensor.meta().device();
            let result = data.iter().map(|&v| v.sin()).collect();
            (node.requires_grad, result, shape, dtype, device)
        };

        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(meta, result)?,
            requires_grad,
            op: TensorNodeOp::Sin { input },
        });
        Ok(out)
    }

    /// Helper: elementwise cos for create_graph backward.
    fn cg_cos(&mut self, input: TensorNodeId) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result, shape, dtype, device) = {
            let node = self.node(input)?;
            let data = node.tensor.contiguous_values_as_f64()?;
            let shape = node.tensor.meta().shape().to_vec();
            let dtype = node.tensor.meta().dtype();
            let device = node.tensor.meta().device();
            let result = data.iter().map(|&v| v.cos()).collect();
            (node.requires_grad, result, shape, dtype, device)
        };

        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(meta, result)?,
            requires_grad,
            op: TensorNodeOp::Cos { input },
        });
        Ok(out)
    }

    /// Helper: elementwise pow for create_graph backward.
    fn cg_pow(
        &mut self,
        base: TensorNodeId,
        exponent: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, result, shape, dtype, device, exp_val) = {
            let base_node = self.node(base)?;
            let exp_node = self.node(exponent)?;
            let base_data = base_node.tensor.contiguous_values_as_f64()?;
            let exp_data = exp_node.tensor.contiguous_values_as_f64()?;
            let shape = base_node.tensor.meta().shape().to_vec();
            let dtype = base_node.tensor.meta().dtype();
            let device = base_node.tensor.meta().device();
            let rg = base_node.requires_grad || exp_node.requires_grad;
            let exp_val = if exp_data.len() == 1 { exp_data[0] } else { exp_data[0] };
            let result: Vec<f64> = base_data.iter().zip(exp_data.iter()).map(|(&b, &e)| b.powf(e)).collect();
            (rg, result, shape, dtype, device, exp_val)
        };

        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(meta, result)?,
            requires_grad,
            op: TensorNodeOp::Pow {
                input: base,
                exponent: exp_val,
            },
        });
        Ok(out)
    }

    /// Helper: expand a scalar node to a larger shape for create_graph backward.
    /// Creates an Expand node so the second backward can sum gradients back.
    fn cg_expand(
        &mut self,
        input: TensorNodeId,
        expanded_data: Vec<f64>,
        shape: Vec<usize>,
    ) -> Result<TensorNodeId, AutogradError> {
        let (requires_grad, original_shape, dtype, device) = {
            let node = self.node(input)?;
            (
                node.requires_grad,
                node.tensor.meta().shape().to_vec(),
                node.tensor.meta().dtype(),
                node.tensor.meta().device(),
            )
        };

        let meta = ft_core::TensorMeta::from_shape(shape, dtype, device);
        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(meta, expanded_data)?,
            requires_grad,
            op: TensorNodeOp::Expand { input, original_shape },
        });
        Ok(out)
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
                | TensorNodeOp::Max { lhs, rhs }
                | TensorNodeOp::Atan2 { lhs, rhs }
                | TensorNodeOp::Fmod { lhs, rhs }
                | TensorNodeOp::Remainder { lhs, rhs } => {
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
                | TensorNodeOp::Rsqrt { input }
                | TensorNodeOp::Erf { input }
                | TensorNodeOp::Erfc { input }
                | TensorNodeOp::Hardswish { input }
                | TensorNodeOp::Hardsigmoid { input }
                | TensorNodeOp::Hardtanh { input }
                | TensorNodeOp::Softplus { input }
                | TensorNodeOp::Mish { input }
                | TensorNodeOp::Square { input }
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
                | TensorNodeOp::Norm { input, .. }
                | TensorNodeOp::NormDim { input, .. }
                | TensorNodeOp::CumSum { input, .. }
                | TensorNodeOp::CumProd { input, .. }
                | TensorNodeOp::Softmax { input, .. }
                | TensorNodeOp::LogSoftmax { input, .. }
                | TensorNodeOp::Reshape { input, .. }
                | TensorNodeOp::View { input, .. }
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
                | TensorNodeOp::ScatterAdd { input, .. }
                | TensorNodeOp::IndexPut { input, .. }
                | TensorNodeOp::Sort { input, .. }
                | TensorNodeOp::TopK { input, .. }
                | TensorNodeOp::Flip { input, .. }
                | TensorNodeOp::Repeat { input, .. }
                | TensorNodeOp::Roll { input, .. }
                | TensorNodeOp::Pad { input, .. }
                | TensorNodeOp::CastF32 { input }
                | TensorNodeOp::CastF64 { input } => {
                    stack.push(input);
                }
                TensorNodeOp::Cat { ref inputs, .. } | TensorNodeOp::Stack { ref inputs, .. } => {
                    for &id in inputs {
                        stack.push(id);
                    }
                }
                TensorNodeOp::CustomFunction { ref inputs, .. } => {
                    for &id in inputs {
                        stack.push(id);
                    }
                }
                TensorNodeOp::Where { condition, x, y } => {
                    stack.push(condition);
                    stack.push(x);
                    stack.push(y);
                }
                TensorNodeOp::Lerp { start, end, .. } => {
                    stack.push(start);
                    stack.push(end);
                }
                TensorNodeOp::Addmm {
                    input, mat1, mat2, ..
                } => {
                    stack.push(input);
                    stack.push(mat1);
                    stack.push(mat2);
                }
                TensorNodeOp::Addmv {
                    input, mat, vec: v, ..
                } => {
                    stack.push(input);
                    stack.push(mat);
                    stack.push(v);
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
                | TensorNodeOp::Max { lhs, rhs }
                | TensorNodeOp::Atan2 { lhs, rhs }
                | TensorNodeOp::Fmod { lhs, rhs }
                | TensorNodeOp::Remainder { lhs, rhs } => {
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
                | TensorNodeOp::Rsqrt { input }
                | TensorNodeOp::Erf { input }
                | TensorNodeOp::Erfc { input }
                | TensorNodeOp::Hardswish { input }
                | TensorNodeOp::Hardsigmoid { input }
                | TensorNodeOp::Hardtanh { input }
                | TensorNodeOp::Softplus { input }
                | TensorNodeOp::Mish { input }
                | TensorNodeOp::Square { input }
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
                | TensorNodeOp::Norm { input, .. }
                | TensorNodeOp::NormDim { input, .. }
                | TensorNodeOp::CumSum { input, .. }
                | TensorNodeOp::CumProd { input, .. }
                | TensorNodeOp::Softmax { input, .. }
                | TensorNodeOp::LogSoftmax { input, .. }
                | TensorNodeOp::Reshape { input, .. }
                | TensorNodeOp::View { input, .. }
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
                | TensorNodeOp::ScatterAdd { input, .. }
                | TensorNodeOp::IndexPut { input, .. }
                | TensorNodeOp::Sort { input, .. }
                | TensorNodeOp::TopK { input, .. }
                | TensorNodeOp::Flip { input, .. }
                | TensorNodeOp::Repeat { input, .. }
                | TensorNodeOp::Roll { input, .. }
                | TensorNodeOp::Pad { input, .. }
                | TensorNodeOp::CastF32 { input }
                | TensorNodeOp::CastF64 { input } => {
                    pending[input.0] = pending[input.0].saturating_add(1);
                }
                TensorNodeOp::Cat { ref inputs, .. } | TensorNodeOp::Stack { ref inputs, .. } => {
                    for &id in inputs {
                        pending[id.0] = pending[id.0].saturating_add(1);
                    }
                }
                TensorNodeOp::CustomFunction { ref inputs, .. } => {
                    for &id in inputs {
                        pending[id.0] = pending[id.0].saturating_add(1);
                    }
                }
                TensorNodeOp::Where { condition, x, y } => {
                    pending[condition.0] = pending[condition.0].saturating_add(1);
                    pending[x.0] = pending[x.0].saturating_add(1);
                    pending[y.0] = pending[y.0].saturating_add(1);
                }
                TensorNodeOp::Lerp { start, end, .. } => {
                    pending[start.0] = pending[start.0].saturating_add(1);
                    pending[end.0] = pending[end.0].saturating_add(1);
                }
                TensorNodeOp::Addmm {
                    input, mat1, mat2, ..
                } => {
                    pending[input.0] = pending[input.0].saturating_add(1);
                    pending[mat1.0] = pending[mat1.0].saturating_add(1);
                    pending[mat2.0] = pending[mat2.0].saturating_add(1);
                }
                TensorNodeOp::Addmv {
                    input, mat, vec: v, ..
                } => {
                    pending[input.0] = pending[input.0].saturating_add(1);
                    pending[mat.0] = pending[mat.0].saturating_add(1);
                    pending[v.0] = pending[v.0].saturating_add(1);
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

    fn normalize_repeat_shape(
        input_shape: &[usize],
        repeats: &[usize],
    ) -> Result<Vec<usize>, AutogradError> {
        if repeats.len() < input_shape.len() {
            return Err(AutogradError::Dispatch(DispatchError::Kernel(
                ft_kernel_cpu::KernelError::ShapeMismatch {
                    lhs: input_shape.to_vec(),
                    rhs: repeats.to_vec(),
                },
            )));
        }

        let leading_dims = repeats.len() - input_shape.len();
        let mut normalized_shape = vec![1usize; leading_dims];
        normalized_shape.extend_from_slice(input_shape);
        Ok(normalized_shape)
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

    fn normalize_wrapped_index_float(
        idx_f: f64,
        dim_size: usize,
        invalid_reason: &'static str,
        oob_reason: &'static str,
        overflow_reason: &'static str,
    ) -> Result<usize, AutogradError> {
        if !idx_f.is_finite() || idx_f.fract().abs() > f64::EPSILON {
            return Err(AutogradError::Dispatch(
                DispatchKeyError::IncompatibleSet {
                    reason: invalid_reason,
                }
                .into(),
            ));
        }
        if idx_f < isize::MIN as f64 || idx_f > isize::MAX as f64 {
            return Err(AutogradError::Dispatch(
                DispatchKeyError::IncompatibleSet {
                    reason: invalid_reason,
                }
                .into(),
            ));
        }

        let dim_size_i =
            isize::try_from(dim_size).map_err(|_| Self::shape_overflow_error(overflow_reason))?;
        let mut idx_i = idx_f as isize;
        if idx_i < 0 {
            idx_i += dim_size_i;
        }
        if idx_i < 0 || idx_i >= dim_size_i {
            return Err(AutogradError::Dispatch(
                DispatchKeyError::IncompatibleSet { reason: oob_reason }.into(),
            ));
        }

        usize::try_from(idx_i).map_err(|_| Self::shape_overflow_error(overflow_reason))
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

    fn apply_tensor_hooks(
        &self,
        node: TensorNodeId,
        incoming: &[f64],
    ) -> Result<Vec<f64>, AutogradError> {
        let Some(hooks) = self.tensor_hooks.get(&node.0) else {
            return Ok(incoming.to_vec());
        };
        let mut current = incoming.to_vec();
        for hook in hooks {
            if let Some(updated) = (hook.callback)(current.as_slice())? {
                Self::ensure_tensor_len(node, current.len(), updated.len())?;
                current = updated;
            }
        }
        Ok(current)
    }

    fn accumulate_persistent_gradients(
        &mut self,
        gradients: &[Option<Vec<f64>>],
    ) -> Result<(), AutogradError> {
        for (idx, gradient) in gradients.iter().enumerate() {
            let Some(gradient) = gradient.as_deref() else {
                continue;
            };
            let node = TensorNodeId(idx);
            match self.persistent_grads.get_mut(&idx) {
                Some(existing) => Self::accumulate_tensor_gradient(node, existing, gradient)?,
                None => {
                    self.persistent_grads.insert(idx, gradient.to_vec());
                }
            }
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
    use std::sync::{Arc, Mutex};

    use ft_core::{DType, DenseTensor, DenseTensorError, Device, ExecutionMode, TensorMeta};
    use ft_dispatch::DispatchError;
    use proptest::prelude::*;

    use super::{
        AutogradError, BackwardOptions, NodeId, ReentrantPolicy, SchedulerTelemetry, Tape,
        TensorHookHandle, TensorNode, TensorNodeId, TensorNodeOp, TensorTape,
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

    // ---- grad_enabled tests (bd-3dpn.1) ----

    #[test]
    fn tape_grad_enabled_default_is_true() {
        let tape = Tape::new();
        assert!(tape.is_grad_enabled());
    }

    #[test]
    fn tape_set_grad_enabled() {
        let mut tape = Tape::new();
        tape.set_grad_enabled(false);
        assert!(!tape.is_grad_enabled());
        tape.set_grad_enabled(true);
        assert!(tape.is_grad_enabled());
    }

    #[test]
    fn tape_leaf_respects_grad_disabled() {
        let mut tape = Tape::new();
        tape.set_grad_enabled(false);
        let x = tape.leaf(2.0, true); // requests grad but disabled
        // backward should fail because the node has requires_grad=false
        let err = tape.backward(x);
        assert!(err.is_err());
    }

    #[test]
    fn tape_ops_respect_grad_disabled() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true); // grad enabled
        tape.set_grad_enabled(false);
        let (y, _) = tape.add(x, x, ExecutionMode::Strict).expect("add");
        tape.set_grad_enabled(true);
        // y was created with grad disabled, so backward from y fails
        let err = tape.backward(y);
        assert!(err.is_err());
    }

    #[test]
    fn tensor_tape_grad_enabled_default_is_true() {
        let tape = TensorTape::new();
        assert!(tape.is_grad_enabled());
    }

    #[test]
    fn tensor_tape_leaf_respects_grad_disabled() {
        let mut tape = TensorTape::new();
        tape.set_grad_enabled(false);
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("leaf");
        tape.set_grad_enabled(true);
        let err = tape.backward(x);
        assert!(err.is_err());
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
    fn tensor_backward_dot_rejects_incoming_gradient_len_mismatch() {
        let mut tape = TensorTape::new();
        let lhs = tape
            .leaf(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("lhs leaf should build");
        let rhs = tape
            .leaf(vec![4.0, 5.0, 6.0], vec![3], true)
            .expect("rhs leaf should build");
        let (out, _) = tape
            .dot(lhs, rhs, ExecutionMode::Strict)
            .expect("dot should succeed");
        tape.nodes[out.0].tensor = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu),
            vec![1.0, 2.0],
        )
        .expect("replacement tensor should build");

        let err = tape
            .backward(out)
            .expect_err("incoming gradient shape mismatch should fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorGradientShapeMismatch { node, expected: 1, actual: 2 }
            if node == out
        ));
    }

    #[test]
    fn tensor_backward_bmm_rejects_incoming_gradient_len_mismatch() {
        let mut tape = TensorTape::new();
        let lhs = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2], true)
            .expect("lhs leaf should build");
        let rhs = tape
            .leaf(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2], true)
            .expect("rhs leaf should build");
        let (out, _) = tape
            .bmm(lhs, rhs, ExecutionMode::Strict)
            .expect("bmm should succeed");
        tape.nodes[out.0].tensor = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![1, 1, 2], DType::F64, Device::Cpu),
            vec![1.0, 2.0],
        )
        .expect("replacement tensor should build");

        let err = tape
            .backward(out)
            .expect_err("incoming gradient shape mismatch should fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorGradientShapeMismatch { .. }
        ));
    }

    #[test]
    fn tensor_backward_index_select_rejects_huge_index_value() {
        let mut tape = TensorTape::new();
        let input = tape
            .leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("input leaf should build");
        let out = tape
            .index_select(input, 0, &[0.0, 1.0])
            .expect("index_select should succeed");
        assert!(matches!(
            tape.nodes[out.0].op,
            TensorNodeOp::IndexSelect { .. }
        ));
        if let TensorNodeOp::IndexSelect { indices, .. } = &mut tape.nodes[out.0].op {
            indices[0] = 1.0e300;
        }

        let err = tape
            .backward(out)
            .expect_err("huge index should fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "index_select backward received invalid index value"
                }
            ))
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
    fn tensor_where_with_offset_views_uses_each_operand_layout() {
        let mut tape = TensorTape::new();
        let cond = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu),
            vec![1.0, 0.0, 1.0],
        )
        .expect("cond should build");
        let x = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(2),
            vec![0.0, 0.0, 10.0, 20.0, 30.0],
        )
        .expect("x offset view should build");
        let y = DenseTensor::from_storage(
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1),
            vec![0.0, -1.0, -2.0, -3.0],
        )
        .expect("y offset view should build");

        let cond_node = tape.leaf_tensor(cond, false);
        let x_node = tape.leaf_tensor(x, false);
        let y_node = tape.leaf_tensor(y, false);

        let out_node = tape
            .tensor_where(cond_node, x_node, y_node)
            .expect("where should succeed with offset views");
        let out = tape.tensor(out_node).expect("output should resolve");
        assert_eq!(
            out.dispatch_values().expect("where output values"),
            &[10.0, -2.0, 30.0]
        );
    }

    #[test]
    fn tensor_where_rejects_shape_mismatch_fail_closed() {
        let mut tape = TensorTape::new();
        let cond = tape
            .leaf(vec![1.0, 0.0], vec![2], false)
            .expect("cond should build");
        let x = tape
            .leaf(vec![10.0, 20.0, 30.0], vec![3], false)
            .expect("x should build");
        let y = tape
            .leaf(vec![-1.0, -2.0, -3.0], vec![3], false)
            .expect("y should build");

        let err = tape
            .tensor_where(cond, x, y)
            .expect_err("shape mismatch must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "where requires condition, x, and y to have the same shape"
                }
            ))
        ));
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

        let report = tape
            .backward_with_options(
                out,
                BackwardOptions::strict_default().with_retain_graph(true),
            )
            .expect("backward should succeed");
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
                    retain_graph: false,
                    create_graph: false,
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
                    retain_graph: false,
                    create_graph: false,
                },
            )
            .expect("hardened overflow should fallback");

        assert!(report.telemetry.reentrant_guard_triggered);
        assert!(report.telemetry.hardened_fallback_used);
    }

    #[test]
    fn unknown_node_returns_error() {
        let mut tape = Tape::new();
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

            let first = tape
                .backward_with_options(out, BackwardOptions::strict_default().with_retain_graph(true))
                .expect("backward should succeed");
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
                    retain_graph: false,
                    create_graph: false,
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
                        retain_graph: false,
                        create_graph: false,
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
        let mut tape = TensorTape::new();
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

    // ---- Graph consumption tests (bd-3dpn.2) ----

    #[test]
    fn default_backward_consumes_scalar_graph() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (z, _) = tape.add(x, y, ExecutionMode::Strict).expect("add");
        let report = tape.backward(z).expect("first backward should succeed");
        assert_eq!(report.gradient(x), Some(1.0));
        assert_eq!(report.gradient(y), Some(1.0));
        assert!(
            tape.consumed,
            "tape should be consumed after default backward"
        );
    }

    #[test]
    fn second_backward_on_consumed_scalar_graph_returns_error() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (z, _) = tape.add(x, y, ExecutionMode::Strict).expect("add");
        let _ = tape.backward(z).expect("first backward should succeed");
        let err = tape
            .backward(z)
            .expect_err("second backward on consumed graph should fail");
        assert!(matches!(err, AutogradError::GraphConsumed));
        assert!(
            err.to_string().contains("already consumed"),
            "error message should mention consumption: {}",
            err
        );
    }

    #[test]
    fn retain_graph_allows_second_scalar_backward() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0, true);
        let y = tape.leaf(3.0, true);
        let (sum, _) = tape.add(x, y, ExecutionMode::Strict).expect("add");
        let (out, _) = tape.mul(sum, x, ExecutionMode::Strict).expect("mul");

        let opts = BackwardOptions::strict_default().with_retain_graph(true);
        let first = tape
            .backward_with_options(out, opts)
            .expect("first backward with retain_graph=true");
        assert!(
            !tape.consumed,
            "tape should NOT be consumed when retain_graph=true"
        );

        let second = tape.backward(out).expect("second backward should succeed");
        assert_eq!(first.gradient(x), second.gradient(x));
        assert_eq!(first.gradient(y), second.gradient(y));
        assert!(
            tape.consumed,
            "tape should be consumed after default backward"
        );
    }

    #[test]
    fn default_backward_consumes_tensor_graph() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let y = tape.leaf(vec![3.0, 4.0], vec![2], true).expect("y");
        let (z, _) = tape.add(x, y, ExecutionMode::Strict).expect("add");
        let report = tape.backward(z).expect("first backward should succeed");
        assert!(report.gradient(x).is_some());
        assert!(
            tape.consumed,
            "tensor tape should be consumed after default backward"
        );
    }

    #[test]
    fn second_backward_on_consumed_tensor_graph_returns_error() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let y = tape.leaf(vec![3.0, 4.0], vec![2], true).expect("y");
        let (z, _) = tape.add(x, y, ExecutionMode::Strict).expect("add");
        let _ = tape.backward(z).expect("first backward should succeed");
        let err = tape
            .backward(z)
            .expect_err("second backward on consumed tensor graph should fail");
        assert!(matches!(err, AutogradError::TensorGraphConsumed));
        assert!(
            err.to_string().contains("already consumed"),
            "error message should mention consumption: {}",
            err
        );
    }

    #[test]
    fn retain_graph_allows_second_tensor_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let y = tape.leaf(vec![3.0, 4.0], vec![2], true).expect("y");
        let (z, _) = tape.add(x, y, ExecutionMode::Strict).expect("add");

        let opts = BackwardOptions::strict_default().with_retain_graph(true);
        let first = tape
            .backward_with_options(z, opts)
            .expect("first backward with retain_graph=true");
        assert!(
            !tape.consumed,
            "tensor tape should NOT be consumed when retain_graph=true"
        );

        let second = tape.backward(z).expect("second backward should succeed");
        assert_eq!(first.gradient(x), second.gradient(x));
        assert_eq!(first.gradient(y), second.gradient(y));
        assert!(
            tape.consumed,
            "tensor tape should be consumed after default backward"
        );
    }

    #[test]
    fn gan_style_shared_graph_two_backward_passes() {
        // Simulates GAN training: discriminator and generator share part of the graph.
        // D_loss backward with retain_graph=true, then G_loss backward.
        let mut tape = Tape::new();

        // Shared "generated output" node
        let z = tape.leaf(0.5, true);
        // Discriminator path: D_loss = -log(1 - z)
        let one = tape.leaf(1.0, false);
        let (one_minus_z, _) = tape.sub(one, z, ExecutionMode::Strict).expect("sub");
        let (d_loss, _) = tape.neg(one_minus_z, ExecutionMode::Strict).expect("neg");

        // Generator path: G_loss = -log(z) ≈ -z for testing (use neg as proxy)
        let (g_loss, _) = tape.neg(z, ExecutionMode::Strict).expect("neg");

        // First backward (discriminator) with retain_graph=true
        let d_report = tape
            .backward_with_options(
                d_loss,
                BackwardOptions::strict_default().with_retain_graph(true),
            )
            .expect("D backward with retain_graph=true should succeed");
        let d_grad_z = d_report.gradient(z).expect("z gradient from D");

        // Second backward (generator) consumes the graph
        let g_report = tape
            .backward(g_loss)
            .expect("G backward should succeed after retained graph");
        let g_grad_z = g_report.gradient(z).expect("z gradient from G");

        // d(-(1-z))/dz = 1.0, d(-z)/dz = -1.0
        assert!((d_grad_z - 1.0).abs() < 1e-12, "D grad z = {d_grad_z}");
        assert!((g_grad_z - (-1.0)).abs() < 1e-12, "G grad z = {g_grad_z}");

        // Graph is now consumed — third backward should fail
        let err = tape
            .backward(g_loss)
            .expect_err("third backward on consumed graph should fail");
        assert!(matches!(err, AutogradError::GraphConsumed));
    }

    #[test]
    fn graph_consumption_reduces_memory_footprint() {
        // Verify that consuming the graph marks the tape as consumed,
        // which prevents further backward computation (the memory optimization).
        // We measure the tape size before and after backward to confirm consumption.
        let mut tape = TensorTape::new();
        let n = 1000;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x = tape.leaf(data.clone(), vec![n], true).expect("x");
        let y = tape.leaf(data, vec![n], true).expect("y");
        let (sum, _) = tape.add(x, y, ExecutionMode::Strict).expect("add");
        let (out, _) = tape.mul(sum, x, ExecutionMode::Strict).expect("mul");

        let node_count_before = tape.node_count();
        assert_eq!(node_count_before, 4, "should have 4 nodes: x, y, sum, out");

        let report = tape.backward(out).expect("backward");
        assert!(tape.consumed, "tape should be consumed");
        assert!(report.gradient(x).is_some());
        assert!(report.gradient(y).is_some());

        // Verify the graph is consumed — backward fails
        let err = tape
            .backward(out)
            .expect_err("consumed graph should reject backward");
        assert!(matches!(err, AutogradError::TensorGraphConsumed));

        // Node count is unchanged (nodes aren't removed, just marked consumed)
        assert_eq!(tape.node_count(), node_count_before);
    }

    // ── f32 dtype tests ──────────────────────────────────────────────

    #[test]
    fn f32_leaf_and_values_roundtrip() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], false)
            .unwrap();
        assert_eq!(tape.dtype(a).unwrap(), DType::F32);
        let vals = tape.values_f32(a).unwrap();
        assert_eq!(vals, vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn f32_unary_neg_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![1.0f32, -2.0, 3.0], vec![3], false)
            .unwrap();
        let (out, _) = tape.neg(a, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(out).unwrap(), DType::F32);
        let vals = tape.values_f32(out).unwrap();
        assert_eq!(vals, vec![-1.0f32, 2.0, -3.0]);
    }

    #[test]
    fn f32_unary_abs_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![-1.0f32, 2.0, -3.0], vec![3], false)
            .unwrap();
        let (out, _) = tape.abs(a, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(out).unwrap(), DType::F32);
        let vals = tape.values_f32(out).unwrap();
        assert_eq!(vals, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn f32_binary_add_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape.leaf_f32(vec![1.0f32, 2.0], vec![2], false).unwrap();
        let b = tape.leaf_f32(vec![3.0f32, 4.0], vec![2], false).unwrap();
        let (out, _) = tape.add(a, b, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(out).unwrap(), DType::F32);
        let vals = tape.values_f32(out).unwrap();
        assert_eq!(vals, vec![4.0f32, 6.0]);
    }

    #[test]
    fn f32_binary_mul_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape.leaf_f32(vec![2.0f32, 3.0], vec![2], false).unwrap();
        let b = tape.leaf_f32(vec![4.0f32, 5.0], vec![2], false).unwrap();
        let (out, _) = tape.mul(a, b, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(out).unwrap(), DType::F32);
        let vals = tape.values_f32(out).unwrap();
        assert_eq!(vals, vec![8.0f32, 15.0]);
    }

    #[test]
    fn f32_mixed_binary_promotes_to_f64() {
        let mut tape = TensorTape::new();
        let a = tape.leaf_f32(vec![1.0f32, 2.0], vec![2], false).unwrap();
        let b = tape.leaf(vec![3.0f64, 4.0], vec![2], false).unwrap();
        let (out, _) = tape.add(a, b, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(out).unwrap(), DType::F64);
        let vals = tape.values(out).unwrap();
        assert_eq!(vals, vec![4.0f64, 6.0]);
    }

    #[test]
    fn f32_cast_roundtrip() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.5f64, 2.5, 3.5], vec![3], false).unwrap();
        assert_eq!(tape.dtype(a).unwrap(), DType::F64);

        let b = tape.to_f32(a).unwrap();
        assert_eq!(tape.dtype(b).unwrap(), DType::F32);
        let vals32 = tape.values_f32(b).unwrap();
        assert_eq!(vals32, vec![1.5f32, 2.5, 3.5]);

        let c = tape.to_f64(b).unwrap();
        assert_eq!(tape.dtype(c).unwrap(), DType::F64);
        let vals64 = tape.values(c).unwrap();
        assert_eq!(vals64, vec![1.5f64, 2.5, 3.5]);
    }

    #[test]
    fn f32_cast_noop_same_dtype() {
        let mut tape = TensorTape::new();
        let a = tape.leaf_f32(vec![1.0f32, 2.0], vec![2], false).unwrap();
        let b = tape.to_f32(a).unwrap();
        assert_eq!(a, b); // same node — no-op
    }

    #[test]
    fn f32_backward_through_unary() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![1.0f32, 2.0, 3.0], vec![3], true)
            .unwrap();
        let (b, _) = tape.neg(a, ExecutionMode::Strict).unwrap();
        let (c, _) = tape.sum(b, ExecutionMode::Strict).unwrap();
        let report = tape.backward(c).unwrap();
        // d(sum(neg(x)))/dx = -1 for each element
        let grad = report.gradient(a).unwrap();
        assert_eq!(grad, &[-1.0, -1.0, -1.0]);
    }

    #[test]
    fn f32_backward_through_cast() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0f64, 2.0, 3.0], vec![3], true).unwrap();
        let b = tape.to_f32(a).unwrap();
        let c = tape.to_f64(b).unwrap();
        let (d, _) = tape.sum(c, ExecutionMode::Strict).unwrap();
        let report = tape.backward(d).unwrap();
        // Cast is identity for gradients, sum gradient is all 1s
        let grad = report.gradient(a).unwrap();
        assert_eq!(grad, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn tensor_requires_grad_and_grad_fn_introspection() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        assert!(tape.tensor_requires_grad(x).expect("requires_grad"));
        assert!(tape.tensor_is_leaf(x).expect("is_leaf"));
        assert!(tape.tensor_grad_fn(x).expect("grad_fn").is_none());

        let (y, _) = tape.neg(x, ExecutionMode::Strict).expect("neg");
        assert!(tape.tensor_requires_grad(y).expect("requires_grad"));
        assert!(!tape.tensor_is_leaf(y).expect("is_leaf"));
        assert_eq!(
            tape.tensor_grad_fn(y).expect("grad_fn"),
            Some("Neg".to_string())
        );
    }

    #[test]
    fn tensor_requires_grad_toggle_on_leaf_roundtrip() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], false).expect("x");
        assert!(!tape.tensor_requires_grad(x).expect("requires_grad"));

        tape.set_tensor_requires_grad(x, true)
            .expect("set requires_grad true");
        assert!(tape.tensor_requires_grad(x).expect("requires_grad"));

        tape.set_tensor_requires_grad(x, false)
            .expect("set requires_grad false");
        assert!(!tape.tensor_requires_grad(x).expect("requires_grad"));
    }

    #[test]
    fn tensor_requires_grad_toggle_rejects_non_leaf() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let (y, _) = tape.neg(x, ExecutionMode::Strict).expect("neg");

        let err = tape
            .set_tensor_requires_grad(y, false)
            .expect_err("non-leaf requires_grad_ must fail");
        assert!(matches!(
            err,
            AutogradError::TensorRequiresGradNonLeaf { .. }
        ));
    }

    #[test]
    fn tensor_detach_in_place_turns_non_leaf_into_leaf() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let (y, _) = tape.neg(x, ExecutionMode::Strict).expect("neg");

        assert!(!tape.tensor_is_leaf(y).expect("is_leaf"));
        assert!(tape.tensor_requires_grad(y).expect("requires_grad"));

        tape.detach_tensor_in_place(y).expect("detach_");

        assert!(tape.tensor_is_leaf(y).expect("is_leaf"));
        assert!(!tape.tensor_requires_grad(y).expect("requires_grad"));
        assert!(tape.tensor_grad_fn(y).expect("grad_fn").is_none());

        let (z, _) = tape.add(x, y, ExecutionMode::Strict).expect("add");
        let report = tape.backward(z).expect("backward");
        assert!(report.gradient(x).is_some(), "x should keep gradient flow");
        assert!(
            report.gradient(y).is_none(),
            "detached node should not receive gradients"
        );
    }

    #[test]
    fn tensor_detach_in_place_on_leaf_is_noop() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], false).expect("x");
        tape.detach_tensor_in_place(x).expect("detach leaf");
        assert!(tape.tensor_is_leaf(x).expect("is_leaf"));
        assert!(!tape.tensor_requires_grad(x).expect("requires_grad"));
        assert!(tape.tensor_grad_fn(x).expect("grad_fn").is_none());
    }

    #[test]
    fn tensor_register_hook_modifies_gradient() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let (y, _) = tape.neg(x, ExecutionMode::Strict).expect("neg");
        tape.register_tensor_hook(y, |grad| {
            Ok(Some(
                grad.iter().map(|value| value * 2.0).collect::<Vec<_>>(),
            ))
        })
        .expect("register hook");

        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(y).expect("grad y"), &[2.0, 2.0]);
        assert_eq!(report.gradient(x).expect("grad x"), &[-2.0, -2.0]);
    }

    #[test]
    fn tensor_register_hook_chains_in_registration_order() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let (y, _) = tape.neg(x, ExecutionMode::Strict).expect("neg");

        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let order_first = Arc::clone(&order);
        tape.register_tensor_hook(y, move |grad| {
            order_first.lock().expect("order lock").push("first");
            Ok(Some(
                grad.iter().map(|value| value + 1.0).collect::<Vec<_>>(),
            ))
        })
        .expect("register first");
        let order_second = Arc::clone(&order);
        tape.register_tensor_hook(y, move |grad| {
            order_second.lock().expect("order lock").push("second");
            Ok(Some(
                grad.iter().map(|value| value * 2.0).collect::<Vec<_>>(),
            ))
        })
        .expect("register second");

        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(y).expect("grad y"), &[4.0, 4.0]);
        assert_eq!(report.gradient(x).expect("grad x"), &[-4.0, -4.0]);
        let order_values = order.lock().expect("order lock");
        assert_eq!(order_values.as_slice(), &["first", "second"]);
    }

    #[test]
    fn tensor_remove_hook_disables_callback() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let (y, _) = tape.neg(x, ExecutionMode::Strict).expect("neg");

        let handle: TensorHookHandle = tape
            .register_tensor_hook(y, |grad| {
                Ok(Some(
                    grad.iter().map(|value| value * 3.0).collect::<Vec<_>>(),
                ))
            })
            .expect("register hook");
        assert!(tape.remove_tensor_hook(handle).expect("remove hook"));
        assert!(
            !tape
                .remove_tensor_hook(handle)
                .expect("second remove is no-op")
        );

        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(y).expect("grad y"), &[1.0, 1.0]);
        assert_eq!(report.gradient(x).expect("grad x"), &[-1.0, -1.0]);
    }

    #[test]
    fn tensor_register_hook_none_is_observation_only() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let (y, _) = tape.neg(x, ExecutionMode::Strict).expect("neg");
        tape.register_tensor_hook(y, |_grad| Ok(None))
            .expect("register hook");

        let report = tape.backward(y).expect("backward");
        assert_eq!(report.gradient(y).expect("grad y"), &[1.0, 1.0]);
        assert_eq!(report.gradient(x).expect("grad x"), &[-1.0, -1.0]);
    }

    #[test]
    fn tensor_register_hook_error_propagates() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let (y, _) = tape.neg(x, ExecutionMode::Strict).expect("neg");
        tape.register_tensor_hook(y, |_grad| Err(AutogradError::TensorGraphConsumed))
            .expect("register hook");

        let err = tape.backward(y).expect_err("hook error should propagate");
        assert!(matches!(err, AutogradError::TensorGraphConsumed));
    }

    #[test]
    fn tensor_register_hook_rejects_shape_mismatch() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let (y, _) = tape.neg(x, ExecutionMode::Strict).expect("neg");
        tape.register_tensor_hook(y, |_grad| Ok(Some(vec![1.0])))
            .expect("register hook");

        let err = tape
            .backward(y)
            .expect_err("hook shape mismatch must fail closed");
        assert!(matches!(
            err,
            AutogradError::TensorGradientShapeMismatch { .. }
        ));
    }

    // ---- CustomFunction / apply_function tests ----

    #[test]
    fn custom_function_identity_forward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("x");

        let y = tape
            .apply_function(
                &[x],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    Ok((vals.to_vec(), shape.to_vec()))
                },
                |_ctx, grad_outputs| Ok(vec![Some(grad_outputs[0].to_vec())]),
            )
            .expect("identity function");

        let vals = tape.values(y).expect("values");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn custom_function_identity_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("x");

        let y = tape
            .apply_function(
                &[x],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    Ok((vals.to_vec(), shape.to_vec()))
                },
                |_ctx, grad_outputs| Ok(vec![Some(grad_outputs[0].to_vec())]),
            )
            .expect("identity function");

        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("gradient exists");
        assert_eq!(grad, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn custom_function_double_forward_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![2.0, 3.0, 5.0], vec![3], true).expect("x");

        // f(x) = 2*x, df/dx = 2
        let y = tape
            .apply_function(
                &[x],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    let doubled: Vec<f64> = vals.iter().map(|v| v * 2.0).collect();
                    Ok((doubled, shape.to_vec()))
                },
                |_ctx, grad_outputs| {
                    let grad: Vec<f64> = grad_outputs[0].iter().map(|g| g * 2.0).collect();
                    Ok(vec![Some(grad)])
                },
            )
            .expect("double function");

        let vals = tape.values(y).expect("values");
        assert_eq!(vals, vec![4.0, 6.0, 10.0]);

        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("gradient exists");
        assert_eq!(grad, &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn custom_function_save_for_backward() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("x");

        // Custom ReLU: forward = max(0, x), backward = grad * (x > 0)
        let y = tape
            .apply_function(
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
            .expect("custom relu");

        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("gradient exists");
        // All inputs > 0, so gradient = 1.0
        assert_eq!(grad, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn custom_function_save_for_backward_with_negatives() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![-1.0, 2.0, -3.0, 4.0], vec![4], true)
            .expect("x");

        // Custom ReLU with negatives
        let y = tape
            .apply_function(
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
            .expect("custom relu");

        let vals = tape.values(y).expect("values");
        assert_eq!(vals, vec![0.0, 2.0, 0.0, 4.0]);

        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("gradient exists");
        assert_eq!(grad, &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn custom_function_straight_through_estimator() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf(vec![1.3, 2.7, -0.5], vec![3], true)
            .expect("x");

        // STE: forward = round(x), backward = grad (straight-through)
        let y = tape
            .apply_function(
                &[x],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    let rounded: Vec<f64> = vals.iter().map(|v| v.round()).collect();
                    Ok((rounded, shape.to_vec()))
                },
                |_ctx, grad_outputs| Ok(vec![Some(grad_outputs[0].to_vec())]),
            )
            .expect("STE");

        let vals = tape.values(y).expect("values");
        assert_eq!(vals, vec![1.0, 3.0, -1.0]);

        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("gradient exists");
        assert_eq!(grad, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn custom_function_multi_input() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![2.0, 3.0], vec![2], true).expect("a");
        let b = tape.leaf(vec![4.0, 5.0], vec![2], true).expect("b");

        // f(a, b) = a * b; grad_a = grad * b, grad_b = grad * a
        let y = tape
            .apply_function(
                &[a, b],
                |ctx, inputs| {
                    let (a_vals, a_shape) = &inputs[0];
                    let (b_vals, _b_shape) = &inputs[1];
                    ctx.save_for_backward(a_vals.to_vec(), a_shape.to_vec());
                    ctx.save_for_backward(b_vals.to_vec(), a_shape.to_vec());
                    let product: Vec<f64> = a_vals
                        .iter()
                        .zip(b_vals.iter())
                        .map(|(x, y)| x * y)
                        .collect();
                    Ok((product, a_shape.to_vec()))
                },
                |ctx, grad_outputs| {
                    let saved_a = &ctx.saved_tensors()[0];
                    let saved_b = &ctx.saved_tensors()[1];
                    let grad = grad_outputs[0];
                    let grad_a: Vec<f64> =
                        grad.iter().zip(saved_b.iter()).map(|(g, b)| g * b).collect();
                    let grad_b: Vec<f64> =
                        grad.iter().zip(saved_a.iter()).map(|(g, a)| g * a).collect();
                    Ok(vec![Some(grad_a), Some(grad_b)])
                },
            )
            .expect("mul function");

        let vals = tape.values(y).expect("values");
        assert_eq!(vals, vec![8.0, 15.0]);

        let report = tape.backward(y).expect("backward");
        let grad_a = report.gradient(a).expect("gradient a");
        let grad_b = report.gradient(b).expect("gradient b");
        // grad_a = grad * b = [1,1] * [4,5] = [4,5]
        assert_eq!(grad_a, &[4.0, 5.0]);
        // grad_b = grad * a = [1,1] * [2,3] = [2,3]
        assert_eq!(grad_b, &[2.0, 3.0]);
    }

    #[test]
    fn custom_function_none_gradient_for_input() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("a");
        let b = tape.leaf(vec![3.0, 4.0], vec![2], true).expect("b");

        // Function that only depends on first input, returns None for second
        let y = tape
            .apply_function(
                &[a, b],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    Ok((vals.to_vec(), shape.to_vec()))
                },
                |_ctx, grad_outputs| Ok(vec![Some(grad_outputs[0].to_vec()), None]),
            )
            .expect("partial grad function");

        let report = tape.backward(y).expect("backward");
        let grad_a = report.gradient(a).expect("gradient a");
        assert_eq!(grad_a, &[1.0, 1.0]);
    }

    #[test]
    fn custom_function_needs_input_grad() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0], vec![1], true).expect("a");
        let b = tape.leaf(vec![2.0], vec![1], false).expect("b no grad");

        let needs_grad_observed = Arc::new(Mutex::new(Vec::new()));
        let needs_grad_clone = Arc::clone(&needs_grad_observed);

        let y = tape
            .apply_function(
                &[a, b],
                |ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    Ok((vals.to_vec(), shape.to_vec()))
                },
                move |ctx, grad_outputs| {
                    let mut lock = needs_grad_clone.lock().unwrap();
                    *lock = ctx.needs_input_grad().to_vec();
                    Ok(vec![Some(grad_outputs[0].to_vec()), None])
                },
            )
            .expect("function");

        let report = tape.backward(y).expect("backward");
        let observed = needs_grad_observed.lock().unwrap();
        assert_eq!(*observed, vec![true, false]);
    }

    #[test]
    fn custom_function_composed_with_standard_ops() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![2.0, 3.0], vec![2], true).expect("x");

        // Custom: f(x) = x * 3, then standard neg
        let tripled = tape
            .apply_function(
                &[x],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    let tripled: Vec<f64> = vals.iter().map(|v| v * 3.0).collect();
                    Ok((tripled, shape.to_vec()))
                },
                |_ctx, grad_outputs| {
                    let grad: Vec<f64> = grad_outputs[0].iter().map(|g| g * 3.0).collect();
                    Ok(vec![Some(grad)])
                },
            )
            .expect("triple");

        let (y, _) = tape.neg(tripled, ExecutionMode::Strict).expect("neg");

        let vals = tape.values(y).expect("values");
        assert_eq!(vals, vec![-6.0, -9.0]);

        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("gradient");
        // d/dx(-3x) = -3
        assert_eq!(grad, &[-3.0, -3.0]);
    }

    #[test]
    fn custom_function_no_saved_tensors() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![5.0], vec![1], true).expect("x");

        let y = tape
            .apply_function(
                &[x],
                |ctx, inputs| {
                    // Don't save anything
                    assert!(ctx.saved_tensors().is_empty());
                    let (vals, shape) = &inputs[0];
                    Ok((vals.to_vec(), shape.to_vec()))
                },
                |ctx, grad_outputs| {
                    assert!(ctx.saved_tensors().is_empty());
                    Ok(vec![Some(grad_outputs[0].to_vec())])
                },
            )
            .expect("no-save function");

        let report = tape.backward(y).expect("backward");
        let grad = report.gradient(x).expect("gradient");
        assert_eq!(grad, &[1.0]);
    }

    #[test]
    fn custom_function_grad_fn_label() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0], vec![1], true).expect("x");

        let y = tape
            .apply_function(
                &[x],
                |_ctx, inputs| {
                    let (vals, shape) = &inputs[0];
                    Ok((vals.to_vec(), shape.to_vec()))
                },
                |_ctx, grad_outputs| Ok(vec![Some(grad_outputs[0].to_vec())]),
            )
            .expect("function");

        let label = tape.tensor_grad_fn(y).expect("grad_fn").expect("Some");
        assert_eq!(label, "CustomFunction");
    }

    // ---- create_graph tests (bd-3dpn.3) ----

    #[test]
    fn create_graph_records_gradient_ops_in_tape() {
        // After backward with create_graph=True, the tape should have more nodes
        // than before (gradient ops were recorded).
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![2.0], vec![1], true).expect("x");
        let (x2, _) = tape.mul(x, x, ExecutionMode::Strict).expect("x*x");
        let (x3, _) = tape.mul(x2, x, ExecutionMode::Strict).expect("x^3");

        let node_count_before = tape.node_count();

        let report = tape
            .backward_with_options(
                x3,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .expect("backward with create_graph");

        // New gradient computation nodes should have been added
        assert!(
            tape.node_count() > node_count_before,
            "tape should have more nodes after create_graph backward"
        );

        // The gradient node for x should exist
        assert!(
            report.gradient_node(x).is_some(),
            "gradient_node for x should exist"
        );
    }

    #[test]
    fn create_graph_second_derivative_x_cubed() {
        // f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
        // At x=2: f(2)=8, f'(2)=12, f''(2)=12
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![2.0], vec![1], true).expect("x");
        let (x2, _) = tape.mul(x, x, ExecutionMode::Strict).expect("x*x");
        let (x3, _) = tape.mul(x2, x, ExecutionMode::Strict).expect("x^3");

        // First backward with create_graph=true
        let report1 = tape
            .backward_with_options(
                x3,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .expect("first backward");

        // f'(2) = 12
        let grad_x = report1.gradient(x).expect("x gradient");
        assert!(
            (grad_x[0] - 12.0).abs() < 1e-10,
            "f'(2) should be 12, got {}",
            grad_x[0]
        );

        // Get the gradient node for second backward
        let dx_node = report1.gradient_node(x).expect("gradient node for x");

        // Second backward through the gradient graph
        let report2 = tape.backward(dx_node).expect("second backward");
        let grad2_x = report2.gradient(x).expect("x second gradient");
        assert!(
            (grad2_x[0] - 12.0).abs() < 1e-10,
            "f''(2) should be 12, got {}",
            grad2_x[0]
        );
    }

    #[test]
    fn create_graph_second_derivative_exp() {
        // f(x) = exp(x), f'(x) = exp(x), f''(x) = exp(x)
        // At x=1: f(1) = e, f'(1) = e, f''(1) = e
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0], vec![1], true).expect("x");
        let (ex, _) = tape.exp(x, ExecutionMode::Strict).expect("exp(x)");

        let report1 = tape
            .backward_with_options(
                ex,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .expect("first backward");

        let grad_x = report1.gradient(x).expect("x gradient");
        let expected = 1.0_f64.exp();
        assert!(
            (grad_x[0] - expected).abs() < 1e-10,
            "f'(1) should be e={}, got {}",
            expected,
            grad_x[0]
        );

        let dx_node = report1.gradient_node(x).expect("gradient node for x");
        let report2 = tape.backward(dx_node).expect("second backward");
        let grad2_x = report2.gradient(x).expect("x second gradient");
        assert!(
            (grad2_x[0] - expected).abs() < 1e-10,
            "f''(1) should be e={}, got {}",
            expected,
            grad2_x[0]
        );
    }

    #[test]
    fn create_graph_second_derivative_sin() {
        // f(x) = sin(x), f'(x) = cos(x), f''(x) = -sin(x)
        // At x=1: f''(1) = -sin(1)
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0], vec![1], true).expect("x");
        let (sx, _) = tape.sin(x, ExecutionMode::Strict).expect("sin(x)");

        let report1 = tape
            .backward_with_options(
                sx,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .expect("first backward");

        let grad_x = report1.gradient(x).expect("x gradient");
        assert!(
            (grad_x[0] - 1.0_f64.cos()).abs() < 1e-10,
            "f'(1) should be cos(1)"
        );

        let dx_node = report1.gradient_node(x).expect("gradient node");
        let report2 = tape.backward(dx_node).expect("second backward");
        let grad2_x = report2.gradient(x).expect("second gradient");
        let expected = -1.0_f64.sin();
        assert!(
            (grad2_x[0] - expected).abs() < 1e-10,
            "f''(1) should be -sin(1)={}, got {}",
            expected,
            grad2_x[0]
        );
    }

    #[test]
    fn create_graph_second_derivative_matches_finite_differences() {
        // Numerical validation: f(x) = x^3 + 2*x^2
        // f'(x) = 3x^2 + 4x, f''(x) = 6x + 4
        // Check f''(x) via finite differences of f'(x)
        let eps = 1e-5;
        let x_val = 3.0;

        // Compute f'(x+eps) numerically
        let compute_first_deriv = |val: f64| -> f64 {
            let mut tape = TensorTape::new();
            let x = tape.leaf(vec![val], vec![1], true).expect("x");
            let (x2, _) = tape.mul(x, x, ExecutionMode::Strict).expect("x^2");
            let (x3, _) = tape.mul(x2, x, ExecutionMode::Strict).expect("x^3");
            let two = tape.leaf(vec![2.0], vec![1], false).expect("2");
            let (two_x2, _) = tape.mul(two, x2, ExecutionMode::Strict).expect("2x^2");
            let (out, _) = tape.add(x3, two_x2, ExecutionMode::Strict).expect("x^3 + 2x^2");
            let report = tape.backward(out).expect("backward");
            report.gradient(x).expect("grad")[0]
        };

        let fd_second_deriv = (compute_first_deriv(x_val + eps) - compute_first_deriv(x_val - eps)) / (2.0 * eps);

        // Compute f''(x) analytically via create_graph
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![x_val], vec![1], true).expect("x");
        let (x2, _) = tape.mul(x, x, ExecutionMode::Strict).expect("x^2");
        let (x3, _) = tape.mul(x2, x, ExecutionMode::Strict).expect("x^3");
        let two = tape.leaf(vec![2.0], vec![1], false).expect("2");
        let (two_x2, _) = tape.mul(two, x2, ExecutionMode::Strict).expect("2x^2");
        let (out, _) = tape.add(x3, two_x2, ExecutionMode::Strict).expect("x^3 + 2x^2");

        let report1 = tape
            .backward_with_options(
                out,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .expect("first backward");

        let dx_node = report1.gradient_node(x).expect("gradient node");
        let report2 = tape.backward(dx_node).expect("second backward");
        let analytic_second_deriv = report2.gradient(x).expect("second gradient")[0];

        // f''(3) = 6*3 + 4 = 22
        assert!(
            (analytic_second_deriv - 22.0).abs() < 1e-8,
            "analytic f''(3) should be 22, got {}",
            analytic_second_deriv
        );
        assert!(
            (analytic_second_deriv - fd_second_deriv).abs() < 1e-4,
            "analytic {} should match finite-diff {}", analytic_second_deriv, fd_second_deriv
        );
    }

    #[test]
    fn create_graph_default_false_does_not_record_gradient_ops() {
        // With create_graph=false (default), no gradient nodes should be created
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![2.0], vec![1], true).expect("x");
        let (x2, _) = tape.mul(x, x, ExecutionMode::Strict).expect("x*x");

        let report = tape.backward(x2).expect("backward");
        // gradient_node should be None when create_graph is false
        assert!(
            report.gradient_node(x).is_none(),
            "gradient_node should be None when create_graph=false"
        );
    }

    #[test]
    fn create_graph_scalar_output_second_derivative() {
        // f(x) = x^2 (scalar, single element tensor), f''(x) = 2
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![5.0], vec![1], true).expect("x");
        let (x2, _) = tape.mul(x, x, ExecutionMode::Strict).expect("x^2");

        let report1 = tape
            .backward_with_options(
                x2,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .expect("first backward");

        let grad_x = report1.gradient(x).expect("first gradient");
        assert!(
            (grad_x[0] - 10.0).abs() < 1e-10,
            "f'(5) = 2*5 = 10, got {}",
            grad_x[0]
        );

        let dx_node = report1.gradient_node(x).expect("gradient node");
        let report2 = tape.backward(dx_node).expect("second backward");
        let grad2_x = report2.gradient(x).expect("second gradient");
        assert!(
            (grad2_x[0] - 2.0).abs() < 1e-10,
            "f''(x) = 2, got {}",
            grad2_x[0]
        );
    }

    #[test]
    fn create_graph_multi_element_tensor() {
        // f(x) = sum(x^2) where x = [1,2,3]
        // df/dx_i = 2*x_i, d²f/dx_i² = 2
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], true).expect("x");
        let (x2, _) = tape.mul(x, x, ExecutionMode::Strict).expect("x^2");
        let (s, _) = tape.sum(x2, ExecutionMode::Strict).expect("sum(x^2)");

        let report1 = tape
            .backward_with_options(
                s,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .expect("first backward");

        let grad_x = report1.gradient(x).expect("first gradient");
        assert_eq!(grad_x, &[2.0, 4.0, 6.0]);

        let dx_node = report1.gradient_node(x).expect("gradient node");
        let report2 = tape.backward(dx_node).expect("second backward");
        let grad2_x = report2.gradient(x).expect("second gradient");
        // d²(sum(x^2))/dx_i² = 2 for all i
        for &v in grad2_x {
            assert!(
                (v - 2.0).abs() < 1e-10,
                "second derivative of sum(x^2) should be 2, got {}",
                v
            );
        }
    }

    #[test]
    fn create_graph_gradient_penalty_style() {
        // WGAN-GP style: L = ||grad(f(x))||^2
        // f(x) = sum(x^3), grad_f = 3*x^2
        // L = sum((3*x^2)^2) = sum(9*x^4)
        // dL/dx = 36*x^3
        // At x = [1, 2]: dL/dx = [36, 288]
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0], vec![2], true).expect("x");
        let (x2, _) = tape.mul(x, x, ExecutionMode::Strict).expect("x^2");
        let (x3, _) = tape.mul(x2, x, ExecutionMode::Strict).expect("x^3");
        let (f, _) = tape.sum(x3, ExecutionMode::Strict).expect("sum(x^3)");

        // First backward with create_graph to get gradient nodes
        let report1 = tape
            .backward_with_options(
                f,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .expect("first backward");

        let dx_node = report1.gradient_node(x).expect("gradient node for x");

        // Compute ||grad||^2 = sum(grad^2)
        let (grad_sq, _) = tape.mul(dx_node, dx_node, ExecutionMode::Strict).expect("grad^2");
        let (penalty, _) = tape.sum(grad_sq, ExecutionMode::Strict).expect("sum(grad^2)");

        // Second backward to get dL/dx
        let report2 = tape.backward(penalty).expect("second backward");
        let grad_penalty = report2.gradient(x).expect("gradient penalty grad");

        // dL/dx_i = 2 * 3x_i^2 * 3 * 2 * x_i = 36*x_i^3
        // At x=1: 36, at x=2: 36*8=288
        assert!(
            (grad_penalty[0] - 36.0).abs() < 1e-6,
            "dL/dx[0] should be 36, got {}",
            grad_penalty[0]
        );
        assert!(
            (grad_penalty[1] - 288.0).abs() < 1e-6,
            "dL/dx[1] should be 288, got {}",
            grad_penalty[1]
        );
    }

    #[test]
    fn create_graph_physics_double_backward() {
        // Physics-informed: d²u/dx² for u(x) = x^4
        // u'(x) = 4x^3, u''(x) = 12x^2
        // At x=2: u''(2) = 48
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![2.0], vec![1], true).expect("x");
        let (x2, _) = tape.mul(x, x, ExecutionMode::Strict).expect("x^2");
        let (x4, _) = tape.mul(x2, x2, ExecutionMode::Strict).expect("x^4");

        let report1 = tape
            .backward_with_options(
                x4,
                BackwardOptions {
                    create_graph: true,
                    ..BackwardOptions::strict_default()
                },
            )
            .expect("first backward");

        let dx_node = report1.gradient_node(x).expect("gradient node");
        let report2 = tape.backward(dx_node).expect("second backward");
        let second_deriv = report2.gradient(x).expect("second derivative");
        assert!(
            (second_deriv[0] - 48.0).abs() < 1e-8,
            "u''(2) should be 48, got {}",
            second_deriv[0]
        );
    }

    // ── F32 typed dispatch: reductions preserve dtype ──────────────────

    #[test]
    fn f32_sum_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![1.0f32, 2.0, 3.0], vec![3], false)
            .unwrap();
        let (s, _) = tape.sum(a, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(s).unwrap(), DType::F32);
        let vals = tape.values_f32(s).unwrap();
        assert_eq!(vals, vec![6.0f32]);
    }

    #[test]
    fn f32_mean_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![2.0f32, 4.0, 6.0], vec![3], false)
            .unwrap();
        let (m, _) = tape.mean(a, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(m).unwrap(), DType::F32);
        let vals = tape.values_f32(m).unwrap();
        assert_eq!(vals, vec![4.0f32]);
    }

    #[test]
    fn f32_pow_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![2.0f32, 3.0], vec![2], false)
            .unwrap();
        let (p, _) = tape.pow(a, 2.0, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(p).unwrap(), DType::F32);
        let vals = tape.values_f32(p).unwrap();
        assert_eq!(vals, vec![4.0f32, 9.0]);
    }

    #[test]
    fn f32_clamp_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![0.5f32, 1.5, 2.5, 3.5], vec![4], false)
            .unwrap();
        let (c, _) = tape.tensor_clamp(a, 1.0, 3.0, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(c).unwrap(), DType::F32);
        let vals = tape.values_f32(c).unwrap();
        assert_eq!(vals, vec![1.0f32, 1.5, 2.5, 3.0]);
    }

    #[test]
    fn f32_softmax_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![1.0f32, 2.0, 3.0], vec![1, 3], false)
            .unwrap();
        let (s, _) = tape.softmax(a, 1, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(s).unwrap(), DType::F32);
        let vals = tape.values_f32(s).unwrap();
        let total: f32 = vals.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax should sum to 1, got {total}");
    }

    #[test]
    fn f32_cumsum_preserves_dtype() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![1.0f32, 2.0, 3.0], vec![3], false)
            .unwrap();
        let (c, _) = tape.cumsum(a, 0, ExecutionMode::Strict).unwrap();
        assert_eq!(tape.dtype(c).unwrap(), DType::F32);
        let vals = tape.values_f32(c).unwrap();
        assert_eq!(vals, vec![1.0f32, 3.0, 6.0]);
    }

    // ── F32 autograd: forward f32 → backward gradients correct ────────

    #[test]
    fn f32_backward_mul_sum_chain() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf_f32(vec![2.0f32, 3.0, 4.0], vec![3], true)
            .unwrap();
        let y = tape
            .leaf_f32(vec![5.0f32, 6.0, 7.0], vec![3], true)
            .unwrap();
        // z = sum(x * y) = 2*5 + 3*6 + 4*7 = 10 + 18 + 28 = 56
        let (xy, _) = tape.mul(x, y, ExecutionMode::Strict).unwrap();
        let (z, _) = tape.sum(xy, ExecutionMode::Strict).unwrap();
        let report = tape.backward(z).unwrap();
        // dz/dx = y, dz/dy = x
        let grad_x = report.gradient(x).unwrap();
        let grad_y = report.gradient(y).unwrap();
        assert_eq!(grad_x, &[5.0, 6.0, 7.0]);
        assert_eq!(grad_y, &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn f32_backward_pow_sum() {
        let mut tape = TensorTape::new();
        let x = tape
            .leaf_f32(vec![1.0f32, 2.0, 3.0], vec![3], true)
            .unwrap();
        // y = sum(x^3) → dy/dx = 3x^2
        let (p, _) = tape.pow(x, 3.0, ExecutionMode::Strict).unwrap();
        let (s, _) = tape.sum(p, ExecutionMode::Strict).unwrap();
        let report = tape.backward(s).unwrap();
        let grad = report.gradient(x).unwrap();
        // 3*1^2=3, 3*2^2=12, 3*3^2=27
        assert!((grad[0] - 3.0).abs() < 1e-4);
        assert!((grad[1] - 12.0).abs() < 1e-4);
        assert!((grad[2] - 27.0).abs() < 1e-4);
    }

    // ── F32 numerical precision ────────────────────────────────────────

    #[test]
    fn f32_vs_f64_precision_comparison() {
        // Verify f32 results are close but not identical to f64
        let data_f32 = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let data_f64: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();

        let mut tape = TensorTape::new();
        let a32 = tape.leaf_f32(data_f32, vec![10], false).unwrap();
        let a64 = tape.leaf(data_f64, vec![10], false).unwrap();

        let (s32, _) = tape.sum(a32, ExecutionMode::Strict).unwrap();
        let (s64, _) = tape.sum(a64, ExecutionMode::Strict).unwrap();

        let v32 = tape.values_f32(s32).unwrap()[0] as f64;
        let v64 = tape.values(s64).unwrap()[0];

        // Should be close (within f32 precision) but may differ slightly
        assert!(
            (v32 - v64).abs() < 1e-6,
            "f32 sum={v32}, f64 sum={v64}, diff={}",
            (v32 - v64).abs()
        );
    }

    #[test]
    fn f32_softmax_numerical_stability() {
        // Large values should not overflow in f32 softmax
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![80.0f32, 81.0, 82.0], vec![1, 3], false)
            .unwrap();
        let (s, _) = tape.softmax(a, 1, ExecutionMode::Strict).unwrap();
        let vals = tape.values_f32(s).unwrap();
        for v in &vals {
            assert!(v.is_finite(), "softmax value should be finite, got {v}");
            assert!(*v >= 0.0 && *v <= 1.0, "softmax value should be in [0,1], got {v}");
        }
        let total: f32 = vals.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax should sum to 1, got {total}");
    }

    // ── F32 training loop ──────────────────────────────────────────────

    #[test]
    fn f32_training_loop_manual_linear() {
        // Simple linear regression: y = 2x + 1, train w and b in f32
        let mut tape = TensorTape::new();
        let mut w = tape.leaf_f32(vec![0.0f32], vec![1], true).unwrap();
        let mut b = tape.leaf_f32(vec![0.0f32], vec![1], true).unwrap();

        let lr = 0.01f64;

        for _ in 0..500 {
            // x=1 → target=3
            let x1 = tape.leaf_f32(vec![1.0f32], vec![1], false).unwrap();
            let t1 = tape.leaf_f32(vec![3.0f32], vec![1], false).unwrap();

            // pred = w*x + b
            let (wx, _) = tape.mul(w, x1, ExecutionMode::Strict).unwrap();
            let (pred, _) = tape.add(wx, b, ExecutionMode::Strict).unwrap();

            // loss = (pred - target)^2
            let (diff, _) = tape.sub(pred, t1, ExecutionMode::Strict).unwrap();
            let (loss, _) = tape.pow(diff, 2.0, ExecutionMode::Strict).unwrap();
            let (loss_s, _) = tape.sum(loss, ExecutionMode::Strict).unwrap();

            let report = tape.backward(loss_s).unwrap();
            let gw = report.gradient(w).unwrap()[0];
            let gb = report.gradient(b).unwrap()[0];

            // SGD update
            let w_val = tape.values_f32(w).unwrap()[0] as f64 - lr * gw;
            let b_val = tape.values_f32(b).unwrap()[0] as f64 - lr * gb;

            tape = TensorTape::new();
            w = tape.leaf_f32(vec![w_val as f32], vec![1], true).unwrap();
            b = tape.leaf_f32(vec![b_val as f32], vec![1], true).unwrap();
        }

        let w_final = tape.values_f32(w).unwrap()[0];
        let b_final = tape.values_f32(b).unwrap()[0];
        // Should converge toward w≈2, b≈1 (only training on x=1)
        // With single sample, w+b → 3 is the convergence
        let pred = w_final + b_final;
        assert!(
            (pred - 3.0).abs() < 0.1,
            "prediction should be ~3.0, got {pred} (w={w_final}, b={b_final})"
        );
    }

    // ── F32 memory: storage uses f32 internally ────────────────────────

    #[test]
    fn f32_storage_is_actually_f32() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], false)
            .unwrap();
        let tensor = tape.tensor(a).unwrap();
        match tensor.typed_storage() {
            ft_core::TensorStorage::F32(data) => {
                assert_eq!(data.len(), 4);
                assert_eq!(data, &[1.0f32, 2.0, 3.0, 4.0]);
            }
            ft_core::TensorStorage::F64(_) => {
                panic!("f32 leaf should produce F32 storage, not F64");
            }
        }
    }

    #[test]
    fn f32_reduction_output_is_f32_storage() {
        let mut tape = TensorTape::new();
        let a = tape
            .leaf_f32(vec![1.0f32, 2.0, 3.0], vec![3], false)
            .unwrap();
        let (s, _) = tape.sum(a, ExecutionMode::Strict).unwrap();
        let tensor = tape.tensor(s).unwrap();
        match tensor.typed_storage() {
            ft_core::TensorStorage::F32(data) => {
                assert_eq!(data, &[6.0f32]);
            }
            ft_core::TensorStorage::F64(_) => {
                panic!("f32 sum should produce F32 storage, not F64");
            }
        }
    }

    // ── to_dtype tests ─────────────────────────────────────────────────

    #[test]
    fn to_dtype_f64_to_f32() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.5, 2.5, 3.5], vec![3], false).unwrap();
        let b = tape.to_dtype(a, DType::F32).unwrap();
        assert_eq!(tape.dtype(b).unwrap(), DType::F32);
        assert_eq!(tape.values_f32(b).unwrap(), vec![1.5f32, 2.5, 3.5]);
    }

    #[test]
    fn to_dtype_f32_to_f64() {
        let mut tape = TensorTape::new();
        let a = tape.leaf_f32(vec![1.0f32, 2.0], vec![2], false).unwrap();
        let b = tape.to_dtype(a, DType::F64).unwrap();
        assert_eq!(tape.dtype(b).unwrap(), DType::F64);
        assert_eq!(tape.values(b).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn to_dtype_noop_same_type() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0], vec![2], false).unwrap();
        let b = tape.to_dtype(a, DType::F64).unwrap();
        assert_eq!(a, b); // no-op returns same node
    }

    #[test]
    fn to_dtype_rejects_non_float_target() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0], vec![2], false).unwrap();
        let err = tape.to_dtype(a, DType::I64).unwrap_err();
        assert!(
            matches!(
                err,
                AutogradError::DenseTensor(ft_core::DenseTensorError::UnsupportedDType(DType::I64))
            ),
            "expected UnsupportedDType(I64), got {err:?}"
        );
    }

    #[test]
    fn to_dtype_rejects_bool_target() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0], vec![2], false).unwrap();
        let err = tape.to_dtype(a, DType::Bool).unwrap_err();
        assert!(
            matches!(
                err,
                AutogradError::DenseTensor(ft_core::DenseTensorError::UnsupportedDType(
                    DType::Bool
                ))
            ),
            "expected UnsupportedDType(Bool), got {err:?}"
        );
    }

    // ── view() zero-copy tests ─────────────────────────────────────────

    #[test]
    fn view_same_shape_returns_same_data() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        let b = tape.view(a, vec![2, 2]).unwrap();
        assert_eq!(tape.values(b).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tape.shape(b).unwrap(), &[2, 2]);
    }

    #[test]
    fn view_flatten() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false).unwrap();
        let b = tape.view(a, vec![6]).unwrap();
        assert_eq!(tape.values(b).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tape.shape(b).unwrap(), &[6]);
    }

    #[test]
    fn view_unflatten() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], false).unwrap();
        let b = tape.view(a, vec![2, 3]).unwrap();
        assert_eq!(tape.values(b).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tape.shape(b).unwrap(), &[2, 3]);
    }

    #[test]
    fn view_shares_storage() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0, 3.0, 4.0], vec![4], false).unwrap();
        let b = tape.view(a, vec![2, 2]).unwrap();
        // Both should share the same storage_id
        let tensor_a = tape.tensor(a).unwrap();
        let tensor_b = tape.tensor(b).unwrap();
        assert!(tensor_a.shares_storage_with(tensor_b));
    }

    #[test]
    fn view_backward_gradient_correct() {
        let mut tape = TensorTape::new();
        let x = tape.leaf(vec![1.0, 2.0, 3.0, 4.0], vec![4], true).unwrap();
        let v = tape.view(x, vec![2, 2]).unwrap();
        let (s, _) = tape.sum(v, ExecutionMode::Strict).unwrap();
        let report = tape.backward(s).unwrap();
        let grad = report.gradient(x).unwrap();
        assert_eq!(grad, &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn view_wrong_numel_errors() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        assert!(tape.view(a, vec![2, 2]).is_err());
    }

    #[test]
    fn view_of_view_shares_storage() {
        let mut tape = TensorTape::new();
        let a = tape.leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], false).unwrap();
        let b = tape.view(a, vec![2, 3]).unwrap();
        let c = tape.view(b, vec![3, 2]).unwrap();
        let tensor_a = tape.tensor(a).unwrap();
        let tensor_c = tape.tensor(c).unwrap();
        assert!(tensor_a.shares_storage_with(tensor_c));
        assert_eq!(tape.values(c).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn view_f32_shares_storage() {
        let mut tape = TensorTape::new();
        let a = tape.leaf_f32(vec![1.0f32, 2.0, 3.0], vec![3], false).unwrap();
        let b = tape.view(a, vec![1, 3]).unwrap();
        assert_eq!(tape.dtype(b).unwrap(), DType::F32);
        assert_eq!(tape.values_f32(b).unwrap(), vec![1.0f32, 2.0, 3.0]);
        let tensor_a = tape.tensor(a).unwrap();
        let tensor_b = tape.tensor(b).unwrap();
        assert!(tensor_a.shares_storage_with(tensor_b));
    }
}
