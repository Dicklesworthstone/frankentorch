#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use ft_core::{DType, DenseTensor, DenseTensorError, Device, ExecutionMode, ScalarTensor};
use ft_dispatch::{
    BinaryOp, DispatchDecision, DispatchError, dispatch_scalar_binary,
    dispatch_tensor_binary_contiguous_f64,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorNodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeOp {
    Leaf,
    Add { lhs: NodeId, rhs: NodeId },
    Sub { lhs: NodeId, rhs: NodeId },
    Div { lhs: NodeId, rhs: NodeId },
    Mul { lhs: NodeId, rhs: NodeId },
}

#[derive(Debug, Clone, PartialEq)]
struct Node {
    tensor: ScalarTensor,
    requires_grad: bool,
    op: NodeOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub struct TensorOperationEvent {
    pub op: BinaryOp,
    pub lhs: TensorNodeId,
    pub rhs: TensorNodeId,
    pub out: TensorNodeId,
    pub decision: DispatchDecision,
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
}

impl fmt::Display for AutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNode(node) => write!(f, "unknown node id {}", node.0),
            Self::UnknownTensorNode(node) => write!(f, "unknown tensor node id {}", node.0),
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

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: NodeId,
        rhs: NodeId,
        mode: ExecutionMode,
    ) -> Result<(NodeId, OperationEvent), AutogradError> {
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

        let mut grads = vec![0.0; self.nodes.len()];
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
            }
        }

        let gradients = grads
            .iter()
            .enumerate()
            .map(|(idx, grad)| {
                if self.nodes[idx].requires_grad {
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
            dependency_snapshot: pending,
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
                | NodeOp::Mul { lhs, rhs } => {
                    stack.push(lhs);
                    stack.push(rhs);
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
                | NodeOp::Mul { lhs, rhs } => {
                    pending[lhs.0] = pending[lhs.0].saturating_add(1);
                    pending[rhs.0] = pending[rhs.0].saturating_add(1);
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

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        mode: ExecutionMode,
    ) -> Result<(TensorNodeId, TensorOperationEvent), AutogradError> {
        let (requires_grad, output_meta, outcome) = {
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
            let output_meta = ft_core::TensorMeta::from_shape(
                lhs_meta.shape().to_vec(),
                lhs_meta.dtype(),
                lhs_meta.device(),
            );
            (requires_grad, output_meta, outcome)
        };

        let out = TensorNodeId(self.nodes.len());
        self.nodes.push(TensorNode {
            tensor: DenseTensor::from_storage(output_meta, outcome.values)?,
            requires_grad,
            op: match op {
                BinaryOp::Add => TensorNodeOp::Add { lhs, rhs },
                BinaryOp::Sub => TensorNodeOp::Sub { lhs, rhs },
                BinaryOp::Div => TensorNodeOp::Div { lhs, rhs },
                BinaryOp::Mul => TensorNodeOp::Mul { lhs, rhs },
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

    pub fn backward_with_options(
        &self,
        root: TensorNodeId,
        options: BackwardOptions,
    ) -> Result<TensorBackwardReport, AutogradError> {
        if root.0 >= self.nodes.len() {
            return Err(AutogradError::UnknownTensorNode(root));
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

        let mut grads = self
            .nodes
            .iter()
            .map(|node| vec![0.0; node.tensor.meta().numel()])
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
                    let lhs_values = self.nodes[lhs.0].tensor.dispatch_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.dispatch_values()?;
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
                    let lhs_values = self.nodes[lhs.0].tensor.dispatch_values()?;
                    let rhs_values = self.nodes[rhs.0].tensor.dispatch_values()?;
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
            }
        }

        let gradients = grads
            .iter()
            .enumerate()
            .map(|(idx, grad)| {
                if self.nodes[idx].requires_grad {
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
            dependency_snapshot: pending,
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
                | TensorNodeOp::Mul { lhs, rhs } => {
                    stack.push(lhs);
                    stack.push(rhs);
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
                | TensorNodeOp::Mul { lhs, rhs } => {
                    pending[lhs.0] = pending[lhs.0].saturating_add(1);
                    pending[rhs.0] = pending[rhs.0].saturating_add(1);
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
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ft_core::{DType, DenseTensor, DenseTensorError, Device, ExecutionMode, TensorMeta};
    use ft_dispatch::DispatchError;
    use proptest::prelude::*;

    use super::{
        AutogradError, BackwardOptions, NodeId, ReentrantPolicy, SchedulerTelemetry, Tape,
        TensorNodeId, TensorTape,
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
}
