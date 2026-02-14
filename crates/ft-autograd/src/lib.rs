#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use ft_core::{DType, Device, ExecutionMode, ScalarTensor};
use ft_dispatch::{BinaryOp, DispatchDecision, DispatchError, dispatch_scalar_binary};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeOp {
    Leaf,
    Add { lhs: NodeId, rhs: NodeId },
    Mul { lhs: NodeId, rhs: NodeId },
}

#[derive(Debug, Clone, PartialEq)]
struct Node {
    tensor: ScalarTensor,
    requires_grad: bool,
    op: NodeOp,
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
pub struct OperationEvent {
    pub op: BinaryOp,
    pub lhs: NodeId,
    pub rhs: NodeId,
    pub out: NodeId,
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

#[derive(Debug, Clone, PartialEq)]
pub enum AutogradError {
    UnknownNode(NodeId),
    Dispatch(DispatchError),
    ReentrantDepthExceeded { current: usize, max: usize },
    DependencyUnderflow { node: NodeId },
}

impl fmt::Display for AutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNode(node) => write!(f, "unknown node id {}", node.0),
            Self::Dispatch(error) => write!(f, "dispatch failure: {error}"),
            Self::ReentrantDepthExceeded { current, max } => write!(
                f,
                "reentrant backward depth exceeded: current={current} max={max}"
            ),
            Self::DependencyUnderflow { node } => {
                write!(f, "dependency scheduler underflow at node {}", node.0)
            }
        }
    }
}

impl std::error::Error for AutogradError {}

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

        let mut queue = ReadyQueue::default();
        queue.push(root);

        let mut steps = Vec::new();
        let mut execution_order = Vec::new();

        while let Some(node_id) = queue.pop() {
            if !reachable[node_id.0] {
                continue;
            }

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
                NodeOp::Add { lhs, rhs } | NodeOp::Mul { lhs, rhs } => {
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
                NodeOp::Add { lhs, rhs } | NodeOp::Mul { lhs, rhs } => {
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

#[cfg(test)]
mod tests {
    use ft_core::ExecutionMode;

    use super::{BackwardOptions, NodeId, ReentrantPolicy, Tape};

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
}
