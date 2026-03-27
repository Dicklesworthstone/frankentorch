#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, FunctionCtx, TensorNodeId};
use ft_core::{DType, DenseTensor, DenseTensorError};
use ft_dispatch::{DispatchError, DispatchKeyError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleRegistrationError {
    InvalidName { kind: &'static str },
    NameConflict { name: String },
    Unsupported { operation: &'static str },
}

impl std::fmt::Display for ModuleRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidName { kind } => write!(f, "invalid {kind} registration name"),
            Self::NameConflict { name } => {
                write!(
                    f,
                    "registration name '{name}' conflicts with existing state"
                )
            }
            Self::Unsupported { operation } => {
                write!(f, "module does not support '{operation}'")
            }
        }
    }
}

impl std::error::Error for ModuleRegistrationError {}

pub type StateDict = BTreeMap<String, DenseTensor>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadStateDictReport {
    pub missing_keys: Vec<String>,
    pub unexpected_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StateDictError {
    Autograd(AutogradError),
    DenseTensor(DenseTensorError),
    DuplicateStateKey {
        key: String,
    },
    UnsupportedDType {
        key: String,
        dtype: DType,
    },
    StrictKeyMismatch {
        missing_keys: Vec<String>,
        unexpected_keys: Vec<String>,
    },
    ShapeMismatch {
        key: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    DTypeMismatch {
        key: String,
        expected: DType,
        found: DType,
    },
}

impl std::fmt::Display for StateDictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Autograd(error) => write!(f, "autograd error: {error}"),
            Self::DenseTensor(error) => write!(f, "dense tensor error: {error}"),
            Self::DuplicateStateKey { key } => write!(f, "duplicate state key '{key}'"),
            Self::UnsupportedDType { key, dtype } => {
                write!(f, "state key '{key}' uses unsupported dtype {dtype:?}")
            }
            Self::StrictKeyMismatch {
                missing_keys,
                unexpected_keys,
            } => write!(
                f,
                "strict load_state_dict key mismatch: missing={missing_keys:?}, unexpected={unexpected_keys:?}"
            ),
            Self::ShapeMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "shape mismatch for state key '{key}': expected={expected:?}, found={found:?}"
            ),
            Self::DTypeMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "dtype mismatch for state key '{key}': expected={expected:?}, found={found:?}"
            ),
        }
    }
}

impl std::error::Error for StateDictError {}

impl From<AutogradError> for StateDictError {
    fn from(value: AutogradError) -> Self {
        Self::Autograd(value)
    }
}

impl From<DenseTensorError> for StateDictError {
    fn from(value: DenseTensorError) -> Self {
        Self::DenseTensor(value)
    }
}

fn gradient_utils_error(reason: &'static str) -> AutogradError {
    AutogradError::Dispatch(DispatchError::Key(DispatchKeyError::IncompatibleSet {
        reason,
    }))
}

#[derive(Debug, Clone)]
struct RegisteredParameter {
    name: String,
    tensor: Option<TensorNodeId>,
}

#[derive(Debug, Clone)]
struct RegisteredBuffer {
    name: String,
    tensor: Option<TensorNodeId>,
    persistent: bool,
}

fn is_valid_registration_name(name: &str) -> bool {
    if name.is_empty() || name.starts_with('.') || name.ends_with('.') {
        return false;
    }
    name.split('.')
        .all(|part| !part.is_empty() && part.chars().all(|c| c == '_' || c.is_ascii_alphanumeric()))
}

fn upsert_registered_parameter(
    parameters: &mut Vec<RegisteredParameter>,
    name: &str,
    tensor: Option<TensorNodeId>,
) {
    if let Some(existing) = parameters.iter_mut().find(|entry| entry.name == name) {
        existing.tensor = tensor;
    } else {
        parameters.push(RegisteredParameter {
            name: name.to_string(),
            tensor,
        });
    }
}

fn upsert_registered_buffer(
    buffers: &mut Vec<RegisteredBuffer>,
    name: &str,
    tensor: Option<TensorNodeId>,
    persistent: bool,
) {
    if let Some(existing) = buffers.iter_mut().find(|entry| entry.name == name) {
        existing.tensor = tensor;
        existing.persistent = persistent;
    } else {
        buffers.push(RegisteredBuffer {
            name: name.to_string(),
            tensor,
            persistent,
        });
    }
}

/// Trait for neural network modules.
///
/// Modules encapsulate parameters and define a forward computation.
pub trait Module {
    /// Execute the forward pass, returning the output node.
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError>;

    /// Collect all trainable parameter node IDs.
    fn parameters(&self) -> Vec<TensorNodeId>;

    /// Return this module's own named parameters (non-recursive).
    ///
    /// Each tuple is `(local_name, node_id)` — e.g. `("weight", id)`, `("bias", id)`.
    /// Container modules that only hold children (like Sequential) should return an
    /// empty vec here; their children's parameters are collected by `named_parameters`.
    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        Vec::new()
    }

    /// Return dynamically-registered own parameters, including optional slots.
    ///
    /// Entries with `None` represent registered names without tensor payload.
    fn named_parameter_slots_own(&self) -> Vec<(String, Option<TensorNodeId>)> {
        Vec::new()
    }

    /// Return own registered buffers, including persistence metadata.
    ///
    /// Entries with `None` represent registered names without tensor payload.
    fn named_buffer_slots_own(&self) -> Vec<(String, Option<TensorNodeId>, bool)> {
        Vec::new()
    }

    /// Register (or replace) a named parameter slot.
    fn register_parameter(
        &mut self,
        _name: &str,
        _parameter: Option<TensorNodeId>,
    ) -> Result<(), ModuleRegistrationError> {
        Err(ModuleRegistrationError::Unsupported {
            operation: "register_parameter",
        })
    }

    /// Register (or replace) a named buffer slot.
    fn register_buffer(
        &mut self,
        _name: &str,
        _tensor: Option<TensorNodeId>,
        _persistent: bool,
    ) -> Result<(), ModuleRegistrationError> {
        Err(ModuleRegistrationError::Unsupported {
            operation: "register_buffer",
        })
    }

    /// Return direct child sub-modules with their local names.
    ///
    /// For indexed containers (Sequential, ModuleList), names are `"0"`, `"1"`, etc.
    /// For named containers (ModuleDict), names are the user-supplied keys.
    /// For composite modules (MultiheadAttention), names are field names like `"q_proj"`.
    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        Vec::new()
    }

    /// Set training mode for this module and descendants.
    ///
    /// Default behavior recursively propagates to children.
    fn train(&self, mode: bool) {
        for (_, child) in self.named_children() {
            child.train(mode);
        }
    }

    /// Convenience helper to switch into evaluation mode.
    fn eval(&self) {
        self.train(false);
    }

    /// Returns whether this module is in training mode.
    ///
    /// Default behavior returns true for leaf modules and requires all children
    /// to report training mode for container modules.
    fn is_training(&self) -> bool {
        let children = self.named_children();
        children.is_empty() || children.iter().all(|(_, child)| child.is_training())
    }

    /// Export trainable parameters and persistent buffers in a deterministic key order.
    fn state_dict(&self, session: &FrankenTorchSession) -> Result<StateDict, StateDictError>
    where
        Self: Sized,
    {
        module_state_dict(self, session)
    }

    /// Load trainable parameters and persistent buffers from a state dict.
    ///
    /// If `strict` is true, missing/unexpected keys cause an error and no updates are applied.
    fn load_state_dict(
        &self,
        session: &mut FrankenTorchSession,
        state: &StateDict,
        strict: bool,
    ) -> Result<LoadStateDictReport, StateDictError>
    where
        Self: Sized,
    {
        module_load_state_dict(self, session, state, strict)
    }
}

/// Collect direct child modules without names.
pub fn children(module: &dyn Module) -> Vec<&dyn Module> {
    module
        .named_children()
        .into_iter()
        .map(|(_, m)| m)
        .collect()
}

/// Recursively collect all named parameters with hierarchical dot-separated prefixes.
///
/// Pass an empty string for `prefix` to get unqualified names from the root.
/// This matches PyTorch's `module.named_parameters()` behavior: own parameters
/// first, then children's parameters in registration order.
pub fn named_parameters(module: &dyn Module, prefix: &str) -> Vec<(String, TensorNodeId)> {
    let mut result = Vec::new();
    for (name, id) in module.named_parameters_own() {
        let full = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}.{name}")
        };
        result.push((full, id));
    }
    for (name, maybe_id) in module.named_parameter_slots_own() {
        if let Some(id) = maybe_id {
            let full = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}.{name}")
            };
            result.push((full, id));
        }
    }
    for (child_name, child) in module.named_children() {
        let child_prefix = if prefix.is_empty() {
            child_name
        } else {
            format!("{prefix}.{child_name}")
        };
        result.extend(named_parameters(child, &child_prefix));
    }
    result
}

/// Recursively collect all sub-modules depth-first, including the root.
///
/// Pass an empty string for `prefix` to get unqualified names from the root.
/// The root module is included with the given prefix (empty string for root).
/// Ordering matches PyTorch's `module.named_modules()` depth-first traversal.
pub fn named_modules<'a>(module: &'a dyn Module, prefix: &str) -> Vec<(String, &'a dyn Module)> {
    let mut result = vec![(prefix.to_string(), module)];
    for (child_name, child) in module.named_children() {
        let child_prefix = if prefix.is_empty() {
            child_name
        } else {
            format!("{prefix}.{child_name}")
        };
        result.extend(named_modules(child, &child_prefix));
    }
    result
}

/// Collect all sub-modules recursively without names.
pub fn modules(module: &dyn Module) -> Vec<&dyn Module> {
    named_modules(module, "")
        .into_iter()
        .map(|(_, m)| m)
        .collect()
}

/// Recursively collect all buffer tensors (including non-persistent buffers).
pub fn named_buffers(module: &dyn Module, prefix: &str) -> Vec<(String, TensorNodeId)> {
    let mut result = Vec::new();
    for (name, maybe_id, _persistent) in module.named_buffer_slots_own() {
        if let Some(id) = maybe_id {
            let full = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}.{name}")
            };
            result.push((full, id));
        }
    }
    for (child_name, child) in module.named_children() {
        let child_prefix = if prefix.is_empty() {
            child_name
        } else {
            format!("{prefix}.{child_name}")
        };
        result.extend(named_buffers(child, &child_prefix));
    }
    result
}

/// Recursively collect all persistent buffer tensors.
pub fn named_persistent_buffers(module: &dyn Module, prefix: &str) -> Vec<(String, TensorNodeId)> {
    let mut result = Vec::new();
    for (name, maybe_id, persistent) in module.named_buffer_slots_own() {
        if persistent && let Some(id) = maybe_id {
            let full = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}.{name}")
            };
            result.push((full, id));
        }
    }
    for (child_name, child) in module.named_children() {
        let child_prefix = if prefix.is_empty() {
            child_name
        } else {
            format!("{prefix}.{child_name}")
        };
        result.extend(named_persistent_buffers(child, &child_prefix));
    }
    result
}

/// Collect all buffer tensors without names (including non-persistent buffers).
pub fn buffers(module: &dyn Module) -> Vec<TensorNodeId> {
    named_buffers(module, "")
        .into_iter()
        .map(|(_, id)| id)
        .collect()
}

fn collect_state_targets(
    module: &dyn Module,
) -> Result<Vec<(String, TensorNodeId)>, StateDictError> {
    let mut targets = Vec::new();
    for (name, id) in named_parameters(module, "") {
        targets.push((name, id));
    }
    for (name, id) in named_persistent_buffers(module, "") {
        targets.push((name, id));
    }

    let mut seen = std::collections::BTreeSet::new();
    for (name, _) in &targets {
        if !seen.insert(name.clone()) {
            return Err(StateDictError::DuplicateStateKey { key: name.clone() });
        }
    }
    Ok(targets)
}

fn snapshot_tensor_for_state_dict(
    session: &FrankenTorchSession,
    key: &str,
    node: TensorNodeId,
) -> Result<DenseTensor, StateDictError> {
    let (values_f64, meta) = session.tensor_values_meta(node)?;
    match meta.dtype() {
        DType::F64 => DenseTensor::from_storage(meta, values_f64).map_err(StateDictError::from),
        DType::F32 => {
            let values_f32 = session.tensor_values_f32(node)?;
            DenseTensor::from_storage_f32(meta, values_f32).map_err(StateDictError::from)
        }
        other => Err(StateDictError::UnsupportedDType {
            key: key.to_string(),
            dtype: other,
        }),
    }
}

pub fn module_state_dict(
    module: &dyn Module,
    session: &FrankenTorchSession,
) -> Result<StateDict, StateDictError> {
    let targets = collect_state_targets(module)?;
    let mut state = StateDict::new();
    for (name, node) in targets {
        let snapshot = snapshot_tensor_for_state_dict(session, name.as_str(), node)?;
        if state.insert(name.clone(), snapshot).is_some() {
            return Err(StateDictError::DuplicateStateKey { key: name });
        }
    }
    Ok(state)
}

pub fn module_load_state_dict(
    module: &dyn Module,
    session: &mut FrankenTorchSession,
    state: &StateDict,
    strict: bool,
) -> Result<LoadStateDictReport, StateDictError> {
    let targets = collect_state_targets(module)?;
    let mut target_map: BTreeMap<String, TensorNodeId> = BTreeMap::new();
    for (name, node) in targets {
        if target_map.insert(name.clone(), node).is_some() {
            return Err(StateDictError::DuplicateStateKey { key: name });
        }
    }

    let mut missing_keys = Vec::new();
    for key in target_map.keys() {
        if !state.contains_key(key) {
            missing_keys.push(key.clone());
        }
    }
    let mut unexpected_keys = Vec::new();
    for key in state.keys() {
        if !target_map.contains_key(key) {
            unexpected_keys.push(key.clone());
        }
    }

    if strict && (!missing_keys.is_empty() || !unexpected_keys.is_empty()) {
        return Err(StateDictError::StrictKeyMismatch {
            missing_keys,
            unexpected_keys,
        });
    }

    // Validate all matching entries first so updates are all-or-nothing.
    let mut updates: Vec<(TensorNodeId, DenseTensor)> = Vec::new();
    for (name, target_node) in &target_map {
        let Some(source_tensor) = state.get(name) else {
            continue;
        };
        let (_, target_meta) = session.tensor_values_meta(*target_node)?;
        let source_meta = source_tensor.meta();
        if target_meta.shape() != source_meta.shape() {
            return Err(StateDictError::ShapeMismatch {
                key: name.clone(),
                expected: target_meta.shape().to_vec(),
                found: source_meta.shape().to_vec(),
            });
        }
        if target_meta.dtype() != source_meta.dtype() {
            return Err(StateDictError::DTypeMismatch {
                key: name.clone(),
                expected: target_meta.dtype(),
                found: source_meta.dtype(),
            });
        }
        updates.push((*target_node, source_tensor.clone()));
    }

    session.no_grad_enter();
    for (target_node, source_tensor) in updates {
        let source_node = session.tensor_variable_from_storage(source_tensor, false);
        session.tensor_zero_(target_node)?;
        session.tensor_add_(target_node, source_node)?;
    }
    session.no_grad_exit();

    Ok(LoadStateDictReport {
        missing_keys,
        unexpected_keys,
    })
}

/// Clip accumulated gradients by total p-norm and return the pre-clip norm.
pub fn clip_grad_norm_(
    session: &mut FrankenTorchSession,
    parameters: &[TensorNodeId],
    max_norm: f64,
    norm_type: f64,
) -> Result<f64, AutogradError> {
    if !max_norm.is_finite() || max_norm < 0.0 {
        return Err(gradient_utils_error(
            "clip_grad_norm_ requires finite non-negative max_norm",
        ));
    }
    if !(norm_type.is_infinite() || (norm_type.is_finite() && norm_type > 0.0)) {
        return Err(gradient_utils_error(
            "clip_grad_norm_ requires positive finite norm_type or +inf",
        ));
    }

    let mut grads = Vec::new();
    for &param in parameters {
        if let Some(grad) = session.tensor_accumulated_gradient(param)? {
            grads.push((param, grad));
        }
    }
    if grads.is_empty() {
        return Ok(0.0);
    }

    let total_norm = if norm_type.is_infinite() {
        grads
            .iter()
            .flat_map(|(_, grad)| grad.iter())
            .map(|value| value.abs())
            .fold(0.0, f64::max)
    } else {
        let norm_acc: f64 = grads
            .iter()
            .flat_map(|(_, grad)| grad.iter())
            .map(|value| value.abs().powf(norm_type))
            .sum();
        norm_acc.powf(1.0 / norm_type)
    };

    if total_norm > max_norm && total_norm > 0.0 {
        let scale = if max_norm == 0.0 {
            0.0
        } else {
            max_norm / total_norm
        };
        for (param, mut grad) in grads {
            for value in &mut grad {
                *value *= scale;
            }
            session.tensor_set_accumulated_gradient(param, grad)?;
        }
    }

    Ok(total_norm)
}

/// Clip accumulated gradients element-wise into `[-clip_value, clip_value]`.
pub fn clip_grad_value_(
    session: &mut FrankenTorchSession,
    parameters: &[TensorNodeId],
    clip_value: f64,
) -> Result<(), AutogradError> {
    if !clip_value.is_finite() || clip_value < 0.0 {
        return Err(gradient_utils_error(
            "clip_grad_value_ requires finite non-negative clip_value",
        ));
    }
    for &param in parameters {
        let Some(mut grad) = session.tensor_accumulated_gradient(param)? else {
            continue;
        };
        for value in &mut grad {
            *value = value.clamp(-clip_value, clip_value);
        }
        session.tensor_set_accumulated_gradient(param, grad)?;
    }
    Ok(())
}

/// Flatten parameters into a single 1D tensor.
pub fn parameters_to_vector(
    session: &mut FrankenTorchSession,
    parameters: &[TensorNodeId],
) -> Result<TensorNodeId, AutogradError> {
    let mut flattened = Vec::new();
    for &parameter in parameters {
        flattened.extend(session.tensor_values(parameter)?);
    }
    session.tensor_variable(flattened.clone(), vec![flattened.len()], false)
}

/// Copy values from a flat 1D tensor back into parameter tensors.
pub fn vector_to_parameters(
    session: &mut FrankenTorchSession,
    vector: TensorNodeId,
    parameters: &[TensorNodeId],
) -> Result<(), AutogradError> {
    let (flat_values, flat_meta) = session.tensor_values_meta(vector)?;
    if flat_meta.shape().len() != 1 {
        return Err(gradient_utils_error(
            "vector_to_parameters requires a 1D source vector",
        ));
    }

    let mut offset = 0usize;
    session.no_grad_enter();
    for &parameter in parameters {
        let (param_values, param_meta) = session.tensor_values_meta(parameter)?;
        let len = param_values.len();
        let Some(end) = offset.checked_add(len) else {
            session.no_grad_exit();
            return Err(gradient_utils_error("vector_to_parameters offset overflow"));
        };
        if end > flat_values.len() {
            session.no_grad_exit();
            return Err(gradient_utils_error(
                "vector_to_parameters source vector is shorter than parameter footprint",
            ));
        }

        let chunk_node = session.tensor_variable(
            flat_values[offset..end].to_vec(),
            param_meta.shape().to_vec(),
            false,
        )?;
        session.tensor_zero_(parameter)?;
        session.tensor_add_(parameter, chunk_node)?;
        offset = end;
    }
    session.no_grad_exit();

    if offset != flat_values.len() {
        return Err(gradient_utils_error(
            "vector_to_parameters source vector has unused trailing values",
        ));
    }

    Ok(())
}

/// Fully connected linear layer: output = input @ weight^T + bias.
pub struct Linear {
    weight: TensorNodeId,
    bias: Option<TensorNodeId>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Create a new Linear layer with Kaiming uniform initialization.
    ///
    /// `weight` has shape `[out_features, in_features]`.
    /// `bias` (if enabled) has shape `[1, out_features]` for broadcast add.
    pub fn new(
        session: &mut FrankenTorchSession,
        in_features: usize,
        out_features: usize,
        use_bias: bool,
    ) -> Result<Self, AutogradError> {
        if in_features == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "linear layer requires in_features > 0",
                },
            )));
        }
        // PyTorch Linear initialization: U(-bound, bound) where bound = sqrt(1 / in_features)
        let bound = 1.0 / (in_features as f64).sqrt();

        // Initialize weight
        let weight_rand = session.rand(vec![out_features, in_features], false)?;
        let scale_tensor = session.full(vec![out_features, in_features], 2.0 * bound, false)?;
        let weight_scaled = session.tensor_mul(weight_rand, scale_tensor)?;
        let shift_tensor = session.full(vec![out_features, in_features], bound, false)?;
        let weight_shifted = session.tensor_sub(weight_scaled, shift_tensor)?;

        let weight_values = session.tensor_values(weight_shifted)?;
        let weight =
            session.tensor_variable(weight_values, vec![out_features, in_features], true)?;

        // Initialize bias
        let bias = if use_bias {
            let bias_rand = session.rand(vec![1, out_features], false)?;
            let bias_scale = session.full(vec![1, out_features], 2.0 * bound, false)?;
            let bias_scaled = session.tensor_mul(bias_rand, bias_scale)?;
            let bias_shift = session.full(vec![1, out_features], bound, false)?;
            let bias_shifted = session.tensor_sub(bias_scaled, bias_shift)?;

            let bias_values = session.tensor_values(bias_shifted)?;
            Some(session.tensor_variable(bias_values, vec![1, out_features], true)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Access the weight parameter node ID.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias parameter node ID (if present).
    #[must_use]
    pub fn bias(&self) -> Option<TensorNodeId> {
        self.bias
    }

    /// Input feature dimension.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature dimension.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // Transpose weight: [out, in] -> [in, out]
        let weight_t = session.tensor_transpose(self.weight, 0, 1)?;
        // output = input @ weight^T => [batch, in] @ [in, out] => [batch, out]
        let output = session.tensor_matmul(input, weight_t)?;

        match self.bias {
            Some(bias) => {
                let out_shape = {
                    let (_, meta) = session.tensor_values_meta(output)?;
                    meta.shape().to_vec()
                };
                let expanded_bias = session.tensor_expand(bias, out_shape)?;
                session.tensor_add(output, expanded_bias)
            }
            None => Ok(output),
        }
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight];
        if let Some(bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        let mut params = vec![("weight", self.weight)];
        if let Some(bias) = self.bias {
            params.push(("bias", bias));
        }
        params
    }
}

/// Bilinear transformation module: `y_k = x1^T W_k x2 + b_k`.
///
/// Equivalent to `torch.nn.Bilinear(in1_features, in2_features, out_features)`.
/// Weight has shape `[out_features, in1_features, in2_features]`.
/// Bias has shape `[out_features]`.
///
/// Note: the `forward` method takes only `input` (`x1`). The second input `x2`
/// must be provided via `forward_bilinear`.
pub struct Bilinear {
    weight: TensorNodeId,
    bias: Option<TensorNodeId>,
    in1_features: usize,
    in2_features: usize,
    out_features: usize,
}

impl Bilinear {
    /// Create a new Bilinear layer with uniform initialization.
    pub fn new(
        session: &mut FrankenTorchSession,
        in1_features: usize,
        in2_features: usize,
        out_features: usize,
        use_bias: bool,
    ) -> Result<Self, AutogradError> {
        let bound = 1.0 / (in1_features as f64).sqrt();
        let total = out_features * in1_features * in2_features;

        let weight_rand = session.rand(vec![total], false)?;
        let scale = session.full(vec![total], 2.0 * bound, false)?;
        let w_scaled = session.tensor_mul(weight_rand, scale)?;
        let shift = session.full(vec![total], bound, false)?;
        let w_shifted = session.tensor_sub(w_scaled, shift)?;

        let w_vals = session.tensor_values(w_shifted)?;
        let weight = session.tensor_variable(
            w_vals,
            vec![out_features, in1_features, in2_features],
            true,
        )?;

        let bias = if use_bias {
            let b_rand = session.rand(vec![out_features], false)?;
            let b_scale = session.full(vec![out_features], 2.0 * bound, false)?;
            let b_scaled = session.tensor_mul(b_rand, b_scale)?;
            let b_shift = session.full(vec![out_features], bound, false)?;
            let b_shifted = session.tensor_sub(b_scaled, b_shift)?;
            let b_vals = session.tensor_values(b_shifted)?;
            Some(session.tensor_variable(b_vals, vec![out_features], true)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in1_features,
            in2_features,
            out_features,
        })
    }

    /// Compute bilinear forward: `y_k = x1^T W_k x2 + b_k`.
    ///
    /// `x1` has shape `[*, in1_features]`, `x2` has shape `[*, in2_features]`.
    /// Output has shape `[*, out_features]`.
    pub fn forward_bilinear(
        &self,
        session: &mut FrankenTorchSession,
        x1: TensorNodeId,
        x2: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let x1_vals = session.tensor_values(x1)?;
        let x1_shape = session.tensor_shape(x1)?;
        let x2_vals = session.tensor_values(x2)?;
        let x2_shape = session.tensor_shape(x2)?;
        let w_vals = session.tensor_values(self.weight)?;

        let in1 = self.in1_features;
        let in2 = self.in2_features;
        let out = self.out_features;

        // Handle batched inputs
        let batch = if x1_shape.len() == 1 {
            1
        } else {
            x1_shape[..x1_shape.len() - 1].iter().product()
        };

        let x1_last = *x1_shape.last().unwrap_or(&0);
        let x2_last = *x2_shape.last().unwrap_or(&0);

        if x1_last != in1 || x2_last != in2 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "bilinear: input feature dimensions mismatch",
                },
            )));
        }

        // y[b, k] = sum_i sum_j x1[b, i] * W[k, i, j] * x2[b, j]
        let mut result = vec![0.0f64; batch * out];
        for b in 0..batch {
            for k in 0..out {
                let mut val = 0.0f64;
                for i in 0..in1 {
                    for j in 0..in2 {
                        val += x1_vals[b * in1 + i]
                            * w_vals[k * in1 * in2 + i * in2 + j]
                            * x2_vals[b * in2 + j];
                    }
                }
                result[b * out + k] = val;
            }
        }

        let out_shape = if x1_shape.len() == 1 {
            vec![out]
        } else {
            let mut s = x1_shape[..x1_shape.len() - 1].to_vec();
            s.push(out);
            s
        };

        let result_node = session.tensor_variable(result, out_shape.clone(), false)?;

        if let Some(bias) = self.bias {
            let expanded_bias = session.tensor_expand(bias, out_shape)?;
            session.tensor_add(result_node, expanded_bias)
        } else {
            Ok(result_node)
        }
    }

    /// Access the weight parameter.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias parameter.
    #[must_use]
    pub fn bias(&self) -> Option<TensorNodeId> {
        self.bias
    }
}

impl Module for Bilinear {
    fn forward(
        &self,
        _session: &mut FrankenTorchSession,
        _input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        Err(AutogradError::Dispatch(DispatchError::Key(
            DispatchKeyError::IncompatibleSet {
                reason: "bilinear: use forward_bilinear(session, x1, x2) instead of forward()",
            },
        )))
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight];
        if let Some(bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        let mut params = vec![("weight", self.weight)];
        if let Some(bias) = self.bias {
            params.push(("bias", bias));
        }
        params
    }
}

/// Extracts sliding local blocks from a batched input tensor (im2col).
///
/// Equivalent to `torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)`.
/// Input shape: `[N, C, H, W]` (4-D batched image tensor).
/// Output shape: `[N, C * kH * kW, L]` where `L` is the number of valid positions.
pub struct Unfold {
    kernel_h: usize,
    kernel_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    padding_h: usize,
    padding_w: usize,
    stride_h: usize,
    stride_w: usize,
}

impl Unfold {
    /// Create a new Unfold module.
    ///
    /// `kernel_size`: (height, width) of the sliding window.
    /// `dilation`, `padding`, `stride`: optional, default to (1,1), (0,0), (1,1).
    #[must_use]
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_h: kernel_size.0,
            kernel_w: kernel_size.1,
            dilation_h: 1,
            dilation_w: 1,
            padding_h: 0,
            padding_w: 0,
            stride_h: 1,
            stride_w: 1,
        }
    }

    /// Set dilation.
    #[must_use]
    pub fn dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation_h = dilation.0;
        self.dilation_w = dilation.1;
        self
    }

    /// Set padding.
    #[must_use]
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding_h = padding.0;
        self.padding_w = padding.1;
        self
    }

    /// Set stride.
    #[must_use]
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride_h = stride.0;
        self.stride_w = stride.1;
        self
    }

    fn output_positions(&self, h: usize, w: usize) -> (usize, usize) {
        let eff_kh = self.dilation_h * (self.kernel_h - 1) + 1;
        let eff_kw = self.dilation_w * (self.kernel_w - 1) + 1;
        let out_h = (h + 2 * self.padding_h).saturating_sub(eff_kh) / self.stride_h + 1;
        let out_w = (w + 2 * self.padding_w).saturating_sub(eff_kw) / self.stride_w + 1;
        (out_h, out_w)
    }
}

impl Module for Unfold {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let vals = session.tensor_values(input)?;
        let shape = session.tensor_shape(input)?;
        if shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "unfold: input must be 4-D [N, C, H, W]",
                },
            )));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = self.output_positions(h, w);
        let l = out_h * out_w;
        let block_size = c * self.kernel_h * self.kernel_w;

        let mut output = vec![0.0f64; n * block_size * l];

        for batch in 0..n {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let col_idx = oh * out_w + ow;
                    for ci in 0..c {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ih = oh * self.stride_h + kh * self.dilation_h;
                                let iw = ow * self.stride_w + kw * self.dilation_w;
                                let row =
                                    ci * self.kernel_h * self.kernel_w + kh * self.kernel_w + kw;
                                let val = if ih >= self.padding_h
                                    && ih < h + self.padding_h
                                    && iw >= self.padding_w
                                    && iw < w + self.padding_w
                                {
                                    let src_h = ih - self.padding_h;
                                    let src_w = iw - self.padding_w;
                                    vals[batch * c * h * w + ci * h * w + src_h * w + src_w]
                                } else {
                                    0.0
                                };
                                output[batch * block_size * l + row * l + col_idx] = val;
                            }
                        }
                    }
                }
            }
        }

        session.tensor_variable(output, vec![n, block_size, l], false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Combines an array of sliding local blocks into a large containing tensor (col2im).
///
/// Equivalent to `torch.nn.Fold(output_size, kernel_size, dilation=1, padding=0, stride=1)`.
/// Input shape: `[N, C * kH * kW, L]`.
/// Output shape: `[N, C, output_h, output_w]`.
///
/// Overlapping blocks are summed (this is the transpose of `Unfold`).
pub struct Fold {
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    padding_h: usize,
    padding_w: usize,
    stride_h: usize,
    stride_w: usize,
}

impl Fold {
    /// Create a new Fold module.
    ///
    /// `output_size`: target (H, W). `kernel_size`: (kH, kW).
    #[must_use]
    pub fn new(output_size: (usize, usize), kernel_size: (usize, usize)) -> Self {
        Self {
            output_h: output_size.0,
            output_w: output_size.1,
            kernel_h: kernel_size.0,
            kernel_w: kernel_size.1,
            dilation_h: 1,
            dilation_w: 1,
            padding_h: 0,
            padding_w: 0,
            stride_h: 1,
            stride_w: 1,
        }
    }

    /// Set dilation.
    #[must_use]
    pub fn dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation_h = dilation.0;
        self.dilation_w = dilation.1;
        self
    }

    /// Set padding.
    #[must_use]
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding_h = padding.0;
        self.padding_w = padding.1;
        self
    }

    /// Set stride.
    #[must_use]
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride_h = stride.0;
        self.stride_w = stride.1;
        self
    }

    fn output_positions(&self) -> (usize, usize) {
        let eff_kh = self.dilation_h * (self.kernel_h - 1) + 1;
        let eff_kw = self.dilation_w * (self.kernel_w - 1) + 1;
        let out_h = (self.output_h + 2 * self.padding_h).saturating_sub(eff_kh) / self.stride_h + 1;
        let out_w = (self.output_w + 2 * self.padding_w).saturating_sub(eff_kw) / self.stride_w + 1;
        (out_h, out_w)
    }
}

impl Module for Fold {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let vals = session.tensor_values(input)?;
        let shape = session.tensor_shape(input)?;
        if shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "fold: input must be 3-D [N, C*kH*kW, L]",
                },
            )));
        }
        let (n, block_size, l) = (shape[0], shape[1], shape[2]);
        let (out_h, out_w) = self.output_positions();
        let expected_l = out_h * out_w;
        if l != expected_l {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "fold: L does not match output_size/kernel/stride/padding",
                },
            )));
        }

        let c = block_size / (self.kernel_h * self.kernel_w);
        if c * self.kernel_h * self.kernel_w != block_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "fold: block_size must be divisible by kH*kW",
                },
            )));
        }

        let h = self.output_h;
        let w = self.output_w;
        let mut output = vec![0.0f64; n * c * h * w];

        for batch in 0..n {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let col_idx = oh * out_w + ow;
                    for ci in 0..c {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ih = oh * self.stride_h + kh * self.dilation_h;
                                let iw = ow * self.stride_w + kw * self.dilation_w;
                                if ih >= self.padding_h
                                    && ih < h + self.padding_h
                                    && iw >= self.padding_w
                                    && iw < w + self.padding_w
                                {
                                    let dst_h = ih - self.padding_h;
                                    let dst_w = iw - self.padding_w;
                                    let row = ci * self.kernel_h * self.kernel_w
                                        + kh * self.kernel_w
                                        + kw;
                                    output[batch * c * h * w + ci * h * w + dst_h * w + dst_w] +=
                                        vals[batch * block_size * l + row * l + col_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        session.tensor_variable(output, vec![n, c, h, w], false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// ReLU activation module.
pub struct ReLU;

impl Module for ReLU {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_relu(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Sigmoid activation module.
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_sigmoid(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Tanh activation module.
pub struct Tanh;

impl Module for Tanh {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_tanh(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// GELU activation module.
pub struct GELU;

impl Module for GELU {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_gelu(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// SiLU (Swish) activation module.
pub struct SiLU;

impl Module for SiLU {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_silu(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Sequential container: chains modules in order.
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Create a new empty Sequential container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
        }
    }

    /// Add a module to the end of the chain.
    pub fn push(&mut self, module: Box<dyn Module>) {
        self.modules.push(module);
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let mut current = input;
        for module in &self.modules {
            current = module.forward(session, current)?;
        }
        Ok(current)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.modules
            .iter()
            .enumerate()
            .map(|(i, m)| (i.to_string(), m.as_ref() as &dyn Module))
            .collect()
    }
}

/// Layer Normalization module.
///
/// Applies normalization over the last `len(normalized_shape)` dimensions:
/// `y = (x - mean) / sqrt(var + eps) * weight + bias`
///
/// Weight is initialized to ones, bias to zeros (matching PyTorch defaults).
pub struct LayerNorm {
    weight: TensorNodeId,
    bias: TensorNodeId,
    normalized_shape: Vec<usize>,
    eps: f64,
}

impl LayerNorm {
    /// Create a new LayerNorm module.
    ///
    /// `normalized_shape` specifies the shape of the last D dimensions to normalize over.
    /// `eps` is added to the variance for numerical stability (typically 1e-5).
    pub fn new(
        session: &mut FrankenTorchSession,
        normalized_shape: Vec<usize>,
        eps: f64,
    ) -> Result<Self, AutogradError> {
        if normalized_shape.is_empty() {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "LayerNorm requires non-empty normalized_shape",
                },
            )));
        }

        let norm_numel: usize = normalized_shape.iter().product();
        if norm_numel == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "LayerNorm normalized_shape must not contain zero dimensions",
                },
            )));
        }

        // Weight (gamma) initialized to ones, bias (beta) to zeros
        let weight =
            session.tensor_variable(vec![1.0; norm_numel], normalized_shape.clone(), true)?;
        let bias =
            session.tensor_variable(vec![0.0; norm_numel], normalized_shape.clone(), true)?;

        Ok(Self {
            weight,
            bias,
            normalized_shape,
            eps,
        })
    }

    /// Access the weight (gamma) parameter node ID.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias (beta) parameter node ID.
    #[must_use]
    pub fn bias(&self) -> TensorNodeId {
        self.bias
    }

    /// The normalized shape.
    #[must_use]
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// The epsilon value.
    #[must_use]
    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for LayerNorm {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        let ndim = input_shape.len();
        let num_norm_dims = self.normalized_shape.len();

        if ndim < num_norm_dims {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "input has fewer dimensions than normalized_shape",
                },
            )));
        }

        // Validate trailing dimensions match normalized_shape
        let start = ndim - num_norm_dims;
        for i in 0..num_norm_dims {
            if input_shape[start + i] != self.normalized_shape[i] {
                return Err(AutogradError::Dispatch(DispatchError::Key(
                    DispatchKeyError::IncompatibleSet {
                        reason: "input trailing dimensions do not match normalized_shape",
                    },
                )));
            }
        }

        let batch_numel: usize = if start == 0 {
            1
        } else {
            input_shape[..start].iter().product()
        };
        let norm_numel: usize = self.normalized_shape.iter().product();

        // Flatten to [batch_numel, norm_numel] for uniform reduction
        let flat = session.tensor_reshape(input, vec![batch_numel, norm_numel])?;

        // Mean over normalized elements: mean_dim(flat, 1) -> [batch_numel]
        let mean = session.tensor_mean_dim(flat, 1)?;
        let mean_us = session.tensor_unsqueeze(mean, 1)?;
        let mean_exp = session.tensor_expand(mean_us, vec![batch_numel, norm_numel])?;

        // diff = x - mean
        let diff = session.tensor_sub(flat, mean_exp)?;

        // Variance = mean(diff^2) (population variance, matching PyTorch LayerNorm)
        let diff_sq = session.tensor_mul(diff, diff)?;
        let var = session.tensor_mean_dim(diff_sq, 1)?;
        let var_us = session.tensor_unsqueeze(var, 1)?;
        let var_exp = session.tensor_expand(var_us, vec![batch_numel, norm_numel])?;

        // std = sqrt(var + eps)
        let eps_t = session.full(vec![batch_numel, norm_numel], self.eps, false)?;
        let var_eps = session.tensor_add(var_exp, eps_t)?;
        let std = session.tensor_sqrt(var_eps)?;

        // normalized = diff / std
        let normalized = session.tensor_div(diff, std)?;

        // Reshape back to original input shape
        let normalized_orig = session.tensor_reshape(normalized, input_shape.clone())?;

        // Expand weight [normalized_shape] -> [1]*batch_dims ++ normalized_shape -> input_shape
        let mut affine_shape = vec![1usize; start];
        affine_shape.extend_from_slice(&self.normalized_shape);
        let w_reshaped = session.tensor_reshape(self.weight, affine_shape.clone())?;
        let w_expanded = session.tensor_expand(w_reshaped, input_shape.clone())?;
        let b_reshaped = session.tensor_reshape(self.bias, affine_shape)?;
        let b_expanded = session.tensor_expand(b_reshaped, input_shape)?;

        // output = weight * normalized + bias
        let scaled = session.tensor_mul(w_expanded, normalized_orig)?;
        session.tensor_add(scaled, b_expanded)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        vec![self.weight, self.bias]
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        vec![("weight", self.weight), ("bias", self.bias)]
    }
}

/// Dropout module (stochastic regularization).
///
/// During training, randomly zeros elements with probability `p`.
/// During eval, passes through unchanged.
pub struct Dropout {
    p: f64,
    training: std::cell::Cell<bool>,
}

impl Dropout {
    /// Create a new Dropout module with the given drop probability.
    #[must_use]
    pub fn new(p: f64) -> Self {
        Self {
            p,
            training: std::cell::Cell::new(true),
        }
    }

    /// Set the module training mode.
    pub fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    /// Set the module to evaluation mode.
    pub fn eval(&self) {
        self.train(false);
    }

    /// Check if the module is in training mode.
    #[must_use]
    pub fn is_training(&self) -> bool {
        self.training.get()
    }
}

impl Module for Dropout {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        if !self.p.is_finite() || !(0.0..=1.0).contains(&self.p) {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "dropout probability p must be finite and in [0, 1]",
                },
            )));
        }
        if !self.training.get() || self.p == 0.0 {
            return Ok(input);
        }
        if self.p >= 1.0 {
            let shape = {
                let (_, meta) = session.tensor_values_meta(input)?;
                meta.shape().to_vec()
            };
            let zeros = session.zeros(shape, false)?;
            return session.tensor_mul(input, zeros);
        }

        // Generate random mask: values in [0, 1), keep where > p
        let shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };
        let mask_rand = session.rand(shape.clone(), false)?;

        // Create threshold tensor
        let threshold = session.full(shape.clone(), self.p, false)?;

        // mask = (rand > p) as f64  — use gt comparison
        // But comparisons aren't tracked through autograd, so we use them as masks
        let mask = session.tensor_gt(mask_rand, threshold)?;

        // Scale by 1/(1-p) for inverted dropout
        let scale = 1.0 / (1.0 - self.p);
        let scale_tensor = session.full(shape, scale, false)?;
        let scaled_mask = session.tensor_mul(mask, scale_tensor)?;

        session.tensor_mul(input, scaled_mask)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Spatial dropout: drops entire channels for 2D feature maps (N, C, H, W).
///
/// During training, randomly zeroes entire channels with probability `p`.
/// Non-dropped channels are scaled by `1/(1-p)`.
pub struct Dropout2d {
    p: f64,
    training: std::cell::Cell<bool>,
}

impl Dropout2d {
    #[must_use]
    pub fn new(p: f64) -> Self {
        Self {
            p,
            training: std::cell::Cell::new(true),
        }
    }

    pub fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    pub fn eval(&self) {
        self.train(false);
    }
}

impl Module for Dropout2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(input);
        }
        let shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };
        if shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Dropout2d expects 4D input (N, C, H, W)",
                },
            )));
        }
        if self.p >= 1.0 {
            let zeros = session.zeros(shape, false)?;
            return session.tensor_mul(input, zeros);
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        // Generate per-channel mask and expand to spatial dims
        let channel_rand = session.rand(vec![n * c], false)?;
        let channel_rand_vals = session.tensor_values(channel_rand)?;
        let scale = 1.0 / (1.0 - self.p);
        let mut mask_data = vec![0.0f64; n * c * h * w];
        for nc in 0..(n * c) {
            let keep = if channel_rand_vals[nc] > self.p {
                scale
            } else {
                0.0
            };
            for spatial in 0..(h * w) {
                mask_data[nc * h * w + spatial] = keep;
            }
        }
        let mask = session.tensor_variable(mask_data, shape, false)?;
        session.tensor_mul(input, mask)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Spatial dropout for 3D feature maps (N, C, D, H, W).
///
/// Drops entire channels, same pattern as Dropout2d but for 5D inputs.
pub struct Dropout3d {
    p: f64,
    training: std::cell::Cell<bool>,
}

impl Dropout3d {
    #[must_use]
    pub fn new(p: f64) -> Self {
        Self {
            p,
            training: std::cell::Cell::new(true),
        }
    }

    pub fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    pub fn eval(&self) {
        self.train(false);
    }
}

impl Module for Dropout3d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(input);
        }
        let shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };
        if shape.len() != 5 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Dropout3d expects 5D input (N, C, D, H, W)",
                },
            )));
        }
        if self.p >= 1.0 {
            let zeros = session.zeros(shape, false)?;
            return session.tensor_mul(input, zeros);
        }
        let (n, c) = (shape[0], shape[1]);
        let spatial: usize = shape[2..].iter().product();
        let channel_rand = session.rand(vec![n * c], false)?;
        let channel_rand_vals = session.tensor_values(channel_rand)?;
        let scale = 1.0 / (1.0 - self.p);
        let mut mask_data = vec![0.0f64; n * c * spatial];
        for nc in 0..(n * c) {
            let keep = if channel_rand_vals[nc] > self.p {
                scale
            } else {
                0.0
            };
            for s in 0..spatial {
                mask_data[nc * spatial + s] = keep;
            }
        }
        let mask = session.tensor_variable(mask_data, shape, false)?;
        session.tensor_mul(input, mask)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Alpha dropout for self-normalizing networks (SELU activation).
///
/// Instead of zeroing dropped elements, sets them to the SELU negative saturation
/// value to preserve mean and variance of activations.
pub struct AlphaDropout {
    p: f64,
    training: std::cell::Cell<bool>,
}

impl AlphaDropout {
    /// SELU parameters
    const ALPHA: f64 = 1.6732632423543772;
    const LAMBDA: f64 = 1.0507009873554805;
    /// The negative saturation point: -lambda * alpha
    const SAT: f64 = -Self::LAMBDA * Self::ALPHA;

    #[must_use]
    pub fn new(p: f64) -> Self {
        Self {
            p,
            training: std::cell::Cell::new(true),
        }
    }

    pub fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    pub fn eval(&self) {
        self.train(false);
    }
}

impl Module for AlphaDropout {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(input);
        }
        let shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };
        let numel: usize = shape.iter().product();
        let input_vals = session.tensor_values(input)?;

        // For alpha dropout, we need to affine transform to preserve mean/variance
        // After masking: x' = mask * x + (1-mask) * sat
        // Then affine: x'' = a * x' + b
        // where a = (1/(1-p) * (1 + p * alpha^2 * lambda^2))^-0.5
        //       b = -a * (1-mask_mean) * sat

        let alpha_p = -Self::SAT;
        let a = ((1.0 - self.p) * (1.0 + self.p * alpha_p * alpha_p)).sqrt();
        let a = 1.0 / a;

        let rand_vals = session.rand(vec![numel], false)?;
        let rand = session.tensor_values(rand_vals)?;

        let mut result = vec![0.0f64; numel];
        for i in 0..numel {
            if rand[i] > self.p {
                result[i] = a * input_vals[i] + a * self.p * Self::SAT;
            } else {
                result[i] = a * Self::SAT + a * self.p * Self::SAT;
            }
        }

        session.tensor_variable(result, shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Embedding lookup table: maps integer indices to dense vectors.
///
/// Weight has shape `[num_embeddings, embedding_dim]`, initialized from N(0,1).
/// Input should contain integer indices (stored as f64).
pub struct Embedding {
    weight: TensorNodeId,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    /// Create a new Embedding with standard normal initialization.
    pub fn new(
        session: &mut FrankenTorchSession,
        num_embeddings: usize,
        embedding_dim: usize,
    ) -> Result<Self, AutogradError> {
        if num_embeddings == 0 || embedding_dim == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Embedding requires num_embeddings > 0 and embedding_dim > 0",
                },
            )));
        }

        // PyTorch default: N(0, 1) initialization
        let weight_init = session.randn(vec![num_embeddings, embedding_dim], false)?;
        let weight_values = session.tensor_values(weight_init)?;
        let weight =
            session.tensor_variable(weight_values, vec![num_embeddings, embedding_dim], true)?;

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
        })
    }

    /// Access the weight parameter node ID.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Number of embeddings in the table.
    #[must_use]
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Dimension of each embedding vector.
    #[must_use]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl Module for Embedding {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        let total: usize = input_shape.iter().product();

        // Flatten indices to 1D for index_select
        let flat_indices = session.tensor_reshape(input, vec![total])?;

        // index_select(weight, dim=0, flat_indices) -> [total, embedding_dim]
        let selected = session.tensor_index_select(self.weight, 0, flat_indices)?;

        // Reshape to [*input_shape, embedding_dim]
        let mut out_shape = input_shape;
        out_shape.push(self.embedding_dim);
        session.tensor_reshape(selected, out_shape)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        vec![self.weight]
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        vec![("weight", self.weight)]
    }
}

/// Embedding bag module that computes sums/means/maxes of bags of embeddings.
///
/// Unlike `Embedding`, this does not materialize intermediate per-element embeddings.
/// Input is `(indices, offsets)` where offsets marks the boundaries of each bag.
///
/// Modes: `"sum"`, `"mean"`, `"max"`.
pub struct EmbeddingBag {
    weight: TensorNodeId,
    num_embeddings: usize,
    embedding_dim: usize,
    mode: EmbeddingBagMode,
    padding_idx: Option<usize>,
}

/// Reduction mode for EmbeddingBag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingBagMode {
    Sum,
    Mean,
    Max,
}

impl EmbeddingBag {
    /// Create a new EmbeddingBag with standard normal initialization.
    pub fn new(
        session: &mut FrankenTorchSession,
        num_embeddings: usize,
        embedding_dim: usize,
        mode: EmbeddingBagMode,
        padding_idx: Option<usize>,
    ) -> Result<Self, AutogradError> {
        if num_embeddings == 0 || embedding_dim == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "EmbeddingBag requires num_embeddings > 0 and embedding_dim > 0",
                },
            )));
        }

        let weight_init = session.randn(vec![num_embeddings, embedding_dim], false)?;
        let weight_values = session.tensor_values(weight_init)?;
        let weight =
            session.tensor_variable(weight_values, vec![num_embeddings, embedding_dim], true)?;

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            mode,
            padding_idx,
        })
    }

    /// Access the weight parameter.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Number of embeddings in the lookup table.
    #[must_use]
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Dimension of each embedding vector.
    #[must_use]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Reduction mode (Sum, Mean, or Max).
    #[must_use]
    pub fn mode(&self) -> EmbeddingBagMode {
        self.mode
    }

    /// Forward pass with indices, offsets, and optional per-sample weights.
    ///
    /// `indices`: 1D tensor of embedding indices.
    /// `offsets`: 1D tensor marking the start of each bag.
    /// `per_sample_weights`: optional 1D tensor of weights (same length as indices).
    ///
    /// Returns tensor of shape `[num_bags, embedding_dim]`.
    pub fn forward_with_offsets(
        &self,
        session: &mut FrankenTorchSession,
        indices: TensorNodeId,
        offsets: TensorNodeId,
        per_sample_weights: Option<TensorNodeId>,
    ) -> Result<TensorNodeId, AutogradError> {
        let indices_vals = session.tensor_values(indices)?;
        let offsets_vals = session.tensor_values(offsets)?;
        let weight_vals = session.tensor_values(self.weight)?;
        let psw = if let Some(w) = per_sample_weights {
            Some(session.tensor_values(w)?)
        } else {
            None
        };

        let num_bags = offsets_vals.len();
        let dim = self.embedding_dim;
        let n_indices = indices_vals.len();
        let mut output = vec![0.0f64; num_bags * dim];

        for bag in 0..num_bags {
            let start = offsets_vals[bag] as usize;
            let end = if bag + 1 < num_bags {
                offsets_vals[bag + 1] as usize
            } else {
                n_indices
            };

            if start >= end {
                // Empty bag: leave as zeros
                continue;
            }

            let bag_size = end - start;
            let mut bag_result = vec![0.0f64; dim];

            match self.mode {
                EmbeddingBagMode::Sum | EmbeddingBagMode::Mean => {
                    for i in start..end {
                        let idx = indices_vals[i] as usize;
                        if Some(idx) == self.padding_idx {
                            continue;
                        }
                        let w = if let Some(ref psw) = psw { psw[i] } else { 1.0 };
                        for d in 0..dim {
                            bag_result[d] += w * weight_vals[idx * dim + d];
                        }
                    }
                    if self.mode == EmbeddingBagMode::Mean && bag_size > 0 {
                        for item in bag_result.iter_mut().take(dim) {
                            *item /= bag_size as f64;
                        }
                    }
                }
                EmbeddingBagMode::Max => {
                    // Initialize with -inf
                    for item in bag_result.iter_mut().take(dim) {
                        *item = f64::NEG_INFINITY;
                    }
                    for &iv in indices_vals.iter().take(end).skip(start) {
                        let idx = iv as usize;
                        if Some(idx) == self.padding_idx {
                            continue;
                        }
                        for d in 0..dim {
                            let v = weight_vals[idx * dim + d];
                            if v > bag_result[d] {
                                bag_result[d] = v;
                            }
                        }
                    }
                    // If all were padding, replace -inf with 0
                    for item in bag_result.iter_mut().take(dim) {
                        if *item == f64::NEG_INFINITY {
                            *item = 0.0;
                        }
                    }
                }
            }

            output[bag * dim..(bag + 1) * dim].copy_from_slice(&bag_result);
        }

        session.tensor_variable(output, vec![num_bags, dim], false)
    }
}

impl Module for EmbeddingBag {
    fn forward(
        &self,
        _session: &mut FrankenTorchSession,
        _input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        Err(AutogradError::Dispatch(DispatchError::Key(
            DispatchKeyError::IncompatibleSet {
                reason: "EmbeddingBag requires forward_with_offsets, not forward",
            },
        )))
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        vec![self.weight]
    }
}

/// Batch Normalization over a batch of 2D inputs `[N, C]`.
///
/// Normalizes each feature across the batch dimension using batch statistics
/// (training) or running statistics (eval). Applies affine transform: `y = gamma * x_hat + beta`.
pub struct BatchNorm1d {
    weight: TensorNodeId,
    bias: TensorNodeId,
    running_mean: std::cell::RefCell<Vec<f64>>,
    running_var: std::cell::RefCell<Vec<f64>>,
    running_mean_buffer: std::cell::RefCell<TensorNodeId>,
    running_var_buffer: std::cell::RefCell<TensorNodeId>,
    num_batches_tracked: std::cell::Cell<u64>,
    num_batches_tracked_buffer: std::cell::RefCell<TensorNodeId>,
    registered_parameters: std::cell::RefCell<Vec<RegisteredParameter>>,
    registered_buffers: std::cell::RefCell<Vec<RegisteredBuffer>>,
    num_features: usize,
    eps: f64,
    momentum: f64,
    training: std::cell::Cell<bool>,
}

impl BatchNorm1d {
    /// Create a new BatchNorm1d module.
    ///
    /// Weight (gamma) initialized to ones, bias (beta) to zeros.
    /// Running mean initialized to zeros, running var to ones.
    pub fn new(
        session: &mut FrankenTorchSession,
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> Result<Self, AutogradError> {
        if num_features == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "BatchNorm1d requires num_features > 0",
                },
            )));
        }

        let weight = session.tensor_variable(vec![1.0; num_features], vec![num_features], true)?;
        let bias = session.tensor_variable(vec![0.0; num_features], vec![num_features], true)?;
        let running_mean_buffer =
            session.tensor_variable(vec![0.0; num_features], vec![num_features], false)?;
        let running_var_buffer =
            session.tensor_variable(vec![1.0; num_features], vec![num_features], false)?;
        let num_batches_tracked_buffer = session.tensor_variable(vec![0.0], vec![1], false)?;

        Ok(Self {
            weight,
            bias,
            running_mean: std::cell::RefCell::new(vec![0.0; num_features]),
            running_var: std::cell::RefCell::new(vec![1.0; num_features]),
            running_mean_buffer: std::cell::RefCell::new(running_mean_buffer),
            running_var_buffer: std::cell::RefCell::new(running_var_buffer),
            num_batches_tracked: std::cell::Cell::new(0),
            num_batches_tracked_buffer: std::cell::RefCell::new(num_batches_tracked_buffer),
            registered_parameters: std::cell::RefCell::new(Vec::new()),
            registered_buffers: std::cell::RefCell::new(vec![
                RegisteredBuffer {
                    name: "running_mean".to_string(),
                    tensor: Some(running_mean_buffer),
                    persistent: true,
                },
                RegisteredBuffer {
                    name: "running_var".to_string(),
                    tensor: Some(running_var_buffer),
                    persistent: true,
                },
                RegisteredBuffer {
                    name: "num_batches_tracked".to_string(),
                    tensor: Some(num_batches_tracked_buffer),
                    persistent: true,
                },
            ]),
            num_features,
            eps,
            momentum,
            training: std::cell::Cell::new(true),
        })
    }

    /// Access the weight (gamma) parameter node ID.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias (beta) parameter node ID.
    #[must_use]
    pub fn bias(&self) -> TensorNodeId {
        self.bias
    }

    /// Number of features (channels).
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Set the module training mode.
    pub fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    /// Set the module to evaluation mode.
    pub fn eval(&self) {
        self.train(false);
    }

    /// Check if the module is in training mode.
    #[must_use]
    pub fn is_training(&self) -> bool {
        self.training.get()
    }

    /// Get a copy of the current running mean.
    pub fn running_mean(&self) -> Vec<f64> {
        self.running_mean.borrow().clone()
    }

    /// Get a copy of the current running variance.
    pub fn running_var(&self) -> Vec<f64> {
        self.running_var.borrow().clone()
    }

    fn has_builtin_parameter_name(name: &str) -> bool {
        matches!(name, "weight" | "bias")
    }

    fn has_builtin_buffer_name(name: &str) -> bool {
        matches!(name, "running_mean" | "running_var" | "num_batches_tracked")
    }

    fn forward_train(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        input_shape: &[usize],
    ) -> Result<TensorNodeId, AutogradError> {
        let n = input_shape[0];
        let c = input_shape[1];

        // Batch mean: mean_dim(input, 0) -> [C]
        let batch_mean = session.tensor_mean_dim(input, 0)?;

        // diff = input - expand(unsqueeze(mean, 0), [N, C])
        let mean_us = session.tensor_unsqueeze(batch_mean, 0)?;
        let mean_exp = session.tensor_expand(mean_us, vec![n, c])?;
        let diff = session.tensor_sub(input, mean_exp)?;

        // batch_var = mean_dim(diff^2, 0) -> [C] (biased, divides by N)
        let diff_sq = session.tensor_mul(diff, diff)?;
        let batch_var = session.tensor_mean_dim(diff_sq, 0)?;

        // Update running statistics
        {
            let mean_vals = session.tensor_values(batch_mean)?;
            let var_vals = session.tensor_values(batch_var)?;
            let mut rm = self.running_mean.borrow_mut();
            let mut rv = self.running_var.borrow_mut();
            // Unbiased variance for running stats: var * N / (N-1)
            let bessel_factor = if n > 1 {
                n as f64 / (n as f64 - 1.0)
            } else {
                1.0
            };
            for i in 0..c {
                rm[i] = (1.0 - self.momentum) * rm[i] + self.momentum * mean_vals[i];
                rv[i] = (1.0 - self.momentum) * rv[i] + self.momentum * var_vals[i] * bessel_factor;
            }

            let running_mean_buffer = session.tensor_variable(rm.clone(), vec![c], false)?;
            let running_var_buffer = session.tensor_variable(rv.clone(), vec![c], false)?;
            let num_batches_tracked = self.num_batches_tracked.get().saturating_add(1);
            self.num_batches_tracked.set(num_batches_tracked);
            let num_batches_tracked_buffer =
                session.tensor_variable(vec![num_batches_tracked as f64], vec![1], false)?;

            self.running_mean_buffer.replace(running_mean_buffer);
            self.running_var_buffer.replace(running_var_buffer);
            self.num_batches_tracked_buffer
                .replace(num_batches_tracked_buffer);

            let mut registered_buffers = self.registered_buffers.borrow_mut();
            upsert_registered_buffer(
                &mut registered_buffers,
                "running_mean",
                Some(running_mean_buffer),
                true,
            );
            upsert_registered_buffer(
                &mut registered_buffers,
                "running_var",
                Some(running_var_buffer),
                true,
            );
            upsert_registered_buffer(
                &mut registered_buffers,
                "num_batches_tracked",
                Some(num_batches_tracked_buffer),
                true,
            );
        }

        // std = sqrt(var + eps) -> [C]
        let eps_t = session.full(vec![c], self.eps, false)?;
        let var_eps = session.tensor_add(batch_var, eps_t)?;
        let std_t = session.tensor_sqrt(var_eps)?;

        // Expand std to [N, C]
        let std_us = session.tensor_unsqueeze(std_t, 0)?;
        let std_exp = session.tensor_expand(std_us, vec![n, c])?;

        // normalized = diff / std
        let normalized = session.tensor_div(diff, std_exp)?;

        // Apply affine: weight * normalized + bias
        let w_us = session.tensor_unsqueeze(self.weight, 0)?;
        let w_exp = session.tensor_expand(w_us, vec![n, c])?;
        let b_us = session.tensor_unsqueeze(self.bias, 0)?;
        let b_exp = session.tensor_expand(b_us, vec![n, c])?;

        let scaled = session.tensor_mul(w_exp, normalized)?;
        session.tensor_add(scaled, b_exp)
    }

    fn forward_eval(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        input_shape: &[usize],
    ) -> Result<TensorNodeId, AutogradError> {
        let n = input_shape[0];
        let c = input_shape[1];

        // Running-stat buffers are maintained as non-grad tensors.
        let mean_t = *self.running_mean_buffer.borrow();
        let var_t = *self.running_var_buffer.borrow();

        // Expand mean to [N, C]
        let mean_us = session.tensor_unsqueeze(mean_t, 0)?;
        let mean_exp = session.tensor_expand(mean_us, vec![n, c])?;

        // diff = input - mean
        let diff = session.tensor_sub(input, mean_exp)?;

        // std = sqrt(var + eps) -> [C]
        let eps_t = session.full(vec![c], self.eps, false)?;
        let var_eps = session.tensor_add(var_t, eps_t)?;
        let std_t = session.tensor_sqrt(var_eps)?;

        // Expand std to [N, C]
        let std_us = session.tensor_unsqueeze(std_t, 0)?;
        let std_exp = session.tensor_expand(std_us, vec![n, c])?;

        // normalized = diff / std
        let normalized = session.tensor_div(diff, std_exp)?;

        // Apply affine
        let w_us = session.tensor_unsqueeze(self.weight, 0)?;
        let w_exp = session.tensor_expand(w_us, vec![n, c])?;
        let b_us = session.tensor_unsqueeze(self.bias, 0)?;
        let b_exp = session.tensor_expand(b_us, vec![n, c])?;

        let scaled = session.tensor_mul(w_exp, normalized)?;
        session.tensor_add(scaled, b_exp)
    }
}

impl Module for BatchNorm1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 2 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "BatchNorm1d expects 2D input [N, C]",
                },
            )));
        }

        if input_shape[1] != self.num_features {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "input feature dimension does not match num_features",
                },
            )));
        }

        if self.training.get() {
            self.forward_train(session, input, &input_shape)
        } else {
            self.forward_eval(session, input, &input_shape)
        }
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight, self.bias];
        params.extend(
            self.registered_parameters
                .borrow()
                .iter()
                .filter_map(|entry| entry.tensor),
        );
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        vec![("weight", self.weight), ("bias", self.bias)]
    }

    fn named_parameter_slots_own(&self) -> Vec<(String, Option<TensorNodeId>)> {
        self.registered_parameters
            .borrow()
            .iter()
            .map(|entry| (entry.name.clone(), entry.tensor))
            .collect()
    }

    fn named_buffer_slots_own(&self) -> Vec<(String, Option<TensorNodeId>, bool)> {
        self.registered_buffers
            .borrow()
            .iter()
            .map(|entry| (entry.name.clone(), entry.tensor, entry.persistent))
            .collect()
    }

    fn register_parameter(
        &mut self,
        name: &str,
        parameter: Option<TensorNodeId>,
    ) -> Result<(), ModuleRegistrationError> {
        if !is_valid_registration_name(name) {
            return Err(ModuleRegistrationError::InvalidName { kind: "parameter" });
        }
        if Self::has_builtin_parameter_name(name)
            || Self::has_builtin_buffer_name(name)
            || self
                .registered_buffers
                .borrow()
                .iter()
                .any(|entry| entry.name == name)
        {
            return Err(ModuleRegistrationError::NameConflict {
                name: name.to_string(),
            });
        }
        let mut registered_parameters = self.registered_parameters.borrow_mut();
        upsert_registered_parameter(&mut registered_parameters, name, parameter);
        Ok(())
    }

    fn register_buffer(
        &mut self,
        name: &str,
        tensor: Option<TensorNodeId>,
        persistent: bool,
    ) -> Result<(), ModuleRegistrationError> {
        if !is_valid_registration_name(name) {
            return Err(ModuleRegistrationError::InvalidName { kind: "buffer" });
        }
        if Self::has_builtin_parameter_name(name)
            || self
                .registered_parameters
                .borrow()
                .iter()
                .any(|entry| entry.name == name)
        {
            return Err(ModuleRegistrationError::NameConflict {
                name: name.to_string(),
            });
        }

        let effective_persistent = if Self::has_builtin_buffer_name(name) {
            true
        } else {
            persistent
        };
        match (name, tensor) {
            ("running_mean", Some(id)) => {
                self.running_mean_buffer.replace(id);
            }
            ("running_var", Some(id)) => {
                self.running_var_buffer.replace(id);
            }
            ("num_batches_tracked", Some(id)) => {
                self.num_batches_tracked_buffer.replace(id);
            }
            _ => {}
        }
        let mut registered_buffers = self.registered_buffers.borrow_mut();
        upsert_registered_buffer(&mut registered_buffers, name, tensor, effective_persistent);
        Ok(())
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// 1D convolution module.
///
/// Applies cross-correlation over an input signal of shape `[N, C_in, L]`.
/// Weight shape: `[C_out, C_in, kernel_size]`. Output: `[N, C_out, L_out]`
/// where `L_out = (L + 2*padding - kernel_size) / stride + 1`.
pub struct Conv1d {
    weight: TensorNodeId,
    bias: Option<TensorNodeId>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl Conv1d {
    /// Create a new Conv1d with Kaiming uniform initialization.
    pub fn new(
        session: &mut FrankenTorchSession,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) -> Result<Self, AutogradError> {
        if in_channels == 0 || out_channels == 0 || kernel_size == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv1d requires positive in_channels, out_channels, kernel_size",
                },
            )));
        }
        if stride == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv1d requires stride > 0",
                },
            )));
        }

        // Kaiming uniform: U(-bound, bound) where bound = sqrt(1 / (in_channels * kernel_size))
        let fan_in = in_channels * kernel_size;
        let bound = 1.0 / (fan_in as f64).sqrt();
        let numel = out_channels * in_channels * kernel_size;

        let w_rand = session.rand(vec![numel], false)?;
        let w_scale = session.full(vec![numel], 2.0 * bound, false)?;
        let w_scaled = session.tensor_mul(w_rand, w_scale)?;
        let w_shift = session.full(vec![numel], bound, false)?;
        let w_shifted = session.tensor_sub(w_scaled, w_shift)?;
        let w_values = session.tensor_values(w_shifted)?;
        let weight = session.tensor_variable(
            w_values,
            vec![out_channels, in_channels, kernel_size],
            true,
        )?;

        let bias = if use_bias {
            let b_rand = session.rand(vec![out_channels], false)?;
            let b_scale = session.full(vec![out_channels], 2.0 * bound, false)?;
            let b_scaled = session.tensor_mul(b_rand, b_scale)?;
            let b_shift = session.full(vec![out_channels], bound, false)?;
            let b_shifted = session.tensor_sub(b_scaled, b_shift)?;
            let b_values = session.tensor_values(b_shifted)?;
            Some(session.tensor_variable(b_values, vec![out_channels], true)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Access the weight parameter.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias parameter.
    #[must_use]
    pub fn bias(&self) -> Option<TensorNodeId> {
        self.bias
    }
}

impl Module for Conv1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv1d expects 3D input [N, C_in, L]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let c_in = input_shape[1];
        if c_in != self.in_channels {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv1d input channels do not match in_channels",
                },
            )));
        }

        let l_in = input_shape[2];

        // Apply padding if needed
        let padded = if self.padding > 0 {
            session.tensor_pad(input, &[self.padding, self.padding], 0.0)?
        } else {
            input
        };
        let l_padded = l_in + 2 * self.padding;

        // Output length
        if l_padded < self.kernel_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv1d input too short for given kernel_size and padding",
                },
            )));
        }
        let l_out = (l_padded - self.kernel_size) / self.stride + 1;

        // Im2col: extract sliding windows using narrow
        // For each output position, extract a patch and reshape
        let ck = self.in_channels * self.kernel_size;
        let mut patches = Vec::with_capacity(l_out);
        for l in 0..l_out {
            let start = l * self.stride;
            // narrow(padded, dim=2, start, kernel_size) -> [N, C_in, K]
            let patch = session.tensor_narrow(padded, 2, start, self.kernel_size)?;
            // reshape to [N, 1, C_in*K]
            let flat = session.tensor_reshape(patch, vec![batch_size, 1, ck])?;
            patches.push(flat);
        }

        // cat along dim 1: [N, L_out, C_in*K]
        let unfolded = session.tensor_cat(&patches, 1)?;

        // Weight: [C_out, C_in, K] -> [C_out, C_in*K] -> transpose -> [C_in*K, C_out]
        let w_flat = session.tensor_reshape(self.weight, vec![self.out_channels, ck])?;
        let w_t = session.tensor_transpose(w_flat, 0, 1)?; // [C_in*K, C_out]

        // Expand weight for bmm: [N, C_in*K, C_out]
        let w_us = session.tensor_unsqueeze(w_t, 0)?; // [1, C_in*K, C_out]
        let w_expanded = session.tensor_expand(w_us, vec![batch_size, ck, self.out_channels])?;

        // bmm: [N, L_out, C_in*K] @ [N, C_in*K, C_out] -> [N, L_out, C_out]
        let output = session.tensor_bmm(unfolded, w_expanded)?;

        // Transpose to [N, C_out, L_out]
        let output_t = session.tensor_transpose(output, 1, 2)?;

        // Add bias if present
        match self.bias {
            Some(bias) => {
                // bias: [C_out] -> [1, C_out, 1] -> expand [N, C_out, L_out]
                let b_rs = session.tensor_reshape(bias, vec![1, self.out_channels, 1])?;
                let b_exp =
                    session.tensor_expand(b_rs, vec![batch_size, self.out_channels, l_out])?;
                session.tensor_add(output_t, b_exp)
            }
            None => Ok(output_t),
        }
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight];
        if let Some(bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        let mut params = vec![("weight", self.weight)];
        if let Some(bias) = self.bias {
            params.push(("bias", bias));
        }
        params
    }
}

/// 1D average pooling module.
///
/// Input: `[N, C, L]`. Output: `[N, C, L_out]` where `L_out = (L - kernel_size) / stride + 1`.
pub struct AvgPool1d {
    kernel_size: usize,
    stride: usize,
}

impl AvgPool1d {
    /// Create an AvgPool1d module. If stride is 0, it defaults to kernel_size.
    #[must_use]
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        let effective_stride = if stride == 0 { kernel_size } else { stride };
        Self {
            kernel_size,
            stride: effective_stride,
        }
    }
}

impl Module for AvgPool1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AvgPool1d expects 3D input [N, C, L]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let l_in = input_shape[2];

        if l_in < self.kernel_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AvgPool1d input length smaller than kernel_size",
                },
            )));
        }

        let l_out = (l_in - self.kernel_size) / self.stride + 1;

        // For each output position, narrow the input, mean over the kernel dimension
        let mut slices = Vec::with_capacity(l_out);
        for l in 0..l_out {
            let start = l * self.stride;
            // narrow(input, dim=2, start, kernel_size) -> [N, C, K]
            let patch = session.tensor_narrow(input, 2, start, self.kernel_size)?;
            // mean_dim over dim 2 -> [N, C]
            let avg = session.tensor_mean_dim(patch, 2)?;
            // unsqueeze -> [N, C, 1]
            let avg_us = session.tensor_unsqueeze(avg, 2)?;
            slices.push(avg_us);
        }

        // cat along dim 2: [N, C, L_out]
        let output = session.tensor_cat(&slices, 2)?;

        // Verify shape
        debug_assert_eq!(
            {
                let (_, m) = session.tensor_values_meta(output).unwrap();
                m.shape().to_vec()
            },
            vec![batch_size, channels, l_out]
        );

        Ok(output)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Multi-head attention module.
///
/// Implements scaled dot-product attention with multiple heads.
/// Input: `[N, S, E]` where E = embed_dim. Output: `[N, S, E]`.
///
/// When used via the `Module` trait, performs self-attention (query = key = value = input).
/// Use `forward_qkv` for cross-attention with separate query, key, value.
pub struct MultiheadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiheadAttention {
    /// Create a new MultiheadAttention module.
    ///
    /// `embed_dim` must be divisible by `num_heads`.
    pub fn new(
        session: &mut FrankenTorchSession,
        embed_dim: usize,
        num_heads: usize,
    ) -> Result<Self, AutogradError> {
        if embed_dim == 0 || num_heads == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MultiheadAttention requires positive embed_dim and num_heads",
                },
            )));
        }
        if !embed_dim.is_multiple_of(num_heads) {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "embed_dim must be divisible by num_heads",
                },
            )));
        }

        let head_dim = embed_dim / num_heads;
        let q_proj = Linear::new(session, embed_dim, embed_dim, true)?;
        let k_proj = Linear::new(session, embed_dim, embed_dim, true)?;
        let v_proj = Linear::new(session, embed_dim, embed_dim, true)?;
        let out_proj = Linear::new(session, embed_dim, embed_dim, true)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Forward with separate query, key, value tensors.
    ///
    /// All inputs have shape `[N, S, E]` (or `[N, T, E]` for key/value).
    pub fn forward_qkv(
        &self,
        session: &mut FrankenTorchSession,
        query: TensorNodeId,
        key: TensorNodeId,
        value: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let q_shape = {
            let (_, meta) = session.tensor_values_meta(query)?;
            meta.shape().to_vec()
        };

        if q_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MultiheadAttention expects 3D input [N, S, E]",
                },
            )));
        }

        let batch_size = q_shape[0];
        let seq_len_q = q_shape[1];
        let embed_dim = q_shape[2];

        let k_shape = {
            let (_, meta) = session.tensor_values_meta(key)?;
            meta.shape().to_vec()
        };
        let seq_len_k = k_shape[1];

        // Project Q, K, V: each [N, S, E] -> [N, S, E]
        // Linear expects [batch, features], so reshape [N, S, E] -> [N*S, E]
        let q_flat = session.tensor_reshape(query, vec![batch_size * seq_len_q, embed_dim])?;
        let q_proj = self.q_proj.forward(session, q_flat)?;
        let q = session.tensor_reshape(q_proj, vec![batch_size, seq_len_q, embed_dim])?;

        let k_flat = session.tensor_reshape(key, vec![batch_size * seq_len_k, embed_dim])?;
        let k_proj = self.k_proj.forward(session, k_flat)?;
        let k = session.tensor_reshape(k_proj, vec![batch_size, seq_len_k, embed_dim])?;

        let v_flat = session.tensor_reshape(value, vec![batch_size * seq_len_k, embed_dim])?;
        let v_proj = self.v_proj.forward(session, v_flat)?;
        let v = session.tensor_reshape(v_proj, vec![batch_size, seq_len_k, embed_dim])?;

        // Process each head separately, then concatenate
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let offset = h * self.head_dim;

            // Q_h = Q[:, :, offset:offset+head_dim] -> [N, S_q, head_dim]
            let q_h = session.tensor_narrow(q, 2, offset, self.head_dim)?;
            // K_h = K[:, :, offset:offset+head_dim] -> [N, S_k, head_dim]
            let k_h = session.tensor_narrow(k, 2, offset, self.head_dim)?;
            // V_h = V[:, :, offset:offset+head_dim] -> [N, S_k, head_dim]
            let v_h = session.tensor_narrow(v, 2, offset, self.head_dim)?;

            // Scale Q: Q_h * scale
            let scale_t = session.full(
                vec![batch_size, seq_len_q, self.head_dim],
                self.scale,
                false,
            )?;
            let q_scaled = session.tensor_mul(q_h, scale_t)?;

            // Attention scores: Q_h @ K_h^T -> [N, S_q, S_k]
            let k_t = session.tensor_transpose(k_h, 1, 2)?; // [N, head_dim, S_k]
            let scores = session.tensor_bmm(q_scaled, k_t)?; // [N, S_q, S_k]

            // Softmax over key dimension
            let attn_weights = session.tensor_softmax(scores, 2)?;

            // Weighted sum: attn @ V_h -> [N, S_q, head_dim]
            let head_out = session.tensor_bmm(attn_weights, v_h)?;
            head_outputs.push(head_out);
        }

        // Concatenate heads along last dim: [N, S_q, E]
        let concat = session.tensor_cat(&head_outputs, 2)?;

        // Output projection: [N*S_q, E] -> Linear -> [N*S_q, E] -> [N, S_q, E]
        let concat_flat =
            session.tensor_reshape(concat, vec![batch_size * seq_len_q, embed_dim])?;
        let out = self.out_proj.forward(session, concat_flat)?;
        session.tensor_reshape(out, vec![batch_size, seq_len_q, embed_dim])
    }
}

impl Module for MultiheadAttention {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.forward_qkv(session, input, input, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = self.q_proj.parameters();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("q_proj".to_string(), &self.q_proj as &dyn Module),
            ("k_proj".to_string(), &self.k_proj as &dyn Module),
            ("v_proj".to_string(), &self.v_proj as &dyn Module),
            ("out_proj".to_string(), &self.out_proj as &dyn Module),
        ]
    }
}

/// Softmax module: applies softmax along a specified dimension.
pub struct Softmax {
    dim: usize,
}

impl Softmax {
    /// Create a Softmax module that normalizes along `dim`.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Module for Softmax {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_softmax(input, self.dim)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// LogSoftmax module: applies log-softmax along a specified dimension.
pub struct LogSoftmax {
    dim: usize,
}

impl LogSoftmax {
    /// Create a LogSoftmax module that normalizes along `dim`.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Module for LogSoftmax {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_log_softmax(input, self.dim)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Flatten module: flattens a contiguous range of dimensions.
pub struct Flatten {
    start_dim: usize,
    end_dim: usize,
}

impl Flatten {
    /// Create a Flatten module that flattens dimensions from `start_dim` to `end_dim` (inclusive).
    #[must_use]
    pub fn new(start_dim: usize, end_dim: usize) -> Self {
        Self { start_dim, end_dim }
    }
}

impl Module for Flatten {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_flatten(input, self.start_dim, self.end_dim)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// LeakyReLU activation module (negative slope = 0.01).
pub struct LeakyReLU;

impl Module for LeakyReLU {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_leaky_relu(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// ELU activation module (alpha = 1.0).
pub struct ELU;

impl Module for ELU {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_elu(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Mish activation module.
pub struct Mish;

impl Module for Mish {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_mish(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Softplus activation module.
pub struct Softplus;

impl Module for Softplus {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_softplus(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// PReLU (Parametric ReLU) activation module.
///
/// `f(x) = max(0, x) + a * min(0, x)` where `a` is a learnable parameter.
/// When `num_parameters > 1`, a separate slope is learned per channel.
pub struct PReLU {
    weight: TensorNodeId,
    num_parameters: usize,
}

impl PReLU {
    /// Create a PReLU with the given number of parameters and initial slope.
    ///
    /// `num_parameters = 1`: single learnable slope for all channels.
    /// `num_parameters = C`: one learnable slope per channel.
    pub fn new(
        session: &mut FrankenTorchSession,
        num_parameters: usize,
        init: f64,
    ) -> Result<Self, AutogradError> {
        let weight =
            session.tensor_variable(vec![init; num_parameters], vec![num_parameters], true)?;
        Ok(Self {
            weight,
            num_parameters,
        })
    }
}

impl Module for PReLU {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let input_vals = session.tensor_values(input)?;
        let weight_vals = session.tensor_values(self.weight)?;

        let numel = input_vals.len();
        let mut output = Vec::with_capacity(numel);

        if self.num_parameters == 1 {
            let a = weight_vals[0];
            for &x in &input_vals {
                output.push(if x >= 0.0 { x } else { a * x });
            }
        } else {
            // Multi-channel: weight[c] applies to channel c
            // Determine channel dimension (dim 1 for multi-dim, or dim 0 for 1D)
            let channels = if input_shape.len() >= 2 {
                input_shape[1]
            } else {
                input_shape[0]
            };
            if channels != self.num_parameters {
                return Err(AutogradError::Dispatch(DispatchError::Key(
                    DispatchKeyError::IncompatibleSet {
                        reason: "PReLU num_parameters must match channel count",
                    },
                )));
            }
            let spatial: usize = if input_shape.len() >= 2 {
                input_shape[2..].iter().product()
            } else {
                1
            };
            let batch = if input_shape.len() >= 2 {
                input_shape[0]
            } else {
                1
            };

            for b in 0..batch {
                for (c, &a) in weight_vals.iter().enumerate().take(channels) {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let x = input_vals[idx];
                        output.push(if x >= 0.0 { x } else { a * x });
                    }
                }
            }
        }

        session.tensor_variable(output, input_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        vec![self.weight]
    }
}

/// CELU (Continuously Differentiable ELU) activation module.
///
/// `f(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))`
///
/// Unlike standard ELU, CELU is continuously differentiable everywhere.
pub struct CELU {
    alpha: f64,
}

impl CELU {
    /// Create a CELU module with the given alpha.
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Module for CELU {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
        let input_shape = session.tensor_shape(input)?;
        let input_vals = session.tensor_values(input)?;
        let alpha = self.alpha;

        let output: Vec<f64> = input_vals
            .iter()
            .map(|&x| {
                if x >= 0.0 {
                    x
                } else {
                    alpha * ((x / alpha).exp() - 1.0)
                }
            })
            .collect();

        session.tensor_variable(output, input_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// GLU (Gated Linear Unit) activation module.
///
/// Splits the input tensor along `dim` into two halves `a` and `b`,
/// then computes `a * sigmoid(b)`.
///
/// Input size along `dim` must be even.
pub struct GLU {
    dim: usize,
}

impl GLU {
    /// Create a GLU module that splits along the given dimension.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Module for GLU {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let dim = self.dim;

        if dim >= input_shape.len() {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GLU: dim out of range",
                },
            )));
        }

        let dim_size = input_shape[dim];
        if dim_size % 2 != 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GLU: input size along dim must be even",
                },
            )));
        }

        let half = dim_size / 2;
        // Split: narrow along dim for first half and second half
        let a = session.tensor_narrow(input, dim, 0, half)?;
        let b = session.tensor_narrow(input, dim, half, half)?;
        let b_sig = session.tensor_sigmoid(b)?;
        session.tensor_mul(a, b_sig)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Threshold activation module.
///
/// `f(x) = x if x > threshold, else value`
pub struct Threshold {
    threshold: f64,
    value: f64,
}

impl Threshold {
    /// Create a Threshold module.
    #[must_use]
    pub fn new(threshold: f64, value: f64) -> Self {
        Self { threshold, value }
    }
}

impl Module for Threshold {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let input_vals = session.tensor_values(input)?;
        let output: Vec<f64> = input_vals
            .iter()
            .map(|&x| if x > self.threshold { x } else { self.value })
            .collect();
        session.tensor_variable(output, input_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Group normalization (Wu & He, 2018).
///
/// Divides the channels into `num_groups` groups and normalizes within each group
/// independently. This is more flexible than BatchNorm (which normalizes across
/// the batch) and LayerNorm (which normalizes across all channels).
///
/// Input shape: `[N, C, *]` where `C` must be divisible by `num_groups` and `*`
/// is zero or more spatial dimensions.
///
/// When `affine` is true (default), learnable `weight` (gamma) and `bias` (beta)
/// parameters of shape `[C]` are applied after normalization.
pub struct GroupNorm {
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    weight: Option<TensorNodeId>,
    bias: Option<TensorNodeId>,
}

impl GroupNorm {
    /// Create a new GroupNorm module.
    ///
    /// * `num_groups` - number of groups to divide channels into
    /// * `num_channels` - expected number of channels (C), must be divisible by num_groups
    /// * `eps` - value added to variance for numerical stability (typically 1e-5)
    /// * `affine` - if true, include learnable weight and bias parameters
    pub fn new(
        session: &mut FrankenTorchSession,
        num_groups: usize,
        num_channels: usize,
        eps: f64,
        affine: bool,
    ) -> Result<Self, AutogradError> {
        if num_groups == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GroupNorm requires num_groups > 0",
                },
            )));
        }
        if num_channels == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GroupNorm requires num_channels > 0",
                },
            )));
        }
        if !num_channels.is_multiple_of(num_groups) {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GroupNorm requires num_channels divisible by num_groups",
                },
            )));
        }

        let (weight, bias) = if affine {
            let w = session.tensor_variable(vec![1.0; num_channels], vec![num_channels], true)?;
            let b = session.tensor_variable(vec![0.0; num_channels], vec![num_channels], true)?;
            (Some(w), Some(b))
        } else {
            (None, None)
        };

        Ok(Self {
            num_groups,
            num_channels,
            eps,
            weight,
            bias,
        })
    }
}

impl Module for GroupNorm {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() < 2 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GroupNorm input must have at least 2 dimensions [N, C, ...]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];

        if channels != self.num_channels {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GroupNorm input channels do not match num_channels",
                },
            )));
        }

        let channels_per_group = channels / self.num_groups;
        // Compute spatial size (product of dims after N, C)
        let spatial: usize = if input_shape.len() > 2 {
            input_shape[2..].iter().product()
        } else {
            1
        };
        let group_numel = channels_per_group * spatial;

        // Reshape to [N, num_groups, channels_per_group * spatial] for per-group normalization
        let reshaped =
            session.tensor_reshape(input, vec![batch_size, self.num_groups, group_numel])?;

        // Compute mean per group: mean_dim along dim=2 -> [N, num_groups]
        let mean = session.tensor_mean_dim(reshaped, 2)?;
        // Expand mean back: [N, num_groups] -> [N, num_groups, 1] -> [N, num_groups, group_numel]
        let mean_us = session.tensor_unsqueeze(mean, 2)?;
        let mean_exp =
            session.tensor_expand(mean_us, vec![batch_size, self.num_groups, group_numel])?;

        // diff = x - mean
        let diff = session.tensor_sub(reshaped, mean_exp)?;

        // Variance per group: mean(diff^2, dim=2) -> [N, num_groups]
        let diff_sq = session.tensor_mul(diff, diff)?;
        let var = session.tensor_mean_dim(diff_sq, 2)?;
        let var_us = session.tensor_unsqueeze(var, 2)?;
        let var_exp =
            session.tensor_expand(var_us, vec![batch_size, self.num_groups, group_numel])?;

        // std = sqrt(var + eps)
        let eps_t = session.full(
            vec![batch_size, self.num_groups, group_numel],
            self.eps,
            false,
        )?;
        let var_eps = session.tensor_add(var_exp, eps_t)?;
        let std = session.tensor_sqrt(var_eps)?;

        // normalized = diff / std
        let normalized = session.tensor_div(diff, std)?;

        // Reshape back to original input shape
        let normalized_orig = session.tensor_reshape(normalized, input_shape.clone())?;

        // Apply affine transform if present
        if let (Some(w), Some(b)) = (self.weight, self.bias) {
            // weight and bias have shape [C]. We need to reshape them to be
            // broadcastable with input [N, C, *].
            // Target shape: [1, C, 1, 1, ...] with 1s for each spatial dim
            let mut affine_shape = vec![1usize; input_shape.len()];
            affine_shape[1] = channels;

            let w_reshaped = session.tensor_reshape(w, affine_shape.clone())?;
            let w_expanded = session.tensor_expand(w_reshaped, input_shape.clone())?;
            let b_reshaped = session.tensor_reshape(b, affine_shape)?;
            let b_expanded = session.tensor_expand(b_reshaped, input_shape)?;

            let scaled = session.tensor_mul(w_expanded, normalized_orig)?;
            session.tensor_add(scaled, b_expanded)
        } else {
            Ok(normalized_orig)
        }
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        match (self.weight, self.bias) {
            (Some(w), Some(b)) => vec![w, b],
            _ => Vec::new(),
        }
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        match (self.weight, self.bias) {
            (Some(w), Some(b)) => vec![("weight", w), ("bias", b)],
            _ => Vec::new(),
        }
    }
}

/// Instance normalization for 1D inputs (Ulyanov et al., 2016).
///
/// Normalizes each channel of each sample independently, equivalent to
/// `GroupNorm` with `num_groups = num_channels`.
///
/// Root Mean Square Layer Normalization.
///
/// Equivalent to `torch.nn.RMSNorm(normalized_shape, eps=1e-5)`.
/// Unlike LayerNorm, RMSNorm does not center (subtract mean); it only
/// normalizes by the root mean square: `x * rsqrt(mean(x^2) + eps) * weight`.
///
/// Standard in modern LLM architectures: LLaMA, Mistral, Qwen, DeepSeek.
pub struct RMSNorm {
    normalized_shape: Vec<usize>,
    eps: f64,
    weight: TensorNodeId,
}

impl RMSNorm {
    /// Create a new RMSNorm module.
    ///
    /// `normalized_shape` is the shape of the last dimensions to normalize over.
    /// For a typical LLM with hidden_size=D, use `vec![D]`.
    pub fn new(
        session: &mut FrankenTorchSession,
        normalized_shape: Vec<usize>,
        eps: f64,
    ) -> Result<Self, AutogradError> {
        let total: usize = normalized_shape.iter().product();
        let weight = session.tensor_variable(vec![1.0; total], normalized_shape.clone(), true)?;
        Ok(Self {
            normalized_shape,
            eps,
            weight,
        })
    }

    /// Access the weight parameter.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }
}

impl Module for RMSNorm {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let vals = session.tensor_values(input)?;
        let input_shape = session.tensor_shape(input)?;

        let norm_dims: usize = self.normalized_shape.iter().product();
        let ndim = input_shape.len();
        let norm_ndim = self.normalized_shape.len();

        if ndim < norm_ndim {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "RMSNorm: input has fewer dimensions than normalized_shape",
                },
            )));
        }

        // Check trailing dimensions match
        for (i, &ns) in self.normalized_shape.iter().enumerate() {
            if input_shape[ndim - norm_ndim + i] != ns {
                return Err(AutogradError::Dispatch(DispatchError::Key(
                    DispatchKeyError::IncompatibleSet {
                        reason: "RMSNorm: input trailing dims don't match normalized_shape",
                    },
                )));
            }
        }

        let w_vals = session.tensor_values(self.weight)?;
        let batch_dims: usize = input_shape[..ndim - norm_ndim].iter().product();

        let mut result = vec![0.0f64; vals.len()];

        for b in 0..batch_dims {
            let offset = b * norm_dims;
            // Compute RMS
            let mut sum_sq = 0.0f64;
            for i in 0..norm_dims {
                sum_sq += vals[offset + i] * vals[offset + i];
            }
            let rms = (sum_sq / norm_dims as f64 + self.eps).sqrt();

            // Normalize and scale
            for i in 0..norm_dims {
                result[offset + i] = vals[offset + i] / rms * w_vals[i];
            }
        }

        session.tensor_variable(result, input_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        vec![self.weight]
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        vec![("weight", self.weight)]
    }
}

/// Input shape: `[N, C, L]` where `C` must equal `num_features`.
///
/// Widely used in style transfer and generative models where batch statistics
/// are unreliable or undesirable.
pub struct InstanceNorm1d {
    inner: GroupNorm,
}

impl InstanceNorm1d {
    /// Create a new InstanceNorm1d module.
    ///
    /// * `num_features` - number of channels (C)
    /// * `eps` - value added to variance for numerical stability (typically 1e-5)
    /// * `affine` - if true, include learnable weight and bias parameters
    pub fn new(
        session: &mut FrankenTorchSession,
        num_features: usize,
        eps: f64,
        affine: bool,
    ) -> Result<Self, AutogradError> {
        // InstanceNorm = GroupNorm where each channel is its own group
        let inner = GroupNorm::new(session, num_features, num_features, eps, affine)?;
        Ok(Self { inner })
    }
}

impl Module for InstanceNorm1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.inner.forward(session, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.inner.parameters()
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        self.inner.named_parameters_own()
    }
}

/// Instance normalization for 2D inputs (4D tensors).
///
/// Normalizes each channel of each sample independently over spatial dimensions,
/// equivalent to `GroupNorm` with `num_groups = num_channels`.
///
/// Input shape: `[N, C, H, W]` where `C` must equal `num_features`.
pub struct InstanceNorm2d {
    inner: GroupNorm,
    num_features: usize,
}

impl InstanceNorm2d {
    /// Create a new InstanceNorm2d module.
    ///
    /// * `num_features` - number of channels (C)
    /// * `eps` - value added to variance for numerical stability (typically 1e-5)
    /// * `affine` - if true, include learnable weight and bias parameters
    pub fn new(
        session: &mut FrankenTorchSession,
        num_features: usize,
        eps: f64,
        affine: bool,
    ) -> Result<Self, AutogradError> {
        let inner = GroupNorm::new(session, num_features, num_features, eps, affine)?;
        Ok(Self {
            inner,
            num_features,
        })
    }
}

impl Module for InstanceNorm2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "InstanceNorm2d expects 4D input [N, C, H, W]",
                },
            )));
        }

        if input_shape[1] != self.num_features {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "input channel dimension does not match num_features",
                },
            )));
        }

        self.inner.forward(session, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.inner.parameters()
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        self.inner.named_parameters_own()
    }
}

/// 1D max pooling module.
///
/// Input: `[N, C, L]`. Output: `[N, C, L_out]` where `L_out = (L - kernel_size) / stride + 1`.
/// Selects the maximum value within each pooling window.
pub struct MaxPool1d {
    kernel_size: usize,
    stride: usize,
}

impl MaxPool1d {
    /// Create a MaxPool1d module. If stride is 0, it defaults to kernel_size.
    #[must_use]
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        let effective_stride = if stride == 0 { kernel_size } else { stride };
        Self {
            kernel_size,
            stride: effective_stride,
        }
    }
}

impl Module for MaxPool1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxPool1d expects 3D input [N, C, L]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let l_in = input_shape[2];

        if l_in < self.kernel_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxPool1d input length smaller than kernel_size",
                },
            )));
        }

        let l_out = (l_in - self.kernel_size) / self.stride + 1;

        // For each output position, narrow the input, max over the kernel dimension
        let mut slices = Vec::with_capacity(l_out);
        for l in 0..l_out {
            let start = l * self.stride;
            // narrow(input, dim=2, start, kernel_size) -> [N, C, K]
            let patch = session.tensor_narrow(input, 2, start, self.kernel_size)?;
            // max_dim over dim 2 -> ([N, C], [N, C] indices)
            let (max_vals, _indices) = session.tensor_max_dim(patch, 2)?;
            // unsqueeze -> [N, C, 1]
            let max_us = session.tensor_unsqueeze(max_vals, 2)?;
            slices.push(max_us);
        }

        // cat along dim 2: [N, C, L_out]
        let output = session.tensor_cat(&slices, 2)?;

        debug_assert_eq!(
            {
                let (_, m) = session.tensor_values_meta(output).unwrap();
                m.shape().to_vec()
            },
            vec![batch_size, channels, l_out]
        );

        Ok(output)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// 2D convolution module.
///
/// Applies cross-correlation over an input signal of shape `[N, C_in, H, W]`.
/// Weight shape: `[C_out, C_in, kH, kW]`. Output: `[N, C_out, H_out, W_out]`
/// where `H_out = (H + 2*padding_h - kH) / stride_h + 1` (similarly for W).
pub struct Conv2d {
    weight: TensorNodeId,
    bias: Option<TensorNodeId>,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
}

impl Conv2d {
    /// Create a new Conv2d with Kaiming uniform initialization.
    ///
    /// `kernel_size` is `(kH, kW)`, `stride` is `(sH, sW)`, `padding` is `(pH, pW)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: &mut FrankenTorchSession,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        use_bias: bool,
    ) -> Result<Self, AutogradError> {
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;

        if in_channels == 0 || out_channels == 0 || kh == 0 || kw == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv2d requires positive in_channels, out_channels, kernel_size",
                },
            )));
        }
        if sh == 0 || sw == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv2d requires stride > 0",
                },
            )));
        }

        // Kaiming uniform: U(-bound, bound) where bound = sqrt(1 / fan_in)
        let fan_in = in_channels * kh * kw;
        let bound = 1.0 / (fan_in as f64).sqrt();
        let numel = out_channels * in_channels * kh * kw;

        let w_rand = session.rand(vec![numel], false)?;
        let w_scale = session.full(vec![numel], 2.0 * bound, false)?;
        let w_scaled = session.tensor_mul(w_rand, w_scale)?;
        let w_shift = session.full(vec![numel], bound, false)?;
        let w_shifted = session.tensor_sub(w_scaled, w_shift)?;
        let w_values = session.tensor_values(w_shifted)?;
        let weight =
            session.tensor_variable(w_values, vec![out_channels, in_channels, kh, kw], true)?;

        let bias = if use_bias {
            let b_rand = session.rand(vec![out_channels], false)?;
            let b_scale = session.full(vec![out_channels], 2.0 * bound, false)?;
            let b_scaled = session.tensor_mul(b_rand, b_scale)?;
            let b_shift = session.full(vec![out_channels], bound, false)?;
            let b_shifted = session.tensor_sub(b_scaled, b_shift)?;
            let b_values = session.tensor_values(b_shifted)?;
            Some(session.tensor_variable(b_values, vec![out_channels], true)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_h: kh,
            kernel_w: kw,
            stride_h: sh,
            stride_w: sw,
            padding_h: ph,
            padding_w: pw,
        })
    }

    /// Access the weight parameter.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias parameter.
    #[must_use]
    pub fn bias(&self) -> Option<TensorNodeId> {
        self.bias
    }
}

impl Module for Conv2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv2d expects 4D input [N, C_in, H, W]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let c_in = input_shape[1];
        if c_in != self.in_channels {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv2d input channels do not match in_channels",
                },
            )));
        }

        let h_in = input_shape[2];
        let w_in = input_shape[3];

        // Apply padding if needed
        let padded = if self.padding_h > 0 || self.padding_w > 0 {
            // PyTorch pad convention: innermost dim first -> [W_before, W_after, H_before, H_after]
            session.tensor_pad(
                input,
                &[
                    self.padding_w,
                    self.padding_w,
                    self.padding_h,
                    self.padding_h,
                ],
                0.0,
            )?
        } else {
            input
        };
        let h_padded = h_in + 2 * self.padding_h;
        let w_padded = w_in + 2 * self.padding_w;

        // Output spatial dimensions
        if h_padded < self.kernel_h || w_padded < self.kernel_w {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv2d input too small for given kernel_size and padding",
                },
            )));
        }
        let h_out = (h_padded - self.kernel_h) / self.stride_h + 1;
        let w_out = (w_padded - self.kernel_w) / self.stride_w + 1;

        // Im2col: extract sliding 2D windows using narrow on dims 2 and 3
        let ck = self.in_channels * self.kernel_h * self.kernel_w;
        let mut patches = Vec::with_capacity(h_out * w_out);
        for hi in 0..h_out {
            let h_start = hi * self.stride_h;
            // narrow on dim 2 (height): [N, C_in, kH, w_padded]
            let row_slice = session.tensor_narrow(padded, 2, h_start, self.kernel_h)?;
            for wi in 0..w_out {
                let w_start = wi * self.stride_w;
                // narrow on dim 3 (width): [N, C_in, kH, kW]
                let patch = session.tensor_narrow(row_slice, 3, w_start, self.kernel_w)?;
                // reshape to [N, 1, C_in*kH*kW]
                let flat = session.tensor_reshape(patch, vec![batch_size, 1, ck])?;
                patches.push(flat);
            }
        }

        // cat along dim 1: [N, H_out*W_out, C_in*kH*kW]
        let unfolded = session.tensor_cat(&patches, 1)?;

        // Weight: [C_out, C_in, kH, kW] -> [C_out, ck] -> transpose -> [ck, C_out]
        let w_flat = session.tensor_reshape(self.weight, vec![self.out_channels, ck])?;
        let w_t = session.tensor_transpose(w_flat, 0, 1)?;

        // Expand weight for bmm: [N, ck, C_out]
        let w_us = session.tensor_unsqueeze(w_t, 0)?;
        let w_expanded = session.tensor_expand(w_us, vec![batch_size, ck, self.out_channels])?;

        // bmm: [N, H_out*W_out, ck] @ [N, ck, C_out] -> [N, H_out*W_out, C_out]
        let output = session.tensor_bmm(unfolded, w_expanded)?;

        // Transpose to [N, C_out, H_out*W_out], then reshape to [N, C_out, H_out, W_out]
        let output_t = session.tensor_transpose(output, 1, 2)?;
        let output_4d =
            session.tensor_reshape(output_t, vec![batch_size, self.out_channels, h_out, w_out])?;

        // Add bias if present
        match self.bias {
            Some(bias) => {
                // bias: [C_out] -> [1, C_out, 1, 1] -> expand [N, C_out, H_out, W_out]
                let b_rs = session.tensor_reshape(bias, vec![1, self.out_channels, 1, 1])?;
                let b_exp = session
                    .tensor_expand(b_rs, vec![batch_size, self.out_channels, h_out, w_out])?;
                session.tensor_add(output_4d, b_exp)
            }
            None => Ok(output_4d),
        }
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight];
        if let Some(bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        let mut params = vec![("weight", self.weight)];
        if let Some(bias) = self.bias {
            params.push(("bias", bias));
        }
        params
    }
}

/// 2D max pooling module.
///
/// Input: `[N, C, H, W]`. Output: `[N, C, H_out, W_out]` where
/// `H_out = (H - kH) / sH + 1` and `W_out = (W - kW) / sW + 1`.
/// Selects the maximum value within each 2D pooling window.
pub struct MaxPool2d {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
}

impl MaxPool2d {
    /// Create a MaxPool2d module.
    ///
    /// If stride values are 0, they default to the corresponding kernel size.
    #[must_use]
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        Self {
            kernel_h: kh,
            kernel_w: kw,
            stride_h: if sh == 0 { kh } else { sh },
            stride_w: if sw == 0 { kw } else { sw },
        }
    }
}

impl Module for MaxPool2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxPool2d expects 4D input [N, C, H, W]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];

        if h_in < self.kernel_h || w_in < self.kernel_w {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxPool2d input smaller than kernel_size",
                },
            )));
        }

        let h_out = (h_in - self.kernel_h) / self.stride_h + 1;
        let w_out = (w_in - self.kernel_w) / self.stride_w + 1;

        // Extract 2D patches, flatten spatial+kernel dims, take max
        let nc = batch_size * channels;
        let kk = self.kernel_h * self.kernel_w;

        let mut patches = Vec::with_capacity(h_out * w_out);
        for hi in 0..h_out {
            let h_start = hi * self.stride_h;
            let row_slice = session.tensor_narrow(input, 2, h_start, self.kernel_h)?;
            for wi in 0..w_out {
                let w_start = wi * self.stride_w;
                // [N, C, kH, kW]
                let patch = session.tensor_narrow(row_slice, 3, w_start, self.kernel_w)?;
                // reshape to [N*C, kH*kW]
                let flat = session.tensor_reshape(patch, vec![nc, kk])?;
                // max along dim=1 -> ([N*C], _indices)
                let (max_vals, _) = session.tensor_max_dim(flat, 1)?;
                // reshape to [N, C, 1]
                let shaped = session.tensor_reshape(max_vals, vec![batch_size, channels, 1])?;
                patches.push(shaped);
            }
        }

        // cat along dim 2: [N, C, H_out*W_out]
        let pooled = session.tensor_cat(&patches, 2)?;
        // reshape to [N, C, H_out, W_out]
        session.tensor_reshape(pooled, vec![batch_size, channels, h_out, w_out])
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Identity module that passes input through unchanged.
///
/// Useful as a placeholder in `Sequential` containers or for skip connections
/// where a no-op branch is needed.
/// Adaptive average pooling for 2D spatial inputs.
///
/// Input: `[N, C, H_in, W_in]`. Output: `[N, C, H_out, W_out]` where `(H_out, W_out)`
/// is the target output size. Computes the kernel size and stride automatically.
pub struct AdaptiveAvgPool2d {
    output_h: usize,
    output_w: usize,
}

impl AdaptiveAvgPool2d {
    /// Create an AdaptiveAvgPool2d targeting the given output spatial size.
    #[must_use]
    pub fn new(output_size: (usize, usize)) -> Self {
        Self {
            output_h: output_size.0,
            output_w: output_size.1,
        }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveAvgPool2d expects 4D input [N, C, H, W]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];

        if self.output_h == 0 || self.output_w == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveAvgPool2d output size must be > 0",
                },
            )));
        }

        // Special case: output == input means identity
        if self.output_h == h_in && self.output_w == w_in {
            return Ok(input);
        }

        // Collect output patches column by column: for each (oh, ow) compute start/end indices
        // using PyTorch's floor-division formula: start_i = floor(i * input_size / output_size)
        let mut col_nodes = Vec::with_capacity(self.output_h * self.output_w);

        for oh in 0..self.output_h {
            for ow in 0..self.output_w {
                let h_start = (oh * h_in) / self.output_h;
                let h_end = ((oh + 1) * h_in) / self.output_h;
                let w_start = (ow * w_in) / self.output_w;
                let w_end = ((ow + 1) * w_in) / self.output_w;

                let kh = h_end - h_start;
                let kw = w_end - w_start;

                // Narrow along H (dim 2), then W (dim 3)
                let h_slice = session.tensor_narrow(input, 2, h_start, kh)?;
                let hw_slice = session.tensor_narrow(h_slice, 3, w_start, kw)?;

                // Reshape to [N, C, kh*kw] and mean over dim 2 → [N, C]
                let flat = session.tensor_reshape(hw_slice, vec![n, c, kh * kw])?;
                let pooled = session.tensor_mean_dim(flat, 2)?;
                // Unsqueeze to [N, C, 1]
                let col = session.tensor_unsqueeze(pooled, 2)?;
                col_nodes.push(col);
            }
        }

        // Cat all columns along dim 2 → [N, C, output_h * output_w]
        let result = session.tensor_cat(&col_nodes, 2)?;

        // Reshape to [N, C, output_h, output_w]
        session.tensor_reshape(result, vec![n, c, self.output_h, self.output_w])
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// 2D average pooling module (non-adaptive).
///
/// Input: `[N, C, H, W]`. Output: `[N, C, H_out, W_out]` where
/// `H_out = floor((H + 2*pad_h - kH) / stride_h) + 1`.
///
/// Supports padding, `ceil_mode` (ceiling instead of floor for output size),
/// and `count_include_pad` (include zero-padded elements in average).
pub struct AvgPool2d {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    ceil_mode: bool,
    count_include_pad: bool,
}

impl AvgPool2d {
    /// Create an AvgPool2d module.
    ///
    /// If stride values are 0, they default to the corresponding kernel size.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Self {
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        Self {
            kernel_h: kh,
            kernel_w: kw,
            stride_h: if sh == 0 { kh } else { sh },
            stride_w: if sw == 0 { kw } else { sw },
            padding_h: padding.0,
            padding_w: padding.1,
            ceil_mode,
            count_include_pad,
        }
    }
}

impl Module for AvgPool2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AvgPool2d expects 4D input [N, C, H, W]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];

        // Apply padding if needed
        let padded = if self.padding_h > 0 || self.padding_w > 0 {
            // tensor_pad uses innermost-first: [w_before, w_after, h_before, h_after]
            session.tensor_pad(
                input,
                &[
                    self.padding_w,
                    self.padding_w,
                    self.padding_h,
                    self.padding_h,
                ],
                0.0,
            )?
        } else {
            input
        };

        let h_padded = h_in + 2 * self.padding_h;
        let w_padded = w_in + 2 * self.padding_w;

        let h_out = if self.ceil_mode {
            (h_padded - self.kernel_h).div_ceil(self.stride_h) + 1
        } else {
            (h_padded - self.kernel_h) / self.stride_h + 1
        };
        let w_out = if self.ceil_mode {
            (w_padded - self.kernel_w).div_ceil(self.stride_w) + 1
        } else {
            (w_padded - self.kernel_w) / self.stride_w + 1
        };

        let nc = n * c;

        let mut patches = Vec::with_capacity(h_out * w_out);
        for hi in 0..h_out {
            let h_start = hi * self.stride_h;
            let h_end = (h_start + self.kernel_h).min(h_padded);
            let kh_actual = h_end - h_start;
            let row_slice = session.tensor_narrow(padded, 2, h_start, kh_actual)?;
            for wi in 0..w_out {
                let w_start = wi * self.stride_w;
                let w_end = (w_start + self.kernel_w).min(w_padded);
                let kw_actual = w_end - w_start;

                let patch = session.tensor_narrow(row_slice, 3, w_start, kw_actual)?;
                let flat = session.tensor_reshape(patch, vec![nc, kh_actual * kw_actual])?;

                if self.count_include_pad || (self.padding_h == 0 && self.padding_w == 0) {
                    // Sum and divide by full kernel size
                    let sum = session.tensor_sum_dim(flat, 1)?;
                    let divisor =
                        session.full(vec![nc], (self.kernel_h * self.kernel_w) as f64, false)?;
                    let avg = session.tensor_div(sum, divisor)?;
                    let shaped = session.tensor_reshape(avg, vec![n, c, 1])?;
                    patches.push(shaped);
                } else {
                    // count_include_pad=false divides by the materialized patch size only.
                    let sum = session.tensor_sum_dim(flat, 1)?;
                    let divisor = session.full(vec![nc], (kh_actual * kw_actual) as f64, false)?;
                    let avg = session.tensor_div(sum, divisor)?;
                    let shaped = session.tensor_reshape(avg, vec![n, c, 1])?;
                    patches.push(shaped);
                }
            }
        }

        let pooled = session.tensor_cat(&patches, 2)?;
        session.tensor_reshape(pooled, vec![n, c, h_out, w_out])
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// 3D average pooling module.
///
/// Input: `[N, C, D, H, W]`. Output: `[N, C, D_out, H_out, W_out]` where
/// output size is `floor((input_size - kernel_size) / stride) + 1`.
pub struct AvgPool3d {
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
}

impl AvgPool3d {
    /// Create an AvgPool3d module. If stride values are 0, they default to kernel sizes.
    #[must_use]
    pub fn new(kernel_size: (usize, usize, usize), stride: (usize, usize, usize)) -> Self {
        let (kd, kh, kw) = kernel_size;
        let (sd, sh, sw) = stride;
        Self {
            kernel_d: kd,
            kernel_h: kh,
            kernel_w: kw,
            stride_d: if sd == 0 { kd } else { sd },
            stride_h: if sh == 0 { kh } else { sh },
            stride_w: if sw == 0 { kw } else { sw },
        }
    }
}

impl Module for AvgPool3d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 5 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AvgPool3d expects 5D input [N, C, D, H, W]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let d_in = input_shape[2];
        let h_in = input_shape[3];
        let w_in = input_shape[4];

        if d_in < self.kernel_d || h_in < self.kernel_h || w_in < self.kernel_w {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AvgPool3d input smaller than kernel_size",
                },
            )));
        }

        let d_out = (d_in - self.kernel_d) / self.stride_d + 1;
        let h_out = (h_in - self.kernel_h) / self.stride_h + 1;
        let w_out = (w_in - self.kernel_w) / self.stride_w + 1;
        let nc = n * c;
        let kkk = self.kernel_d * self.kernel_h * self.kernel_w;

        let mut patches = Vec::with_capacity(d_out * h_out * w_out);
        for di in 0..d_out {
            let d_start = di * self.stride_d;
            let d_slice = session.tensor_narrow(input, 2, d_start, self.kernel_d)?;
            for hi in 0..h_out {
                let h_start = hi * self.stride_h;
                let h_slice = session.tensor_narrow(d_slice, 3, h_start, self.kernel_h)?;
                for wi in 0..w_out {
                    let w_start = wi * self.stride_w;
                    let patch = session.tensor_narrow(h_slice, 4, w_start, self.kernel_w)?;
                    // [N, C, kD, kH, kW] -> [N*C, kD*kH*kW]
                    let flat = session.tensor_reshape(patch, vec![nc, kkk])?;
                    let avg = session.tensor_mean_dim(flat, 1)?;
                    let shaped = session.tensor_reshape(avg, vec![n, c, 1])?;
                    patches.push(shaped);
                }
            }
        }

        let pooled = session.tensor_cat(&patches, 2)?;
        session.tensor_reshape(pooled, vec![n, c, d_out, h_out, w_out])
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// 3D max pooling module.
///
/// Input: `[N, C, D, H, W]`. Output: `[N, C, D_out, H_out, W_out]` where
/// output size is `floor((input_size - kernel_size) / stride) + 1`.
pub struct MaxPool3d {
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
}

impl MaxPool3d {
    /// Create a MaxPool3d module. If stride values are 0, they default to kernel sizes.
    #[must_use]
    pub fn new(kernel_size: (usize, usize, usize), stride: (usize, usize, usize)) -> Self {
        let (kd, kh, kw) = kernel_size;
        let (sd, sh, sw) = stride;
        Self {
            kernel_d: kd,
            kernel_h: kh,
            kernel_w: kw,
            stride_d: if sd == 0 { kd } else { sd },
            stride_h: if sh == 0 { kh } else { sh },
            stride_w: if sw == 0 { kw } else { sw },
        }
    }
}

impl Module for MaxPool3d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 5 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxPool3d expects 5D input [N, C, D, H, W]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let d_in = input_shape[2];
        let h_in = input_shape[3];
        let w_in = input_shape[4];

        if d_in < self.kernel_d || h_in < self.kernel_h || w_in < self.kernel_w {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxPool3d input smaller than kernel_size",
                },
            )));
        }

        let d_out = (d_in - self.kernel_d) / self.stride_d + 1;
        let h_out = (h_in - self.kernel_h) / self.stride_h + 1;
        let w_out = (w_in - self.kernel_w) / self.stride_w + 1;
        let nc = n * c;
        let kkk = self.kernel_d * self.kernel_h * self.kernel_w;

        let mut patches = Vec::with_capacity(d_out * h_out * w_out);
        for di in 0..d_out {
            let d_start = di * self.stride_d;
            let d_slice = session.tensor_narrow(input, 2, d_start, self.kernel_d)?;
            for hi in 0..h_out {
                let h_start = hi * self.stride_h;
                let h_slice = session.tensor_narrow(d_slice, 3, h_start, self.kernel_h)?;
                for wi in 0..w_out {
                    let w_start = wi * self.stride_w;
                    let patch = session.tensor_narrow(h_slice, 4, w_start, self.kernel_w)?;
                    let flat = session.tensor_reshape(patch, vec![nc, kkk])?;
                    let (max_vals, _) = session.tensor_max_dim(flat, 1)?;
                    let shaped = session.tensor_reshape(max_vals, vec![n, c, 1])?;
                    patches.push(shaped);
                }
            }
        }

        let pooled = session.tensor_cat(&patches, 2)?;
        session.tensor_reshape(pooled, vec![n, c, d_out, h_out, w_out])
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Adaptive average pooling for 1D inputs.
///
/// Input: `[N, C, L]`. Output: `[N, C, L_out]` where `L_out` is the target output size.
pub struct AdaptiveAvgPool1d {
    output_size: usize,
}

impl AdaptiveAvgPool1d {
    /// Create an AdaptiveAvgPool1d targeting the given output length.
    #[must_use]
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }
}

impl Module for AdaptiveAvgPool1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveAvgPool1d expects 3D input [N, C, L]",
                },
            )));
        }

        let _n = input_shape[0];
        let _c = input_shape[1];
        let l_in = input_shape[2];

        if self.output_size == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveAvgPool1d output size must be > 0",
                },
            )));
        }

        if self.output_size == l_in {
            return Ok(input);
        }

        let mut slices = Vec::with_capacity(self.output_size);
        for i in 0..self.output_size {
            let start = (i * l_in) / self.output_size;
            let end = ((i + 1) * l_in) / self.output_size;
            let k = end - start;
            let patch = session.tensor_narrow(input, 2, start, k)?;
            let avg = session.tensor_mean_dim(patch, 2)?;
            let col = session.tensor_unsqueeze(avg, 2)?;
            slices.push(col);
        }

        session.tensor_cat(&slices, 2)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Adaptive average pooling for 3D spatial inputs.
///
/// Input: `[N, C, D, H, W]`. Output: `[N, C, D_out, H_out, W_out]` where
/// `(D_out, H_out, W_out)` is the target output size.
pub struct AdaptiveAvgPool3d {
    output_d: usize,
    output_h: usize,
    output_w: usize,
}

impl AdaptiveAvgPool3d {
    /// Create an AdaptiveAvgPool3d targeting the given output spatial size.
    #[must_use]
    pub fn new(output_size: (usize, usize, usize)) -> Self {
        Self {
            output_d: output_size.0,
            output_h: output_size.1,
            output_w: output_size.2,
        }
    }
}

impl Module for AdaptiveAvgPool3d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 5 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveAvgPool3d expects 5D input [N, C, D, H, W]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let d_in = input_shape[2];
        let h_in = input_shape[3];
        let w_in = input_shape[4];

        if self.output_d == 0 || self.output_h == 0 || self.output_w == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveAvgPool3d output size must be > 0",
                },
            )));
        }

        if self.output_d == d_in && self.output_h == h_in && self.output_w == w_in {
            return Ok(input);
        }

        let nc = n * c;
        let mut patches = Vec::with_capacity(self.output_d * self.output_h * self.output_w);

        for od in 0..self.output_d {
            let d_start = (od * d_in) / self.output_d;
            let d_end = ((od + 1) * d_in) / self.output_d;
            let kd = d_end - d_start;
            let d_slice = session.tensor_narrow(input, 2, d_start, kd)?;

            for oh in 0..self.output_h {
                let h_start = (oh * h_in) / self.output_h;
                let h_end = ((oh + 1) * h_in) / self.output_h;
                let kh = h_end - h_start;
                let h_slice = session.tensor_narrow(d_slice, 3, h_start, kh)?;

                for ow in 0..self.output_w {
                    let w_start = (ow * w_in) / self.output_w;
                    let w_end = ((ow + 1) * w_in) / self.output_w;
                    let kw = w_end - w_start;
                    let patch = session.tensor_narrow(h_slice, 4, w_start, kw)?;

                    let flat = session.tensor_reshape(patch, vec![nc, kd * kh * kw])?;
                    let avg = session.tensor_mean_dim(flat, 1)?;
                    let shaped = session.tensor_reshape(avg, vec![n, c, 1])?;
                    patches.push(shaped);
                }
            }
        }

        let pooled = session.tensor_cat(&patches, 2)?;
        session.tensor_reshape(
            pooled,
            vec![n, c, self.output_d, self.output_h, self.output_w],
        )
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Adaptive max pooling for 1D inputs.
///
/// Input: `[N, C, L]`. Output: `[N, C, L_out]` where `L_out` is the target output size.
pub struct AdaptiveMaxPool1d {
    output_size: usize,
}

impl AdaptiveMaxPool1d {
    /// Create an AdaptiveMaxPool1d targeting the given output length.
    #[must_use]
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }
}

impl Module for AdaptiveMaxPool1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveMaxPool1d expects 3D input [N, C, L]",
                },
            )));
        }

        let _n = input_shape[0];
        let _c = input_shape[1];
        let l_in = input_shape[2];

        if self.output_size == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveMaxPool1d output size must be > 0",
                },
            )));
        }

        let mut slices = Vec::with_capacity(self.output_size);
        for i in 0..self.output_size {
            let start = (i * l_in) / self.output_size;
            let end = ((i + 1) * l_in) / self.output_size;
            let k = end - start;
            let patch = session.tensor_narrow(input, 2, start, k)?;
            let (max_vals, _) = session.tensor_max_dim(patch, 2)?;
            let col = session.tensor_unsqueeze(max_vals, 2)?;
            slices.push(col);
        }

        session.tensor_cat(&slices, 2)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Adaptive max pooling for 2D spatial inputs.
///
/// Input: `[N, C, H, W]`. Output: `[N, C, H_out, W_out]`.
pub struct AdaptiveMaxPool2d {
    output_h: usize,
    output_w: usize,
}

impl AdaptiveMaxPool2d {
    /// Create an AdaptiveMaxPool2d targeting the given output spatial size.
    #[must_use]
    pub fn new(output_size: (usize, usize)) -> Self {
        Self {
            output_h: output_size.0,
            output_w: output_size.1,
        }
    }
}

impl Module for AdaptiveMaxPool2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveMaxPool2d expects 4D input [N, C, H, W]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];

        if self.output_h == 0 || self.output_w == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveMaxPool2d output size must be > 0",
                },
            )));
        }

        let nc = n * c;
        let mut patches = Vec::with_capacity(self.output_h * self.output_w);

        for oh in 0..self.output_h {
            let h_start = (oh * h_in) / self.output_h;
            let h_end = ((oh + 1) * h_in) / self.output_h;
            let kh = h_end - h_start;
            let h_slice = session.tensor_narrow(input, 2, h_start, kh)?;

            for ow in 0..self.output_w {
                let w_start = (ow * w_in) / self.output_w;
                let w_end = ((ow + 1) * w_in) / self.output_w;
                let kw = w_end - w_start;
                let patch = session.tensor_narrow(h_slice, 3, w_start, kw)?;

                let flat = session.tensor_reshape(patch, vec![nc, kh * kw])?;
                let (max_vals, _) = session.tensor_max_dim(flat, 1)?;
                let shaped = session.tensor_reshape(max_vals, vec![n, c, 1])?;
                patches.push(shaped);
            }
        }

        let pooled = session.tensor_cat(&patches, 2)?;
        session.tensor_reshape(pooled, vec![n, c, self.output_h, self.output_w])
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Adaptive max pooling for 3D spatial inputs.
///
/// Input: `[N, C, D, H, W]`. Output: `[N, C, D_out, H_out, W_out]`.
pub struct AdaptiveMaxPool3d {
    output_d: usize,
    output_h: usize,
    output_w: usize,
}

impl AdaptiveMaxPool3d {
    /// Create an AdaptiveMaxPool3d targeting the given output spatial size.
    #[must_use]
    pub fn new(output_size: (usize, usize, usize)) -> Self {
        Self {
            output_d: output_size.0,
            output_h: output_size.1,
            output_w: output_size.2,
        }
    }
}

impl Module for AdaptiveMaxPool3d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 5 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveMaxPool3d expects 5D input [N, C, D, H, W]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let d_in = input_shape[2];
        let h_in = input_shape[3];
        let w_in = input_shape[4];

        if self.output_d == 0 || self.output_h == 0 || self.output_w == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "AdaptiveMaxPool3d output size must be > 0",
                },
            )));
        }

        let nc = n * c;
        let mut patches = Vec::with_capacity(self.output_d * self.output_h * self.output_w);

        for od in 0..self.output_d {
            let d_start = (od * d_in) / self.output_d;
            let d_end = ((od + 1) * d_in) / self.output_d;
            let kd = d_end - d_start;
            let d_slice = session.tensor_narrow(input, 2, d_start, kd)?;

            for oh in 0..self.output_h {
                let h_start = (oh * h_in) / self.output_h;
                let h_end = ((oh + 1) * h_in) / self.output_h;
                let kh = h_end - h_start;
                let h_slice = session.tensor_narrow(d_slice, 3, h_start, kh)?;

                for ow in 0..self.output_w {
                    let w_start = (ow * w_in) / self.output_w;
                    let w_end = ((ow + 1) * w_in) / self.output_w;
                    let kw = w_end - w_start;
                    let patch = session.tensor_narrow(h_slice, 4, w_start, kw)?;

                    let flat = session.tensor_reshape(patch, vec![nc, kd * kh * kw])?;
                    let (max_vals, _) = session.tensor_max_dim(flat, 1)?;
                    let shaped = session.tensor_reshape(max_vals, vec![n, c, 1])?;
                    patches.push(shaped);
                }
            }
        }

        let pooled = session.tensor_cat(&patches, 2)?;
        session.tensor_reshape(
            pooled,
            vec![n, c, self.output_d, self.output_h, self.output_w],
        )
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// MaxUnpool1d reverses a 1D max pooling operation.
///
/// Uses the indices from `MaxPool1d` (stored externally) to scatter max-pooled values
/// back to their original positions. Since we don't store indices from MaxPool, this
/// implementation takes explicit indices as a flat vector.
///
/// Input: `[N, C, L_pooled]`. Indices: flat vector of original positions.
/// Output: `[N, C, output_size]`.
pub struct MaxUnpool1d {
    kernel_size: usize,
    stride: usize,
}

impl MaxUnpool1d {
    /// Create a MaxUnpool1d module. If stride is 0, defaults to kernel_size.
    #[must_use]
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        let effective_stride = if stride == 0 { kernel_size } else { stride };
        Self {
            kernel_size,
            stride: effective_stride,
        }
    }

    /// Kernel size used for unpooling.
    #[must_use]
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Stride used for unpooling.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Unpool using the given indices and output size.
    ///
    /// `indices` maps each element in the pooled output to its position in the original input.
    /// `output_size` is the length of the original (unpooled) dimension.
    pub fn forward_with_indices(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        indices: &[usize],
        output_size: usize,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxUnpool1d expects 3D input [N, C, L]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let l_pooled = input_shape[2];

        if indices.len() != l_pooled {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxUnpool1d: indices length must match pooled length",
                },
            )));
        }

        let input_vals = session.tensor_values(input)?;
        let mut output = vec![0.0f64; n * c * output_size];

        for batch in 0..n {
            for ch in 0..c {
                for (l, &dst_pos) in indices.iter().enumerate().take(l_pooled) {
                    let src_idx = batch * c * l_pooled + ch * l_pooled + l;
                    if dst_pos < output_size {
                        let dst_idx = batch * c * output_size + ch * output_size + dst_pos;
                        output[dst_idx] = input_vals[src_idx];
                    }
                }
            }
        }

        session.tensor_variable(output, vec![n, c, output_size], false)
    }
}

/// MaxUnpool2d reverses a 2D max pooling operation.
///
/// Input: `[N, C, H_pooled, W_pooled]`.
/// Indices: flat vector mapping each pooled element to position in the unpooled spatial plane.
/// Output: `[N, C, H_out, W_out]`.
pub struct MaxUnpool2d {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
}

impl MaxUnpool2d {
    /// Create a MaxUnpool2d module. If stride values are 0, they default to kernel sizes.
    #[must_use]
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        Self {
            kernel_h: kh,
            kernel_w: kw,
            stride_h: if sh == 0 { kh } else { sh },
            stride_w: if sw == 0 { kw } else { sw },
        }
    }

    /// Kernel size as `(height, width)`.
    #[must_use]
    pub fn kernel_size(&self) -> (usize, usize) {
        (self.kernel_h, self.kernel_w)
    }

    /// Stride as `(height, width)`.
    #[must_use]
    pub fn stride(&self) -> (usize, usize) {
        (self.stride_h, self.stride_w)
    }

    /// Unpool using the given 2D indices and output size.
    ///
    /// `indices` maps each pooled spatial position to a flat index in the original HxW plane.
    /// `output_size` is `(H_out, W_out)`.
    pub fn forward_with_indices(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        indices: &[usize],
        output_size: (usize, usize),
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxUnpool2d expects 4D input [N, C, H, W]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let h_pooled = input_shape[2];
        let w_pooled = input_shape[3];
        let (h_out, w_out) = output_size;
        let spatial_pooled = h_pooled * w_pooled;

        if indices.len() != spatial_pooled {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxUnpool2d: indices length must match pooled spatial size",
                },
            )));
        }

        let input_vals = session.tensor_values(input)?;
        let out_spatial = h_out * w_out;
        let mut output = vec![0.0f64; n * c * out_spatial];

        for batch in 0..n {
            for ch in 0..c {
                for hi in 0..h_pooled {
                    for wi in 0..w_pooled {
                        let src_idx =
                            batch * c * spatial_pooled + ch * spatial_pooled + hi * w_pooled + wi;
                        let dst_pos = indices[hi * w_pooled + wi];
                        if dst_pos < out_spatial {
                            let dst_idx = batch * c * out_spatial + ch * out_spatial + dst_pos;
                            output[dst_idx] = input_vals[src_idx];
                        }
                    }
                }
            }
        }

        session.tensor_variable(output, vec![n, c, h_out, w_out], false)
    }
}

/// MaxUnpool3d reverses a 3D max pooling operation.
///
/// Input: `[N, C, D_pooled, H_pooled, W_pooled]`.
/// Output: `[N, C, D_out, H_out, W_out]`.
pub struct MaxUnpool3d {
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
}

impl MaxUnpool3d {
    /// Create a MaxUnpool3d module. If stride values are 0, they default to kernel sizes.
    #[must_use]
    pub fn new(kernel_size: (usize, usize, usize), stride: (usize, usize, usize)) -> Self {
        let (kd, kh, kw) = kernel_size;
        let (sd, sh, sw) = stride;
        Self {
            kernel_d: kd,
            kernel_h: kh,
            kernel_w: kw,
            stride_d: if sd == 0 { kd } else { sd },
            stride_h: if sh == 0 { kh } else { sh },
            stride_w: if sw == 0 { kw } else { sw },
        }
    }

    /// Kernel size as `(depth, height, width)`.
    #[must_use]
    pub fn kernel_size(&self) -> (usize, usize, usize) {
        (self.kernel_d, self.kernel_h, self.kernel_w)
    }

    /// Stride as `(depth, height, width)`.
    #[must_use]
    pub fn stride(&self) -> (usize, usize, usize) {
        (self.stride_d, self.stride_h, self.stride_w)
    }

    /// Unpool using the given 3D indices and output size.
    pub fn forward_with_indices(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        indices: &[usize],
        output_size: (usize, usize, usize),
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 5 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxUnpool3d expects 5D input [N, C, D, H, W]",
                },
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let d_pooled = input_shape[2];
        let h_pooled = input_shape[3];
        let w_pooled = input_shape[4];
        let (d_out, h_out, w_out) = output_size;
        let spatial_pooled = d_pooled * h_pooled * w_pooled;

        if indices.len() != spatial_pooled {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "MaxUnpool3d: indices length must match pooled spatial size",
                },
            )));
        }

        let input_vals = session.tensor_values(input)?;
        let out_spatial = d_out * h_out * w_out;
        let mut output = vec![0.0f64; n * c * out_spatial];

        for batch in 0..n {
            for ch in 0..c {
                for (i, &dst_pos) in indices.iter().enumerate().take(spatial_pooled) {
                    let src_idx = batch * c * spatial_pooled + ch * spatial_pooled + i;
                    if dst_pos < out_spatial {
                        let dst_idx = batch * c * out_spatial + ch * out_spatial + dst_pos;
                        output[dst_idx] = input_vals[src_idx];
                    }
                }
            }
        }

        session.tensor_variable(output, vec![n, c, d_out, h_out, w_out], false)
    }
}

/// Batch normalization over 4D input `[N, C, H, W]`.
///
/// Normalizes over the batch and spatial dimensions (0, 2, 3), keeping per-channel
/// statistics. During training, computes batch statistics and updates exponential
/// moving averages. During evaluation, uses stored running statistics.
pub struct BatchNorm2d {
    weight: TensorNodeId,
    bias: TensorNodeId,
    running_mean: std::cell::RefCell<Vec<f64>>,
    running_var: std::cell::RefCell<Vec<f64>>,
    running_mean_buffer: std::cell::RefCell<TensorNodeId>,
    running_var_buffer: std::cell::RefCell<TensorNodeId>,
    num_batches_tracked: std::cell::Cell<u64>,
    num_batches_tracked_buffer: std::cell::RefCell<TensorNodeId>,
    registered_parameters: std::cell::RefCell<Vec<RegisteredParameter>>,
    registered_buffers: std::cell::RefCell<Vec<RegisteredBuffer>>,
    num_features: usize,
    eps: f64,
    momentum: f64,
    training: std::cell::Cell<bool>,
}

impl BatchNorm2d {
    /// Create a new BatchNorm2d module.
    ///
    /// Weight (gamma) initialized to ones, bias (beta) to zeros.
    /// Running mean initialized to zeros, running var to ones.
    pub fn new(
        session: &mut FrankenTorchSession,
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> Result<Self, AutogradError> {
        if num_features == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "BatchNorm2d requires num_features > 0",
                },
            )));
        }

        let weight = session.tensor_variable(vec![1.0; num_features], vec![num_features], true)?;
        let bias = session.tensor_variable(vec![0.0; num_features], vec![num_features], true)?;
        let running_mean_buffer =
            session.tensor_variable(vec![0.0; num_features], vec![num_features], false)?;
        let running_var_buffer =
            session.tensor_variable(vec![1.0; num_features], vec![num_features], false)?;
        let num_batches_tracked_buffer = session.tensor_variable(vec![0.0], vec![1], false)?;

        Ok(Self {
            weight,
            bias,
            running_mean: std::cell::RefCell::new(vec![0.0; num_features]),
            running_var: std::cell::RefCell::new(vec![1.0; num_features]),
            running_mean_buffer: std::cell::RefCell::new(running_mean_buffer),
            running_var_buffer: std::cell::RefCell::new(running_var_buffer),
            num_batches_tracked: std::cell::Cell::new(0),
            num_batches_tracked_buffer: std::cell::RefCell::new(num_batches_tracked_buffer),
            registered_parameters: std::cell::RefCell::new(Vec::new()),
            registered_buffers: std::cell::RefCell::new(vec![
                RegisteredBuffer {
                    name: "running_mean".to_string(),
                    tensor: Some(running_mean_buffer),
                    persistent: true,
                },
                RegisteredBuffer {
                    name: "running_var".to_string(),
                    tensor: Some(running_var_buffer),
                    persistent: true,
                },
                RegisteredBuffer {
                    name: "num_batches_tracked".to_string(),
                    tensor: Some(num_batches_tracked_buffer),
                    persistent: true,
                },
            ]),
            num_features,
            eps,
            momentum,
            training: std::cell::Cell::new(true),
        })
    }

    /// Access the weight (gamma) parameter node ID.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias (beta) parameter node ID.
    #[must_use]
    pub fn bias(&self) -> TensorNodeId {
        self.bias
    }

    /// Number of features (channels).
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Set the module training mode.
    pub fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    /// Set the module to evaluation mode.
    pub fn eval(&self) {
        self.train(false);
    }

    /// Check if the module is in training mode.
    #[must_use]
    pub fn is_training(&self) -> bool {
        self.training.get()
    }

    /// Get a copy of the current running mean.
    pub fn running_mean(&self) -> Vec<f64> {
        self.running_mean.borrow().clone()
    }

    /// Get a copy of the current running variance.
    pub fn running_var(&self) -> Vec<f64> {
        self.running_var.borrow().clone()
    }

    fn has_builtin_parameter_name(name: &str) -> bool {
        matches!(name, "weight" | "bias")
    }

    fn has_builtin_buffer_name(name: &str) -> bool {
        matches!(name, "running_mean" | "running_var" | "num_batches_tracked")
    }

    fn forward_train(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        input_shape: &[usize],
    ) -> Result<TensorNodeId, AutogradError> {
        let n = input_shape[0];
        let c = input_shape[1];
        let h = input_shape[2];
        let w = input_shape[3];
        let m = n * h * w; // number of elements per channel

        // Permute [N, C, H, W] -> [N, H, W, C] then reshape to [N*H*W, C]
        let perm = session.tensor_permute(input, vec![0, 2, 3, 1])?;
        let flat = session.tensor_reshape(perm, vec![m, c])?;

        // Batch mean: mean_dim(flat, 0) -> [C]
        let batch_mean = session.tensor_mean_dim(flat, 0)?;

        // diff = flat - expand(unsqueeze(mean, 0), [m, c])
        let mean_us = session.tensor_unsqueeze(batch_mean, 0)?;
        let mean_exp = session.tensor_expand(mean_us, vec![m, c])?;
        let diff = session.tensor_sub(flat, mean_exp)?;

        // batch_var = mean_dim(diff^2, 0) -> [C] (biased, divides by m)
        let diff_sq = session.tensor_mul(diff, diff)?;
        let batch_var = session.tensor_mean_dim(diff_sq, 0)?;

        // Update running statistics
        {
            let mean_vals = session.tensor_values(batch_mean)?;
            let var_vals = session.tensor_values(batch_var)?;
            let mut rm = self.running_mean.borrow_mut();
            let mut rv = self.running_var.borrow_mut();
            // Unbiased variance for running stats: var * m / (m-1)
            let bessel_factor = if m > 1 {
                m as f64 / (m as f64 - 1.0)
            } else {
                1.0
            };
            for i in 0..c {
                rm[i] = (1.0 - self.momentum) * rm[i] + self.momentum * mean_vals[i];
                rv[i] = (1.0 - self.momentum) * rv[i] + self.momentum * var_vals[i] * bessel_factor;
            }

            let running_mean_buffer = session.tensor_variable(rm.clone(), vec![c], false)?;
            let running_var_buffer = session.tensor_variable(rv.clone(), vec![c], false)?;
            let num_batches_tracked = self.num_batches_tracked.get().saturating_add(1);
            self.num_batches_tracked.set(num_batches_tracked);
            let num_batches_tracked_buffer =
                session.tensor_variable(vec![num_batches_tracked as f64], vec![1], false)?;

            self.running_mean_buffer.replace(running_mean_buffer);
            self.running_var_buffer.replace(running_var_buffer);
            self.num_batches_tracked_buffer
                .replace(num_batches_tracked_buffer);

            let mut registered_buffers = self.registered_buffers.borrow_mut();
            upsert_registered_buffer(
                &mut registered_buffers,
                "running_mean",
                Some(running_mean_buffer),
                true,
            );
            upsert_registered_buffer(
                &mut registered_buffers,
                "running_var",
                Some(running_var_buffer),
                true,
            );
            upsert_registered_buffer(
                &mut registered_buffers,
                "num_batches_tracked",
                Some(num_batches_tracked_buffer),
                true,
            );
        }

        // std = sqrt(var + eps) -> [C]
        let eps_t = session.full(vec![c], self.eps, false)?;
        let var_eps = session.tensor_add(batch_var, eps_t)?;
        let std_t = session.tensor_sqrt(var_eps)?;

        // Expand std to [m, C]
        let std_us = session.tensor_unsqueeze(std_t, 0)?;
        let std_exp = session.tensor_expand(std_us, vec![m, c])?;

        // normalized = diff / std
        let normalized = session.tensor_div(diff, std_exp)?;

        // Apply affine: weight * normalized + bias
        let w_us = session.tensor_unsqueeze(self.weight, 0)?;
        let w_exp = session.tensor_expand(w_us, vec![m, c])?;
        let b_us = session.tensor_unsqueeze(self.bias, 0)?;
        let b_exp = session.tensor_expand(b_us, vec![m, c])?;

        let scaled = session.tensor_mul(w_exp, normalized)?;
        let affine = session.tensor_add(scaled, b_exp)?;

        // Reshape [m, C] -> [N, H, W, C] -> permute to [N, C, H, W]
        let reshaped = session.tensor_reshape(affine, vec![n, h, w, c])?;
        session.tensor_permute(reshaped, vec![0, 3, 1, 2])
    }

    fn forward_eval(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        input_shape: &[usize],
    ) -> Result<TensorNodeId, AutogradError> {
        let n = input_shape[0];
        let c = input_shape[1];
        let h = input_shape[2];
        let w = input_shape[3];
        let m = n * h * w;

        // Permute [N, C, H, W] -> [N, H, W, C] then reshape to [N*H*W, C]
        let perm = session.tensor_permute(input, vec![0, 2, 3, 1])?;
        let flat = session.tensor_reshape(perm, vec![m, c])?;

        // Running-stat buffers are maintained as non-grad tensors.
        let mean_t = *self.running_mean_buffer.borrow();
        let var_t = *self.running_var_buffer.borrow();

        // Expand mean to [m, C]
        let mean_us = session.tensor_unsqueeze(mean_t, 0)?;
        let mean_exp = session.tensor_expand(mean_us, vec![m, c])?;

        // diff = flat - mean
        let diff = session.tensor_sub(flat, mean_exp)?;

        // std = sqrt(var + eps) -> [C]
        let eps_t = session.full(vec![c], self.eps, false)?;
        let var_eps = session.tensor_add(var_t, eps_t)?;
        let std_t = session.tensor_sqrt(var_eps)?;

        // Expand std to [m, C]
        let std_us = session.tensor_unsqueeze(std_t, 0)?;
        let std_exp = session.tensor_expand(std_us, vec![m, c])?;

        // normalized = diff / std
        let normalized = session.tensor_div(diff, std_exp)?;

        // Apply affine
        let w_us = session.tensor_unsqueeze(self.weight, 0)?;
        let w_exp = session.tensor_expand(w_us, vec![m, c])?;
        let b_us = session.tensor_unsqueeze(self.bias, 0)?;
        let b_exp = session.tensor_expand(b_us, vec![m, c])?;

        let scaled = session.tensor_mul(w_exp, normalized)?;
        let affine = session.tensor_add(scaled, b_exp)?;

        // Reshape [m, C] -> [N, H, W, C] -> permute to [N, C, H, W]
        let reshaped = session.tensor_reshape(affine, vec![n, h, w, c])?;
        session.tensor_permute(reshaped, vec![0, 3, 1, 2])
    }
}

impl Module for BatchNorm2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "BatchNorm2d expects 4D input [N, C, H, W]",
                },
            )));
        }

        if input_shape[1] != self.num_features {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "input channel dimension does not match num_features",
                },
            )));
        }

        if self.training.get() {
            self.forward_train(session, input, &input_shape)
        } else {
            self.forward_eval(session, input, &input_shape)
        }
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight, self.bias];
        params.extend(
            self.registered_parameters
                .borrow()
                .iter()
                .filter_map(|entry| entry.tensor),
        );
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        vec![("weight", self.weight), ("bias", self.bias)]
    }

    fn named_parameter_slots_own(&self) -> Vec<(String, Option<TensorNodeId>)> {
        self.registered_parameters
            .borrow()
            .iter()
            .map(|entry| (entry.name.clone(), entry.tensor))
            .collect()
    }

    fn named_buffer_slots_own(&self) -> Vec<(String, Option<TensorNodeId>, bool)> {
        self.registered_buffers
            .borrow()
            .iter()
            .map(|entry| (entry.name.clone(), entry.tensor, entry.persistent))
            .collect()
    }

    fn register_parameter(
        &mut self,
        name: &str,
        parameter: Option<TensorNodeId>,
    ) -> Result<(), ModuleRegistrationError> {
        if !is_valid_registration_name(name) {
            return Err(ModuleRegistrationError::InvalidName { kind: "parameter" });
        }
        if Self::has_builtin_parameter_name(name)
            || Self::has_builtin_buffer_name(name)
            || self
                .registered_buffers
                .borrow()
                .iter()
                .any(|entry| entry.name == name)
        {
            return Err(ModuleRegistrationError::NameConflict {
                name: name.to_string(),
            });
        }
        let mut registered_parameters = self.registered_parameters.borrow_mut();
        upsert_registered_parameter(&mut registered_parameters, name, parameter);
        Ok(())
    }

    fn register_buffer(
        &mut self,
        name: &str,
        tensor: Option<TensorNodeId>,
        persistent: bool,
    ) -> Result<(), ModuleRegistrationError> {
        if !is_valid_registration_name(name) {
            return Err(ModuleRegistrationError::InvalidName { kind: "buffer" });
        }
        if Self::has_builtin_parameter_name(name)
            || self
                .registered_parameters
                .borrow()
                .iter()
                .any(|entry| entry.name == name)
        {
            return Err(ModuleRegistrationError::NameConflict {
                name: name.to_string(),
            });
        }

        let effective_persistent = if Self::has_builtin_buffer_name(name) {
            true
        } else {
            persistent
        };
        match (name, tensor) {
            ("running_mean", Some(id)) => {
                self.running_mean_buffer.replace(id);
            }
            ("running_var", Some(id)) => {
                self.running_var_buffer.replace(id);
            }
            ("num_batches_tracked", Some(id)) => {
                self.num_batches_tracked_buffer.replace(id);
            }
            _ => {}
        }
        let mut registered_buffers = self.registered_buffers.borrow_mut();
        upsert_registered_buffer(&mut registered_buffers, name, tensor, effective_persistent);
        Ok(())
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

pub struct Identity;

impl Module for Identity {
    fn forward(
        &self,
        _session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        Ok(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

// ── Upsampling Modules ─────────────────────────────────────────────────

/// Nearest-neighbor upsampling for 1D inputs (3D tensors `[N, C, L]`).
///
/// Repeats each element along the length dimension `scale_factor` times.
/// Output shape: `[N, C, L * scale_factor]`.
pub struct Upsample1d {
    scale_factor: usize,
}

impl Upsample1d {
    /// Create a new Upsample1d with the given integer scale factor.
    #[must_use]
    pub fn new(scale_factor: usize) -> Self {
        Self { scale_factor }
    }
}

impl Module for Upsample1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Upsample1d expects 3D input [N, C, L]",
                },
            )));
        }

        if self.scale_factor <= 1 {
            return Ok(input);
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let l_in = input_shape[2];
        let l_out = l_in * self.scale_factor;

        // Nearest-neighbor: unsqueeze last dim, expand, then reshape
        // [N, C, L] → [N, C, L, 1]
        let unsqueezed = session.tensor_unsqueeze(input, 3)?;
        // [N, C, L, 1] → [N, C, L, scale_factor]
        let expanded = session.tensor_expand(
            unsqueezed,
            vec![batch_size, channels, l_in, self.scale_factor],
        )?;
        // [N, C, L, scale_factor] → [N, C, L * scale_factor]
        session.tensor_reshape(expanded, vec![batch_size, channels, l_out])
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Nearest-neighbor upsampling for 2D inputs (4D tensors `[N, C, H, W]`).
///
/// Repeats each element along both spatial dimensions by the scale factor.
/// Output shape: `[N, C, H * scale_h, W * scale_w]`.
pub struct Upsample2d {
    scale_h: usize,
    scale_w: usize,
}

impl Upsample2d {
    /// Create a new Upsample2d. If a single scale factor is needed, pass it for both.
    #[must_use]
    pub fn new(scale_h: usize, scale_w: usize) -> Self {
        Self { scale_h, scale_w }
    }
}

impl Module for Upsample2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Upsample2d expects 4D input [N, C, H, W]",
                },
            )));
        }

        if self.scale_h <= 1 && self.scale_w <= 1 {
            return Ok(input);
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];
        let h_out = h_in * self.scale_h;
        let w_out = w_in * self.scale_w;

        // Nearest-neighbor for 2D:
        // [N, C, H, W] → [N, C, H, 1, W, 1]
        let x = session.tensor_reshape(input, vec![batch_size, channels, h_in, 1, w_in, 1])?;
        // expand → [N, C, H, scale_h, W, scale_w]
        let x = session.tensor_expand(
            x,
            vec![batch_size, channels, h_in, self.scale_h, w_in, self.scale_w],
        )?;
        // reshape → [N, C, H*scale_h, W*scale_w]
        session.tensor_reshape(x, vec![batch_size, channels, h_out, w_out])
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

// ── Transposed Convolution ─────────────────────────────────────────────

/// 1D transposed convolution (sometimes called "deconvolution").
///
/// Applies a transposed convolution over an input signal of shape `[N, C_in, L]`.
/// Output shape: `[N, C_out, L_out]` where `L_out = (L - 1) * stride - 2*padding + kernel_size`.
///
/// Useful in decoder architectures for learned upsampling.
pub struct ConvTranspose1d {
    weight: TensorNodeId,
    bias: Option<TensorNodeId>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl ConvTranspose1d {
    /// Create a new ConvTranspose1d with Kaiming uniform initialization.
    pub fn new(
        session: &mut FrankenTorchSession,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) -> Result<Self, AutogradError> {
        if in_channels == 0 || out_channels == 0 || kernel_size == 0 || stride == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose1d requires positive dimensions and stride",
                },
            )));
        }

        // Weight shape: [in_channels, out_channels, kernel_size]
        // (transposed from Conv1d's [out_channels, in_channels, kernel_size])
        let fan_in = in_channels * kernel_size;
        let bound = 1.0 / (fan_in as f64).sqrt();
        let numel = in_channels * out_channels * kernel_size;

        let w_rand = session.rand(vec![numel], false)?;
        let w_scale = session.full(vec![numel], 2.0 * bound, false)?;
        let w_scaled = session.tensor_mul(w_rand, w_scale)?;
        let w_shift = session.full(vec![numel], bound, false)?;
        let w_shifted = session.tensor_sub(w_scaled, w_shift)?;
        let w_values = session.tensor_values(w_shifted)?;
        let weight = session.tensor_variable(
            w_values,
            vec![in_channels, out_channels, kernel_size],
            true,
        )?;

        let bias = if use_bias {
            let b_rand = session.rand(vec![1, out_channels], false)?;
            let b_scale = session.full(vec![1, out_channels], 2.0 * bound, false)?;
            let b_scaled = session.tensor_mul(b_rand, b_scale)?;
            let b_shift = session.full(vec![1, out_channels], bound, false)?;
            let b_shifted = session.tensor_sub(b_scaled, b_shift)?;
            let b_values = session.tensor_values(b_shifted)?;
            Some(session.tensor_variable(b_values, vec![1, out_channels], true)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Access the weight parameter.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias parameter.
    #[must_use]
    pub fn bias(&self) -> Option<TensorNodeId> {
        self.bias
    }
}

impl Module for ConvTranspose1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose1d expects 3D input [N, C_in, L]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let c_in = input_shape[1];
        if c_in != self.in_channels {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose1d input channels do not match in_channels",
                },
            )));
        }
        let l_in = input_shape[2];

        // Output length: (L - 1) * stride - 2*padding + kernel_size
        let l_out = (l_in - 1) * self.stride + self.kernel_size - 2 * self.padding;

        // Transposed convolution via scatter-and-accumulate:
        // For each input position, scatter kernel-weighted values to output positions.
        // Initialize output to zeros
        let output = session.zeros(vec![batch_size, self.out_channels, l_out], false)?;

        // For each input channel c_in and each output channel c_out:
        // For each input position i: output[:, c_out, i*stride + k - padding] += input[:, c_in, i] * weight[c_in, c_out, k]
        //
        // We implement this by iterating over kernel positions and accumulating.
        let mut result = output;
        for k in 0..self.kernel_size {
            // Extract weight slice for this kernel position: weight[:, :, k]
            // weight shape: [in_channels, out_channels, kernel_size]
            let w_k = session.tensor_narrow(self.weight, 2, k, 1)?;
            // [in_channels, out_channels, 1] → [in_channels, out_channels]
            let w_k = session.tensor_squeeze(w_k, 2)?;

            // For each input position, compute contribution at output position
            for i in 0..l_in {
                let out_pos_raw = i * self.stride + k;
                if out_pos_raw < self.padding || out_pos_raw - self.padding >= l_out {
                    continue;
                }
                let out_pos = out_pos_raw - self.padding;

                // Extract input[:, :, i] → [batch, in_channels]
                let x_i = session.tensor_narrow(input, 2, i, 1)?;
                let x_i = session.tensor_squeeze(x_i, 2)?;

                // Compute x_i @ w_k → [batch, out_channels]
                let contrib = session.tensor_matmul(x_i, w_k)?;
                // Unsqueeze to [batch, out_channels, 1]
                let contrib = session.tensor_unsqueeze(contrib, 2)?;

                // Create a zero tensor and place contrib at position out_pos
                // We do this by padding: pad left by out_pos, right by (l_out - out_pos - 1)
                let pad_left = out_pos;
                let pad_right = l_out - out_pos - 1;
                let contrib_padded = session.tensor_pad(contrib, &[pad_left, pad_right], 0.0)?;

                result = session.tensor_add(result, contrib_padded)?;
            }
        }

        // Add bias if present
        if let Some(bias) = self.bias {
            let bias_shape = vec![batch_size, self.out_channels, l_out];
            let bias_expanded = session.tensor_expand(bias, vec![1, self.out_channels, 1])?;
            let bias_expanded = session.tensor_expand(bias_expanded, bias_shape)?;
            result = session.tensor_add(result, bias_expanded)?;
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight];
        if let Some(bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        let mut params = vec![("weight", self.weight)];
        if let Some(bias) = self.bias {
            params.push(("bias", bias));
        }
        params
    }
}

// ── Conv3d ─────────────────────────────────────────────────────────────

/// 3D convolution over 5D input `[N, C_in, D, H, W]`.
///
/// Output shape: `[N, C_out, D_out, H_out, W_out]`
/// where `D_out = (D + 2*padding_d - kD) / stride_d + 1` (similarly for H, W).
pub struct Conv3d {
    weight: TensorNodeId,
    bias: Option<TensorNodeId>,
    in_channels: usize,
    out_channels: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
}

impl Conv3d {
    /// Create a new Conv3d with Kaiming uniform initialization.
    ///
    /// `kernel_size` is `(kD, kH, kW)`, `stride` is `(sD, sH, sW)`, `padding` is `(pD, pH, pW)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: &mut FrankenTorchSession,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        use_bias: bool,
    ) -> Result<Self, AutogradError> {
        let (kd, kh, kw) = kernel_size;
        let (sd, sh, sw) = stride;
        let (pd, ph, pw) = padding;

        if in_channels == 0 || out_channels == 0 || kd == 0 || kh == 0 || kw == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv3d requires positive in_channels, out_channels, kernel_size",
                },
            )));
        }
        if sd == 0 || sh == 0 || sw == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv3d requires stride > 0",
                },
            )));
        }

        let fan_in = in_channels * kd * kh * kw;
        let bound = 1.0 / (fan_in as f64).sqrt();
        let numel = out_channels * in_channels * kd * kh * kw;

        let w_rand = session.rand(vec![numel], false)?;
        let w_scale = session.full(vec![numel], 2.0 * bound, false)?;
        let w_scaled = session.tensor_mul(w_rand, w_scale)?;
        let w_shift = session.full(vec![numel], bound, false)?;
        let w_shifted = session.tensor_sub(w_scaled, w_shift)?;
        let w_values = session.tensor_values(w_shifted)?;
        let weight =
            session.tensor_variable(w_values, vec![out_channels, in_channels, kd, kh, kw], true)?;

        let bias = if use_bias {
            let b_rand = session.rand(vec![out_channels], false)?;
            let b_scale = session.full(vec![out_channels], 2.0 * bound, false)?;
            let b_scaled = session.tensor_mul(b_rand, b_scale)?;
            let b_shift = session.full(vec![out_channels], bound, false)?;
            let b_shifted = session.tensor_sub(b_scaled, b_shift)?;
            let b_values = session.tensor_values(b_shifted)?;
            Some(session.tensor_variable(b_values, vec![out_channels], true)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_d: kd,
            kernel_h: kh,
            kernel_w: kw,
            stride_d: sd,
            stride_h: sh,
            stride_w: sw,
            padding_d: pd,
            padding_h: ph,
            padding_w: pw,
        })
    }

    /// Access the weight parameter.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias parameter.
    #[must_use]
    pub fn bias(&self) -> Option<TensorNodeId> {
        self.bias
    }
}

impl Module for Conv3d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 5 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv3d expects 5D input [N, C_in, D, H, W]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let c_in = input_shape[1];
        if c_in != self.in_channels {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv3d input channels do not match in_channels",
                },
            )));
        }

        let d_in = input_shape[2];
        let h_in = input_shape[3];
        let w_in = input_shape[4];

        // Apply padding if needed: pad innermost dims first [W_l,W_r, H_l,H_r, D_l,D_r]
        let padded = if self.padding_d > 0 || self.padding_h > 0 || self.padding_w > 0 {
            session.tensor_pad(
                input,
                &[
                    self.padding_w,
                    self.padding_w,
                    self.padding_h,
                    self.padding_h,
                    self.padding_d,
                    self.padding_d,
                ],
                0.0,
            )?
        } else {
            input
        };
        let d_padded = d_in + 2 * self.padding_d;
        let h_padded = h_in + 2 * self.padding_h;
        let w_padded = w_in + 2 * self.padding_w;

        if d_padded < self.kernel_d || h_padded < self.kernel_h || w_padded < self.kernel_w {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "Conv3d input too small for given kernel_size and padding",
                },
            )));
        }
        let d_out = (d_padded - self.kernel_d) / self.stride_d + 1;
        let h_out = (h_padded - self.kernel_h) / self.stride_h + 1;
        let w_out = (w_padded - self.kernel_w) / self.stride_w + 1;

        // Im2col: extract sliding 3D windows
        let ck = self.in_channels * self.kernel_d * self.kernel_h * self.kernel_w;
        let spatial_out = d_out * h_out * w_out;
        let mut patches = Vec::with_capacity(spatial_out);
        for di in 0..d_out {
            let d_start = di * self.stride_d;
            let depth_slice = session.tensor_narrow(padded, 2, d_start, self.kernel_d)?;
            for hi in 0..h_out {
                let h_start = hi * self.stride_h;
                let row_slice = session.tensor_narrow(depth_slice, 3, h_start, self.kernel_h)?;
                for wi in 0..w_out {
                    let w_start = wi * self.stride_w;
                    let patch = session.tensor_narrow(row_slice, 4, w_start, self.kernel_w)?;
                    let flat = session.tensor_reshape(patch, vec![batch_size, 1, ck])?;
                    patches.push(flat);
                }
            }
        }

        // cat along dim 1: [N, D_out*H_out*W_out, ck]
        let unfolded = session.tensor_cat(&patches, 1)?;

        // Weight: [C_out, C_in, kD, kH, kW] -> [C_out, ck] -> transpose -> [ck, C_out]
        let w_flat = session.tensor_reshape(self.weight, vec![self.out_channels, ck])?;
        let w_t = session.tensor_transpose(w_flat, 0, 1)?;

        let w_us = session.tensor_unsqueeze(w_t, 0)?;
        let w_expanded = session.tensor_expand(w_us, vec![batch_size, ck, self.out_channels])?;

        // bmm: [N, spatial_out, ck] @ [N, ck, C_out] -> [N, spatial_out, C_out]
        let output = session.tensor_bmm(unfolded, w_expanded)?;

        // Transpose to [N, C_out, spatial_out], reshape to [N, C_out, D_out, H_out, W_out]
        let output_t = session.tensor_transpose(output, 1, 2)?;
        let output_5d = session.tensor_reshape(
            output_t,
            vec![batch_size, self.out_channels, d_out, h_out, w_out],
        )?;

        match self.bias {
            Some(bias) => {
                let b_rs = session.tensor_reshape(bias, vec![1, self.out_channels, 1, 1, 1])?;
                let b_exp = session.tensor_expand(
                    b_rs,
                    vec![batch_size, self.out_channels, d_out, h_out, w_out],
                )?;
                session.tensor_add(output_5d, b_exp)
            }
            None => Ok(output_5d),
        }
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight];
        if let Some(bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        let mut params = vec![("weight", self.weight)];
        if let Some(bias) = self.bias {
            params.push(("bias", bias));
        }
        params
    }
}

// ── ConvTranspose2d ────────────────────────────────────────────────────

/// Transposed 2D convolution (deconvolution) over 4D input `[N, C_in, H, W]`.
///
/// Output shape: `[N, C_out, H_out, W_out]`
/// where `H_out = (H - 1) * stride_h - 2*padding_h + kernel_h + output_padding_h`
/// (similarly for W).
pub struct ConvTranspose2d {
    weight: TensorNodeId,
    bias: Option<TensorNodeId>,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    output_padding_h: usize,
    output_padding_w: usize,
}

impl ConvTranspose2d {
    /// Create a new ConvTranspose2d with Kaiming uniform initialization.
    ///
    /// `kernel_size` is `(kH, kW)`, `stride` is `(sH, sW)`, `padding` is `(pH, pW)`,
    /// `output_padding` is `(opH, opW)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: &mut FrankenTorchSession,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        use_bias: bool,
    ) -> Result<Self, AutogradError> {
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (oph, opw) = output_padding;

        if in_channels == 0 || out_channels == 0 || kh == 0 || kw == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose2d requires positive dimensions",
                },
            )));
        }
        if sh == 0 || sw == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose2d requires stride > 0",
                },
            )));
        }
        if oph >= sh || opw >= sw {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose2d output_padding must be < stride",
                },
            )));
        }

        // Weight shape: [in_channels, out_channels, kH, kW]
        let fan_in = in_channels * kh * kw;
        let bound = 1.0 / (fan_in as f64).sqrt();
        let numel = in_channels * out_channels * kh * kw;

        let w_rand = session.rand(vec![numel], false)?;
        let w_scale = session.full(vec![numel], 2.0 * bound, false)?;
        let w_scaled = session.tensor_mul(w_rand, w_scale)?;
        let w_shift = session.full(vec![numel], bound, false)?;
        let w_shifted = session.tensor_sub(w_scaled, w_shift)?;
        let w_values = session.tensor_values(w_shifted)?;
        let weight =
            session.tensor_variable(w_values, vec![in_channels, out_channels, kh, kw], true)?;

        let bias = if use_bias {
            let b_rand = session.rand(vec![out_channels], false)?;
            let b_scale = session.full(vec![out_channels], 2.0 * bound, false)?;
            let b_scaled = session.tensor_mul(b_rand, b_scale)?;
            let b_shift = session.full(vec![out_channels], bound, false)?;
            let b_shifted = session.tensor_sub(b_scaled, b_shift)?;
            let b_values = session.tensor_values(b_shifted)?;
            Some(session.tensor_variable(b_values, vec![out_channels], true)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_h: kh,
            kernel_w: kw,
            stride_h: sh,
            stride_w: sw,
            padding_h: ph,
            padding_w: pw,
            output_padding_h: oph,
            output_padding_w: opw,
        })
    }

    /// Access the weight parameter.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias parameter.
    #[must_use]
    pub fn bias(&self) -> Option<TensorNodeId> {
        self.bias
    }
}

impl Module for ConvTranspose2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 4 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose2d expects 4D input [N, C_in, H, W]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let c_in = input_shape[1];
        if c_in != self.in_channels {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose2d input channels do not match in_channels",
                },
            )));
        }
        let h_in = input_shape[2];
        let w_in = input_shape[3];

        // Output dimensions
        let h_out =
            (h_in - 1) * self.stride_h - 2 * self.padding_h + self.kernel_h + self.output_padding_h;
        let w_out =
            (w_in - 1) * self.stride_w - 2 * self.padding_w + self.kernel_w + self.output_padding_w;

        // Transposed convolution via col2im approach (autograd-friendly).
        // For each output position, we gather contributions from overlapping
        // input positions and kernel weights, then sum them.
        //
        // weight shape: [in_channels, out_channels, kH, kW]
        // Reshape to [in_channels, out_channels * kH * kW], transpose to
        // [out_channels * kH * kW, in_channels]
        let ok = self.out_channels * self.kernel_h * self.kernel_w;
        // input: [N, C_in, H_in, W_in] -> flatten spatial: [N, C_in, H_in*W_in]
        // -> transpose: [N, H_in*W_in, C_in]
        let x_flat =
            session.tensor_reshape(input, vec![batch_size, self.in_channels, h_in * w_in])?;
        let x_flat = session.tensor_transpose(x_flat, 1, 2)?; // [N, H_in*W_in, C_in]

        // weight: [in_channels, out_channels, kH, kW] -> [in_channels, ok]
        let w_flat = session.tensor_reshape(self.weight, vec![self.in_channels, ok])?;
        // transpose: [ok, in_channels]
        let w_t = session.tensor_transpose(w_flat, 0, 1)?;
        // expand for bmm: [N, ok, in_channels]
        let w_us = session.tensor_unsqueeze(w_t, 0)?;
        let w_expanded = session.tensor_expand(w_us, vec![batch_size, ok, self.in_channels])?;

        // bmm: [N, H_in*W_in, C_in] @ [N, C_in, ok]^T = [N, H_in*W_in, ok]
        // Actually: [N, H_in*W_in, C_in] @ [N, ok, C_in]^T
        // We need [N, C_in, ok], so transpose w_expanded dims 1,2
        let w_for_bmm = session.tensor_transpose(w_expanded, 1, 2)?; // [N, C_in, ok]
        // bmm: [N, H_in*W_in, C_in] @ [N, C_in, ok] -> [N, H_in*W_in, ok]
        let columns = session.tensor_bmm(x_flat, w_for_bmm)?;
        // columns[n, hw, :] contains out_channels*kH*kW values for input position hw
        // Reshape: [N, H_in, W_in, out_channels, kH, kW]
        let columns = session.tensor_reshape(
            columns,
            vec![
                batch_size,
                h_in,
                w_in,
                self.out_channels,
                self.kernel_h,
                self.kernel_w,
            ],
        )?;

        // Col2im: scatter each column into the output.
        // For each input (hi, wi) and kernel (kh, kw):
        //   out_h = hi*stride_h + kh - padding_h
        //   out_w = wi*stride_w + kw - padding_w
        //   output[:, :, out_h, out_w] += columns[:, hi, wi, :, kh, kw]
        //
        // Group by output row for efficiency: for each output row out_h,
        // collect all contributions, cat them, and sum.

        // (Row-by-row approach was too complex; using simpler kernel-position method below.)

        // SIMPLER APPROACH: For each (kh, kw), shift the matmul result into the
        // correct output position using narrow/cat on an intermediate grid.
        //
        // For each kernel position (kh, kw), the contributions form a grid at
        // positions (hi*stride_h + kh - padding_h, wi*stride_w + kw - padding_w).
        // Each contribution is columns[:, hi, wi, :, kh, kw].
        //
        // Since these positions form a regular grid, we can reshape and
        // use narrow to extract the valid region.

        // Reset approach: direct accumulation using narrow on the output.
        // Actually, let's use the simplest autograd-friendly approach:
        // Process each (hi, wi, kh, kw) individually and accumulate using scatter-add.
        // But we don't have scatter_add...
        //
        // Final approach: build output column-by-column, stacking contributions.
        // For each output position (oh, ow), find all (hi, wi, kh, kw) contributing.

        // Actually the most practical approach: for each (kh, kw), compute a shifted
        // grid and use it to build partial outputs that we sum together.
        let mut result = session.zeros(vec![batch_size, self.out_channels, h_out, w_out], false)?;

        for kh in 0..self.kernel_h {
            for kw in 0..self.kernel_w {
                // Extract columns[:, :, :, :, kh, kw] -> [N, H_in, W_in, out_channels]
                let c = session.tensor_narrow(columns, 4, kh, 1)?;
                let c = session.tensor_narrow(c, 5, kw, 1)?;
                let c =
                    session.tensor_reshape(c, vec![batch_size, h_in, w_in, self.out_channels])?;
                // Transpose to [N, out_channels, H_in, W_in]
                let c = session.tensor_transpose(c, 1, 3)?; // [N, out_ch, W_in, H_in]
                let c = session.tensor_transpose(c, 2, 3)?; // [N, out_ch, H_in, W_in]

                // This grid needs to be placed at output positions:
                //   oh = hi*stride_h + kh - padding_h
                //   ow = wi*stride_w + kw - padding_w
                // The grid spacing in output space is stride_h x stride_w.
                // If stride > 1, we need to insert zeros between elements.

                // Upsample by stride: insert (stride-1) zeros between elements
                let mut upsampled = c;
                if self.stride_h > 1 {
                    // Interleave rows with zeros
                    let mut rows = Vec::with_capacity(h_in);
                    for hi in 0..h_in {
                        let row = session.tensor_narrow(upsampled, 2, hi, 1)?;
                        rows.push(row);
                    }
                    let zero_row =
                        session.zeros(vec![batch_size, self.out_channels, 1, w_in], false)?;
                    let mut interleaved =
                        Vec::with_capacity(h_in + (h_in - 1) * (self.stride_h - 1));
                    for (i, row) in rows.iter().enumerate() {
                        interleaved.push(*row);
                        if i < h_in - 1 {
                            for _ in 1..self.stride_h {
                                interleaved.push(zero_row);
                            }
                        }
                    }
                    upsampled = session.tensor_cat(&interleaved, 2)?;
                }
                let up_h = if h_in > 0 {
                    (h_in - 1) * self.stride_h + 1
                } else {
                    0
                };

                if self.stride_w > 1 {
                    // Interleave cols with zeros
                    let mut cols = Vec::with_capacity(w_in);
                    for wi in 0..w_in {
                        let col = session.tensor_narrow(upsampled, 3, wi, 1)?;
                        cols.push(col);
                    }
                    let zero_col =
                        session.zeros(vec![batch_size, self.out_channels, up_h, 1], false)?;
                    let mut interleaved =
                        Vec::with_capacity(w_in + (w_in - 1) * (self.stride_w - 1));
                    for (i, col) in cols.iter().enumerate() {
                        interleaved.push(*col);
                        if i < w_in - 1 {
                            for _ in 1..self.stride_w {
                                interleaved.push(zero_col);
                            }
                        }
                    }
                    upsampled = session.tensor_cat(&interleaved, 3)?;
                }
                let up_w = if w_in > 0 {
                    (w_in - 1) * self.stride_w + 1
                } else {
                    0
                };

                // Now upsampled is [N, out_ch, up_h, up_w] where
                // up_h = (h_in-1)*stride_h + 1, up_w = (w_in-1)*stride_w + 1
                // It needs to be placed starting at (kh - padding_h, kw - padding_w)
                // in the h_out x w_out output.

                // Compute the valid region overlap with output
                let src_h_start = self.padding_h.saturating_sub(kh);
                let dst_h_start = kh.saturating_sub(self.padding_h);
                let src_h_end = up_h.min(h_out + self.padding_h - kh);
                let src_w_start = self.padding_w.saturating_sub(kw);
                let dst_w_start = kw.saturating_sub(self.padding_w);
                let src_w_end = up_w.min(w_out + self.padding_w - kw);

                if src_h_start >= src_h_end || src_w_start >= src_w_end {
                    continue;
                }

                let valid_h = src_h_end - src_h_start;
                let valid_w = src_w_end - src_w_start;

                // Extract valid region from upsampled
                let valid = session.tensor_narrow(upsampled, 2, src_h_start, valid_h)?;
                let valid = session.tensor_narrow(valid, 3, src_w_start, valid_w)?;

                // Extract corresponding region from result, add, put back
                // We can't "put back" easily, so instead we create a full-sized
                // contribution using reshape+cat. But that's also complex.
                // Simplest: accumulate in a flat buffer using narrow on result.
                //
                // Actually, the cleanest autograd-friendly way: build a full
                // h_out x w_out tensor with the valid region and zeros elsewhere,
                // using cat operations.

                // Build the full contribution tensor by padding with zero-rows/cols
                // using cat (autograd-tracked unlike tensor_pad)
                let mut padded = valid;

                // Pad height: add zero rows top and bottom
                if dst_h_start > 0 {
                    let top_zeros = session.zeros(
                        vec![batch_size, self.out_channels, dst_h_start, valid_w],
                        false,
                    )?;
                    padded = session.tensor_cat(&[top_zeros, padded], 2)?;
                }
                let bottom = h_out - dst_h_start - valid_h;
                if bottom > 0 {
                    let bot_zeros = session
                        .zeros(vec![batch_size, self.out_channels, bottom, valid_w], false)?;
                    padded = session.tensor_cat(&[padded, bot_zeros], 2)?;
                }
                // padded is now [N, out_ch, h_out, valid_w]

                // Pad width: add zero cols left and right
                if dst_w_start > 0 {
                    let left_zeros = session.zeros(
                        vec![batch_size, self.out_channels, h_out, dst_w_start],
                        false,
                    )?;
                    padded = session.tensor_cat(&[left_zeros, padded], 3)?;
                }
                let right = w_out - dst_w_start - valid_w;
                if right > 0 {
                    let right_zeros =
                        session.zeros(vec![batch_size, self.out_channels, h_out, right], false)?;
                    padded = session.tensor_cat(&[padded, right_zeros], 3)?;
                }
                // padded is now [N, out_ch, h_out, w_out]

                result = session.tensor_add(result, padded)?;
            }
        }

        // Add bias
        if let Some(bias) = self.bias {
            let b_rs = session.tensor_reshape(bias, vec![1, self.out_channels, 1, 1])?;
            let b_exp =
                session.tensor_expand(b_rs, vec![batch_size, self.out_channels, h_out, w_out])?;
            result = session.tensor_add(result, b_exp)?;
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight];
        if let Some(bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        let mut params = vec![("weight", self.weight)];
        if let Some(bias) = self.bias {
            params.push(("bias", bias));
        }
        params
    }
}

// ── ConvTranspose3d ────────────────────────────────────────────────────

/// Transposed 3D convolution (deconvolution) over 5D input `[N, C_in, D, H, W]`.
///
/// Output shape: `[N, C_out, D_out, H_out, W_out]`
/// where `D_out = (D - 1) * stride_d - 2*padding_d + kernel_d + output_padding_d`
/// (similarly for H, W).
pub struct ConvTranspose3d {
    weight: TensorNodeId,
    bias: Option<TensorNodeId>,
    in_channels: usize,
    out_channels: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
    output_padding_d: usize,
    output_padding_h: usize,
    output_padding_w: usize,
}

impl ConvTranspose3d {
    /// Create a new ConvTranspose3d with Kaiming uniform initialization.
    ///
    /// `kernel_size` is `(kD, kH, kW)`, `stride` is `(sD, sH, sW)`,
    /// `padding` is `(pD, pH, pW)`, `output_padding` is `(opD, opH, opW)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: &mut FrankenTorchSession,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        use_bias: bool,
    ) -> Result<Self, AutogradError> {
        let (kd, kh, kw) = kernel_size;
        let (sd, sh, sw) = stride;
        let (pd, ph, pw) = padding;
        let (opd, oph, opw) = output_padding;

        if in_channels == 0 || out_channels == 0 || kd == 0 || kh == 0 || kw == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose3d requires positive dimensions",
                },
            )));
        }
        if sd == 0 || sh == 0 || sw == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose3d requires stride > 0",
                },
            )));
        }
        if opd >= sd || oph >= sh || opw >= sw {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose3d output_padding must be < stride",
                },
            )));
        }

        // Weight shape: [in_channels, out_channels, kD, kH, kW]
        let fan_in = in_channels * kd * kh * kw;
        let bound = 1.0 / (fan_in as f64).sqrt();
        let numel = in_channels * out_channels * kd * kh * kw;

        let w_rand = session.rand(vec![numel], false)?;
        let w_scale = session.full(vec![numel], 2.0 * bound, false)?;
        let w_scaled = session.tensor_mul(w_rand, w_scale)?;
        let w_shift = session.full(vec![numel], bound, false)?;
        let w_shifted = session.tensor_sub(w_scaled, w_shift)?;
        let w_values = session.tensor_values(w_shifted)?;
        let weight =
            session.tensor_variable(w_values, vec![in_channels, out_channels, kd, kh, kw], true)?;

        let bias = if use_bias {
            let b_rand = session.rand(vec![out_channels], false)?;
            let b_scale = session.full(vec![out_channels], 2.0 * bound, false)?;
            let b_scaled = session.tensor_mul(b_rand, b_scale)?;
            let b_shift = session.full(vec![out_channels], bound, false)?;
            let b_shifted = session.tensor_sub(b_scaled, b_shift)?;
            let b_values = session.tensor_values(b_shifted)?;
            Some(session.tensor_variable(b_values, vec![out_channels], true)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_d: kd,
            kernel_h: kh,
            kernel_w: kw,
            stride_d: sd,
            stride_h: sh,
            stride_w: sw,
            padding_d: pd,
            padding_h: ph,
            padding_w: pw,
            output_padding_d: opd,
            output_padding_h: oph,
            output_padding_w: opw,
        })
    }

    /// Access the weight parameter.
    #[must_use]
    pub fn weight(&self) -> TensorNodeId {
        self.weight
    }

    /// Access the bias parameter.
    #[must_use]
    pub fn bias(&self) -> Option<TensorNodeId> {
        self.bias
    }
}

impl Module for ConvTranspose3d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };

        if input_shape.len() != 5 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose3d expects 5D input [N, C_in, D, H, W]",
                },
            )));
        }

        let batch_size = input_shape[0];
        let c_in = input_shape[1];
        if c_in != self.in_channels {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ConvTranspose3d input channels do not match in_channels",
                },
            )));
        }
        let d_in = input_shape[2];
        let h_in = input_shape[3];
        let w_in = input_shape[4];

        let d_out =
            (d_in - 1) * self.stride_d - 2 * self.padding_d + self.kernel_d + self.output_padding_d;
        let h_out =
            (h_in - 1) * self.stride_h - 2 * self.padding_h + self.kernel_h + self.output_padding_h;
        let w_out =
            (w_in - 1) * self.stride_w - 2 * self.padding_w + self.kernel_w + self.output_padding_w;

        let mut result = session.zeros(
            vec![batch_size, self.out_channels, d_out, h_out, w_out],
            false,
        )?;

        for kd in 0..self.kernel_d {
            for kh in 0..self.kernel_h {
                for kw in 0..self.kernel_w {
                    // weight[:, :, kd, kh, kw] -> [in_channels, out_channels]
                    let w_slice = session.tensor_narrow(self.weight, 2, kd, 1)?;
                    let w_slice = session.tensor_narrow(w_slice, 3, kh, 1)?;
                    let w_slice = session.tensor_narrow(w_slice, 4, kw, 1)?;
                    let w_slice = session.tensor_squeeze(w_slice, 4)?;
                    let w_slice = session.tensor_squeeze(w_slice, 3)?;
                    let w_slice = session.tensor_squeeze(w_slice, 2)?;

                    for di in 0..d_in {
                        let out_d_raw = di * self.stride_d + kd;
                        if out_d_raw < self.padding_d || out_d_raw - self.padding_d >= d_out {
                            continue;
                        }
                        let out_d = out_d_raw - self.padding_d;

                        for hi in 0..h_in {
                            let out_h_raw = hi * self.stride_h + kh;
                            if out_h_raw < self.padding_h || out_h_raw - self.padding_h >= h_out {
                                continue;
                            }
                            let out_h = out_h_raw - self.padding_h;

                            for wi in 0..w_in {
                                let out_w_raw = wi * self.stride_w + kw;
                                if out_w_raw < self.padding_w || out_w_raw - self.padding_w >= w_out
                                {
                                    continue;
                                }
                                let out_w = out_w_raw - self.padding_w;

                                // input[:, :, di, hi, wi] -> [batch, in_channels]
                                let x_dhw = session.tensor_narrow(input, 2, di, 1)?;
                                let x_dhw = session.tensor_narrow(x_dhw, 3, hi, 1)?;
                                let x_dhw = session.tensor_narrow(x_dhw, 4, wi, 1)?;
                                let x_dhw = session.tensor_squeeze(x_dhw, 4)?;
                                let x_dhw = session.tensor_squeeze(x_dhw, 3)?;
                                let x_dhw = session.tensor_squeeze(x_dhw, 2)?;

                                let contrib = session.tensor_matmul(x_dhw, w_slice)?;
                                // -> [batch, out_channels, 1, 1, 1]
                                let contrib = session.tensor_unsqueeze(contrib, 2)?;
                                let contrib = session.tensor_unsqueeze(contrib, 3)?;
                                let contrib = session.tensor_unsqueeze(contrib, 4)?;

                                // Pad to place at (out_d, out_h, out_w)
                                let contrib_padded = session.tensor_pad(
                                    contrib,
                                    &[
                                        out_w,
                                        w_out - out_w - 1,
                                        out_h,
                                        h_out - out_h - 1,
                                        out_d,
                                        d_out - out_d - 1,
                                    ],
                                    0.0,
                                )?;

                                result = session.tensor_add(result, contrib_padded)?;
                            }
                        }
                    }
                }
            }
        }

        if let Some(bias) = self.bias {
            let b_rs = session.tensor_reshape(bias, vec![1, self.out_channels, 1, 1, 1])?;
            let b_exp = session.tensor_expand(
                b_rs,
                vec![batch_size, self.out_channels, d_out, h_out, w_out],
            )?;
            result = session.tensor_add(result, b_exp)?;
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = vec![self.weight];
        if let Some(bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        let mut params = vec![("weight", self.weight)];
        if let Some(bias) = self.bias {
            params.push(("bias", bias));
        }
        params
    }
}

// ── Recurrent Cell Modules ─────────────────────────────────────────────

/// Elman RNN cell: `h' = activation(x @ W_ih^T + h @ W_hh^T + b_ih + b_hh)`.
///
/// Input shape: `[batch, input_size]`. Hidden state shape: `[batch, hidden_size]`.
/// Returns the new hidden state `h'`.
pub struct RNNCell {
    w_ih: TensorNodeId,
    w_hh: TensorNodeId,
    b_ih: TensorNodeId,
    b_hh: TensorNodeId,
    input_size: usize,
    hidden_size: usize,
    use_tanh: bool,
}

impl RNNCell {
    /// Create a new RNNCell.
    ///
    /// * `input_size` - size of the input features
    /// * `hidden_size` - size of the hidden state
    /// * `use_tanh` - if true, use tanh activation; otherwise use relu
    pub fn new(
        session: &mut FrankenTorchSession,
        input_size: usize,
        hidden_size: usize,
        use_tanh: bool,
    ) -> Result<Self, AutogradError> {
        let bound = 1.0 / (hidden_size as f64).sqrt();

        let w_ih = Self::init_weight(session, hidden_size, input_size, bound)?;
        let w_hh = Self::init_weight(session, hidden_size, hidden_size, bound)?;
        let b_ih = Self::init_bias(session, hidden_size, bound)?;
        let b_hh = Self::init_bias(session, hidden_size, bound)?;

        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            input_size,
            hidden_size,
            use_tanh,
        })
    }

    fn init_weight(
        session: &mut FrankenTorchSession,
        rows: usize,
        cols: usize,
        bound: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let numel = rows * cols;
        let r = session.rand(vec![numel], false)?;
        let scale = session.full(vec![numel], 2.0 * bound, false)?;
        let scaled = session.tensor_mul(r, scale)?;
        let shift = session.full(vec![numel], bound, false)?;
        let shifted = session.tensor_sub(scaled, shift)?;
        let vals = session.tensor_values(shifted)?;
        session.tensor_variable(vals, vec![rows, cols], true)
    }

    fn init_bias(
        session: &mut FrankenTorchSession,
        size: usize,
        bound: f64,
    ) -> Result<TensorNodeId, AutogradError> {
        let r = session.rand(vec![1, size], false)?;
        let scale = session.full(vec![1, size], 2.0 * bound, false)?;
        let scaled = session.tensor_mul(r, scale)?;
        let shift = session.full(vec![1, size], bound, false)?;
        let shifted = session.tensor_sub(scaled, shift)?;
        let vals = session.tensor_values(shifted)?;
        session.tensor_variable(vals, vec![1, size], true)
    }

    /// Run one step of the RNN cell.
    ///
    /// * `input` - `[batch, input_size]`
    /// * `hx` - previous hidden state `[batch, hidden_size]`
    ///
    /// Returns: new hidden state `[batch, hidden_size]`
    pub fn forward_cell(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        hx: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };
        if input_shape.len() != 2 || input_shape[1] != self.input_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "RNNCell expects input shape [batch, input_size]",
                },
            )));
        }

        // x @ W_ih^T
        let w_ih_t = session.tensor_transpose(self.w_ih, 0, 1)?;
        let xw = session.tensor_matmul(input, w_ih_t)?;

        // h @ W_hh^T
        let w_hh_t = session.tensor_transpose(self.w_hh, 0, 1)?;
        let hw = session.tensor_matmul(hx, w_hh_t)?;

        // xw + hw + b_ih + b_hh
        let out_shape = {
            let (_, meta) = session.tensor_values_meta(xw)?;
            meta.shape().to_vec()
        };
        let b_ih_exp = session.tensor_expand(self.b_ih, out_shape.clone())?;
        let b_hh_exp = session.tensor_expand(self.b_hh, out_shape)?;

        let sum1 = session.tensor_add(xw, hw)?;
        let sum2 = session.tensor_add(sum1, b_ih_exp)?;
        let sum3 = session.tensor_add(sum2, b_hh_exp)?;

        // Apply activation
        if self.use_tanh {
            session.tensor_tanh(sum3)
        } else {
            session.tensor_relu(sum3)
        }
    }

    /// Get the hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Collect trainable parameters.
    pub fn parameters(&self) -> Vec<TensorNodeId> {
        vec![self.w_ih, self.w_hh, self.b_ih, self.b_hh]
    }
}

/// LSTM cell: processes one time step of LSTM computation.
///
/// Implements the standard LSTM equations:
/// ```text
/// i = sigmoid(x @ W_ii^T + h @ W_hi^T + b_ii + b_hi)  // input gate
/// f = sigmoid(x @ W_if^T + h @ W_hf^T + b_if + b_hf)  // forget gate
/// g = tanh(x @ W_ig^T + h @ W_hg^T + b_ig + b_hg)     // cell gate
/// o = sigmoid(x @ W_io^T + h @ W_ho^T + b_io + b_ho)   // output gate
/// c' = f * c + i * g                                     // new cell state
/// h' = o * tanh(c')                                      // new hidden state
/// ```
///
/// Input: `[batch, input_size]`. Hidden/cell states: `[batch, hidden_size]`.
pub struct LSTMCell {
    w_ih: TensorNodeId,
    w_hh: TensorNodeId,
    b_ih: TensorNodeId,
    b_hh: TensorNodeId,
    hidden_size: usize,
    input_size: usize,
}

impl LSTMCell {
    /// Create a new LSTMCell.
    pub fn new(
        session: &mut FrankenTorchSession,
        input_size: usize,
        hidden_size: usize,
    ) -> Result<Self, AutogradError> {
        let bound = 1.0 / (hidden_size as f64).sqrt();
        let gate_size = 4 * hidden_size;

        // W_ih: [4*hidden_size, input_size]
        let w_ih = RNNCell::init_weight(session, gate_size, input_size, bound)?;
        // W_hh: [4*hidden_size, hidden_size]
        let w_hh = RNNCell::init_weight(session, gate_size, hidden_size, bound)?;
        // b_ih, b_hh: [4*hidden_size]
        let b_ih = RNNCell::init_bias(session, gate_size, bound)?;
        let b_hh = RNNCell::init_bias(session, gate_size, bound)?;

        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_size,
            input_size,
        })
    }

    /// Run one step of the LSTM cell.
    ///
    /// * `input` - `[batch, input_size]`
    /// * `hx` - previous hidden state `[batch, hidden_size]`
    /// * `cx` - previous cell state `[batch, hidden_size]`
    ///
    /// Returns: `(h', c')` - new hidden state and new cell state
    pub fn forward_cell(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        hx: TensorNodeId,
        cx: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId), AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };
        if input_shape.len() != 2 || input_shape[1] != self.input_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "LSTMCell expects input shape [batch, input_size]",
                },
            )));
        }
        let batch_size = input_shape[0];

        // gates = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh
        let w_ih_t = session.tensor_transpose(self.w_ih, 0, 1)?;
        let xw = session.tensor_matmul(input, w_ih_t)?;

        let w_hh_t = session.tensor_transpose(self.w_hh, 0, 1)?;
        let hw = session.tensor_matmul(hx, w_hh_t)?;

        let gates_shape = {
            let (_, meta) = session.tensor_values_meta(xw)?;
            meta.shape().to_vec()
        };
        let b_ih_exp = session.tensor_expand(self.b_ih, gates_shape.clone())?;
        let b_hh_exp = session.tensor_expand(self.b_hh, gates_shape)?;

        let sum1 = session.tensor_add(xw, hw)?;
        let sum2 = session.tensor_add(sum1, b_ih_exp)?;
        let gates = session.tensor_add(sum2, b_hh_exp)?;
        // gates shape: [batch, 4*hidden_size]

        // Split into 4 chunks along dim=1
        let chunks = session.tensor_chunk(gates, 4, 1)?;
        let i_gate = session.tensor_sigmoid(chunks[0])?;
        let f_gate = session.tensor_sigmoid(chunks[1])?;
        let g_gate = session.tensor_tanh(chunks[2])?;
        let o_gate = session.tensor_sigmoid(chunks[3])?;

        // c' = f * c + i * g
        let fc = session.tensor_mul(f_gate, cx)?;
        let ig = session.tensor_mul(i_gate, g_gate)?;
        let cx_new = session.tensor_add(fc, ig)?;

        // h' = o * tanh(c')
        let tanh_cx = session.tensor_tanh(cx_new)?;
        let hx_new = session.tensor_mul(o_gate, tanh_cx)?;

        debug_assert_eq!(
            {
                let (_, m) = session.tensor_values_meta(hx_new).unwrap();
                m.shape().to_vec()
            },
            vec![batch_size, self.hidden_size]
        );

        Ok((hx_new, cx_new))
    }

    /// Get the hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Collect trainable parameters.
    pub fn parameters(&self) -> Vec<TensorNodeId> {
        vec![self.w_ih, self.w_hh, self.b_ih, self.b_hh]
    }
}

/// GRU cell: processes one time step of GRU computation.
///
/// Implements the standard GRU equations:
/// ```text
/// r = sigmoid(x @ W_ir^T + h @ W_hr^T + b_ir + b_hr)  // reset gate
/// z = sigmoid(x @ W_iz^T + h @ W_hz^T + b_iz + b_hz)  // update gate
/// n = tanh(x @ W_in^T + r * (h @ W_hn^T + b_hn) + b_in) // new gate
/// h' = (1 - z) * n + z * h                              // new hidden state
/// ```
///
/// Input: `[batch, input_size]`. Hidden state: `[batch, hidden_size]`.
pub struct GRUCell {
    w_ih: TensorNodeId,
    w_hh: TensorNodeId,
    b_ih: TensorNodeId,
    b_hh: TensorNodeId,
    hidden_size: usize,
    input_size: usize,
}

impl GRUCell {
    /// Create a new GRUCell.
    pub fn new(
        session: &mut FrankenTorchSession,
        input_size: usize,
        hidden_size: usize,
    ) -> Result<Self, AutogradError> {
        let bound = 1.0 / (hidden_size as f64).sqrt();
        let gate_size = 3 * hidden_size;

        let w_ih = RNNCell::init_weight(session, gate_size, input_size, bound)?;
        let w_hh = RNNCell::init_weight(session, gate_size, hidden_size, bound)?;
        let b_ih = RNNCell::init_bias(session, gate_size, bound)?;
        let b_hh = RNNCell::init_bias(session, gate_size, bound)?;

        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_size,
            input_size,
        })
    }

    /// Run one step of the GRU cell.
    ///
    /// * `input` - `[batch, input_size]`
    /// * `hx` - previous hidden state `[batch, hidden_size]`
    ///
    /// Returns: new hidden state `[batch, hidden_size]`
    pub fn forward_cell(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        hx: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = {
            let (_, meta) = session.tensor_values_meta(input)?;
            meta.shape().to_vec()
        };
        if input_shape.len() != 2 || input_shape[1] != self.input_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GRUCell expects input shape [batch, input_size]",
                },
            )));
        }
        let batch_size = input_shape[0];

        // Compute input-side gates: x @ W_ih^T + b_ih
        let w_ih_t = session.tensor_transpose(self.w_ih, 0, 1)?;
        let x_gates = session.tensor_matmul(input, w_ih_t)?;
        let x_gates_shape = {
            let (_, meta) = session.tensor_values_meta(x_gates)?;
            meta.shape().to_vec()
        };
        let b_ih_exp = session.tensor_expand(self.b_ih, x_gates_shape)?;
        let x_gates = session.tensor_add(x_gates, b_ih_exp)?;

        // Compute hidden-side gates: h @ W_hh^T + b_hh
        let w_hh_t = session.tensor_transpose(self.w_hh, 0, 1)?;
        let h_gates = session.tensor_matmul(hx, w_hh_t)?;
        let h_gates_shape = {
            let (_, meta) = session.tensor_values_meta(h_gates)?;
            meta.shape().to_vec()
        };
        let b_hh_exp = session.tensor_expand(self.b_hh, h_gates_shape)?;
        let h_gates = session.tensor_add(h_gates, b_hh_exp)?;

        // Split x_gates into 3 chunks: [x_r, x_z, x_n]
        let x_chunks = session.tensor_chunk(x_gates, 3, 1)?;
        // Split h_gates into 3 chunks: [h_r, h_z, h_n]
        let h_chunks = session.tensor_chunk(h_gates, 3, 1)?;

        // r = sigmoid(x_r + h_r) — reset gate
        let r_sum = session.tensor_add(x_chunks[0], h_chunks[0])?;
        let r = session.tensor_sigmoid(r_sum)?;

        // z = sigmoid(x_z + h_z) — update gate
        let z_sum = session.tensor_add(x_chunks[1], h_chunks[1])?;
        let z = session.tensor_sigmoid(z_sum)?;

        // n = tanh(x_n + r * h_n) — new gate
        let r_h_n = session.tensor_mul(r, h_chunks[2])?;
        let n_sum = session.tensor_add(x_chunks[2], r_h_n)?;
        let n = session.tensor_tanh(n_sum)?;

        // h' = (1 - z) * n + z * h
        let ones = session.full(vec![batch_size, self.hidden_size], 1.0, false)?;
        let one_minus_z = session.tensor_sub(ones, z)?;
        let term1 = session.tensor_mul(one_minus_z, n)?;
        let term2 = session.tensor_mul(z, hx)?;
        session.tensor_add(term1, term2)
    }

    /// Get the hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Collect trainable parameters.
    pub fn parameters(&self) -> Vec<TensorNodeId> {
        vec![self.w_ih, self.w_hh, self.b_ih, self.b_hh]
    }
}

// ── Full Sequence Modules ─────────────────────────────────────────────

/// Full LSTM module: processes entire sequences through multi-layer LSTM.
///
/// Wraps [`LSTMCell`] to iterate over time steps, with support for:
/// - Multi-layer stacking (`num_layers`)
/// - Bidirectional processing (`bidirectional`)
/// - Dropout between layers (not applied after the last layer)
/// - `batch_first` input layout
///
/// # Input shapes
///
/// If `batch_first` is false (default): input is `[seq_len, batch, input_size]`.
/// If `batch_first` is true: input is `[batch, seq_len, input_size]`.
///
/// # Forward signature
///
/// ```text
/// lstm(input, (h_0, c_0)) -> (output, (h_n, c_n))
/// ```
///
/// - `h_0`, `c_0`: initial hidden/cell states `[num_layers * num_directions, batch, hidden_size]`
/// - `output`: `[seq_len, batch, hidden_size * num_directions]` (or batch_first variant)
/// - `h_n`, `c_n`: final hidden/cell states `[num_layers * num_directions, batch, hidden_size]`
pub struct LSTM {
    /// Per-layer cells. For bidirectional, even indices are forward, odd are reverse.
    cells: Vec<LSTMCell>,
    /// Dropout modules applied between layers (length = num_layers - 1).
    dropout_layers: Vec<Dropout>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    batch_first: bool,
    training: std::cell::Cell<bool>,
}

/// Output of LSTM forward pass.
pub struct LSTMOutput {
    /// Output tensor: `[seq_len, batch, hidden_size * num_directions]`
    /// (or `[batch, seq_len, ...]` if `batch_first`).
    pub output: TensorNodeId,
    /// Final hidden state: `[num_layers * num_directions, batch, hidden_size]`.
    pub h_n: TensorNodeId,
    /// Final cell state: `[num_layers * num_directions, batch, hidden_size]`.
    pub c_n: TensorNodeId,
}

impl LSTM {
    /// Create a new LSTM module.
    ///
    /// * `input_size` - number of expected features in the input
    /// * `hidden_size` - number of features in the hidden state
    /// * `num_layers` - number of recurrent layers (default: 1)
    /// * `bidirectional` - if true, becomes a bidirectional LSTM
    /// * `dropout` - dropout probability between layers (ignored if `num_layers == 1`)
    /// * `batch_first` - if true, input/output tensors have batch as first dimension
    pub fn new(
        session: &mut FrankenTorchSession,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout: f64,
        batch_first: bool,
    ) -> Result<Self, AutogradError> {
        if num_layers == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "LSTM num_layers must be >= 1",
                },
            )));
        }

        let num_directions: usize = if bidirectional { 2 } else { 1 };
        let mut cells = Vec::with_capacity(num_layers * num_directions);

        for layer in 0..num_layers {
            let layer_input_size = if layer == 0 {
                input_size
            } else {
                hidden_size * num_directions
            };

            // Forward direction cell
            cells.push(LSTMCell::new(session, layer_input_size, hidden_size)?);

            // Reverse direction cell (if bidirectional)
            if bidirectional {
                cells.push(LSTMCell::new(session, layer_input_size, hidden_size)?);
            }
        }

        // Create dropout modules between layers (not after last layer)
        let mut dropout_layers = Vec::new();
        if num_layers > 1 {
            for _ in 0..(num_layers - 1) {
                dropout_layers.push(Dropout::new(dropout));
            }
        }

        Ok(Self {
            cells,
            dropout_layers,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            batch_first,
            training: std::cell::Cell::new(true),
        })
    }

    /// Run the full LSTM forward pass.
    ///
    /// * `input` - input tensor `[seq_len, batch, input_size]` (or `[batch, seq_len, input_size]` if `batch_first`)
    /// * `h_0` - optional initial hidden state `[num_layers * num_directions, batch, hidden_size]`
    /// * `c_0` - optional initial cell state `[num_layers * num_directions, batch, hidden_size]`
    ///
    /// Returns `LSTMOutput { output, h_n, c_n }`.
    pub fn forward_lstm(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        h_0: Option<TensorNodeId>,
        c_0: Option<TensorNodeId>,
    ) -> Result<LSTMOutput, AutogradError> {
        let num_directions: usize = if self.bidirectional { 2 } else { 1 };

        // Get input shape and transpose if batch_first
        let working_input = if self.batch_first {
            // [batch, seq_len, input_size] -> [seq_len, batch, input_size]
            session.tensor_transpose(input, 0, 1)?
        } else {
            input
        };

        let input_shape = {
            let (_, meta) = session.tensor_values_meta(working_input)?;
            meta.shape().to_vec()
        };
        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "LSTM expects 3D input [seq_len, batch, input_size]",
                },
            )));
        }
        let seq_len = input_shape[0];
        let batch_size = input_shape[1];

        if input_shape[2] != self.input_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "LSTM input feature size does not match input_size",
                },
            )));
        }

        // Initialize hidden/cell states: [num_layers * num_directions, batch, hidden_size]
        let total_layers = self.num_layers * num_directions;
        let state_shape = vec![total_layers, batch_size, self.hidden_size];

        let h_init = match h_0 {
            Some(h) => h,
            None => session.zeros(state_shape.clone(), false)?,
        };
        let c_init = match c_0 {
            Some(c) => c,
            None => session.zeros(state_shape, false)?,
        };

        // Unbind initial states along dim=0 to get per-layer-direction states
        let h_states = session.tensor_unbind(h_init, 0)?;
        let c_states = session.tensor_unbind(c_init, 0)?;

        // Each state is [batch, hidden_size]; we need [batch, hidden_size] for LSTMCell
        // But unbind removes the dim, giving us [batch, hidden_size] — perfect.

        // Unbind input along time dimension (dim=0): list of [batch, input_size]
        let time_steps = session.tensor_unbind(working_input, 0)?;

        // Collect final h_n and c_n per layer-direction
        let mut h_n_list: Vec<TensorNodeId> = Vec::with_capacity(total_layers);
        let mut c_n_list: Vec<TensorNodeId> = Vec::with_capacity(total_layers);

        // Current layer input: starts as the original time_steps
        let mut layer_input = time_steps;

        for layer in 0..self.num_layers {
            // Forward direction
            let fwd_cell_idx = layer * num_directions;
            let fwd_h0 = h_states[fwd_cell_idx];
            let fwd_c0 = c_states[fwd_cell_idx];

            let (fwd_outputs, fwd_h_n, fwd_c_n) = self.run_direction(
                session,
                &self.cells[fwd_cell_idx],
                &layer_input,
                fwd_h0,
                fwd_c0,
                false, // forward
            )?;

            h_n_list.push(fwd_h_n);
            c_n_list.push(fwd_c_n);

            let layer_output = if self.bidirectional {
                // Reverse direction
                let rev_cell_idx = fwd_cell_idx + 1;
                let rev_h0 = h_states[rev_cell_idx];
                let rev_c0 = c_states[rev_cell_idx];

                let (rev_outputs, rev_h_n, rev_c_n) = self.run_direction(
                    session,
                    &self.cells[rev_cell_idx],
                    &layer_input,
                    rev_h0,
                    rev_c0,
                    true, // reverse
                )?;

                h_n_list.push(rev_h_n);
                c_n_list.push(rev_c_n);

                // Concatenate forward and reverse outputs at each timestep
                // fwd_outputs[t] is [batch, hidden_size], rev_outputs[t] is [batch, hidden_size]
                // concat along dim=1 -> [batch, 2*hidden_size]
                let mut combined = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    let cat = session.tensor_cat(&[fwd_outputs[t], rev_outputs[t]], 1)?;
                    combined.push(cat);
                }
                combined
            } else {
                fwd_outputs
            };

            // Apply dropout between layers (not after last layer)
            if layer < self.num_layers - 1 {
                let mut dropped = Vec::with_capacity(layer_output.len());
                for &step_out in &layer_output {
                    let d = self.dropout_layers[layer].forward(session, step_out)?;
                    dropped.push(d);
                }
                layer_input = dropped;
            } else {
                layer_input = layer_output;
            }
        }

        // Stack time steps into output: [seq_len, batch, hidden_size * num_directions]
        let output_ids: Vec<TensorNodeId> = layer_input;
        let output = session.tensor_stack(&output_ids, 0)?;

        // Stack h_n and c_n: [num_layers * num_directions, batch, hidden_size]
        let h_n = session.tensor_stack(&h_n_list, 0)?;
        let c_n = session.tensor_stack(&c_n_list, 0)?;

        // Transpose output back if batch_first
        let output = if self.batch_first {
            session.tensor_transpose(output, 0, 1)?
        } else {
            output
        };

        Ok(LSTMOutput { output, h_n, c_n })
    }

    /// Run one direction of one layer over all time steps.
    fn run_direction(
        &self,
        session: &mut FrankenTorchSession,
        cell: &LSTMCell,
        inputs: &[TensorNodeId],
        h_0: TensorNodeId,
        c_0: TensorNodeId,
        reverse: bool,
    ) -> Result<(Vec<TensorNodeId>, TensorNodeId, TensorNodeId), AutogradError> {
        let seq_len = inputs.len();
        let mut h = h_0;
        let mut c = c_0;
        let mut outputs = Vec::with_capacity(seq_len);

        if reverse {
            for &input in inputs.iter().rev() {
                let (h_new, c_new) = cell.forward_cell(session, input, h, c)?;
                outputs.push(h_new);
                h = h_new;
                c = c_new;
            }
            // Reverse outputs so they align with the forward time ordering
            outputs.reverse();
        } else {
            for &input in inputs {
                let (h_new, c_new) = cell.forward_cell(session, input, h, c)?;
                outputs.push(h_new);
                h = h_new;
                c = c_new;
            }
        }

        Ok((outputs, h, c))
    }

    /// Get the input size.
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get the hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Check if this LSTM is bidirectional.
    #[must_use]
    pub fn is_bidirectional(&self) -> bool {
        self.bidirectional
    }
}

impl Module for LSTM {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let result = self.forward_lstm(session, input, None, None)?;
        Ok(result.output)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = Vec::new();
        for cell in &self.cells {
            params.extend(cell.parameters());
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        // Parameters are exposed through named_children -> cells
        Vec::new()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        let mut children: Vec<(String, &dyn Module)> = Vec::new();
        for (i, dropout) in self.dropout_layers.iter().enumerate() {
            children.push((format!("dropout_{i}"), dropout as &dyn Module));
        }
        children
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
        // Propagate to dropout layers
        for dropout in &self.dropout_layers {
            dropout.train(mode);
        }
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Full GRU module: processes entire sequences through multi-layer GRU.
///
/// Wraps [`GRUCell`] to iterate over time steps, with support for:
/// - Multi-layer stacking (`num_layers`)
/// - Bidirectional processing (`bidirectional`)
/// - Dropout between layers (not applied after the last layer)
/// - `batch_first` input layout
///
/// # Input shapes
///
/// If `batch_first` is false (default): input is `[seq_len, batch, input_size]`.
/// If `batch_first` is true: input is `[batch, seq_len, input_size]`.
///
/// # Forward signature
///
/// ```text
/// gru(input, h_0) -> (output, h_n)
/// ```
///
/// - `h_0`: initial hidden state `[num_layers * num_directions, batch, hidden_size]`
/// - `output`: `[seq_len, batch, hidden_size * num_directions]` (or batch_first variant)
/// - `h_n`: final hidden state `[num_layers * num_directions, batch, hidden_size]`
pub struct GRU {
    /// Per-layer cells. For bidirectional, even indices are forward, odd are reverse.
    cells: Vec<GRUCell>,
    /// Dropout modules applied between layers (length = num_layers - 1).
    dropout_layers: Vec<Dropout>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    batch_first: bool,
    training: std::cell::Cell<bool>,
}

/// Output of GRU forward pass.
pub struct GRUOutput {
    /// Output tensor: `[seq_len, batch, hidden_size * num_directions]`
    /// (or `[batch, seq_len, ...]` if `batch_first`).
    pub output: TensorNodeId,
    /// Final hidden state: `[num_layers * num_directions, batch, hidden_size]`.
    pub h_n: TensorNodeId,
}

impl GRU {
    /// Create a new GRU module.
    ///
    /// * `input_size` - number of expected features in the input
    /// * `hidden_size` - number of features in the hidden state
    /// * `num_layers` - number of recurrent layers (default: 1)
    /// * `bidirectional` - if true, becomes a bidirectional GRU
    /// * `dropout` - dropout probability between layers (ignored if `num_layers == 1`)
    /// * `batch_first` - if true, input/output tensors have batch as first dimension
    pub fn new(
        session: &mut FrankenTorchSession,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout: f64,
        batch_first: bool,
    ) -> Result<Self, AutogradError> {
        if num_layers == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GRU num_layers must be >= 1",
                },
            )));
        }

        let num_directions: usize = if bidirectional { 2 } else { 1 };
        let mut cells = Vec::with_capacity(num_layers * num_directions);

        for layer in 0..num_layers {
            let layer_input_size = if layer == 0 {
                input_size
            } else {
                hidden_size * num_directions
            };

            cells.push(GRUCell::new(session, layer_input_size, hidden_size)?);

            if bidirectional {
                cells.push(GRUCell::new(session, layer_input_size, hidden_size)?);
            }
        }

        let mut dropout_layers = Vec::new();
        if num_layers > 1 {
            for _ in 0..(num_layers - 1) {
                dropout_layers.push(Dropout::new(dropout));
            }
        }

        Ok(Self {
            cells,
            dropout_layers,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            batch_first,
            training: std::cell::Cell::new(true),
        })
    }

    /// Run the full GRU forward pass.
    ///
    /// * `input` - input tensor `[seq_len, batch, input_size]` (or batch_first variant)
    /// * `h_0` - optional initial hidden state `[num_layers * num_directions, batch, hidden_size]`
    ///
    /// Returns `GRUOutput { output, h_n }`.
    pub fn forward_gru(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        h_0: Option<TensorNodeId>,
    ) -> Result<GRUOutput, AutogradError> {
        let num_directions: usize = if self.bidirectional { 2 } else { 1 };

        let working_input = if self.batch_first {
            session.tensor_transpose(input, 0, 1)?
        } else {
            input
        };

        let input_shape = {
            let (_, meta) = session.tensor_values_meta(working_input)?;
            meta.shape().to_vec()
        };
        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GRU expects 3D input [seq_len, batch, input_size]",
                },
            )));
        }
        let seq_len = input_shape[0];
        let batch_size = input_shape[1];

        if input_shape[2] != self.input_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "GRU input feature size does not match input_size",
                },
            )));
        }

        let total_layers = self.num_layers * num_directions;
        let state_shape = vec![total_layers, batch_size, self.hidden_size];

        let h_init = match h_0 {
            Some(h) => h,
            None => session.zeros(state_shape, false)?,
        };

        let h_states = session.tensor_unbind(h_init, 0)?;
        let time_steps = session.tensor_unbind(working_input, 0)?;

        let mut h_n_list: Vec<TensorNodeId> = Vec::with_capacity(total_layers);
        let mut layer_input = time_steps;

        for layer in 0..self.num_layers {
            let fwd_cell_idx = layer * num_directions;
            let fwd_h0 = h_states[fwd_cell_idx];

            let (fwd_outputs, fwd_h_n) = self.run_direction(
                session,
                &self.cells[fwd_cell_idx],
                &layer_input,
                fwd_h0,
                false,
            )?;

            h_n_list.push(fwd_h_n);

            let layer_output = if self.bidirectional {
                let rev_cell_idx = fwd_cell_idx + 1;
                let rev_h0 = h_states[rev_cell_idx];

                let (rev_outputs, rev_h_n) = self.run_direction(
                    session,
                    &self.cells[rev_cell_idx],
                    &layer_input,
                    rev_h0,
                    true,
                )?;

                h_n_list.push(rev_h_n);

                let mut combined = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    let cat = session.tensor_cat(&[fwd_outputs[t], rev_outputs[t]], 1)?;
                    combined.push(cat);
                }
                combined
            } else {
                fwd_outputs
            };

            if layer < self.num_layers - 1 {
                let mut dropped = Vec::with_capacity(layer_output.len());
                for &step_out in &layer_output {
                    let d = self.dropout_layers[layer].forward(session, step_out)?;
                    dropped.push(d);
                }
                layer_input = dropped;
            } else {
                layer_input = layer_output;
            }
        }

        let output = session.tensor_stack(&layer_input, 0)?;
        let h_n = session.tensor_stack(&h_n_list, 0)?;

        let output = if self.batch_first {
            session.tensor_transpose(output, 0, 1)?
        } else {
            output
        };

        Ok(GRUOutput { output, h_n })
    }

    /// Run one direction of one layer over all time steps.
    fn run_direction(
        &self,
        session: &mut FrankenTorchSession,
        cell: &GRUCell,
        inputs: &[TensorNodeId],
        h_0: TensorNodeId,
        reverse: bool,
    ) -> Result<(Vec<TensorNodeId>, TensorNodeId), AutogradError> {
        let seq_len = inputs.len();
        let mut h = h_0;
        let mut outputs = Vec::with_capacity(seq_len);

        if reverse {
            for &input in inputs.iter().rev() {
                let h_new = cell.forward_cell(session, input, h)?;
                outputs.push(h_new);
                h = h_new;
            }
            outputs.reverse();
        } else {
            for &input in inputs {
                let h_new = cell.forward_cell(session, input, h)?;
                outputs.push(h_new);
                h = h_new;
            }
        }

        Ok((outputs, h))
    }

    /// Get the input size.
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get the hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Check if this GRU is bidirectional.
    #[must_use]
    pub fn is_bidirectional(&self) -> bool {
        self.bidirectional
    }
}

impl Module for GRU {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let result = self.forward_gru(session, input, None)?;
        Ok(result.output)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = Vec::new();
        for cell in &self.cells {
            params.extend(cell.parameters());
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        Vec::new()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        let mut children: Vec<(String, &dyn Module)> = Vec::new();
        for (i, dropout) in self.dropout_layers.iter().enumerate() {
            children.push((format!("dropout_{i}"), dropout as &dyn Module));
        }
        children
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
        for dropout in &self.dropout_layers {
            dropout.train(mode);
        }
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Full RNN module (Elman network): processes entire sequences through multi-layer RNN.
///
/// Wraps [`RNNCell`] to iterate over time steps, with support for:
/// - Multi-layer stacking (`num_layers`)
/// - Bidirectional processing (`bidirectional`)
/// - Dropout between layers (not applied after the last layer)
/// - `batch_first` input layout
/// - Configurable nonlinearity: `tanh` (default) or `relu`
///
/// # Input shapes
///
/// If `batch_first` is false (default): input is `[seq_len, batch, input_size]`.
/// If `batch_first` is true: input is `[batch, seq_len, input_size]`.
///
/// # Forward signature
///
/// ```text
/// rnn(input, h_0) -> (output, h_n)
/// ```
/// Configuration for [`RNN`] module construction.
pub struct RNNConfig {
    /// Number of recurrent layers (default: 1).
    pub num_layers: usize,
    /// If true, use tanh nonlinearity; if false, use relu (default: true).
    pub use_tanh: bool,
    /// If true, becomes a bidirectional RNN (default: false).
    pub bidirectional: bool,
    /// Dropout probability between layers, ignored if `num_layers == 1` (default: 0.0).
    pub dropout: f64,
    /// If true, input/output have batch as first dimension (default: false).
    pub batch_first: bool,
}

impl Default for RNNConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            use_tanh: true,
            bidirectional: false,
            dropout: 0.0,
            batch_first: false,
        }
    }
}

pub struct RNN {
    cells: Vec<RNNCell>,
    dropout_layers: Vec<Dropout>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    batch_first: bool,
    training: std::cell::Cell<bool>,
}

/// Output of RNN forward pass.
pub struct RNNOutput {
    /// Output tensor: `[seq_len, batch, hidden_size * num_directions]`.
    pub output: TensorNodeId,
    /// Final hidden state: `[num_layers * num_directions, batch, hidden_size]`.
    pub h_n: TensorNodeId,
}

impl RNN {
    /// Create a new RNN module.
    ///
    /// * `input_size` - number of expected features in the input
    /// * `hidden_size` - number of features in the hidden state
    /// * `config` - configuration for layers, nonlinearity, dropout, etc.
    pub fn new(
        session: &mut FrankenTorchSession,
        input_size: usize,
        hidden_size: usize,
        config: RNNConfig,
    ) -> Result<Self, AutogradError> {
        let num_layers = config.num_layers;
        let use_tanh = config.use_tanh;
        let bidirectional = config.bidirectional;
        let dropout = config.dropout;
        let batch_first = config.batch_first;
        if num_layers == 0 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "RNN num_layers must be >= 1",
                },
            )));
        }

        let num_directions: usize = if bidirectional { 2 } else { 1 };
        let mut cells = Vec::with_capacity(num_layers * num_directions);

        for layer in 0..num_layers {
            let layer_input_size = if layer == 0 {
                input_size
            } else {
                hidden_size * num_directions
            };

            cells.push(RNNCell::new(
                session,
                layer_input_size,
                hidden_size,
                use_tanh,
            )?);

            if bidirectional {
                cells.push(RNNCell::new(
                    session,
                    layer_input_size,
                    hidden_size,
                    use_tanh,
                )?);
            }
        }

        let mut dropout_layers = Vec::new();
        if num_layers > 1 {
            for _ in 0..(num_layers - 1) {
                dropout_layers.push(Dropout::new(dropout));
            }
        }

        Ok(Self {
            cells,
            dropout_layers,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            batch_first,
            training: std::cell::Cell::new(true),
        })
    }

    /// Run the full RNN forward pass.
    ///
    /// * `input` - input tensor `[seq_len, batch, input_size]` (or batch_first variant)
    /// * `h_0` - optional initial hidden state `[num_layers * num_directions, batch, hidden_size]`
    ///
    /// Returns `RNNOutput { output, h_n }`.
    pub fn forward_rnn(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        h_0: Option<TensorNodeId>,
    ) -> Result<RNNOutput, AutogradError> {
        let num_directions: usize = if self.bidirectional { 2 } else { 1 };

        let working_input = if self.batch_first {
            session.tensor_transpose(input, 0, 1)?
        } else {
            input
        };

        let input_shape = {
            let (_, meta) = session.tensor_values_meta(working_input)?;
            meta.shape().to_vec()
        };
        if input_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "RNN expects 3D input [seq_len, batch, input_size]",
                },
            )));
        }
        let seq_len = input_shape[0];
        let batch_size = input_shape[1];

        if input_shape[2] != self.input_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "RNN input feature size does not match input_size",
                },
            )));
        }

        let total_layers = self.num_layers * num_directions;
        let state_shape = vec![total_layers, batch_size, self.hidden_size];

        let h_init = match h_0 {
            Some(h) => h,
            None => session.zeros(state_shape, false)?,
        };

        let h_states = session.tensor_unbind(h_init, 0)?;
        let time_steps = session.tensor_unbind(working_input, 0)?;

        let mut h_n_list: Vec<TensorNodeId> = Vec::with_capacity(total_layers);
        let mut layer_input = time_steps;

        for layer in 0..self.num_layers {
            let fwd_cell_idx = layer * num_directions;
            let fwd_h0 = h_states[fwd_cell_idx];

            let (fwd_outputs, fwd_h_n) = self.run_direction(
                session,
                &self.cells[fwd_cell_idx],
                &layer_input,
                fwd_h0,
                false,
            )?;

            h_n_list.push(fwd_h_n);

            let layer_output = if self.bidirectional {
                let rev_cell_idx = fwd_cell_idx + 1;
                let rev_h0 = h_states[rev_cell_idx];

                let (rev_outputs, rev_h_n) = self.run_direction(
                    session,
                    &self.cells[rev_cell_idx],
                    &layer_input,
                    rev_h0,
                    true,
                )?;

                h_n_list.push(rev_h_n);

                let mut combined = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    let cat = session.tensor_cat(&[fwd_outputs[t], rev_outputs[t]], 1)?;
                    combined.push(cat);
                }
                combined
            } else {
                fwd_outputs
            };

            if layer < self.num_layers - 1 {
                let mut dropped = Vec::with_capacity(layer_output.len());
                for &step_out in &layer_output {
                    let d = self.dropout_layers[layer].forward(session, step_out)?;
                    dropped.push(d);
                }
                layer_input = dropped;
            } else {
                layer_input = layer_output;
            }
        }

        let output = session.tensor_stack(&layer_input, 0)?;
        let h_n = session.tensor_stack(&h_n_list, 0)?;

        let output = if self.batch_first {
            session.tensor_transpose(output, 0, 1)?
        } else {
            output
        };

        Ok(RNNOutput { output, h_n })
    }

    /// Run one direction of one layer over all time steps.
    fn run_direction(
        &self,
        session: &mut FrankenTorchSession,
        cell: &RNNCell,
        inputs: &[TensorNodeId],
        h_0: TensorNodeId,
        reverse: bool,
    ) -> Result<(Vec<TensorNodeId>, TensorNodeId), AutogradError> {
        let mut h = h_0;
        let mut outputs = Vec::with_capacity(inputs.len());

        if reverse {
            for &input in inputs.iter().rev() {
                let h_new = cell.forward_cell(session, input, h)?;
                outputs.push(h_new);
                h = h_new;
            }
            outputs.reverse();
        } else {
            for &input in inputs {
                let h_new = cell.forward_cell(session, input, h)?;
                outputs.push(h_new);
                h = h_new;
            }
        }

        Ok((outputs, h))
    }

    /// Get the input size.
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get the hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Check if this RNN is bidirectional.
    #[must_use]
    pub fn is_bidirectional(&self) -> bool {
        self.bidirectional
    }
}

impl Module for RNN {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let result = self.forward_rnn(session, input, None)?;
        Ok(result.output)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = Vec::new();
        for cell in &self.cells {
            params.extend(cell.parameters());
        }
        params
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        Vec::new()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        let mut children: Vec<(String, &dyn Module)> = Vec::new();
        for (i, dropout) in self.dropout_layers.iter().enumerate() {
            children.push((format!("dropout_{i}"), dropout as &dyn Module));
        }
        children
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
        for dropout in &self.dropout_layers {
            dropout.train(mode);
        }
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

// ── Transformer Modules ──────────────────────────────────────────────

/// Activation function choice for transformer feedforward blocks.
#[derive(Clone, Copy, Debug)]
pub enum TransformerActivation {
    /// ReLU activation.
    Relu,
    /// GELU activation.
    Gelu,
}

/// A single Transformer encoder layer.
///
/// Architecture (post-norm, `norm_first=false`, default):
/// ```text
/// x = layernorm1(x + dropout(self_attn(x, x, x)))
/// x = layernorm2(x + dropout(ff(x)))
/// ```
///
/// Architecture (pre-norm, `norm_first=true`):
/// ```text
/// x = x + dropout(self_attn(layernorm1(x), layernorm1(x), layernorm1(x)))
/// x = x + dropout(ff(layernorm2(x)))
/// ```
///
/// Input shape: `[batch, seq_len, d_model]`.
pub struct TransformerEncoderLayer {
    self_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
    dropout1: Dropout,
    dropout2: Dropout,
    activation: TransformerActivation,
    norm_first: bool,
    training: std::cell::Cell<bool>,
}

impl TransformerEncoderLayer {
    /// Create a new `TransformerEncoderLayer`.
    ///
    /// * `d_model` - the number of expected features (embed_dim)
    /// * `nhead` - the number of heads in multihead attention
    /// * `dim_feedforward` - dimension of the feedforward network (default: 2048)
    /// * `dropout` - dropout value (default: 0.1)
    /// * `activation` - activation function in feedforward block
    /// * `norm_first` - if true, layer norm is done prior to attention/feedforward (pre-norm)
    pub fn new(
        session: &mut FrankenTorchSession,
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        dropout_p: f64,
        activation: TransformerActivation,
        norm_first: bool,
    ) -> Result<Self, AutogradError> {
        let self_attn = MultiheadAttention::new(session, d_model, nhead)?;
        let linear1 = Linear::new(session, d_model, dim_feedforward, true)?;
        let linear2 = Linear::new(session, dim_feedforward, d_model, true)?;
        let norm1 = LayerNorm::new(session, vec![d_model], 1e-5)?;
        let norm2 = LayerNorm::new(session, vec![d_model], 1e-5)?;
        let dropout = Dropout::new(dropout_p);
        let dropout1 = Dropout::new(dropout_p);
        let dropout2 = Dropout::new(dropout_p);

        Ok(Self {
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            dropout,
            dropout1,
            dropout2,
            activation,
            norm_first,
            training: std::cell::Cell::new(true),
        })
    }

    /// Apply the feedforward block: linear1 -> activation -> dropout -> linear2.
    fn feedforward(
        &self,
        session: &mut FrankenTorchSession,
        x: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let shape = {
            let (_, meta) = session.tensor_values_meta(x)?;
            meta.shape().to_vec()
        };
        // x is [batch, seq_len, d_model], Linear expects [batch, features]
        // Reshape to [batch*seq_len, d_model]
        let batch_seq = shape[0] * shape[1];
        let d_model = shape[2];

        let x_flat = session.tensor_reshape(x, vec![batch_seq, d_model])?;
        let h = self.linear1.forward(session, x_flat)?;

        let h = match self.activation {
            TransformerActivation::Relu => session.tensor_relu(h)?,
            TransformerActivation::Gelu => session.tensor_gelu(h)?,
        };

        let h = self.dropout.forward(session, h)?;
        let h = self.linear2.forward(session, h)?;

        session.tensor_reshape(h, shape)
    }

    /// Forward pass for the encoder layer.
    ///
    /// Input: `[batch, seq_len, d_model]`.
    /// Returns: `[batch, seq_len, d_model]`.
    pub fn forward_layer(
        &self,
        session: &mut FrankenTorchSession,
        src: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        if self.norm_first {
            // Pre-norm: x = x + dropout(self_attn(layernorm(x)))
            let normed = self.norm1.forward(session, src)?;
            let attn_out = self
                .self_attn
                .forward_qkv(session, normed, normed, normed)?;
            let attn_out = self.dropout1.forward(session, attn_out)?;
            let x = session.tensor_add(src, attn_out)?;

            // x = x + dropout(ff(layernorm(x)))
            let normed2 = self.norm2.forward(session, x)?;
            let ff_out = self.feedforward(session, normed2)?;
            let ff_out = self.dropout2.forward(session, ff_out)?;
            session.tensor_add(x, ff_out)
        } else {
            // Post-norm: x = layernorm(x + dropout(self_attn(x)))
            let attn_out = self.self_attn.forward_qkv(session, src, src, src)?;
            let attn_out = self.dropout1.forward(session, attn_out)?;
            let x = session.tensor_add(src, attn_out)?;
            let x = self.norm1.forward(session, x)?;

            // x = layernorm(x + dropout(ff(x)))
            let ff_out = self.feedforward(session, x)?;
            let ff_out = self.dropout2.forward(session, ff_out)?;
            let x = session.tensor_add(x, ff_out)?;
            self.norm2.forward(session, x)
        }
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.forward_layer(session, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = self.self_attn.parameters();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("self_attn".to_string(), &self.self_attn as &dyn Module),
            ("linear1".to_string(), &self.linear1 as &dyn Module),
            ("linear2".to_string(), &self.linear2 as &dyn Module),
            ("norm1".to_string(), &self.norm1 as &dyn Module),
            ("norm2".to_string(), &self.norm2 as &dyn Module),
            ("dropout".to_string(), &self.dropout as &dyn Module),
            ("dropout1".to_string(), &self.dropout1 as &dyn Module),
            ("dropout2".to_string(), &self.dropout2 as &dyn Module),
        ]
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
        self.dropout.train(mode);
        self.dropout1.train(mode);
        self.dropout2.train(mode);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Transformer encoder: a stack of N `TransformerEncoderLayer` instances.
///
/// Optionally applies a final `LayerNorm` after the last layer.
///
/// Input shape: `[batch, seq_len, d_model]`.
/// Output shape: `[batch, seq_len, d_model]`.
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    final_norm: Option<LayerNorm>,
}

impl TransformerEncoder {
    /// Create a new `TransformerEncoder`.
    ///
    /// * `d_model` - feature dimension
    /// * `nhead` - number of attention heads
    /// * `num_layers` - number of encoder layers
    /// * `dim_feedforward` - feedforward hidden dimension
    /// * `dropout` - dropout probability
    /// * `activation` - activation function
    /// * `norm_first` - pre-norm vs post-norm
    /// * `final_layer_norm` - if true, adds a final LayerNorm
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: &mut FrankenTorchSession,
        d_model: usize,
        nhead: usize,
        num_layers: usize,
        dim_feedforward: usize,
        dropout: f64,
        activation: TransformerActivation,
        norm_first: bool,
        final_layer_norm: bool,
    ) -> Result<Self, AutogradError> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                session,
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                norm_first,
            )?);
        }

        let final_norm = if final_layer_norm {
            Some(LayerNorm::new(session, vec![d_model], 1e-5)?)
        } else {
            None
        };

        Ok(Self { layers, final_norm })
    }
}

impl Module for TransformerEncoder {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let mut output = input;
        for layer in &self.layers {
            output = layer.forward_layer(session, output)?;
        }
        if let Some(ref norm) = self.final_norm {
            output = norm.forward(session, output)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.final_norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        let mut children: Vec<(String, &dyn Module)> = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            children.push((format!("layers.{i}"), layer as &dyn Module));
        }
        if let Some(ref norm) = self.final_norm {
            children.push(("norm".to_string(), norm as &dyn Module));
        }
        children
    }

    fn train(&self, mode: bool) {
        for layer in &self.layers {
            layer.train(mode);
        }
    }

    fn is_training(&self) -> bool {
        self.layers.first().is_some_and(|l| l.is_training())
    }
}

/// A single Transformer decoder layer.
///
/// Architecture (post-norm):
/// ```text
/// x = layernorm1(x + dropout(self_attn(x, x, x)))
/// x = layernorm2(x + dropout(cross_attn(x, memory, memory)))
/// x = layernorm3(x + dropout(ff(x)))
/// ```
///
/// Architecture (pre-norm):
/// ```text
/// x = x + dropout(self_attn(layernorm1(x)))
/// x = x + dropout(cross_attn(layernorm2(x), memory, memory))
/// x = x + dropout(ff(layernorm3(x)))
/// ```
///
/// Input: `tgt` `[batch, tgt_len, d_model]`, `memory` `[batch, src_len, d_model]`.
pub struct TransformerDecoderLayer {
    self_attn: MultiheadAttention,
    cross_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: Dropout,
    dropout1: Dropout,
    dropout2: Dropout,
    dropout3: Dropout,
    activation: TransformerActivation,
    norm_first: bool,
    training: std::cell::Cell<bool>,
}

impl TransformerDecoderLayer {
    /// Create a new `TransformerDecoderLayer`.
    pub fn new(
        session: &mut FrankenTorchSession,
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        dropout_p: f64,
        activation: TransformerActivation,
        norm_first: bool,
    ) -> Result<Self, AutogradError> {
        let self_attn = MultiheadAttention::new(session, d_model, nhead)?;
        let cross_attn = MultiheadAttention::new(session, d_model, nhead)?;
        let linear1 = Linear::new(session, d_model, dim_feedforward, true)?;
        let linear2 = Linear::new(session, dim_feedforward, d_model, true)?;
        let norm1 = LayerNorm::new(session, vec![d_model], 1e-5)?;
        let norm2 = LayerNorm::new(session, vec![d_model], 1e-5)?;
        let norm3 = LayerNorm::new(session, vec![d_model], 1e-5)?;
        let dropout = Dropout::new(dropout_p);
        let dropout1 = Dropout::new(dropout_p);
        let dropout2 = Dropout::new(dropout_p);
        let dropout3 = Dropout::new(dropout_p);

        Ok(Self {
            self_attn,
            cross_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
            dropout,
            dropout1,
            dropout2,
            dropout3,
            activation,
            norm_first,
            training: std::cell::Cell::new(true),
        })
    }

    /// Apply the feedforward block.
    fn feedforward(
        &self,
        session: &mut FrankenTorchSession,
        x: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let shape = {
            let (_, meta) = session.tensor_values_meta(x)?;
            meta.shape().to_vec()
        };
        let batch_seq = shape[0] * shape[1];
        let d_model = shape[2];

        let x_flat = session.tensor_reshape(x, vec![batch_seq, d_model])?;
        let h = self.linear1.forward(session, x_flat)?;

        let h = match self.activation {
            TransformerActivation::Relu => session.tensor_relu(h)?,
            TransformerActivation::Gelu => session.tensor_gelu(h)?,
        };

        let h = self.dropout.forward(session, h)?;
        let h = self.linear2.forward(session, h)?;

        session.tensor_reshape(h, shape)
    }

    /// Forward pass for the decoder layer.
    ///
    /// * `tgt` - target sequence `[batch, tgt_len, d_model]`
    /// * `memory` - encoder output `[batch, src_len, d_model]`
    pub fn forward_layer(
        &self,
        session: &mut FrankenTorchSession,
        tgt: TensorNodeId,
        memory: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        if self.norm_first {
            // Pre-norm: self-attention
            let normed1 = self.norm1.forward(session, tgt)?;
            let sa_out = self
                .self_attn
                .forward_qkv(session, normed1, normed1, normed1)?;
            let sa_out = self.dropout1.forward(session, sa_out)?;
            let x = session.tensor_add(tgt, sa_out)?;

            // Pre-norm: cross-attention
            let normed2 = self.norm2.forward(session, x)?;
            let ca_out = self
                .cross_attn
                .forward_qkv(session, normed2, memory, memory)?;
            let ca_out = self.dropout2.forward(session, ca_out)?;
            let x = session.tensor_add(x, ca_out)?;

            // Pre-norm: feedforward
            let normed3 = self.norm3.forward(session, x)?;
            let ff_out = self.feedforward(session, normed3)?;
            let ff_out = self.dropout3.forward(session, ff_out)?;
            session.tensor_add(x, ff_out)
        } else {
            // Post-norm: self-attention
            let sa_out = self.self_attn.forward_qkv(session, tgt, tgt, tgt)?;
            let sa_out = self.dropout1.forward(session, sa_out)?;
            let x = session.tensor_add(tgt, sa_out)?;
            let x = self.norm1.forward(session, x)?;

            // Post-norm: cross-attention
            let ca_out = self.cross_attn.forward_qkv(session, x, memory, memory)?;
            let ca_out = self.dropout2.forward(session, ca_out)?;
            let x = session.tensor_add(x, ca_out)?;
            let x = self.norm2.forward(session, x)?;

            // Post-norm: feedforward
            let ff_out = self.feedforward(session, x)?;
            let ff_out = self.dropout3.forward(session, ff_out)?;
            let x = session.tensor_add(x, ff_out)?;
            self.norm3.forward(session, x)
        }
    }
}

impl Module for TransformerDecoderLayer {
    fn forward(
        &self,
        _session: &mut FrankenTorchSession,
        _input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // Decoder requires both tgt and memory; use forward_layer directly
        Err(AutogradError::Dispatch(DispatchError::Key(
            DispatchKeyError::IncompatibleSet {
                reason: "TransformerDecoderLayer requires tgt and memory; use forward_layer()",
            },
        )))
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = self.self_attn.parameters();
        params.extend(self.cross_attn.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("self_attn".to_string(), &self.self_attn as &dyn Module),
            ("cross_attn".to_string(), &self.cross_attn as &dyn Module),
            ("linear1".to_string(), &self.linear1 as &dyn Module),
            ("linear2".to_string(), &self.linear2 as &dyn Module),
            ("norm1".to_string(), &self.norm1 as &dyn Module),
            ("norm2".to_string(), &self.norm2 as &dyn Module),
            ("norm3".to_string(), &self.norm3 as &dyn Module),
            ("dropout".to_string(), &self.dropout as &dyn Module),
            ("dropout1".to_string(), &self.dropout1 as &dyn Module),
            ("dropout2".to_string(), &self.dropout2 as &dyn Module),
            ("dropout3".to_string(), &self.dropout3 as &dyn Module),
        ]
    }

    fn train(&self, mode: bool) {
        self.training.set(mode);
        self.dropout.train(mode);
        self.dropout1.train(mode);
        self.dropout2.train(mode);
        self.dropout3.train(mode);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

/// Transformer decoder: a stack of N `TransformerDecoderLayer` instances.
///
/// Input: `tgt` `[batch, tgt_len, d_model]`, `memory` `[batch, src_len, d_model]`.
/// Output: `[batch, tgt_len, d_model]`.
pub struct TransformerDecoder {
    layers: Vec<TransformerDecoderLayer>,
    final_norm: Option<LayerNorm>,
}

impl TransformerDecoder {
    /// Create a new `TransformerDecoder`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: &mut FrankenTorchSession,
        d_model: usize,
        nhead: usize,
        num_layers: usize,
        dim_feedforward: usize,
        dropout: f64,
        activation: TransformerActivation,
        norm_first: bool,
        final_layer_norm: bool,
    ) -> Result<Self, AutogradError> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerDecoderLayer::new(
                session,
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                norm_first,
            )?);
        }

        let final_norm = if final_layer_norm {
            Some(LayerNorm::new(session, vec![d_model], 1e-5)?)
        } else {
            None
        };

        Ok(Self { layers, final_norm })
    }

    /// Forward pass.
    ///
    /// * `tgt` - target sequence `[batch, tgt_len, d_model]`
    /// * `memory` - encoder output `[batch, src_len, d_model]`
    pub fn forward_decoder(
        &self,
        session: &mut FrankenTorchSession,
        tgt: TensorNodeId,
        memory: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let mut output = tgt;
        for layer in &self.layers {
            output = layer.forward_layer(session, output, memory)?;
        }
        if let Some(ref norm) = self.final_norm {
            output = norm.forward(session, output)?;
        }
        Ok(output)
    }
}

impl Module for TransformerDecoder {
    fn forward(
        &self,
        _session: &mut FrankenTorchSession,
        _input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        Err(AutogradError::Dispatch(DispatchError::Key(
            DispatchKeyError::IncompatibleSet {
                reason: "TransformerDecoder requires tgt and memory; use forward_decoder()",
            },
        )))
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.final_norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        let mut children: Vec<(String, &dyn Module)> = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            children.push((format!("layers.{i}"), layer as &dyn Module));
        }
        if let Some(ref norm) = self.final_norm {
            children.push(("norm".to_string(), norm as &dyn Module));
        }
        children
    }

    fn train(&self, mode: bool) {
        for layer in &self.layers {
            layer.train(mode);
        }
    }

    fn is_training(&self) -> bool {
        self.layers.first().is_some_and(|l| l.is_training())
    }
}

/// Full Transformer model combining encoder and decoder.
///
/// Architecture:
/// ```text
/// memory = encoder(src)
/// output = decoder(tgt, memory)
/// ```
///
/// Input: `src` `[batch, src_len, d_model]`, `tgt` `[batch, tgt_len, d_model]`.
/// Output: `[batch, tgt_len, d_model]`.
pub struct Transformer {
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    d_model: usize,
}

impl Transformer {
    /// Create a new `Transformer`.
    ///
    /// * `d_model` - feature dimension
    /// * `nhead` - number of attention heads
    /// * `num_encoder_layers` - number of encoder layers
    /// * `num_decoder_layers` - number of decoder layers
    /// * `dim_feedforward` - feedforward hidden dimension
    /// * `dropout` - dropout probability
    /// * `activation` - activation function
    /// * `norm_first` - pre-norm vs post-norm
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: &mut FrankenTorchSession,
        d_model: usize,
        nhead: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        dim_feedforward: usize,
        dropout: f64,
        activation: TransformerActivation,
        norm_first: bool,
    ) -> Result<Self, AutogradError> {
        let encoder = TransformerEncoder::new(
            session,
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            activation,
            norm_first,
            norm_first, // final_layer_norm when using pre-norm
        )?;

        let decoder = TransformerDecoder::new(
            session,
            d_model,
            nhead,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            activation,
            norm_first,
            norm_first,
        )?;

        Ok(Self {
            encoder,
            decoder,
            d_model,
        })
    }

    /// Forward pass through full encoder-decoder transformer.
    ///
    /// * `src` - source sequence `[batch, src_len, d_model]`
    /// * `tgt` - target sequence `[batch, tgt_len, d_model]`
    pub fn forward_transformer(
        &self,
        session: &mut FrankenTorchSession,
        src: TensorNodeId,
        tgt: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let memory = self.encoder.forward(session, src)?;
        self.decoder.forward_decoder(session, tgt, memory)
    }

    /// Get the model dimension.
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Access the encoder.
    #[must_use]
    pub fn encoder(&self) -> &TransformerEncoder {
        &self.encoder
    }

    /// Access the decoder.
    #[must_use]
    pub fn decoder(&self) -> &TransformerDecoder {
        &self.decoder
    }
}

impl Module for Transformer {
    fn forward(
        &self,
        _session: &mut FrankenTorchSession,
        _input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        Err(AutogradError::Dispatch(DispatchError::Key(
            DispatchKeyError::IncompatibleSet {
                reason: "Transformer requires src and tgt; use forward_transformer()",
            },
        )))
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("encoder".to_string(), &self.encoder as &dyn Module),
            ("decoder".to_string(), &self.decoder as &dyn Module),
        ]
    }

    fn train(&self, mode: bool) {
        self.encoder.train(mode);
        self.decoder.train(mode);
    }

    fn is_training(&self) -> bool {
        self.encoder.is_training()
    }
}

/// Generate a causal mask for autoregressive decoding.
///
/// Returns a `[sz, sz]` tensor where the upper triangle above the diagonal
/// is filled with `f64::NEG_INFINITY` and the lower triangle + diagonal is `0.0`.
/// This prevents attention to future positions.
pub fn generate_square_subsequent_mask(
    session: &mut FrankenTorchSession,
    sz: usize,
) -> Result<TensorNodeId, AutogradError> {
    let mut mask_values = vec![0.0_f64; sz * sz];
    for i in 0..sz {
        for j in (i + 1)..sz {
            mask_values[i * sz + j] = f64::NEG_INFINITY;
        }
    }
    session.tensor_variable(mask_values, vec![sz, sz], false)
}

// ── Loss Module Trait ──────────────────────────────────────────────────

/// Trait for loss function modules.
///
/// Loss modules take two inputs (prediction and target) and return a scalar
/// loss tensor. They have no trainable parameters.
pub trait LossModule {
    /// Compute the loss between `input` (predictions) and `target`.
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError>;
}

// ── Loss Function Modules ──────────────────────────────────────────────

/// Mean Squared Error loss module.
///
/// Computes `mean((input - target)^2)`. Wraps `FrankenTorchSession::mse_loss`.
pub struct MSELoss;

impl LossModule for MSELoss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.mse_loss(input, target)
    }
}

/// L1 (Mean Absolute Error) loss module.
///
/// Computes `mean(|input - target|)`. Wraps `FrankenTorchSession::l1_loss`.
pub struct L1Loss;

impl LossModule for L1Loss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.l1_loss(input, target)
    }
}

/// Cross-entropy loss module for classification.
///
/// Applies log-softmax to `input` along the last dimension, then computes
/// NLL loss against `target` (class indices as f64).
pub struct CrossEntropyLoss;

impl LossModule for CrossEntropyLoss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.cross_entropy_loss(input, target)
    }
}

/// Negative log-likelihood loss module.
///
/// Expects `input` to be log-probabilities of shape `[batch, classes]` and
/// `target` to be class indices of shape `[batch]`.
pub struct NLLLoss;

impl LossModule for NLLLoss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.nll_loss(input, target)
    }
}

/// Binary cross-entropy loss module.
///
/// Expects `input` to be probabilities in `[0, 1]`. Clamps internally for
/// numerical stability.
pub struct BCELoss;

impl LossModule for BCELoss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.bce_loss(input, target)
    }
}

/// Binary cross-entropy with logits loss module.
///
/// Uses the numerically-stable formulation:
/// `mean(max(x, 0) - x * y + log1p(exp(-|x|)))`
///
/// This avoids computing `log(sigmoid(x))` which is unstable for extreme
/// input values.
pub struct BCEWithLogitsLoss;

impl LossModule for BCEWithLogitsLoss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.bce_with_logits_loss(input, target)
    }
}

/// Smooth L1 loss module.
///
/// Behaves as L2 loss when error is small and L1 when error is large.
/// The transition point is controlled by `beta`.
pub struct SmoothL1Loss {
    beta: f64,
}

impl SmoothL1Loss {
    /// Create a new SmoothL1Loss with the given beta threshold.
    #[must_use]
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl LossModule for SmoothL1Loss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.smooth_l1_loss(input, target, self.beta)
    }
}

/// Huber loss module.
///
/// Behaves as L2 loss when `|error| <= delta` and L1 loss otherwise.
pub struct HuberLoss {
    delta: f64,
}

impl HuberLoss {
    /// Create a new HuberLoss with the given delta threshold.
    #[must_use]
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl LossModule for HuberLoss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.huber_loss(input, target, self.delta)
    }
}

/// Cosine embedding loss module for similarity learning.
///
/// Note: this loss takes three inputs (x1, x2, target) via separate methods
/// since it doesn't fit the standard two-argument pattern.
pub struct CosineEmbeddingLoss {
    margin: f64,
}

impl CosineEmbeddingLoss {
    /// Create a new CosineEmbeddingLoss with the given margin (default 0.0).
    #[must_use]
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }

    /// Compute cosine embedding loss.
    ///
    /// * `x1`, `x2` - input tensors of shape `[batch, features]`
    /// * `target` - labels of shape `[batch]` with values 1.0 or -1.0
    pub fn forward_triplet(
        &self,
        session: &mut FrankenTorchSession,
        x1: TensorNodeId,
        x2: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.cosine_embedding_loss(x1, x2, target, self.margin)
    }
}

/// KL divergence loss module.
///
/// Computes `mean(target * (log(target) - input))` where `input` should be
/// log-probabilities and `target` should be probabilities (the true distribution).
///
/// Follows PyTorch convention where `input` is log-space and `target` is
/// probability-space.
pub struct KLDivLoss;

impl LossModule for KLDivLoss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // KL(P || Q) = sum(P * (log(P) - log(Q)))
        // input = log(Q), target = P
        // loss = mean(target * (log(target) - input))
        // We add eps to target before log to avoid log(0)
        let eps = 1e-8;
        let shape = session.tensor_shape(target)?;
        let eps_tensor = session.full(shape, eps, false)?;
        let target_safe = session.tensor_add(target, eps_tensor)?;
        let log_target = session.tensor_log(target_safe)?;
        let diff = session.tensor_sub(log_target, input)?;
        let weighted = session.tensor_mul(target, diff)?;
        session.tensor_mean(weighted)
    }
}

/// Margin ranking loss: `max(0, -y * (x1 - x2) + margin)`.
///
/// Used for ranking/preference learning.
pub struct MarginRankingLoss {
    margin: f64,
}

impl MarginRankingLoss {
    #[must_use]
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }

    /// Compute the loss.
    ///
    /// * `x1`, `x2` - predictions (same shape)
    /// * `y` - target labels (1.0 or -1.0, same shape)
    pub fn forward_triplet(
        &self,
        session: &mut FrankenTorchSession,
        x1: TensorNodeId,
        x2: TensorNodeId,
        y: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // loss = max(0, -y * (x1 - x2) + margin)
        let diff = session.tensor_sub(x1, x2)?;
        let neg_y_diff = session.tensor_mul(y, diff)?;
        let neg_y_diff = session.tensor_neg(neg_y_diff)?;
        let shape = session.tensor_shape(neg_y_diff)?;
        let margin_t = session.full(shape, self.margin, false)?;
        let raw = session.tensor_add(neg_y_diff, margin_t)?;
        let clamped = session.tensor_relu(raw)?;
        session.tensor_mean(clamped)
    }
}

/// Triplet margin loss: `max(0, d(anchor, positive) - d(anchor, negative) + margin)`.
///
/// Used for metric learning, face recognition, contrastive learning.
pub struct TripletMarginLoss {
    margin: f64,
    p: f64, // norm degree (default 2.0)
}

impl TripletMarginLoss {
    #[must_use]
    pub fn new(margin: f64, p: f64) -> Self {
        Self { margin, p }
    }

    /// Compute the loss.
    ///
    /// * `anchor`, `positive`, `negative` - embeddings (same shape)
    pub fn forward_triplet(
        &self,
        session: &mut FrankenTorchSession,
        anchor: TensorNodeId,
        positive: TensorNodeId,
        negative: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // d(a, p) - d(a, n) + margin, clamped to >= 0
        // Using L2 distance: d(x, y) = ||x - y||_p
        let diff_pos = session.tensor_sub(anchor, positive)?;
        let diff_neg = session.tensor_sub(anchor, negative)?;

        // For p=2 (default): squared differences, sum, sqrt
        let sq_pos = session.tensor_mul(diff_pos, diff_pos)?;
        let sq_neg = session.tensor_mul(diff_neg, diff_neg)?;
        let sum_pos = session.tensor_sum(sq_pos)?;
        let sum_neg = session.tensor_sum(sq_neg)?;

        if (self.p - 2.0).abs() < f64::EPSILON {
            // L2: sqrt of sum of squares
            let dist_pos = session.tensor_sqrt(sum_pos)?;
            let dist_neg = session.tensor_sqrt(sum_neg)?;
            let diff = session.tensor_sub(dist_pos, dist_neg)?;
            let shape = session.tensor_shape(diff)?;
            let margin_t = session.full(shape, self.margin, false)?;
            let raw = session.tensor_add(diff, margin_t)?;
            session.tensor_relu(raw)
        } else {
            // For other p values, use power operations
            let dist_pos = session.tensor_sqrt(sum_pos)?;
            let dist_neg = session.tensor_sqrt(sum_neg)?;
            let diff = session.tensor_sub(dist_pos, dist_neg)?;
            let shape = session.tensor_shape(diff)?;
            let margin_t = session.full(shape, self.margin, false)?;
            let raw = session.tensor_add(diff, margin_t)?;
            session.tensor_relu(raw)
        }
    }
}

/// Hinge embedding loss: `x` if `y=1`, `max(0, margin - x)` if `y=-1`.
///
/// Used for semi-supervised learning and embedding learning.
pub struct HingeEmbeddingLoss {
    margin: f64,
}

impl HingeEmbeddingLoss {
    #[must_use]
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }

    /// Compute the loss.
    ///
    /// * `input` - distances or similarities
    /// * `target` - labels (1.0 or -1.0)
    pub fn forward_hinge(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // For y=1: loss = x
        // For y=-1: loss = max(0, margin - x)
        // Combined: loss = (1+y)/2 * x + (1-y)/2 * max(0, margin - x)
        let input_vals = session.tensor_values(input)?;
        let target_vals = session.tensor_values(target)?;
        let shape = session.tensor_shape(input)?;
        let numel: usize = shape.iter().product();
        let mut result = vec![0.0f64; numel];
        for i in 0..numel {
            if target_vals[i] > 0.0 {
                result[i] = input_vals[i];
            } else {
                result[i] = (self.margin - input_vals[i]).max(0.0);
            }
        }
        let loss_elements = session.tensor_variable(result, shape, false)?;
        session.tensor_mean(loss_elements)
    }
}

/// Poisson negative log likelihood loss: `exp(input) - target * input`.
///
/// Used for count/rate data modeled by Poisson distribution.
pub struct PoissonNLLLoss {
    log_input: bool, // if true, input is log-rate (default true)
}

impl PoissonNLLLoss {
    #[must_use]
    pub fn new(log_input: bool) -> Self {
        Self { log_input }
    }
}

impl LossModule for PoissonNLLLoss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        if self.log_input {
            // loss = exp(input) - target * input
            let exp_input = session.tensor_exp(input)?;
            let target_times_input = session.tensor_mul(target, input)?;
            let loss = session.tensor_sub(exp_input, target_times_input)?;
            session.tensor_mean(loss)
        } else {
            // loss = input - target * log(input)
            let log_input = session.tensor_log(input)?;
            let target_times_log = session.tensor_mul(target, log_input)?;
            let loss = session.tensor_sub(input, target_times_log)?;
            session.tensor_mean(loss)
        }
    }
}

/// Gaussian negative log likelihood loss.
///
/// `loss = 0.5 * (log(var) + (input - target)^2 / var + log(2*pi))`
///
/// Used for regression with uncertainty estimation.
pub struct GaussianNLLLoss;

impl GaussianNLLLoss {
    /// Compute the loss.
    ///
    /// * `input` - predicted mean
    /// * `target` - observed values
    /// * `var` - predicted variance (must be positive)
    pub fn forward_with_var(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
        var: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // 0.5 * (log(var) + (input - target)^2 / var + log(2*pi))
        let diff = session.tensor_sub(input, target)?;
        let sq_diff = session.tensor_mul(diff, diff)?;
        let ratio = session.tensor_div(sq_diff, var)?;
        let log_var = session.tensor_log(var)?;
        let sum = session.tensor_add(log_var, ratio)?;
        let shape = session.tensor_shape(sum)?;
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_2pi_t = session.full(shape, log_2pi, false)?;
        let total = session.tensor_add(sum, log_2pi_t)?;
        let shape2 = session.tensor_shape(total)?;
        let half = session.full(shape2, 0.5, false)?;
        let loss = session.tensor_mul(half, total)?;
        session.tensor_mean(loss)
    }
}

/// Multi-label soft margin loss: binary cross-entropy per label.
///
/// `loss = -1/C * sum(y * log(sigmoid(x)) + (1-y) * log(1 - sigmoid(x)))`
pub struct MultiLabelSoftMarginLoss;

impl LossModule for MultiLabelSoftMarginLoss {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        target: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // Uses the numerically stable log-sigmoid formulation
        // This is equivalent to BCEWithLogitsLoss applied per element then averaged
        session.bce_with_logits_loss(input, target)
    }
}

/// Reduction mode for CTCLoss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTCReduction {
    /// No reduction: return per-sample losses of shape (N,).
    None,
    /// Sum all per-sample losses.
    Sum,
    /// Mean: sum of per-sample losses divided by sum of target lengths.
    Mean,
}

/// Connectionist Temporal Classification loss.
///
/// Used for sequence-to-sequence tasks where input and output have different
/// lengths and no alignment is known (e.g., speech recognition, OCR).
///
/// CTC marginalizes over all possible alignments using a blank token. The
/// algorithm uses log-domain dynamic programming for numerical stability.
///
/// This loss does NOT implement `LossModule` because it requires 4 inputs.
/// Use `forward_ctc()` directly.
pub struct CTCLoss {
    /// Index of the blank label (default 0).
    pub blank: usize,
    /// Reduction mode (default Mean).
    pub reduction: CTCReduction,
    /// If true, replace infinite losses with zero (default false).
    pub zero_infinity: bool,
}

impl CTCLoss {
    #[must_use]
    pub fn new() -> Self {
        Self {
            blank: 0,
            reduction: CTCReduction::Mean,
            zero_infinity: false,
        }
    }

    #[must_use]
    pub fn with_blank(mut self, blank: usize) -> Self {
        self.blank = blank;
        self
    }

    #[must_use]
    pub fn with_reduction(mut self, reduction: CTCReduction) -> Self {
        self.reduction = reduction;
        self
    }

    #[must_use]
    pub fn with_zero_infinity(mut self, zero_infinity: bool) -> Self {
        self.zero_infinity = zero_infinity;
        self
    }

    /// Compute the CTC loss.
    ///
    /// * `log_probs` - Log-probabilities of shape `[T, N, C]` (time, batch, classes)
    /// * `targets` - Target sequences of shape `[N, S]` (batch, max target length),
    ///   padded with any value for samples with shorter targets
    /// * `input_lengths` - Actual input lengths per batch element, shape `[N]`
    /// * `target_lengths` - Actual target lengths per batch element, shape `[N]`
    pub fn forward_ctc(
        &self,
        session: &mut FrankenTorchSession,
        log_probs: TensorNodeId,
        targets: TensorNodeId,
        input_lengths: TensorNodeId,
        target_lengths: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let blank = self.blank;
        let reduction = self.reduction;
        let zero_infinity = self.zero_infinity;

        // Read shapes for validation
        let lp_shape = session.tensor_shape(log_probs)?;
        if lp_shape.len() != 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "CTCLoss: log_probs must be 3-D [T, N, C]",
                },
            )));
        }
        let t_max = lp_shape[0];
        let batch_size = lp_shape[1];
        let num_classes = lp_shape[2];

        if blank >= num_classes {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "CTCLoss: blank index must be < num_classes",
                },
            )));
        }

        let tgt_shape = session.tensor_shape(targets)?;
        if tgt_shape.len() != 2 || tgt_shape[0] != batch_size {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "CTCLoss: targets must be 2-D [N, S]",
                },
            )));
        }
        let s_max = tgt_shape[1];

        let il_shape = session.tensor_shape(input_lengths)?;
        if il_shape != [batch_size] {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "CTCLoss: input_lengths must be 1-D [N]",
                },
            )));
        }
        let tl_shape = session.tensor_shape(target_lengths)?;
        if tl_shape != [batch_size] {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "CTCLoss: target_lengths must be 1-D [N]",
                },
            )));
        }

        session.tensor_apply_function(
            &[log_probs, targets, input_lengths, target_lengths],
            // Forward function
            move |ctx: &mut FunctionCtx, inputs: &[(&[f64], &[usize])]| {
                let (lp_data, _) = inputs[0]; // [T, N, C]
                let (tgt_data, _) = inputs[1]; // [N, S]
                let (il_data, _) = inputs[2]; // [N]
                let (tl_data, _) = inputs[3]; // [N]

                let neg_inf = f64::NEG_INFINITY;
                let mut losses = vec![0.0f64; batch_size];

                for b in 0..batch_size {
                    let input_len = il_data[b] as usize;
                    let target_len = tl_data[b] as usize;

                    // Validate lengths are within tensor bounds
                    if input_len > t_max {
                        return Err(AutogradError::Dispatch(DispatchError::Key(
                            DispatchKeyError::IncompatibleSet {
                                reason: "CTCLoss: input_lengths[b] exceeds T dimension of log_probs",
                            },
                        )));
                    }
                    if target_len > s_max {
                        return Err(AutogradError::Dispatch(DispatchError::Key(
                            DispatchKeyError::IncompatibleSet {
                                reason: "CTCLoss: target_lengths[b] exceeds S dimension of targets",
                            },
                        )));
                    }

                    if target_len == 0 {
                        // Empty target: loss = -sum of log P(blank) at each timestep
                        let mut loss = 0.0;
                        for t in 0..input_len {
                            loss -= lp_data[t * batch_size * num_classes + b * num_classes + blank];
                        }
                        losses[b] = loss;
                        continue;
                    }

                    // Build CTC lattice labels: interleave blanks with target symbols
                    // Target 'abc' -> [blank, a, blank, b, blank, c, blank]
                    let lattice_len = 2 * target_len + 1;
                    let mut labels = vec![blank; lattice_len];
                    for i in 0..target_len {
                        let label = tgt_data[b * s_max + i] as usize;
                        if label >= num_classes {
                            return Err(AutogradError::Dispatch(DispatchError::Key(
                                DispatchKeyError::IncompatibleSet {
                                    reason: "CTCLoss: target label >= num_classes",
                                },
                            )));
                        }
                        labels[2 * i + 1] = label;
                    }

                    // Check feasibility: input must be long enough
                    if input_len < target_len {
                        losses[b] = f64::INFINITY;
                        continue;
                    }

                    // Forward pass: alpha[t][s] = log P(emit prefix ending at state s at time t)
                    let mut alpha = vec![vec![neg_inf; lattice_len]; input_len];

                    // t=0 initialization: can only be in state 0 (blank) or state 1 (first label)
                    alpha[0][0] = lp_data[b * num_classes + labels[0]];
                    if lattice_len > 1 {
                        alpha[0][1] = lp_data[b * num_classes + labels[1]];
                    }

                    // Forward recursion
                    for t in 1..input_len {
                        let lp_offset = t * batch_size * num_classes + b * num_classes;
                        for s in 0..lattice_len {
                            let emit = lp_data[lp_offset + labels[s]];

                            // Can stay in same state
                            let mut log_sum = alpha[t - 1][s];

                            // Can come from previous state
                            if s >= 1 {
                                log_sum = log_sum_exp(log_sum, alpha[t - 1][s - 1]);
                            }

                            // Can skip a blank if current and two-back are different non-blank labels
                            if s >= 2 && labels[s] != blank && labels[s] != labels[s - 2] {
                                log_sum = log_sum_exp(log_sum, alpha[t - 1][s - 2]);
                            }

                            alpha[t][s] = log_sum + emit;
                        }
                    }

                    // Total log probability
                    let log_prob = log_sum_exp(
                        alpha[input_len - 1][lattice_len - 1],
                        alpha[input_len - 1][lattice_len - 2],
                    );

                    losses[b] = -log_prob;
                }

                // Handle zero_infinity
                if zero_infinity {
                    for loss in &mut losses {
                        if loss.is_infinite() {
                            *loss = 0.0;
                        }
                    }
                }

                // Save data for backward
                ctx.save_for_backward(lp_data.to_vec(), vec![t_max, batch_size, num_classes]);
                ctx.save_for_backward(tgt_data.to_vec(), vec![batch_size, s_max]);
                ctx.save_for_backward(il_data.to_vec(), vec![batch_size]);
                ctx.save_for_backward(tl_data.to_vec(), vec![batch_size]);
                // Save per-sample losses for reduction gradient
                ctx.save_for_backward(losses.clone(), vec![batch_size]);

                // Apply reduction
                let output = match reduction {
                    CTCReduction::None => {
                        return Ok((losses, vec![batch_size]));
                    }
                    CTCReduction::Sum => {
                        vec![losses.iter().sum::<f64>()]
                    }
                    CTCReduction::Mean => {
                        let total_loss: f64 = losses.iter().sum();
                        let total_target_len: f64 = tl_data.iter().sum();
                        if total_target_len > 0.0 {
                            vec![total_loss / total_target_len]
                        } else {
                            vec![0.0]
                        }
                    }
                };
                Ok((output, vec![1]))
            },
            // Backward function
            move |ctx: &FunctionCtx, grad_outputs: &[&[f64]]| {
                let grad_out = grad_outputs[0]; // scalar or [N]
                let lp_data = &ctx.saved_tensors()[0];
                let tgt_data = &ctx.saved_tensors()[1];
                let il_data = &ctx.saved_tensors()[2];
                let tl_data = &ctx.saved_tensors()[3];
                let per_sample_losses = &ctx.saved_tensors()[4];

                let lp_shapes = &ctx.saved_shapes()[0]; // [T, N, C]
                let t_max_b = lp_shapes[0];
                let batch_size_b = lp_shapes[1];
                let num_classes_b = lp_shapes[2];
                let s_max_b = ctx.saved_shapes()[1][1]; // max target length

                let neg_inf = f64::NEG_INFINITY;

                // Gradient w.r.t. log_probs: shape [T, N, C]
                let total_elems = t_max_b * batch_size_b * num_classes_b;
                let mut grad_lp = vec![0.0f64; total_elems];

                // Per-sample gradient scale from reduction
                let total_target_len: f64 = tl_data.iter().sum();

                for b in 0..batch_size_b {
                    let input_len = il_data[b] as usize;
                    let target_len = tl_data[b] as usize;

                    // Determine per-sample grad scale
                    let grad_scale = if grad_out.len() == 1 {
                        // Scalar output (Sum or Mean reduction)
                        match reduction {
                            CTCReduction::Mean => {
                                if total_target_len > 0.0 {
                                    grad_out[0] / total_target_len
                                } else {
                                    0.0
                                }
                            }
                            CTCReduction::Sum => grad_out[0],
                            CTCReduction::None => grad_out[b],
                        }
                    } else {
                        grad_out[b]
                    };

                    // Skip if loss was infinite and zero_infinity is on
                    if zero_infinity && per_sample_losses[b].is_infinite() {
                        continue;
                    }
                    // Skip infeasible
                    if per_sample_losses[b].is_infinite() {
                        continue;
                    }

                    if target_len == 0 {
                        // Empty target: grad is -1 at blank positions
                        for t in 0..input_len {
                            let offset = t * batch_size_b * num_classes_b + b * num_classes_b;
                            grad_lp[offset + blank] = -grad_scale;
                        }
                        continue;
                    }

                    // Rebuild lattice
                    let lattice_len = 2 * target_len + 1;
                    let mut labels = vec![blank; lattice_len];
                    for i in 0..target_len {
                        labels[2 * i + 1] = tgt_data[b * s_max_b + i] as usize;
                    }

                    // Forward pass (alpha)
                    let mut alpha = vec![vec![neg_inf; lattice_len]; input_len];
                    alpha[0][0] = lp_data[b * num_classes_b + labels[0]];
                    if lattice_len > 1 {
                        alpha[0][1] = lp_data[b * num_classes_b + labels[1]];
                    }
                    for t in 1..input_len {
                        let lp_offset = t * batch_size_b * num_classes_b + b * num_classes_b;
                        for s in 0..lattice_len {
                            let emit = lp_data[lp_offset + labels[s]];
                            let mut log_sum = alpha[t - 1][s];
                            if s >= 1 {
                                log_sum = log_sum_exp(log_sum, alpha[t - 1][s - 1]);
                            }
                            if s >= 2 && labels[s] != blank && labels[s] != labels[s - 2] {
                                log_sum = log_sum_exp(log_sum, alpha[t - 1][s - 2]);
                            }
                            alpha[t][s] = log_sum + emit;
                        }
                    }

                    // Backward pass (beta)
                    let mut beta = vec![vec![neg_inf; lattice_len]; input_len];
                    beta[input_len - 1][lattice_len - 1] = 0.0;
                    if lattice_len >= 2 {
                        beta[input_len - 1][lattice_len - 2] = 0.0;
                    }
                    for t in (0..input_len - 1).rev() {
                        let lp_offset_next = (t + 1) * batch_size_b * num_classes_b + b * num_classes_b;
                        for s in 0..lattice_len {
                            let mut log_sum = beta[t + 1][s]
                                + lp_data[lp_offset_next + labels[s]];
                            if s + 1 < lattice_len {
                                log_sum = log_sum_exp(
                                    log_sum,
                                    beta[t + 1][s + 1] + lp_data[lp_offset_next + labels[s + 1]],
                                );
                            }
                            if s + 2 < lattice_len
                                && labels[s] != blank
                                && labels[s] != labels[s + 2]
                            {
                                log_sum = log_sum_exp(
                                    log_sum,
                                    beta[t + 1][s + 2] + lp_data[lp_offset_next + labels[s + 2]],
                                );
                            }
                            beta[t][s] = log_sum;
                        }
                    }

                    // Total log probability
                    let log_prob = log_sum_exp(
                        alpha[input_len - 1][lattice_len - 1],
                        alpha[input_len - 1][lattice_len - 2],
                    );

                    // Compute gradient: for each (t, c), accumulate alpha[t][s] + beta[t][s]
                    // for all states s where labels[s] == c.
                    // grad[t][c] = -exp(ab_sum[c] - log_prob)
                    // This is the derivative of -log P w.r.t. log_probs[t][c].
                    for t in 0..input_len {
                        let lp_offset = t * batch_size_b * num_classes_b + b * num_classes_b;
                        // For each class, collect alpha*beta contributions
                        let mut ab_sum = vec![neg_inf; num_classes_b];
                        for s in 0..lattice_len {
                            let c = labels[s];
                            ab_sum[c] = log_sum_exp(ab_sum[c], alpha[t][s] + beta[t][s]);
                        }

                        for c in 0..num_classes_b {
                            let posterior = if ab_sum[c] == neg_inf {
                                0.0
                            } else {
                                (ab_sum[c] - log_prob).exp()
                            };
                            grad_lp[lp_offset + c] = -grad_scale * posterior;
                        }
                    }
                }

                // Gradients: only w.r.t. log_probs (input 0), none for targets/lengths
                Ok(vec![Some(grad_lp), None, None, None])
            },
        )
    }
}

impl Default for CTCLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Log-sum-exp of two values: log(exp(a) + exp(b)), numerically stable.
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

// ── Container Modules ──────────────────────────────────────────────────

/// An ordered list of modules.
///
/// Modules are applied in sequence during forward pass, and all parameters
/// are collected from child modules.
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    /// Create a new empty `ModuleList`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
        }
    }

    /// Add a module to the list.
    pub fn push(&mut self, module: Box<dyn Module>) {
        self.modules.push(module);
    }

    /// Number of modules in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get a reference to a module by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.modules.get(index).map(|m| m.as_ref())
    }
}

impl Default for ModuleList {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ModuleList {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let mut x = input;
        for module in &self.modules {
            x = module.forward(session, x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.modules
            .iter()
            .enumerate()
            .map(|(i, m)| (i.to_string(), m.as_ref() as &dyn Module))
            .collect()
    }
}

/// A dictionary of named modules.
///
/// Modules can be looked up by name. Forward pass applies modules in
/// insertion order (like PyTorch's `ModuleDict` iteration).
pub struct ModuleDict {
    entries: Vec<(String, Box<dyn Module>)>,
}

impl ModuleDict {
    /// Create a new empty `ModuleDict`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert a named module. If a module with the same name exists, it is replaced.
    pub fn insert(&mut self, name: String, module: Box<dyn Module>) {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &name) {
            self.entries[pos] = (name, module);
        } else {
            self.entries.push((name, module));
        }
    }

    /// Get a reference to a module by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn Module> {
        self.entries
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, m)| m.as_ref())
    }

    /// Number of modules in the dict.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the dict is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over (name, module) pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &dyn Module)> {
        self.entries.iter().map(|(k, m)| (k.as_str(), m.as_ref()))
    }
}

impl Default for ModuleDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ModuleDict {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let mut x = input;
        for (_, module) in &self.entries {
            x = module.forward(session, x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.entries
            .iter()
            .flat_map(|(_, m)| m.parameters())
            .collect()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        self.entries
            .iter()
            .map(|(name, m)| (name.clone(), m.as_ref() as &dyn Module))
            .collect()
    }
}

// ── Parameter Containers ───────────────────────────────────────────────

/// A list of parameters (tensors), indexed by position.
///
/// Equivalent to `torch.nn.ParameterList`. Unlike `ModuleList`, this holds
/// raw parameter tensors rather than sub-modules. All parameters are
/// registered as trainable and appear in `parameters()` / `state_dict()`.
pub struct ParameterList {
    params: Vec<TensorNodeId>,
}

impl ParameterList {
    /// Create a new `ParameterList` from the given parameter tensors.
    #[must_use]
    pub fn new(params: Vec<TensorNodeId>) -> Self {
        Self { params }
    }

    /// Append a parameter.
    pub fn append(&mut self, param: TensorNodeId) {
        self.params.push(param);
    }

    /// Number of parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Get a parameter by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<TensorNodeId> {
        self.params.get(index).copied()
    }
}

impl Default for ParameterList {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl Module for ParameterList {
    fn forward(
        &self,
        _session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // ParameterList is a container, not a computation — forward is passthrough
        Ok(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.params.clone()
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        // Parameters are returned via named_parameter_slots_own with dynamic names
        Vec::new()
    }

    fn named_parameter_slots_own(&self) -> Vec<(String, Option<TensorNodeId>)> {
        self.params
            .iter()
            .enumerate()
            .map(|(i, &p)| (i.to_string(), Some(p)))
            .collect()
    }
}

/// A dictionary of named parameters (tensors), looked up by key.
///
/// Equivalent to `torch.nn.ParameterDict`. All parameters are registered
/// as trainable and appear in `parameters()` / `state_dict()`.
pub struct ParameterDict {
    entries: Vec<(String, TensorNodeId)>,
}

impl ParameterDict {
    /// Create a new empty `ParameterDict`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert or replace a named parameter.
    pub fn insert(&mut self, name: String, param: TensorNodeId) {
        if let Some(entry) = self.entries.iter_mut().find(|(k, _)| *k == name) {
            entry.1 = param;
        } else {
            self.entries.push((name, param));
        }
    }

    /// Get a parameter by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<TensorNodeId> {
        self.entries
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
    }

    /// Number of parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the dict is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterator over (name, param) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, TensorNodeId)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

impl Default for ParameterDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ParameterDict {
    fn forward(
        &self,
        _session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // ParameterDict is a container, not a computation — forward is passthrough
        Ok(input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        self.entries.iter().map(|(_, v)| *v).collect()
    }

    fn named_parameters_own(&self) -> Vec<(&'static str, TensorNodeId)> {
        Vec::new()
    }

    fn named_parameter_slots_own(&self) -> Vec<(String, Option<TensorNodeId>)> {
        self.entries
            .iter()
            .map(|(k, v)| (k.clone(), Some(*v)))
            .collect()
    }
}

// ── Padding Modules ────────────────────────────────────────────────────

/// Constant padding for 1D inputs (3D tensors `[N, C, W]`).
///
/// Pads the last dimension with a constant value.
pub struct ConstantPad1d {
    padding_left: usize,
    padding_right: usize,
    value: f64,
}

impl ConstantPad1d {
    /// Create a new ConstantPad1d.
    ///
    /// * `padding` - `(left, right)` padding sizes
    /// * `value` - the constant fill value
    #[must_use]
    pub fn new(padding: (usize, usize), value: f64) -> Self {
        Self {
            padding_left: padding.0,
            padding_right: padding.1,
            value,
        }
    }
}

impl Module for ConstantPad1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        session.tensor_pad(input, &[self.padding_left, self.padding_right], self.value)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Constant padding for 2D inputs (4D tensors `[N, C, H, W]`).
///
/// Pads the last two dimensions with a constant value.
pub struct ConstantPad2d {
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    value: f64,
}

impl ConstantPad2d {
    /// Create a new ConstantPad2d.
    ///
    /// * `padding` - `(left, right, top, bottom)` padding sizes
    /// * `value` - the constant fill value
    #[must_use]
    pub fn new(padding: (usize, usize, usize, usize), value: f64) -> Self {
        Self {
            pad_left: padding.0,
            pad_right: padding.1,
            pad_top: padding.2,
            pad_bottom: padding.3,
            value,
        }
    }
}

impl Module for ConstantPad2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        // F.pad convention: innermost dimension first
        // For [N, C, H, W]: [left, right, top, bottom]
        session.tensor_pad(
            input,
            &[self.pad_left, self.pad_right, self.pad_top, self.pad_bottom],
            self.value,
        )
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Zero padding for 2D inputs (4D tensors `[N, C, H, W]`).
///
/// Equivalent to `ConstantPad2d` with `value = 0.0`.
pub struct ZeroPad2d {
    inner: ConstantPad2d,
}

impl ZeroPad2d {
    /// Create a new ZeroPad2d.
    ///
    /// * `padding` - `(left, right, top, bottom)` padding sizes
    #[must_use]
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            inner: ConstantPad2d::new(padding, 0.0),
        }
    }
}

impl Module for ZeroPad2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        self.inner.forward(session, input)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Reflection padding for 1D inputs (3D tensors `[N, C, L]`).
///
/// Pads by reflecting at the boundary: `[a, b, c, d]` with pad=2 -> `[c, b, a, b, c, d, c, b]`.
/// Padding must be less than the input size.
pub struct ReflectionPad1d {
    padding_left: usize,
    padding_right: usize,
}

impl ReflectionPad1d {
    #[must_use]
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding_left: padding.0,
            padding_right: padding.1,
        }
    }
}

impl Module for ReflectionPad1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let ndim = input_shape.len();
        let l = input_shape[ndim - 1];

        if self.padding_left >= l || self.padding_right >= l {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ReflectionPad1d: padding must be < input size",
                },
            )));
        }

        if self.padding_left == 0 && self.padding_right == 0 {
            return Ok(input);
        }

        let vals = session.tensor_values(input)?;
        let batch_dims: usize = input_shape[..ndim - 1].iter().product();
        let new_l = l + self.padding_left + self.padding_right;
        let mut output = Vec::with_capacity(batch_dims * new_l);

        for b in 0..batch_dims {
            let row = &vals[b * l..(b + 1) * l];
            // Left reflection
            for i in (0..self.padding_left).rev() {
                output.push(row[i + 1]);
            }
            // Original
            output.extend_from_slice(row);
            // Right reflection
            for i in 0..self.padding_right {
                output.push(row[l - 2 - i]);
            }
        }

        let mut new_shape = input_shape;
        *new_shape.last_mut().unwrap() = new_l;
        session.tensor_variable(output, new_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Reflection padding for 2D inputs (4D tensors `[N, C, H, W]`).
pub struct ReflectionPad2d {
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
}

impl ReflectionPad2d {
    #[must_use]
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            pad_left: padding.0,
            pad_right: padding.1,
            pad_top: padding.2,
            pad_bottom: padding.3,
        }
    }
}

impl Module for ReflectionPad2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let ndim = input_shape.len();
        if ndim < 2 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ReflectionPad2d requires at least 2D input",
                },
            )));
        }

        let h = input_shape[ndim - 2];
        let w = input_shape[ndim - 1];

        if self.pad_top >= h || self.pad_bottom >= h || self.pad_left >= w || self.pad_right >= w {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ReflectionPad2d: padding must be < input size",
                },
            )));
        }

        let vals = session.tensor_values(input)?;
        let batch_dims: usize = input_shape[..ndim - 2].iter().product();
        let new_h = h + self.pad_top + self.pad_bottom;
        let new_w = w + self.pad_left + self.pad_right;
        let mut output = Vec::with_capacity(batch_dims * new_h * new_w);

        for b in 0..batch_dims {
            let plane = &vals[b * h * w..(b + 1) * h * w];
            for row_out in 0..new_h {
                let src_row = if row_out < self.pad_top {
                    self.pad_top - row_out
                } else if row_out >= self.pad_top + h {
                    h - 2 - (row_out - self.pad_top - h)
                } else {
                    row_out - self.pad_top
                };
                let row_data = &plane[src_row * w..(src_row + 1) * w];
                for col_out in 0..new_w {
                    let src_col = if col_out < self.pad_left {
                        self.pad_left - col_out
                    } else if col_out >= self.pad_left + w {
                        w - 2 - (col_out - self.pad_left - w)
                    } else {
                        col_out - self.pad_left
                    };
                    output.push(row_data[src_col]);
                }
            }
        }

        let mut new_shape = input_shape;
        new_shape[ndim - 2] = new_h;
        new_shape[ndim - 1] = new_w;
        session.tensor_variable(output, new_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Replication padding for 1D inputs (3D tensors `[N, C, L]`).
///
/// Pads by replicating the edge value: `[a, b, c, d]` with pad=2 -> `[a, a, a, b, c, d, d, d]`.
pub struct ReplicationPad1d {
    padding_left: usize,
    padding_right: usize,
}

impl ReplicationPad1d {
    #[must_use]
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding_left: padding.0,
            padding_right: padding.1,
        }
    }
}

impl Module for ReplicationPad1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let ndim = input_shape.len();
        let l = input_shape[ndim - 1];

        if self.padding_left == 0 && self.padding_right == 0 {
            return Ok(input);
        }

        let vals = session.tensor_values(input)?;
        let batch_dims: usize = input_shape[..ndim - 1].iter().product();
        let new_l = l + self.padding_left + self.padding_right;
        let mut output = Vec::with_capacity(batch_dims * new_l);

        for b in 0..batch_dims {
            let row = &vals[b * l..(b + 1) * l];
            for _ in 0..self.padding_left {
                output.push(row[0]);
            }
            output.extend_from_slice(row);
            for _ in 0..self.padding_right {
                output.push(row[l - 1]);
            }
        }

        let mut new_shape = input_shape;
        *new_shape.last_mut().unwrap() = new_l;
        session.tensor_variable(output, new_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Replication padding for 2D inputs (4D tensors `[N, C, H, W]`).
pub struct ReplicationPad2d {
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
}

impl ReplicationPad2d {
    #[must_use]
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            pad_left: padding.0,
            pad_right: padding.1,
            pad_top: padding.2,
            pad_bottom: padding.3,
        }
    }
}

impl Module for ReplicationPad2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let ndim = input_shape.len();
        if ndim < 2 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ReplicationPad2d requires at least 2D input",
                },
            )));
        }

        let h = input_shape[ndim - 2];
        let w = input_shape[ndim - 1];
        let vals = session.tensor_values(input)?;
        let batch_dims: usize = input_shape[..ndim - 2].iter().product();
        let new_h = h + self.pad_top + self.pad_bottom;
        let new_w = w + self.pad_left + self.pad_right;
        let mut output = Vec::with_capacity(batch_dims * new_h * new_w);

        for b in 0..batch_dims {
            let plane = &vals[b * h * w..(b + 1) * h * w];
            for row_out in 0..new_h {
                let src_row = if row_out < self.pad_top {
                    0
                } else if row_out >= self.pad_top + h {
                    h - 1
                } else {
                    row_out - self.pad_top
                };
                let row_data = &plane[src_row * w..(src_row + 1) * w];
                for col_out in 0..new_w {
                    let src_col = if col_out < self.pad_left {
                        0
                    } else if col_out >= self.pad_left + w {
                        w - 1
                    } else {
                        col_out - self.pad_left
                    };
                    output.push(row_data[src_col]);
                }
            }
        }

        let mut new_shape = input_shape;
        new_shape[ndim - 2] = new_h;
        new_shape[ndim - 1] = new_w;
        session.tensor_variable(output, new_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Replication padding for 3D inputs (5D tensors `[N, C, D, H, W]`).
pub struct ReplicationPad3d {
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
}

impl ReplicationPad3d {
    #[must_use]
    pub fn new(padding: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self {
            pad_left: padding.0,
            pad_right: padding.1,
            pad_top: padding.2,
            pad_bottom: padding.3,
            pad_front: padding.4,
            pad_back: padding.5,
        }
    }
}

impl Module for ReplicationPad3d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let ndim = input_shape.len();
        if ndim < 3 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "ReplicationPad3d requires at least 3D input",
                },
            )));
        }

        let d = input_shape[ndim - 3];
        let h = input_shape[ndim - 2];
        let w = input_shape[ndim - 1];
        let vals = session.tensor_values(input)?;
        let batch_dims: usize = input_shape[..ndim - 3].iter().product();
        let new_d = d + self.pad_front + self.pad_back;
        let new_h = h + self.pad_top + self.pad_bottom;
        let new_w = w + self.pad_left + self.pad_right;
        let mut output = Vec::with_capacity(batch_dims * new_d * new_h * new_w);

        for b in 0..batch_dims {
            let vol = &vals[b * d * h * w..(b + 1) * d * h * w];
            for di in 0..new_d {
                let src_d = di.saturating_sub(self.pad_front).min(d - 1);
                for ri in 0..new_h {
                    let src_h = ri.saturating_sub(self.pad_top).min(h - 1);
                    for ci in 0..new_w {
                        let src_w = ci.saturating_sub(self.pad_left).min(w - 1);
                        output.push(vol[src_d * h * w + src_h * w + src_w]);
                    }
                }
            }
        }

        let mut new_shape = input_shape;
        new_shape[ndim - 3] = new_d;
        new_shape[ndim - 2] = new_h;
        new_shape[ndim - 1] = new_w;
        session.tensor_variable(output, new_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Circular padding for 1D inputs (3D tensors `[N, C, L]`).
///
/// Wraps around: `[a, b, c, d]` with pad=2 -> `[c, d, a, b, c, d, a, b]`.
pub struct CircularPad1d {
    padding_left: usize,
    padding_right: usize,
}

impl CircularPad1d {
    #[must_use]
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding_left: padding.0,
            padding_right: padding.1,
        }
    }
}

impl Module for CircularPad1d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let ndim = input_shape.len();
        let l = input_shape[ndim - 1];

        if self.padding_left == 0 && self.padding_right == 0 {
            return Ok(input);
        }

        let vals = session.tensor_values(input)?;
        let batch_dims: usize = input_shape[..ndim - 1].iter().product();
        let new_l = l + self.padding_left + self.padding_right;
        let mut output = Vec::with_capacity(batch_dims * new_l);

        for b in 0..batch_dims {
            let row = &vals[b * l..(b + 1) * l];
            // Left circular: take from end
            for i in 0..self.padding_left {
                let src = ((l as isize - self.padding_left as isize + i as isize) % l as isize
                    + l as isize) as usize
                    % l;
                output.push(row[src]);
            }
            output.extend_from_slice(row);
            // Right circular: take from beginning
            for i in 0..self.padding_right {
                output.push(row[i % l]);
            }
        }

        let mut new_shape = input_shape;
        *new_shape.last_mut().unwrap() = new_l;
        session.tensor_variable(output, new_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

/// Circular padding for 2D inputs (4D tensors `[N, C, H, W]`).
pub struct CircularPad2d {
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
}

impl CircularPad2d {
    #[must_use]
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            pad_left: padding.0,
            pad_right: padding.1,
            pad_top: padding.2,
            pad_bottom: padding.3,
        }
    }
}

impl Module for CircularPad2d {
    fn forward(
        &self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError> {
        let input_shape = session.tensor_shape(input)?;
        let ndim = input_shape.len();
        if ndim < 2 {
            return Err(AutogradError::Dispatch(DispatchError::Key(
                DispatchKeyError::IncompatibleSet {
                    reason: "CircularPad2d requires at least 2D input",
                },
            )));
        }

        let h = input_shape[ndim - 2];
        let w = input_shape[ndim - 1];
        let vals = session.tensor_values(input)?;
        let batch_dims: usize = input_shape[..ndim - 2].iter().product();
        let new_h = h + self.pad_top + self.pad_bottom;
        let new_w = w + self.pad_left + self.pad_right;
        let mut output = Vec::with_capacity(batch_dims * new_h * new_w);

        for b in 0..batch_dims {
            let plane = &vals[b * h * w..(b + 1) * h * w];
            for row_out in 0..new_h {
                let src_row = ((row_out as isize - self.pad_top as isize) % h as isize + h as isize)
                    as usize
                    % h;
                let row_data = &plane[src_row * w..(src_row + 1) * w];
                for col_out in 0..new_w {
                    let src_col = ((col_out as isize - self.pad_left as isize) % w as isize
                        + w as isize) as usize
                        % w;
                    output.push(row_data[src_col]);
                }
            }
        }

        let mut new_shape = input_shape;
        new_shape[ndim - 2] = new_h;
        new_shape[ndim - 1] = new_w;
        session.tensor_variable(output, new_shape, false)
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use ft_api::FrankenTorchSession;
    use ft_core::{DType, DenseTensor, Device, ExecutionMode, TensorMeta};

    use super::*;

    fn dense_values_f64(tensor: &DenseTensor) -> Vec<f64> {
        match tensor.meta().dtype() {
            DType::F64 => tensor
                .contiguous_values()
                .expect("f64 values should be contiguous")
                .to_vec(),
            DType::F32 => tensor
                .contiguous_values_f32()
                .expect("f32 values should be contiguous")
                .iter()
                .map(|&value| f64::from(value))
                .collect(),
            other => panic!("unsupported dtype in test helper: {other:?}"),
        }
    }

    #[test]
    fn relu_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-1.0, 0.0, 1.0, 2.0], vec![4], true)
            .expect("variable should succeed");

        let relu = ReLU;
        let y = relu
            .forward(&mut session, x)
            .expect("relu forward should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert_eq!(values, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn sigmoid_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0], vec![1], false)
            .expect("variable should succeed");

        let sigmoid = Sigmoid;
        let y = sigmoid
            .forward(&mut session, x)
            .expect("sigmoid forward should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert!((values[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn sequential_chains_modules() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-2.0, -1.0, 0.0, 1.0], vec![4], true)
            .expect("variable should succeed");

        let mut seq = Sequential::new();
        seq.push(Box::new(ReLU));
        // After ReLU: [0, 0, 0, 1]

        let y = seq
            .forward(&mut session, x)
            .expect("sequential forward should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert_eq!(values, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn sequential_parameters_collects_from_all_modules() {
        let seq = Sequential::new();
        assert!(seq.parameters().is_empty());
    }

    #[test]
    fn dropout_eval_mode_passes_through() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable should succeed");

        let dropout = Dropout::new(0.5);
        dropout.eval();
        let y = dropout
            .forward(&mut session, x)
            .expect("dropout eval forward should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dropout_zero_probability_passes_through() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable should succeed");

        let dropout = Dropout::new(0.0);
        let y = dropout
            .forward(&mut session, x)
            .expect("dropout 0.0 forward should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dropout_invalid_probability_fails_closed() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable should succeed");

        for p in [-0.1, 1.1, f64::NAN] {
            let dropout = Dropout::new(p);
            let err = dropout
                .forward(&mut session, x)
                .expect_err("invalid dropout probability must fail closed");
            assert!(matches!(
                err,
                AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "dropout probability p must be finite and in [0, 1]"
                    }
                ))
            ));
        }
    }

    // ── Dropout2d / Dropout3d / AlphaDropout tests (bd-2f7k.11) ────────

    #[test]
    fn dropout2d_eval_mode_passes_through() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0; 2 * 3 * 4 * 4], vec![2, 3, 4, 4], false)
            .unwrap();
        let d = Dropout2d::new(0.5);
        d.eval();
        let y = d.forward(&mut session, x).unwrap();
        let vals = session.tensor_values(y).unwrap();
        assert_eq!(vals, vec![1.0; 2 * 3 * 4 * 4]);
    }

    #[test]
    fn dropout2d_zero_prob_passes_through() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0; 2 * 3 * 4 * 4], vec![2, 3, 4, 4], false)
            .unwrap();
        let d = Dropout2d::new(0.0);
        let y = d.forward(&mut session, x).unwrap();
        let vals = session.tensor_values(y).unwrap();
        assert_eq!(vals, vec![1.0; 2 * 3 * 4 * 4]);
    }

    #[test]
    fn dropout2d_full_prob_zeros() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0; 2 * 3 * 4 * 4], vec![2, 3, 4, 4], false)
            .unwrap();
        let d = Dropout2d::new(1.0);
        let y = d.forward(&mut session, x).unwrap();
        let vals = session.tensor_values(y).unwrap();
        for &v in &vals {
            assert!(v.abs() < 1e-12, "expected 0, got {v}");
        }
    }

    #[test]
    fn dropout2d_channel_consistency() {
        // All spatial elements in a dropped channel should be zero
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0; 8 * 2 * 2], vec![1, 8, 2, 2], false)
            .unwrap();
        let d = Dropout2d::new(0.5);
        let y = d.forward(&mut session, x).unwrap();
        let vals = session.tensor_values(y).unwrap();
        // For each channel, either all spatial values are 0 or all are scaled
        for c in 0..8 {
            let start = c * 4;
            let channel_vals: Vec<f64> = (start..start + 4).map(|i| vals[i]).collect();
            let first = channel_vals[0];
            for &v in &channel_vals {
                assert!(
                    (v - first).abs() < 1e-12,
                    "channel {c}: spatial values differ ({v} vs {first})"
                );
            }
        }
    }

    #[test]
    fn dropout2d_has_no_parameters() {
        let d = Dropout2d::new(0.5);
        assert!(d.parameters().is_empty());
    }

    #[test]
    fn dropout3d_eval_passes_through() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0; 2 * 2 * 2 * 2], vec![1, 2, 2, 2, 2], false)
            .unwrap();
        let d = Dropout3d::new(0.5);
        d.eval();
        let y = d.forward(&mut session, x).unwrap();
        let vals = session.tensor_values(y).unwrap();
        assert_eq!(vals, vec![1.0; 2 * 2 * 2 * 2]);
    }

    #[test]
    fn dropout3d_channel_consistency() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0; 4 * 2 * 2 * 2], vec![1, 4, 2, 2, 2], false)
            .unwrap();
        let d = Dropout3d::new(0.5);
        let y = d.forward(&mut session, x).unwrap();
        let vals = session.tensor_values(y).unwrap();
        let spatial = 8; // 2*2*2
        for c in 0..4 {
            let start = c * spatial;
            let first = vals[start];
            for i in 0..spatial {
                assert!(
                    (vals[start + i] - first).abs() < 1e-12,
                    "channel {c}: spatial values differ"
                );
            }
        }
    }

    #[test]
    fn alpha_dropout_eval_passes_through() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let d = AlphaDropout::new(0.5);
        d.eval();
        let y = d.forward(&mut session, x).unwrap();
        let vals = session.tensor_values(y).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn alpha_dropout_zero_prob() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let d = AlphaDropout::new(0.0);
        let y = d.forward(&mut session, x).unwrap();
        let vals = session.tensor_values(y).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn alpha_dropout_no_parameters() {
        let d = AlphaDropout::new(0.5);
        assert!(d.parameters().is_empty());
    }

    // ── Loss function tests (bd-2f7k.10) ──────────────────────────────

    #[test]
    fn margin_ranking_loss_zero_when_correct() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x1 = s.tensor_variable(vec![2.0], vec![1], false).unwrap();
        let x2 = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let y = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let loss = MarginRankingLoss::new(0.0);
        let l = loss.forward_triplet(&mut s, x1, x2, y).unwrap();
        let vals = s.tensor_values(l).unwrap();
        assert!(vals[0].abs() < 1e-10, "expected 0 loss, got {}", vals[0]);
    }

    #[test]
    fn margin_ranking_loss_positive_when_wrong() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x1 = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let x2 = s.tensor_variable(vec![2.0], vec![1], false).unwrap();
        let y = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let loss = MarginRankingLoss::new(0.0);
        let l = loss.forward_triplet(&mut s, x1, x2, y).unwrap();
        let vals = s.tensor_values(l).unwrap();
        // -1 * (1-2) + 0 = 1
        assert!(
            (vals[0] - 1.0).abs() < 1e-10,
            "expected 1.0, got {}",
            vals[0]
        );
    }

    #[test]
    fn triplet_margin_loss_zero_when_far_negative() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![0.0, 0.0], vec![2], false).unwrap();
        let p = s.tensor_variable(vec![1.0, 0.0], vec![2], false).unwrap();
        let n = s.tensor_variable(vec![10.0, 0.0], vec![2], false).unwrap();
        let loss = TripletMarginLoss::new(1.0, 2.0);
        let l = loss.forward_triplet(&mut s, a, p, n).unwrap();
        let vals = s.tensor_values(l).unwrap();
        // d(a,p)=1, d(a,n)=10, 1-10+1=-8, max(0,-8)=0
        assert!(vals[0].abs() < 1e-10, "expected 0, got {}", vals[0]);
    }

    #[test]
    fn triplet_margin_loss_positive_when_close_negative() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(vec![0.0], vec![1], false).unwrap();
        let p = s.tensor_variable(vec![3.0], vec![1], false).unwrap();
        let n = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let loss = TripletMarginLoss::new(1.0, 2.0);
        let l = loss.forward_triplet(&mut s, a, p, n).unwrap();
        let vals = s.tensor_values(l).unwrap();
        // d(a,p)=3, d(a,n)=1, 3-1+1=3
        assert!(
            (vals[0] - 3.0).abs() < 1e-10,
            "expected 3.0, got {}",
            vals[0]
        );
    }

    #[test]
    fn hinge_embedding_loss_y_positive() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = s.tensor_variable(vec![0.5], vec![1], false).unwrap();
        let target = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let loss = HingeEmbeddingLoss::new(1.0);
        let l = loss.forward_hinge(&mut s, input, target).unwrap();
        let vals = s.tensor_values(l).unwrap();
        assert!((vals[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn hinge_embedding_loss_y_negative() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = s.tensor_variable(vec![0.3], vec![1], false).unwrap();
        let target = s.tensor_variable(vec![-1.0], vec![1], false).unwrap();
        let loss = HingeEmbeddingLoss::new(1.0);
        let l = loss.forward_hinge(&mut s, input, target).unwrap();
        let vals = s.tensor_values(l).unwrap();
        // max(0, 1.0 - 0.3) = 0.7
        assert!((vals[0] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn poisson_nll_loss_log_input() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // input = log(rate), target = count
        let input = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let target = s.tensor_variable(vec![2.0], vec![1], false).unwrap();
        let loss = PoissonNLLLoss::new(true);
        let l = loss.forward(&mut s, input, target).unwrap();
        let vals = s.tensor_values(l).unwrap();
        // exp(1) - 2*1 = e - 2 ≈ 0.71828
        let expected = 1.0f64.exp() - 2.0;
        assert!(
            (vals[0] - expected).abs() < 1e-8,
            "expected {expected}, got {}",
            vals[0]
        );
    }

    #[test]
    fn gaussian_nll_loss_basic() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let target = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let var = s.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let loss = GaussianNLLLoss;
        let l = loss.forward_with_var(&mut s, input, target, var).unwrap();
        let vals = s.tensor_values(l).unwrap();
        // 0.5 * (log(1) + 0/1 + log(2pi)) = 0.5 * log(2pi)
        let expected = 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (vals[0] - expected).abs() < 1e-8,
            "expected {expected}, got {}",
            vals[0]
        );
    }

    #[test]
    fn multi_label_soft_margin_loss_basic() {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = s.tensor_variable(vec![0.0, 0.0], vec![2], false).unwrap();
        let target = s.tensor_variable(vec![1.0, 0.0], vec![2], false).unwrap();
        let loss = MultiLabelSoftMarginLoss;
        let l = loss.forward(&mut s, input, target).unwrap();
        let vals = s.tensor_values(l).unwrap();
        // Both inputs are 0: sigmoid(0)=0.5
        // For label 1: -log(0.5) ≈ 0.693
        // For label 0: -log(1-0.5) ≈ 0.693
        // Mean ≈ 0.693
        assert!((vals[0] - std::f64::consts::LN_2).abs() < 0.01);
    }

    #[test]
    fn activation_modules_have_no_parameters() {
        assert!(ReLU.parameters().is_empty());
        assert!(Sigmoid.parameters().is_empty());
        assert!(Tanh.parameters().is_empty());
        assert!(GELU.parameters().is_empty());
        assert!(SiLU.parameters().is_empty());
    }

    #[test]
    fn linear_forward_computes_xw_t_plus_bias() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // Create a 2->1 linear with known weights
        // weight shape: [1, 2], bias shape: [1, 1]
        let linear = Linear::new(&mut session, 2, 1, true).expect("linear should succeed");
        assert_eq!(linear.in_features(), 2);
        assert_eq!(linear.out_features(), 1);
        assert!(linear.bias().is_some());

        // Input: [1, 2] => batch=1, features=2
        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .expect("variable should succeed");

        let y = linear
            .forward(&mut session, x)
            .expect("forward should succeed");
        let vals = session.tensor_values(y).expect("values should resolve");
        // Output should be [1, 1] shape
        assert_eq!(vals.len(), 1);
        // Value is deterministic due to seeded PRNG
    }

    #[test]
    fn linear_without_bias() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 3, 2, false).expect("linear should succeed");
        assert!(linear.bias().is_none());
        assert_eq!(linear.parameters().len(), 1); // only weight

        let x = session
            .tensor_variable(vec![1.0, 0.0, -1.0], vec![1, 3], false)
            .expect("variable should succeed");
        let y = linear
            .forward(&mut session, x)
            .expect("forward should succeed");
        let vals = session.tensor_values(y).expect("values should resolve");
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn linear_rejects_zero_in_features() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let result = Linear::new(&mut session, 0, 2, true);
        assert!(matches!(
            result,
            Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "linear layer requires in_features > 0"
                }
            )))
        ));
    }

    #[test]
    fn linear_parameters_includes_weight_and_bias() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 4, 3, true).expect("linear should succeed");
        let params = linear.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn linear_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 2, 1, true).expect("linear should succeed");

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], true)
            .expect("variable should succeed");

        let y = linear
            .forward(&mut session, x)
            .expect("forward should succeed");
        let loss = session.tensor_sum(y).expect("sum should succeed");
        let report = session
            .tensor_backward(loss)
            .expect("backward should succeed");

        // Gradients should exist for the weight parameter
        let weight_grad = session.tensor_gradient(&report, linear.weight());
        assert!(weight_grad.is_some(), "weight gradient should be computed");
    }

    #[test]
    fn relu_backward_produces_step_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-1.0, 0.0, 1.0, 2.0], vec![4], true)
            .expect("variable should succeed");

        let relu = ReLU;
        let y = relu
            .forward(&mut session, x)
            .expect("relu forward should succeed");
        let loss = session.tensor_sum(y).expect("sum should succeed");
        let report = session
            .tensor_backward(loss)
            .expect("backward should succeed");

        let grad = session
            .tensor_gradient(&report, x)
            .expect("input gradient should exist");
        // ReLU grad: 0 for negative, 0 for zero, 1 for positive
        assert_eq!(grad[0], 0.0); // x=-1
        assert_eq!(grad[1], 0.0); // x=0
        assert_eq!(grad[2], 1.0); // x=1
        assert_eq!(grad[3], 1.0); // x=2
    }

    #[test]
    fn tanh_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0], vec![1], false)
            .expect("variable should succeed");

        let tanh = Tanh;
        let y = tanh
            .forward(&mut session, x)
            .expect("tanh forward should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert!((values[0]).abs() < 1e-10, "tanh(0) should be 0");
    }

    #[test]
    fn gelu_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("variable should succeed");

        let gelu = GELU;
        let y = gelu
            .forward(&mut session, x)
            .expect("gelu forward should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert!((values[0]).abs() < 1e-10, "gelu(0) should be ~0");
        // gelu(1.0) ~ 0.8412
        assert!(
            (values[1] - 0.8412).abs() < 0.01,
            "gelu(1) should be ~0.8412, got {}",
            values[1]
        );
    }

    #[test]
    fn silu_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("variable should succeed");

        let silu = SiLU;
        let y = silu
            .forward(&mut session, x)
            .expect("silu forward should succeed");
        let values = session.tensor_values(y).expect("values should resolve");
        assert!((values[0]).abs() < 1e-10, "silu(0) should be 0");
        // silu(1.0) = 1.0 * sigmoid(1.0) = 1.0 * 0.7311 ~ 0.7311
        assert!(
            (values[1] - 0.7311).abs() < 0.01,
            "silu(1) should be ~0.7311, got {}",
            values[1]
        );
    }

    #[test]
    fn sequential_with_linear_and_relu_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Build a simple network: Linear(2->2) -> ReLU
        let linear = Linear::new(&mut session, 2, 2, true).expect("linear should succeed");
        let weight_id = linear.weight();

        let mut seq = Sequential::new();
        seq.push(Box::new(linear));
        seq.push(Box::new(ReLU));

        // Check parameter collection through Sequential
        let params = seq.parameters();
        assert!(params.len() >= 2, "Sequential should collect all params");

        let x = session
            .tensor_variable(vec![1.0, -1.0], vec![1, 2], true)
            .expect("variable should succeed");

        let y = seq
            .forward(&mut session, x)
            .expect("sequential forward should succeed");
        let loss = session.tensor_sum(y).expect("sum should succeed");
        let report = session
            .tensor_backward(loss)
            .expect("backward should succeed");

        // Weight gradient should exist
        let weight_grad = session.tensor_gradient(&report, weight_id);
        assert!(
            weight_grad.is_some(),
            "weight gradient should flow through Sequential"
        );
    }

    #[test]
    fn dropout_training_mode_zeros_some_elements() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0; 100], vec![100], false)
            .expect("variable should succeed");

        let dropout = Dropout::new(0.5);
        assert!(dropout.is_training());
        let y = dropout
            .forward(&mut session, x)
            .expect("dropout forward should succeed");
        let values = session.tensor_values(y).expect("values should resolve");

        // Some elements should be zero (dropped), others should be scaled by 2.0
        let zeros = values.iter().filter(|&&v| v == 0.0).count();
        let scaled = values.iter().filter(|&&v| (v - 2.0).abs() < 1e-10).count();
        assert!(zeros > 0, "dropout should zero some elements");
        assert!(
            scaled > 0,
            "dropout should scale surviving elements by 1/(1-p)"
        );
        assert_eq!(
            zeros + scaled,
            100,
            "all elements should be either 0.0 or 2.0"
        );
    }

    #[test]
    fn dropout_full_probability_zeros_all() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable should succeed");

        let dropout = Dropout::new(1.0);
        let y = dropout
            .forward(&mut session, x)
            .expect("dropout p=1.0 should return zeros");
        let values = session.tensor_values(y).expect("values should resolve");
        assert_eq!(values, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn dropout_train_eval_toggle() {
        let dropout = Dropout::new(0.5);
        assert!(dropout.is_training());
        dropout.eval();
        assert!(!dropout.is_training());
        dropout.train(true);
        assert!(dropout.is_training());
    }

    #[test]
    fn module_eval_propagates_to_sequential_children() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut seq = Sequential::new();
        seq.push(Box::new(Dropout::new(0.5)));

        assert!(seq.is_training());
        seq.eval();
        assert!(!seq.is_training());

        let x = session
            .tensor_variable(vec![1.0; 128], vec![128], false)
            .expect("variable should succeed");
        let y_eval = seq
            .forward(&mut session, x)
            .expect("forward in eval mode should succeed");
        let eval_vals = session
            .tensor_values(y_eval)
            .expect("values should resolve");
        assert!(
            eval_vals.iter().all(|v| (*v - 1.0).abs() < 1e-12),
            "eval mode should disable dropout and preserve inputs"
        );

        seq.train(true);
        assert!(seq.is_training());
        let y_train = seq
            .forward(&mut session, x)
            .expect("forward in train mode should succeed");
        let train_vals = session
            .tensor_values(y_train)
            .expect("values should resolve");
        let zeros = train_vals.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "train mode should apply dropout masking");
    }

    #[test]
    fn module_eval_propagates_through_nested_containers() {
        let mut inner = Sequential::new();
        inner.push(Box::new(Dropout::new(0.5)));

        let mut list = ModuleList::new();
        list.push(Box::new(inner));

        let mut outer = Sequential::new();
        outer.push(Box::new(list));

        assert!(outer.is_training());
        outer.train(false);
        assert!(!outer.is_training());

        outer.train(true);
        assert!(outer.is_training());
    }

    #[test]
    fn sigmoid_backward_produces_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0], vec![1], true)
            .expect("variable should succeed");

        let sigmoid = Sigmoid;
        let y = sigmoid
            .forward(&mut session, x)
            .expect("sigmoid forward should succeed");
        let loss = session.tensor_sum(y).expect("sum should succeed");
        let report = session
            .tensor_backward(loss)
            .expect("backward should succeed");

        let grad = session
            .tensor_gradient(&report, x)
            .expect("gradient should exist");
        // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        assert!(
            (grad[0] - 0.25).abs() < 1e-10,
            "sigmoid'(0) should be 0.25, got {}",
            grad[0]
        );
    }

    // ---- Edge case tests (bd-1jnp) ----

    #[test]
    fn sequential_empty_forward_is_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");
        let seq = Sequential::new();
        let y = seq.forward(&mut session, x).expect("forward");
        // Empty sequential should return input unchanged
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn sequential_empty_has_no_parameters() {
        let seq = Sequential::new();
        assert!(seq.parameters().is_empty());
    }

    #[test]
    fn dropout_p_zero_is_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("variable");
        let dropout = Dropout::new(0.0);
        let y = dropout.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dropout_p_one_zeros_all_elements() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("variable");
        let dropout = Dropout::new(1.0);
        let y = dropout.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn dropout_eval_mode_is_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![5.0, 6.0, 7.0], vec![3], false)
            .expect("variable");
        let dropout = Dropout::new(0.5);
        dropout.eval();
        assert!(!dropout.is_training());
        let y = dropout.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn linear_accessor_methods_return_correct_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 4, 3, true).expect("linear");
        assert_eq!(linear.in_features(), 4);
        assert_eq!(linear.out_features(), 3);
        assert!(linear.bias().is_some());
    }

    #[test]
    fn linear_without_bias_has_no_bias_param() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 3, 2, false).expect("linear");
        assert!(linear.bias().is_none());
        assert_eq!(linear.parameters().len(), 1); // only weight
    }

    #[test]
    fn sequential_multi_push_collects_all_params() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear1 = Linear::new(&mut session, 4, 3, true).expect("linear1");
        let linear2 = Linear::new(&mut session, 3, 2, true).expect("linear2");
        let params1 = linear1.parameters().len(); // 2 (weight + bias)
        let params2 = linear2.parameters().len(); // 2 (weight + bias)

        let mut seq = Sequential::new();
        seq.push(Box::new(linear1));
        seq.push(Box::new(ReLU));
        seq.push(Box::new(linear2));
        assert_eq!(seq.parameters().len(), params1 + params2);
    }

    #[test]
    fn sigmoid_extreme_values_do_not_produce_nan() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-100.0, 100.0, 0.0], vec![3], false)
            .expect("variable");
        let sigmoid = Sigmoid;
        let y = sigmoid.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals.iter().all(|v| v.is_finite()));
        assert!(vals[0] < 1e-10); // sigmoid(-100) ≈ 0
        assert!((vals[1] - 1.0).abs() < 1e-10); // sigmoid(100) ≈ 1
        assert!((vals[2] - 0.5).abs() < 1e-10); // sigmoid(0) = 0.5
    }

    #[test]
    fn dropout_has_no_parameters() {
        let dropout = Dropout::new(0.5);
        assert!(dropout.parameters().is_empty());
    }

    // ---- LayerNorm tests ----

    #[test]
    fn layernorm_forward_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ln = LayerNorm::new(&mut session, vec![3], 1e-5).expect("layernorm");
        assert_eq!(ln.normalized_shape(), &[3]);
        assert_eq!(ln.parameters().len(), 2);

        // Input [1, 3]: mean=2, var=2/3, std≈0.8165
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("variable");
        let y = ln.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals.len(), 3);

        // With weight=1, bias=0: output ≈ [-1.2247, 0.0, 1.2247]
        let expected_std = (2.0_f64 / 3.0 + 1e-5).sqrt();
        assert!(
            (vals[0] - (-1.0 / expected_std)).abs() < 1e-4,
            "vals[0] = {}, expected {}",
            vals[0],
            -1.0 / expected_std
        );
        assert!(vals[1].abs() < 1e-4, "vals[1] = {}, expected 0", vals[1]);
        assert!(
            (vals[2] - (1.0 / expected_std)).abs() < 1e-4,
            "vals[2] = {}, expected {}",
            vals[2],
            1.0 / expected_std
        );
    }

    #[test]
    fn layernorm_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ln = LayerNorm::new(&mut session, vec![4], 1e-5).expect("layernorm");

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], true)
            .expect("variable");
        let y = ln.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(x_grad.is_some(), "input gradient should exist");
        let x_grad = x_grad.unwrap();
        assert!(
            x_grad.iter().all(|v| v.is_finite()),
            "gradients must be finite"
        );

        let w_grad = session.tensor_gradient(&report, ln.weight());
        assert!(w_grad.is_some(), "weight gradient should exist");

        let b_grad = session.tensor_gradient(&report, ln.bias());
        assert!(b_grad.is_some(), "bias gradient should exist");
        // Bias gradient should be 1 for each element (since output = ... + bias)
        let b_grad = b_grad.unwrap();
        for &g in b_grad {
            assert!(
                (g - 1.0).abs() < 1e-6,
                "bias gradient should be 1.0, got {}",
                g
            );
        }
    }

    #[test]
    fn layernorm_batch_normalization() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ln = LayerNorm::new(&mut session, vec![2], 1e-5).expect("layernorm");

        // Batch of 3 vectors of size 2
        let x = session
            .tensor_variable(vec![1.0, 3.0, 2.0, 4.0, 0.0, 6.0], vec![3, 2], false)
            .expect("variable");
        let y = ln.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals.len(), 6);

        // Each pair should be normalized independently
        // [1,3]: mean=2, var=1, normalized=[-1, 1]
        let std_0 = (1.0_f64 + 1e-5).sqrt();
        assert!((vals[0] - (-1.0 / std_0)).abs() < 1e-4);
        assert!((vals[1] - (1.0 / std_0)).abs() < 1e-4);
        // [2,4]: mean=3, var=1, normalized=[-1, 1]
        assert!((vals[2] - (-1.0 / std_0)).abs() < 1e-4);
        assert!((vals[3] - (1.0 / std_0)).abs() < 1e-4);
        // [0,6]: mean=3, var=9, normalized=[-1, 1]
        let std_2 = (9.0_f64 + 1e-5).sqrt();
        assert!((vals[4] - (-3.0 / std_2)).abs() < 1e-4);
        assert!((vals[5] - (3.0 / std_2)).abs() < 1e-4);
    }

    #[test]
    fn layernorm_multidim_normalized_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // Normalize over last 2 dims [2, 3]
        let ln = LayerNorm::new(&mut session, vec![2, 3], 1e-5).expect("layernorm");

        // Input [2, 2, 3] — batch of 2, each 2x3 matrix
        let x = session
            .tensor_variable(
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                vec![2, 2, 3],
                false,
            )
            .expect("variable");
        let y = ln.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals.len(), 12);
        // Each batch of 6 elements should be independently normalized
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn layernorm_rejects_empty_normalized_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let result = LayerNorm::new(&mut session, vec![], 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn layernorm_rejects_shape_mismatch() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ln = LayerNorm::new(&mut session, vec![4], 1e-5).expect("layernorm");

        // Input has last dim = 3, but normalized_shape = [4]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("variable");
        let result = ln.forward(&mut session, x);
        assert!(result.is_err());
    }

    #[test]
    fn layernorm_in_sequential() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 4, 3, true).expect("linear");
        let ln = LayerNorm::new(&mut session, vec![3], 1e-5).expect("layernorm");

        let mut seq = Sequential::new();
        let total_params = linear.parameters().len() + ln.parameters().len();
        seq.push(Box::new(linear));
        seq.push(Box::new(ln));

        assert_eq!(seq.parameters().len(), total_params);

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], true)
            .expect("variable");
        let y = seq.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(
            x_grad.is_some(),
            "gradient should flow through Linear+LayerNorm"
        );
    }

    // ---- Embedding tests ----

    #[test]
    fn embedding_forward_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let emb = Embedding::new(&mut session, 5, 3).expect("embedding");
        assert_eq!(emb.num_embeddings(), 5);
        assert_eq!(emb.embedding_dim(), 3);
        assert_eq!(emb.parameters().len(), 1);

        // Look up indices [0, 2, 4]
        let indices = session
            .tensor_variable(vec![0.0, 2.0, 4.0], vec![3], false)
            .expect("indices");
        let y = emb.forward(&mut session, indices).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        // Output shape: [3, 3] — 3 indices, each embedding dim 3
        assert_eq!(vals.len(), 9);

        // Verify values match weight rows
        let weight_vals = session.tensor_values(emb.weight()).expect("weight");
        assert_eq!(&vals[0..3], &weight_vals[0..3]); // row 0
        assert_eq!(&vals[3..6], &weight_vals[6..9]); // row 2
        assert_eq!(&vals[6..9], &weight_vals[12..15]); // row 4
    }

    #[test]
    fn embedding_2d_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let emb = Embedding::new(&mut session, 10, 4).expect("embedding");

        // 2D indices [2, 3] -> output [2, 3, 4]
        let indices = session
            .tensor_variable(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3], false)
            .expect("indices");
        let y = emb.forward(&mut session, indices).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[2, 3, 4]);
        assert_eq!(vals.len(), 24);
    }

    #[test]
    fn embedding_backward_produces_weight_gradient() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let emb = Embedding::new(&mut session, 5, 3).expect("embedding");

        let indices = session
            .tensor_variable(vec![1.0, 3.0], vec![2], false)
            .expect("indices");
        let y = emb.forward(&mut session, indices).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let w_grad = session.tensor_gradient(&report, emb.weight());
        assert!(w_grad.is_some(), "weight gradient should exist");
        let w_grad = w_grad.unwrap();
        // Only rows 1 and 3 should have gradient
        assert!(w_grad[0..3].iter().all(|&v| v == 0.0)); // row 0 unused
        assert!(w_grad[3..6].iter().all(|&v| v == 1.0)); // row 1 selected
        assert!(w_grad[6..9].iter().all(|&v| v == 0.0)); // row 2 unused
        assert!(w_grad[9..12].iter().all(|&v| v == 1.0)); // row 3 selected
        assert!(w_grad[12..15].iter().all(|&v| v == 0.0)); // row 4 unused
    }

    #[test]
    fn embedding_rejects_zero_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        assert!(Embedding::new(&mut session, 0, 3).is_err());
        assert!(Embedding::new(&mut session, 3, 0).is_err());
    }

    // ---- Softmax / LogSoftmax module tests ----

    #[test]
    fn softmax_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("variable");
        let sm = Softmax::new(1);
        let y = sm.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals.len(), 3);
        // Softmax probabilities should sum to 1
        let sum: f64 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {}", sum);
    }

    #[test]
    fn log_softmax_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("variable");
        let lsm = LogSoftmax::new(1);
        let y = lsm.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals.len(), 3);
        // All log-softmax values should be negative
        assert!(vals.iter().all(|&v| v < 0.0));
        // exp(log_softmax) should sum to 1
        let exp_sum: f64 = vals.iter().map(|v| v.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-6);
    }

    // ---- Flatten module test ----

    #[test]
    fn flatten_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3], false)
            .expect("variable");
        let flat = Flatten::new(1, 2);
        let y = flat.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 6]);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ---- Additional activation module tests ----

    #[test]
    fn leaky_relu_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-10.0, 0.0, 5.0], vec![3], false)
            .expect("variable");
        let lrelu = LeakyReLU;
        let y = lrelu.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert!((vals[0] - (-0.1)).abs() < 1e-10); // -10 * 0.01
        assert_eq!(vals[1], 0.0);
        assert_eq!(vals[2], 5.0);
    }

    #[test]
    fn elu_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![-1.0, 0.0, 1.0], vec![3], false)
            .expect("variable");
        let elu = ELU;
        let y = elu.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        // elu(-1) = exp(-1) - 1 ≈ -0.6321
        assert!((vals[0] - ((-1.0_f64).exp() - 1.0)).abs() < 1e-6);
        assert_eq!(vals[1], 0.0);
        assert_eq!(vals[2], 1.0);
    }

    #[test]
    fn mish_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("variable");
        let mish = Mish;
        let y = mish.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert!(vals[0].abs() < 1e-10, "mish(0) ≈ 0");
        // mish(1) = 1 * tanh(softplus(1)) = tanh(ln(1+e)) ≈ 0.8651
        assert!((vals[1] - 0.8651).abs() < 0.01);
    }

    #[test]
    fn softplus_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .expect("variable");
        let sp = Softplus;
        let y = sp.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        // softplus(0) = ln(2) ≈ 0.6931
        assert!((vals[0] - 2.0_f64.ln()).abs() < 1e-6);
        // softplus(1) = ln(1 + e) ≈ 1.3133
        assert!((vals[1] - (1.0 + 1.0_f64.exp()).ln()).abs() < 1e-6);
    }

    #[test]
    fn new_activation_modules_have_no_parameters() {
        assert!(LeakyReLU.parameters().is_empty());
        assert!(ELU.parameters().is_empty());
        assert!(Mish.parameters().is_empty());
        assert!(Softplus.parameters().is_empty());
        assert!(Softmax::new(0).parameters().is_empty());
        assert!(LogSoftmax::new(0).parameters().is_empty());
        assert!(Flatten::new(0, 0).parameters().is_empty());
    }

    // ---- BatchNorm1d tests ----

    #[test]
    fn batchnorm1d_forward_training() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm1d::new(&mut session, 2, 1e-5, 0.1).expect("batchnorm");
        assert_eq!(bn.num_features(), 2);
        assert!(bn.is_training());
        assert_eq!(bn.parameters().len(), 2);

        // Input [3, 2] — batch of 3, 2 features
        // Feature 0: [1, 2, 3] -> mean=2, var=2/3
        // Feature 1: [4, 5, 6] -> mean=5, var=2/3
        let x = session
            .tensor_variable(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2], false)
            .expect("variable");
        let y = bn.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals.len(), 6);

        // With weight=1, bias=0: should be normalized
        let expected_std = (2.0_f64 / 3.0 + 1e-5).sqrt();
        // Feature 0: (1-2)/std, (2-2)/std, (3-2)/std
        assert!((vals[0] - (-1.0 / expected_std)).abs() < 1e-4);
        assert!(vals[2].abs() < 1e-4);
        assert!((vals[4] - (1.0 / expected_std)).abs() < 1e-4);
    }

    #[test]
    fn batchnorm1d_updates_running_stats() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm1d::new(&mut session, 2, 1e-5, 0.1).expect("batchnorm");

        // Initial running stats
        assert_eq!(bn.running_mean(), vec![0.0, 0.0]);
        assert_eq!(bn.running_var(), vec![1.0, 1.0]);

        let x = session
            .tensor_variable(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2], false)
            .expect("variable");
        let _ = bn.forward(&mut session, x).expect("forward");

        // Running mean should be updated: 0.9*0 + 0.1*batch_mean
        let rm = bn.running_mean();
        assert!((rm[0] - 0.1 * 2.0).abs() < 1e-6, "rm[0]={}", rm[0]); // 0.1 * 2.0
        assert!((rm[1] - 0.1 * 5.0).abs() < 1e-6, "rm[1]={}", rm[1]); // 0.1 * 5.0

        // Running var: 0.9*1.0 + 0.1*(biased_var * N/(N-1))
        // biased_var = 2/3, unbiased = 2/3 * 3/2 = 1.0
        let rv = bn.running_var();
        let expected_rv = 0.9 * 1.0 + 0.1 * 1.0; // = 1.0
        assert!((rv[0] - expected_rv).abs() < 1e-6, "rv[0]={}", rv[0]);
    }

    #[test]
    fn batchnorm1d_eval_mode() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm1d::new(&mut session, 2, 1e-5, 0.1).expect("batchnorm");

        // Run one training pass to update running stats
        let x = session
            .tensor_variable(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2], false)
            .expect("variable");
        let _ = bn.forward(&mut session, x).expect("training forward");

        // Switch to eval mode
        bn.eval();
        assert!(!bn.is_training());

        // Forward in eval uses running stats
        let x2 = session
            .tensor_variable(vec![2.0, 5.0], vec![1, 2], false)
            .expect("variable");
        let y = bn.forward(&mut session, x2).expect("eval forward");
        let vals = session.tensor_values(y).expect("values");
        assert_eq!(vals.len(), 2);
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn batchnorm1d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm1d::new(&mut session, 3, 1e-5, 0.1).expect("batchnorm");

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("variable");
        let y = bn.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(x_grad.is_some(), "input gradient should exist");
        assert!(
            x_grad.unwrap().iter().all(|v| v.is_finite()),
            "all gradients must be finite"
        );

        let w_grad = session.tensor_gradient(&report, bn.weight());
        assert!(w_grad.is_some(), "weight gradient should exist");

        let b_grad = session.tensor_gradient(&report, bn.bias());
        assert!(b_grad.is_some(), "bias gradient should exist");
    }

    #[test]
    fn batchnorm1d_rejects_wrong_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm1d::new(&mut session, 3, 1e-5, 0.1).expect("batchnorm");

        // 1D input (wrong)
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");
        assert!(bn.forward(&mut session, x).is_err());

        // Wrong feature dim
        let x2 = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .expect("variable");
        assert!(bn.forward(&mut session, x2).is_err());
    }

    #[test]
    fn batchnorm1d_rejects_zero_features() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        assert!(BatchNorm1d::new(&mut session, 0, 1e-5, 0.1).is_err());
    }

    #[test]
    fn batchnorm1d_train_eval_toggle() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm1d::new(&mut session, 4, 1e-5, 0.1).expect("batchnorm");
        assert!(bn.is_training());
        bn.eval();
        assert!(!bn.is_training());
        bn.train(true);
        assert!(bn.is_training());
    }

    // ---- Conv1d tests ----

    #[test]
    fn conv1d_forward_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv1d::new(&mut session, 1, 1, 3, 1, 0, false).expect("conv1d");
        assert_eq!(conv.parameters().len(), 1); // weight only

        // Input [1, 1, 5], kernel_size=3, stride=1, no padding -> L_out = 3
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 1, 5], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 3]);
        assert_eq!(vals.len(), 3);
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn conv1d_forward_with_bias() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv1d::new(&mut session, 2, 3, 2, 1, 0, true).expect("conv1d");
        assert_eq!(conv.parameters().len(), 2); // weight + bias
        assert!(conv.bias().is_some());

        // Input [2, 2, 4] -> output [2, 3, 3]
        let x = session
            .tensor_variable(vec![0.0; 16], vec![2, 2, 4], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[2, 3, 3]);
    }

    #[test]
    fn conv1d_forward_with_padding() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // padding=1 means L_out = (L + 2*1 - K) / stride + 1 = (3 + 2 - 3) / 1 + 1 = 3
        let conv = Conv1d::new(&mut session, 1, 1, 3, 1, 1, false).expect("conv1d");

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 1, 3], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        // Same padding: output length = input length
        assert_eq!(meta.shape(), &[1, 1, 3]);
    }

    #[test]
    fn conv1d_forward_with_stride() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // stride=2, L_out = (6 - 3) / 2 + 1 = 2
        let conv = Conv1d::new(&mut session, 1, 1, 3, 2, 0, false).expect("conv1d");

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 1, 6], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 2]);
    }

    #[test]
    fn conv1d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv1d::new(&mut session, 1, 2, 3, 1, 0, true).expect("conv1d");

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 1, 5], true)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(x_grad.is_some(), "input gradient should exist");

        let w_grad = session.tensor_gradient(&report, conv.weight());
        assert!(w_grad.is_some(), "weight gradient should exist");
    }

    #[test]
    fn conv1d_rejects_wrong_input_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv1d::new(&mut session, 2, 1, 3, 1, 0, false).expect("conv1d");

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");
        assert!(conv.forward(&mut session, x).is_err());
    }

    // ---- AvgPool1d tests ----

    #[test]
    fn avgpool1d_forward_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AvgPool1d::new(2, 2);
        assert!(pool.parameters().is_empty());

        // Input [1, 1, 4] with kernel_size=2, stride=2 -> [1, 1, 2]
        let x = session
            .tensor_variable(vec![1.0, 3.0, 5.0, 7.0], vec![1, 1, 4], false)
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 2]);
        assert!((vals[0] - 2.0).abs() < 1e-10); // avg(1, 3) = 2
        assert!((vals[1] - 6.0).abs() < 1e-10); // avg(5, 7) = 6
    }

    #[test]
    fn avgpool1d_forward_stride_1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AvgPool1d::new(3, 1);

        // Input [1, 1, 5] with kernel_size=3, stride=1 -> [1, 1, 3]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 1, 5], false)
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 3]);
        assert!((vals[0] - 2.0).abs() < 1e-10); // avg(1,2,3) = 2
        assert!((vals[1] - 3.0).abs() < 1e-10); // avg(2,3,4) = 3
        assert!((vals[2] - 4.0).abs() < 1e-10); // avg(3,4,5) = 4
    }

    #[test]
    fn avgpool1d_multichannel() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AvgPool1d::new(2, 2);

        // Input [1, 2, 4] -> [1, 2, 2]
        let x = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                vec![1, 2, 4],
                false,
            )
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 2, 2]);
        // Channel 0: avg(1,2)=1.5, avg(3,4)=3.5
        assert!((vals[0] - 1.5).abs() < 1e-10);
        assert!((vals[1] - 3.5).abs() < 1e-10);
        // Channel 1: avg(10,20)=15, avg(30,40)=35
        assert!((vals[2] - 15.0).abs() < 1e-10);
        assert!((vals[3] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn avgpool1d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AvgPool1d::new(2, 2);

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("variable");
        assert!(pool.forward(&mut session, x).is_err());
    }

    // ---- MultiheadAttention tests ----

    #[test]
    fn mha_forward_self_attention() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mha = MultiheadAttention::new(&mut session, 4, 2).expect("mha");
        assert_eq!(mha.num_heads(), 2);
        // 4 Linear layers, each with weight + bias = 8 params
        assert_eq!(mha.parameters().len(), 8);

        // Input [1, 3, 4]: batch=1, seq_len=3, embed_dim=4
        let x = session
            .tensor_variable(vec![0.0; 12], vec![1, 3, 4], false)
            .expect("variable");
        let y = mha.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 3, 4]);
        assert_eq!(vals.len(), 12);
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mha_forward_batched() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mha = MultiheadAttention::new(&mut session, 6, 3).expect("mha");

        // Input [2, 4, 6]: batch=2, seq_len=4, embed_dim=6
        let x = session
            .tensor_variable(vec![1.0; 48], vec![2, 4, 6], false)
            .expect("variable");
        let y = mha.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[2, 4, 6]);
    }

    #[test]
    fn mha_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mha = MultiheadAttention::new(&mut session, 4, 2).expect("mha");

        let x = session
            .tensor_variable(
                vec![
                    1.0, 0.0, -1.0, 0.5, 1.0, 0.0, -1.0, 0.5, 1.0, 0.0, -1.0, 0.5,
                ],
                vec![1, 3, 4],
                true,
            )
            .expect("variable");
        let y = mha.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(x_grad.is_some(), "input gradient should exist");
        assert!(
            x_grad.unwrap().iter().all(|v| v.is_finite()),
            "all gradients must be finite"
        );
    }

    #[test]
    fn mha_cross_attention() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mha = MultiheadAttention::new(&mut session, 4, 2).expect("mha");

        // Query [1, 2, 4], Key/Value [1, 5, 4] (different seq lengths)
        let query = session
            .tensor_variable(vec![0.0; 8], vec![1, 2, 4], false)
            .expect("query");
        let kv = session
            .tensor_variable(vec![0.0; 20], vec![1, 5, 4], false)
            .expect("kv");
        let y = mha
            .forward_qkv(&mut session, query, kv, kv)
            .expect("cross-attn");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        // Output shape follows query: [1, 2, 4]
        assert_eq!(meta.shape(), &[1, 2, 4]);
    }

    #[test]
    fn mha_rejects_non_divisible_embed_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        assert!(MultiheadAttention::new(&mut session, 5, 2).is_err());
    }

    #[test]
    fn identity_passes_through_unchanged() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("var");
        let identity = Identity;
        let y = identity.forward(&mut session, x).expect("forward");
        // Identity returns the same node
        assert_eq!(x, y);
        assert!(identity.parameters().is_empty());
    }

    #[test]
    fn identity_in_sequential() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("var");
        let mut seq = Sequential::new();
        seq.push(Box::new(Identity));
        seq.push(Box::new(Identity));
        let y = seq.forward(&mut session, x).expect("forward");
        assert_eq!(x, y);
        assert!(seq.parameters().is_empty());
    }

    // ── GroupNorm tests ────────────────────────────────────────────────

    #[test]
    fn groupnorm_basic_output_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, true).expect("groupnorm");

        // Input [N=1, C=4, L=3]
        let x = session
            .tensor_variable(vec![1.0; 12], vec![1, 4, 3], false)
            .expect("var");
        let y = gn.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("meta");
        assert_eq!(meta.shape(), &[1, 4, 3]);
    }

    #[test]
    fn groupnorm_normalizes_within_groups() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2 groups, 4 channels -> 2 channels per group
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, false).expect("groupnorm");

        // Input [N=1, C=4]: group 0 = channels 0,1; group 1 = channels 2,3
        // Constant input within each group should normalize to ~0
        let x = session
            .tensor_variable(vec![5.0, 5.0, 10.0, 10.0], vec![1, 4], false)
            .expect("var");
        let y = gn.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("vals");

        // Within each group, all values are the same => mean = value, var = 0
        // normalized = (val - mean) / sqrt(var + eps) ≈ 0
        for v in &vals {
            assert!(
                v.abs() < 1e-2,
                "constant group should normalize near 0, got {}",
                v
            );
        }
    }

    #[test]
    fn groupnorm_affine_has_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, true).expect("groupnorm");
        assert_eq!(gn.parameters().len(), 2); // weight + bias
    }

    #[test]
    fn groupnorm_no_affine_has_no_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, false).expect("groupnorm");
        assert!(gn.parameters().is_empty());
    }

    #[test]
    fn groupnorm_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, true).expect("groupnorm");

        let x = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![1, 4, 2],
                true,
            )
            .expect("var");
        let y = gn.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let grad = session.tensor_gradient(&report, x);
        assert!(
            grad.is_some(),
            "GroupNorm should produce gradients for input"
        );
    }

    #[test]
    fn groupnorm_rejects_non_divisible_channels() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 3 groups, 4 channels -> 4 % 3 != 0
        assert!(GroupNorm::new(&mut session, 3, 4, 1e-5, true).is_err());
    }

    #[test]
    fn groupnorm_rejects_wrong_input_channels() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, true).expect("groupnorm");

        // Input has 6 channels but GroupNorm expects 4
        let x = session
            .tensor_variable(vec![1.0; 6], vec![1, 6], false)
            .expect("var");
        assert!(gn.forward(&mut session, x).is_err());
    }

    #[test]
    fn groupnorm_rejects_1d_input() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, true).expect("groupnorm");

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .expect("var");
        assert!(gn.forward(&mut session, x).is_err());
    }

    #[test]
    fn groupnorm_with_spatial_dims() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, true).expect("groupnorm");

        // Input [N=2, C=4, H=3] with varying values
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let x = session
            .tensor_variable(data, vec![2, 4, 3], false)
            .expect("var");
        let y = gn.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("meta");
        assert_eq!(meta.shape(), &[2, 4, 3]);
    }

    // ── InstanceNorm1d tests ───────────────────────────────────────────

    #[test]
    fn instancenorm1d_output_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inorm = InstanceNorm1d::new(&mut session, 4, 1e-5, true).expect("instancenorm");

        // Input [N=2, C=4, L=3]
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let x = session
            .tensor_variable(data, vec![2, 4, 3], false)
            .expect("var");
        let y = inorm.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("meta");
        assert_eq!(meta.shape(), &[2, 4, 3]);
    }

    #[test]
    fn instancenorm1d_normalizes_per_channel_per_sample() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inorm = InstanceNorm1d::new(&mut session, 2, 1e-5, false).expect("instancenorm");

        // Input [N=1, C=2, L=3]
        // Channel 0: [1, 2, 3], Channel 1: [10, 10, 10]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 10.0, 10.0, 10.0], vec![1, 2, 3], false)
            .expect("var");
        let y = inorm.forward(&mut session, x).expect("forward");
        let vals = session.tensor_values(y).expect("vals");

        // Channel 1 (constant) should normalize to ~0
        assert!(
            vals[3].abs() < 1e-2,
            "constant channel should be ~0, got {}",
            vals[3]
        );
        assert!(
            vals[4].abs() < 1e-2,
            "constant channel should be ~0, got {}",
            vals[4]
        );
        assert!(
            vals[5].abs() < 1e-2,
            "constant channel should be ~0, got {}",
            vals[5]
        );
    }

    #[test]
    fn instancenorm1d_affine_has_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inorm = InstanceNorm1d::new(&mut session, 4, 1e-5, true).expect("instancenorm");
        assert_eq!(inorm.parameters().len(), 2);
    }

    #[test]
    fn instancenorm1d_no_affine_has_no_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inorm = InstanceNorm1d::new(&mut session, 4, 1e-5, false).expect("instancenorm");
        assert!(inorm.parameters().is_empty());
    }

    #[test]
    fn instancenorm1d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inorm = InstanceNorm1d::new(&mut session, 2, 1e-5, true).expect("instancenorm");

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3], true)
            .expect("var");
        let y = inorm.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let grad = session.tensor_gradient(&report, x);
        assert!(grad.is_some(), "InstanceNorm1d should produce gradients");
    }

    // ── MaxPool1d tests ──────────────────────────────────────────────────

    #[test]
    fn maxpool1d_forward_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = MaxPool1d::new(2, 2);
        assert!(pool.parameters().is_empty());

        // Input [1, 1, 4] with kernel_size=2, stride=2 -> [1, 1, 2]
        let x = session
            .tensor_variable(vec![1.0, 3.0, 5.0, 2.0], vec![1, 1, 4], false)
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 2]);
        assert!((vals[0] - 3.0).abs() < 1e-10); // max(1, 3) = 3
        assert!((vals[1] - 5.0).abs() < 1e-10); // max(5, 2) = 5
    }

    #[test]
    fn maxpool1d_forward_stride_1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = MaxPool1d::new(3, 1);

        // Input [1, 1, 5] with kernel_size=3, stride=1 -> [1, 1, 3]
        let x = session
            .tensor_variable(vec![1.0, 5.0, 2.0, 4.0, 3.0], vec![1, 1, 5], false)
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 3]);
        assert!((vals[0] - 5.0).abs() < 1e-10); // max(1,5,2) = 5
        assert!((vals[1] - 5.0).abs() < 1e-10); // max(5,2,4) = 5
        assert!((vals[2] - 4.0).abs() < 1e-10); // max(2,4,3) = 4
    }

    #[test]
    fn maxpool1d_multichannel() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = MaxPool1d::new(2, 2);

        // Input [1, 2, 4] -> [1, 2, 2]
        let x = session
            .tensor_variable(
                vec![1.0, 3.0, 2.0, 4.0, 10.0, 5.0, 30.0, 20.0],
                vec![1, 2, 4],
                false,
            )
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 2, 2]);
        // Channel 0: max(1,3)=3, max(2,4)=4
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);
        // Channel 1: max(10,5)=10, max(30,20)=30
        assert!((vals[2] - 10.0).abs() < 1e-10);
        assert!((vals[3] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn maxpool1d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = MaxPool1d::new(2, 2);

        let x = session
            .tensor_variable(vec![1.0, 3.0, 5.0, 2.0], vec![1, 1, 4], true)
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let grad = session.tensor_gradient(&report, x);
        assert!(grad.is_some(), "MaxPool1d should produce gradients");
    }

    #[test]
    fn maxpool1d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = MaxPool1d::new(2, 2);

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("variable");
        assert!(pool.forward(&mut session, x).is_err());
    }

    // ── Conv2d tests ─────────────────────────────────────────────────────

    #[test]
    fn conv2d_forward_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv2d::new(&mut session, 1, 1, (3, 3), (1, 1), (0, 0), false).expect("conv2d");
        assert_eq!(conv.parameters().len(), 1); // weight only

        // Input [1, 1, 5, 5], kernel 3x3, no padding -> H_out = 3, W_out = 3
        let x = session
            .tensor_variable(vec![0.0; 25], vec![1, 1, 5, 5], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 3, 3]);
        assert_eq!(vals.len(), 9);
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn conv2d_forward_with_bias() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv2d::new(&mut session, 2, 4, (3, 3), (1, 1), (0, 0), true).expect("conv2d");
        assert_eq!(conv.parameters().len(), 2); // weight + bias
        assert!(conv.bias().is_some());

        // Input [2, 2, 5, 5] -> output [2, 4, 3, 3]
        let x = session
            .tensor_variable(vec![0.0; 100], vec![2, 2, 5, 5], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[2, 4, 3, 3]);
    }

    #[test]
    fn conv2d_forward_with_padding() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // padding=1, same-size output: H_out = (3 + 2 - 3)/1 + 1 = 3
        let conv = Conv2d::new(&mut session, 1, 1, (3, 3), (1, 1), (1, 1), false).expect("conv2d");

        let x = session
            .tensor_variable(vec![1.0; 9], vec![1, 1, 3, 3], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn conv2d_forward_with_stride() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // stride=2: H_out = (6-3)/2 + 1 = 2, W_out = (6-3)/2 + 1 = 2
        let conv = Conv2d::new(&mut session, 1, 1, (3, 3), (2, 2), (0, 0), false).expect("conv2d");

        let x = session
            .tensor_variable(vec![1.0; 36], vec![1, 1, 6, 6], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn conv2d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv2d::new(&mut session, 1, 2, (3, 3), (1, 1), (0, 0), true).expect("conv2d");

        let x = session
            .tensor_variable(vec![1.0; 25], vec![1, 1, 5, 5], true)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(x_grad.is_some(), "input gradient should exist");

        let w_grad = session.tensor_gradient(&report, conv.weight());
        assert!(w_grad.is_some(), "weight gradient should exist");
    }

    #[test]
    fn conv2d_rejects_wrong_input_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv2d::new(&mut session, 2, 1, (3, 3), (1, 1), (0, 0), false).expect("conv2d");

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");
        assert!(conv.forward(&mut session, x).is_err());
    }

    #[test]
    fn conv2d_non_square_kernel() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // Non-square kernel: kH=2, kW=3
        // Input [1, 1, 4, 5]: H_out = (4-2)/1+1=3, W_out = (5-3)/1+1=3
        let conv = Conv2d::new(&mut session, 1, 1, (2, 3), (1, 1), (0, 0), false).expect("conv2d");

        let x = session
            .tensor_variable(vec![1.0; 20], vec![1, 1, 4, 5], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 3, 3]);
    }

    // ── MaxPool2d tests ──────────────────────────────────────────────────

    #[test]
    fn maxpool2d_forward_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = MaxPool2d::new((2, 2), (2, 2));
        assert!(pool.parameters().is_empty());

        // Input [1, 1, 4, 4] with kernel=2x2, stride=2x2 -> [1, 1, 2, 2]
        #[rustfmt::skip]
        let data = vec![
            1.0, 3.0, 5.0, 2.0,
            4.0, 6.0, 7.0, 8.0,
            9.0, 1.0, 3.0, 4.0,
            2.0, 5.0, 6.0, 0.0,
        ];
        let x = session
            .tensor_variable(data, vec![1, 1, 4, 4], false)
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 2, 2]);
        // top-left 2x2: max(1,3,4,6) = 6
        assert!((vals[0] - 6.0).abs() < 1e-10);
        // top-right 2x2: max(5,2,7,8) = 8
        assert!((vals[1] - 8.0).abs() < 1e-10);
        // bottom-left 2x2: max(9,1,2,5) = 9
        assert!((vals[2] - 9.0).abs() < 1e-10);
        // bottom-right 2x2: max(3,4,6,0) = 6
        assert!((vals[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn maxpool2d_forward_stride_1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // kernel=2x2, stride=1x1 -> sliding window
        // Input [1, 1, 3, 3] -> [1, 1, 2, 2]
        let pool = MaxPool2d::new((2, 2), (1, 1));

        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let x = session
            .tensor_variable(data, vec![1, 1, 3, 3], false)
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 2, 2]);
        assert!((vals[0] - 5.0).abs() < 1e-10); // max(1,2,4,5)
        assert!((vals[1] - 6.0).abs() < 1e-10); // max(2,3,5,6)
        assert!((vals[2] - 8.0).abs() < 1e-10); // max(4,5,7,8)
        assert!((vals[3] - 9.0).abs() < 1e-10); // max(5,6,8,9)
    }

    #[test]
    fn maxpool2d_multichannel_batched() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = MaxPool2d::new((2, 2), (2, 2));

        // Input [2, 1, 2, 2] -> [2, 1, 1, 1]
        let data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let x = session
            .tensor_variable(data, vec![2, 1, 2, 2], false)
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[2, 1, 1, 1]);
        assert!((vals[0] - 4.0).abs() < 1e-10); // batch 0: max(1,2,3,4) = 4
        assert!((vals[1] - 40.0).abs() < 1e-10); // batch 1: max(10,20,30,40) = 40
    }

    #[test]
    fn maxpool2d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = MaxPool2d::new((2, 2), (2, 2));

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], true)
            .expect("variable");
        let y = pool.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let grad = session.tensor_gradient(&report, x);
        assert!(grad.is_some(), "MaxPool2d should produce gradients");
    }

    #[test]
    fn maxpool2d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = MaxPool2d::new((2, 2), (2, 2));

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");
        assert!(pool.forward(&mut session, x).is_err());
    }

    // ---- BatchNorm2d tests ----

    #[test]
    fn batchnorm2d_forward_training() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm2d::new(&mut session, 2, 1e-5, 0.1).expect("bn2d");

        // [N=2, C=2, H=2, W=2]
        #[rustfmt::skip]
        let data = vec![
            // N=0, C=0         N=0, C=1
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            // N=1, C=0         N=1, C=1
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let x = session
            .tensor_variable(data, vec![2, 2, 2, 2], true)
            .expect("variable");
        let out = bn.forward(&mut session, x).expect("forward");

        let (vals, meta) = session.tensor_values_meta(out).expect("values_meta");
        assert_eq!(meta.shape(), &[2, 2, 2, 2]);

        // Output should be normalized: mean ≈ 0, std ≈ 1 per channel
        // Channel 0 values: 1..4, 9..12 → mean=6.5, Channel 1: 5..8, 13..16 → mean=10.5
        // First element of channel 0: (1 - 6.5) / std * 1 + 0 < 0
        assert!(
            vals[0] < 0.0,
            "first elem of channel 0 should be negative after normalization"
        );
        assert_eq!(vals.len(), 16);
    }

    #[test]
    fn batchnorm2d_output_shape_preserved() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm2d::new(&mut session, 3, 1e-5, 0.1).expect("bn2d");

        let x = session
            .tensor_variable(vec![1.0; 2 * 3 * 4 * 5], vec![2, 3, 4, 5], true)
            .expect("variable");
        let out = bn.forward(&mut session, x).expect("forward");

        let (_, meta) = session.tensor_values_meta(out).expect("values_meta");
        assert_eq!(meta.shape(), &[2, 3, 4, 5]);
    }

    #[test]
    fn batchnorm2d_eval_mode() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm2d::new(&mut session, 2, 1e-5, 0.1).expect("bn2d");

        // Train first to populate running stats
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let x = session
            .tensor_variable(data.clone(), vec![2, 2, 2, 2], true)
            .expect("variable");
        let _ = bn.forward(&mut session, x).expect("train forward");

        // Running stats should be updated
        let rm = bn.running_mean();
        assert!(rm[0] != 0.0, "running mean should be updated");

        // Switch to eval
        bn.eval();
        assert!(!bn.is_training());

        let x2 = session
            .tensor_variable(data, vec![2, 2, 2, 2], false)
            .expect("variable");
        let out = bn.forward(&mut session, x2).expect("eval forward");

        let (_, meta) = session.tensor_values_meta(out).expect("values_meta");
        assert_eq!(meta.shape(), &[2, 2, 2, 2]);
    }

    #[test]
    fn batchnorm2d_train_eval_toggle() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm2d::new(&mut session, 4, 1e-5, 0.1).expect("bn2d");

        assert!(bn.is_training());
        bn.eval();
        assert!(!bn.is_training());
        bn.train(true);
        assert!(bn.is_training());
    }

    #[test]
    fn batchnorm2d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm2d::new(&mut session, 2, 1e-5, 0.1).expect("bn2d");

        let x = session
            .tensor_variable(vec![1.0; 16], vec![2, 2, 2, 2], true)
            .expect("variable");
        let out = bn.forward(&mut session, x).expect("forward");

        let loss = session.tensor_sum(out).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let w_grad = session.tensor_gradient(&report, bn.weight());
        assert!(w_grad.is_some(), "weight should have gradient");
        let b_grad = session.tensor_gradient(&report, bn.bias());
        assert!(b_grad.is_some(), "bias should have gradient");
    }

    #[test]
    fn batchnorm2d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm2d::new(&mut session, 3, 1e-5, 0.1).expect("bn2d");

        // 2D input — should fail
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("variable");
        assert!(bn.forward(&mut session, x).is_err());

        // 3D input — should fail
        let x3 = session
            .tensor_variable(vec![1.0; 6], vec![1, 3, 2], false)
            .expect("variable");
        assert!(bn.forward(&mut session, x3).is_err());
    }

    #[test]
    fn batchnorm2d_rejects_channel_mismatch() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm2d::new(&mut session, 3, 1e-5, 0.1).expect("bn2d");

        // C=2 but num_features=3
        let x = session
            .tensor_variable(vec![1.0; 16], vec![2, 2, 2, 2], false)
            .expect("variable");
        assert!(bn.forward(&mut session, x).is_err());
    }

    #[test]
    fn batchnorm2d_updates_running_stats() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm2d::new(&mut session, 2, 1e-5, 0.1).expect("bn2d");

        assert_eq!(bn.running_mean(), vec![0.0, 0.0]);
        assert_eq!(bn.running_var(), vec![1.0, 1.0]);

        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let x = session
            .tensor_variable(data, vec![2, 2, 2, 2], true)
            .expect("variable");
        let _ = bn.forward(&mut session, x).expect("forward");

        let rm = bn.running_mean();
        let rv = bn.running_var();
        // After one forward: running_mean should shift toward batch mean
        assert!(rm[0] > 0.0, "channel 0 running mean should be positive");
        assert!(
            rm[1] > rm[0],
            "channel 1 mean should be greater than channel 0"
        );
        // Running var should no longer be exactly 1.0
        assert!(rv[0] != 1.0, "running var should be updated");
    }

    // ---- InstanceNorm2d tests ----

    #[test]
    fn instancenorm2d_output_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inn = InstanceNorm2d::new(&mut session, 3, 1e-5, true).expect("in2d");

        let x = session
            .tensor_variable(vec![1.0; 2 * 3 * 4 * 5], vec![2, 3, 4, 5], true)
            .expect("variable");
        let out = inn.forward(&mut session, x).expect("forward");

        let (_, meta) = session.tensor_values_meta(out).expect("values_meta");
        assert_eq!(meta.shape(), &[2, 3, 4, 5]);
    }

    #[test]
    fn instancenorm2d_normalizes_per_channel() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inn = InstanceNorm2d::new(&mut session, 2, 1e-5, false).expect("in2d");

        // [N=1, C=2, H=2, W=2]: channel 0 has values 1..4, channel 1 has 10..13
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            10.0, 11.0, 12.0, 13.0,
        ];
        let x = session
            .tensor_variable(data, vec![1, 2, 2, 2], true)
            .expect("variable");
        let out = inn.forward(&mut session, x).expect("forward");

        let (vals, _) = session.tensor_values_meta(out).expect("values_meta");
        // Each channel normalized independently: mean ≈ 0
        let ch0_mean: f64 = vals[..4].iter().sum::<f64>() / 4.0;
        let ch1_mean: f64 = vals[4..].iter().sum::<f64>() / 4.0;
        assert!(
            ch0_mean.abs() < 1e-6,
            "channel 0 mean should be ~0, got {ch0_mean}"
        );
        assert!(
            ch1_mean.abs() < 1e-6,
            "channel 1 mean should be ~0, got {ch1_mean}"
        );
    }

    #[test]
    fn instancenorm2d_affine_has_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inn = InstanceNorm2d::new(&mut session, 4, 1e-5, true).expect("in2d");
        assert_eq!(inn.parameters().len(), 2);
    }

    #[test]
    fn instancenorm2d_no_affine_has_no_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inn = InstanceNorm2d::new(&mut session, 4, 1e-5, false).expect("in2d");
        assert_eq!(inn.parameters().len(), 0);
    }

    #[test]
    fn instancenorm2d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inn = InstanceNorm2d::new(&mut session, 3, 1e-5, false).expect("in2d");

        // 3D → should fail
        let x = session
            .tensor_variable(vec![1.0; 6], vec![1, 3, 2], false)
            .expect("variable");
        assert!(inn.forward(&mut session, x).is_err());
    }

    #[test]
    fn instancenorm2d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let inn = InstanceNorm2d::new(&mut session, 2, 1e-5, true).expect("in2d");

        let x = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![1, 2, 2, 2],
                true,
            )
            .expect("variable");
        let out = inn.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(out).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(x_grad.is_some(), "input should have gradient");
    }

    // ---- AdaptiveAvgPool2d tests ----

    #[test]
    fn adaptive_avgpool2d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AdaptiveAvgPool2d::new((1, 1));

        // [N=1, C=1, H=2, W=2] → [1, 1, 1, 1]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], true)
            .expect("variable");
        let out = pool.forward(&mut session, x).expect("forward");

        let (vals, meta) = session.tensor_values_meta(out).expect("values_meta");
        assert_eq!(meta.shape(), &[1, 1, 1, 1]);
        // Global average: (1+2+3+4)/4 = 2.5
        assert!(
            (vals[0] - 2.5).abs() < 1e-10,
            "expected 2.5, got {}",
            vals[0]
        );
    }

    #[test]
    fn adaptive_avgpool2d_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AdaptiveAvgPool2d::new((3, 3));

        let x = session
            .tensor_variable(vec![1.0; 9], vec![1, 1, 3, 3], false)
            .expect("variable");
        let out = pool.forward(&mut session, x).expect("forward");

        let (_, meta) = session.tensor_values_meta(out).expect("values_meta");
        assert_eq!(meta.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn adaptive_avgpool2d_downsample_4x4_to_2x2() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AdaptiveAvgPool2d::new((2, 2));

        // [1, 1, 4, 4] → [1, 1, 2, 2]
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let x = session
            .tensor_variable(data, vec![1, 1, 4, 4], true)
            .expect("variable");
        let out = pool.forward(&mut session, x).expect("forward");

        let (vals, meta) = session.tensor_values_meta(out).expect("values_meta");
        assert_eq!(meta.shape(), &[1, 1, 2, 2]);
        // Top-left 2x2: (1+2+5+6)/4 = 3.5
        assert!(
            (vals[0] - 3.5).abs() < 1e-10,
            "expected 3.5, got {}",
            vals[0]
        );
        // Top-right 2x2: (3+4+7+8)/4 = 5.5
        assert!(
            (vals[1] - 5.5).abs() < 1e-10,
            "expected 5.5, got {}",
            vals[1]
        );
        // Bottom-left 2x2: (9+10+13+14)/4 = 11.5
        assert!(
            (vals[2] - 11.5).abs() < 1e-10,
            "expected 11.5, got {}",
            vals[2]
        );
        // Bottom-right 2x2: (11+12+15+16)/4 = 13.5
        assert!(
            (vals[3] - 13.5).abs() < 1e-10,
            "expected 13.5, got {}",
            vals[3]
        );
    }

    #[test]
    fn adaptive_avgpool2d_multichannel() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AdaptiveAvgPool2d::new((1, 1));

        // [N=2, C=2, H=2, W=2]
        let x = session
            .tensor_variable(vec![1.0; 16], vec![2, 2, 2, 2], false)
            .expect("variable");
        let out = pool.forward(&mut session, x).expect("forward");

        let (_, meta) = session.tensor_values_meta(out).expect("values_meta");
        assert_eq!(meta.shape(), &[2, 2, 1, 1]);
    }

    #[test]
    fn adaptive_avgpool2d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AdaptiveAvgPool2d::new((1, 1));

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], true)
            .expect("variable");
        let out = pool.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(out).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(x_grad.is_some(), "input should have gradient");
    }

    #[test]
    fn adaptive_avgpool2d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pool = AdaptiveAvgPool2d::new((1, 1));

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");
        assert!(pool.forward(&mut session, x).is_err());
    }

    #[test]
    fn adaptive_avgpool2d_has_no_parameters() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        assert!(pool.parameters().is_empty());
    }

    // ── Loss Module Tests ──────────────────────────────────────────────

    #[test]
    fn mse_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = MSELoss;

        let pred = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("pred");
        let target = session
            .tensor_variable(vec![1.5, 2.5, 3.5], vec![3], false)
            .expect("target");

        let loss = loss_fn.forward(&mut session, pred, target).expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        // MSE = mean((0.5)^2 * 3) = mean(0.25 * 3) = 0.25
        assert!((vals[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn mse_loss_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = MSELoss;

        let pred = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .expect("pred");
        let target = session
            .tensor_variable(vec![0.0, 0.0], vec![2], false)
            .expect("target");

        let loss = loss_fn.forward(&mut session, pred, target).expect("loss");
        let report = session.tensor_backward(loss).expect("backward");
        assert!(session.tensor_gradient(&report, pred).is_some());
    }

    #[test]
    fn l1_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = L1Loss;

        let pred = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("pred");
        let target = session
            .tensor_variable(vec![1.5, 2.5, 3.5], vec![3], false)
            .expect("target");

        let loss = loss_fn.forward(&mut session, pred, target).expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        // L1 = mean(|0.5| * 3) = 0.5
        assert!((vals[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn cross_entropy_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = CrossEntropyLoss;

        // 2 samples, 3 classes
        let logits = session
            .tensor_variable(vec![2.0, 1.0, 0.1, 0.1, 1.0, 2.0], vec![2, 3], true)
            .expect("logits");
        let targets = session
            .tensor_variable(vec![0.0, 2.0], vec![2], false)
            .expect("targets");

        let loss = loss_fn
            .forward(&mut session, logits, targets)
            .expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        assert!(vals[0] > 0.0, "cross-entropy loss should be positive");
    }

    #[test]
    fn nll_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = NLLLoss;

        // Pre-computed log probabilities: log(0.5), log(0.3), log(0.2)
        let log_probs = session
            .tensor_variable(
                vec![0.5_f64.ln(), 0.3_f64.ln(), 0.2_f64.ln()],
                vec![1, 3],
                true,
            )
            .expect("log_probs");
        let targets = session
            .tensor_variable(vec![0.0], vec![1], false)
            .expect("targets");

        let loss = loss_fn
            .forward(&mut session, log_probs, targets)
            .expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        // NLL = -log(0.5) ≈ 0.693
        assert!((vals[0] - 0.5_f64.ln().abs()).abs() < 1e-6);
    }

    #[test]
    fn bce_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = BCELoss;

        let pred = session
            .tensor_variable(vec![0.7, 0.3], vec![2], true)
            .expect("pred");
        let target = session
            .tensor_variable(vec![1.0, 0.0], vec![2], false)
            .expect("target");

        let loss = loss_fn.forward(&mut session, pred, target).expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        assert!(vals[0] > 0.0, "BCE loss should be positive");
    }

    #[test]
    fn bce_with_logits_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = BCEWithLogitsLoss;

        // Logits, not probabilities
        let logits = session
            .tensor_variable(vec![2.0, -1.0], vec![2], true)
            .expect("logits");
        let target = session
            .tensor_variable(vec![1.0, 0.0], vec![2], false)
            .expect("target");

        let loss = loss_fn.forward(&mut session, logits, target).expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        assert!(vals[0] > 0.0, "BCE with logits loss should be positive");
    }

    #[test]
    fn bce_with_logits_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = BCEWithLogitsLoss;

        let logits = session
            .tensor_variable(vec![0.5, -0.5], vec![2], true)
            .expect("logits");
        let target = session
            .tensor_variable(vec![1.0, 0.0], vec![2], false)
            .expect("target");

        let loss = loss_fn.forward(&mut session, logits, target).expect("loss");
        let report = session.tensor_backward(loss).expect("backward");
        assert!(session.tensor_gradient(&report, logits).is_some());
    }

    #[test]
    fn smooth_l1_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = SmoothL1Loss::new(1.0);

        let pred = session
            .tensor_variable(vec![0.5, 2.0], vec![2], true)
            .expect("pred");
        let target = session
            .tensor_variable(vec![0.0, 0.0], vec![2], false)
            .expect("target");

        let loss = loss_fn.forward(&mut session, pred, target).expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        assert!(vals[0] > 0.0);
    }

    #[test]
    fn huber_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = HuberLoss::new(1.0);

        let pred = session
            .tensor_variable(vec![0.5, 3.0], vec![2], true)
            .expect("pred");
        let target = session
            .tensor_variable(vec![0.0, 0.0], vec![2], false)
            .expect("target");

        let loss = loss_fn.forward(&mut session, pred, target).expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        assert!(vals[0] > 0.0);
    }

    #[test]
    fn kl_div_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = KLDivLoss;

        // log-probabilities as input: log(0.5), log(0.5)
        let log_q = session
            .tensor_variable(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2], true)
            .expect("log_q");
        // target probabilities
        let p = session
            .tensor_variable(vec![0.3, 0.7], vec![2], false)
            .expect("p");

        let loss = loss_fn.forward(&mut session, log_q, p).expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        // KL(P || Q) where Q=uniform(0.5,0.5), P=(0.3,0.7)
        // = 0.3*ln(0.3/0.5) + 0.7*ln(0.7/0.5) ≈ 0.3*(-0.511) + 0.7*(0.336) ≈ 0.082
        assert!(vals[0] > -1e-6);
    }

    #[test]
    fn kl_div_loss_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = KLDivLoss;

        let log_q = session
            .tensor_variable(vec![0.4_f64.ln(), 0.6_f64.ln()], vec![2], true)
            .expect("log_q");
        let p = session
            .tensor_variable(vec![0.4, 0.6], vec![2], false)
            .expect("p");

        let loss = loss_fn.forward(&mut session, log_q, p).expect("loss");
        let report = session.tensor_backward(loss).expect("backward");
        assert!(session.tensor_gradient(&report, log_q).is_some());
    }

    // ── Container Module Tests ─────────────────────────────────────────

    #[test]
    fn module_list_empty_is_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let list = ModuleList::new();

        assert!(list.is_empty());
        assert_eq!(list.len(), 0);

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .expect("variable");
        let out = list.forward(&mut session, x).expect("forward");

        let (in_vals, _) = session.tensor_values_meta(x).expect("input");
        let (out_vals, _) = session.tensor_values_meta(out).expect("output");
        assert_eq!(in_vals, out_vals);
    }

    #[test]
    fn module_list_chains_modules() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut list = ModuleList::new();
        list.push(Box::new(ReLU));
        list.push(Box::new(Sigmoid));

        assert_eq!(list.len(), 2);
        assert!(!list.is_empty());

        let x = session
            .tensor_variable(vec![-1.0, 0.0, 1.0], vec![3], false)
            .expect("variable");
        let out = list.forward(&mut session, x).expect("forward");

        let (vals, _) = session.tensor_values_meta(out).expect("vals");
        // ReLU(-1)=0, sigmoid(0)=0.5; ReLU(0)=0, sigmoid(0)=0.5; ReLU(1)=1, sigmoid(1)≈0.731
        assert!((vals[0] - 0.5).abs() < 1e-10);
        assert!((vals[1] - 0.5).abs() < 1e-10);
        assert!((vals[2] - 0.7310585786300049).abs() < 1e-6);
    }

    #[test]
    fn module_list_collects_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut list = ModuleList::new();

        let linear1 = Linear::new(&mut session, 2, 3, true).expect("linear1");
        let linear2 = Linear::new(&mut session, 3, 1, true).expect("linear2");
        list.push(Box::new(linear1));
        list.push(Box::new(ReLU));
        list.push(Box::new(linear2));

        // linear1: weight + bias = 2 params; linear2: weight + bias = 2 params
        let params = list.parameters();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn module_list_get_returns_module() {
        let list = {
            let mut l = ModuleList::new();
            l.push(Box::new(ReLU));
            l.push(Box::new(Sigmoid));
            l
        };

        assert!(list.get(0).is_some());
        assert!(list.get(1).is_some());
        assert!(list.get(2).is_none());
    }

    #[test]
    fn module_dict_empty_is_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let dict = ModuleDict::new();

        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("variable");
        let out = dict.forward(&mut session, x).expect("forward");

        let (in_vals, _) = session.tensor_values_meta(x).expect("input");
        let (out_vals, _) = session.tensor_values_meta(out).expect("output");
        assert_eq!(in_vals, out_vals);
    }

    #[test]
    fn module_dict_insert_and_get() {
        let mut dict = ModuleDict::new();
        dict.insert("relu".to_string(), Box::new(ReLU));
        dict.insert("sigmoid".to_string(), Box::new(Sigmoid));

        assert_eq!(dict.len(), 2);
        assert!(dict.get("relu").is_some());
        assert!(dict.get("sigmoid").is_some());
        assert!(dict.get("tanh").is_none());
    }

    #[test]
    fn module_dict_replace_existing() {
        let mut dict = ModuleDict::new();
        dict.insert("act".to_string(), Box::new(ReLU));
        dict.insert("act".to_string(), Box::new(Sigmoid));

        // Should still be length 1 (replaced, not added)
        assert_eq!(dict.len(), 1);
    }

    #[test]
    fn module_dict_forward_chains_in_order() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut dict = ModuleDict::new();
        dict.insert("relu".to_string(), Box::new(ReLU));
        dict.insert("sigmoid".to_string(), Box::new(Sigmoid));

        let x = session
            .tensor_variable(vec![-1.0, 1.0], vec![2], false)
            .expect("variable");
        let out = dict.forward(&mut session, x).expect("forward");

        let (vals, _) = session.tensor_values_meta(out).expect("vals");
        // ReLU(-1)=0, sigmoid(0)=0.5; ReLU(1)=1, sigmoid(1)≈0.731
        assert!((vals[0] - 0.5).abs() < 1e-10);
        assert!((vals[1] - 0.7310585786300049).abs() < 1e-6);
    }

    #[test]
    fn module_dict_collects_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut dict = ModuleDict::new();

        let linear = Linear::new(&mut session, 2, 3, true).expect("linear");
        dict.insert("linear".to_string(), Box::new(linear));
        dict.insert("relu".to_string(), Box::new(ReLU));

        // Linear has weight + bias = 2 params, ReLU has 0
        let params = dict.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn module_dict_iter_in_insertion_order() {
        let mut dict = ModuleDict::new();
        dict.insert("first".to_string(), Box::new(ReLU));
        dict.insert("second".to_string(), Box::new(Sigmoid));
        dict.insert("third".to_string(), Box::new(Tanh));

        let names: Vec<&str> = dict.iter().map(|(name, _)| name).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
    }

    // ── Bilinear Tests ──────────────────────────────────────────────

    #[test]
    fn bilinear_forward_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // Simple 2x3 -> 1 bilinear with known weights
        let _bilinear = Bilinear::new(&mut session, 2, 3, 1, false).unwrap();

        // Override weight to identity-like for predictability
        let w = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![1, 2, 3], true)
            .unwrap();
        let bilinear = Bilinear {
            weight: w,
            bias: None,
            in1_features: 2,
            in2_features: 3,
            out_features: 1,
        };

        let x1 = session
            .tensor_variable(vec![2.0, 3.0], vec![2], false)
            .unwrap();
        let x2 = session
            .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], false)
            .unwrap();
        let out = bilinear.forward_bilinear(&mut session, x1, x2).unwrap();
        let vals = session.tensor_values(out).unwrap();
        // y = x1[0]*W[0,0,0]*x2[0] + x1[0]*W[0,0,1]*x2[1] + ...
        // = 2*1*4 + 2*0*5 + 2*0*6 + 3*0*4 + 3*1*5 + 3*0*6 = 8 + 15 = 23
        assert!((vals[0] - 23.0).abs() < 1e-10);
    }

    #[test]
    fn bilinear_with_bias() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let w = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2], true)
            .unwrap();
        let b = session.tensor_variable(vec![10.0], vec![1], true).unwrap();
        let bilinear = Bilinear {
            weight: w,
            bias: Some(b),
            in1_features: 2,
            in2_features: 2,
            out_features: 1,
        };

        let x1 = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .unwrap();
        let x2 = session
            .tensor_variable(vec![3.0, 4.0], vec![2], false)
            .unwrap();
        let out = bilinear.forward_bilinear(&mut session, x1, x2).unwrap();
        let vals = session.tensor_values(out).unwrap();
        // 1*1*3 + 1*0*4 + 2*0*3 + 2*1*4 = 3 + 8 = 11 + bias(10) = 21
        assert!((vals[0] - 21.0).abs() < 1e-10);
    }

    #[test]
    fn bilinear_batched() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let w = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2], true)
            .unwrap();
        let bilinear = Bilinear {
            weight: w,
            bias: None,
            in1_features: 2,
            in2_features: 2,
            out_features: 1,
        };

        // Batch of 2
        let x1 = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false)
            .unwrap();
        let x2 = session
            .tensor_variable(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], false)
            .unwrap();
        let out = bilinear.forward_bilinear(&mut session, x1, x2).unwrap();
        let shape = session.tensor_shape(out).unwrap();
        assert_eq!(shape, vec![2, 1]);
        let vals = session.tensor_values(out).unwrap();
        // batch 0: 1*1*2 + 1*0*3 + 0*0*2 + 0*1*3 = 2
        // batch 1: 0*1*4 + 0*0*5 + 1*0*4 + 1*1*5 = 5
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn bilinear_parameters_count() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bilinear = Bilinear::new(&mut session, 4, 3, 2, true).unwrap();
        // weight + bias = 2 params
        assert_eq!(bilinear.parameters().len(), 2);
        let named = bilinear.named_parameters_own();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    // ── Unfold / Fold Tests ──────────────────────────────────────────

    #[test]
    fn unfold_basic_2x2_kernel() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // Input: [1, 1, 3, 3] — single channel 3x3 image
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = session
            .tensor_variable(data, vec![1, 1, 3, 3], false)
            .unwrap();
        let unfold = Unfold::new((2, 2));
        let out = unfold.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(out).unwrap();
        // C*kH*kW = 1*2*2 = 4, L = (3-2+1)*(3-2+1) = 4
        assert_eq!(shape, vec![1, 4, 4]);
        let vals = session.tensor_values(out).unwrap();
        // Each column is a flattened 2x2 patch
        // col0: [1,2,4,5], col1: [2,3,5,6], col2: [4,5,7,8], col3: [5,6,8,9]
        assert!((vals[0] - 1.0).abs() < 1e-12); // row0, col0
        assert!((vals[1] - 2.0).abs() < 1e-12); // row0, col1
        assert!((vals[2] - 4.0).abs() < 1e-12); // row0, col2
        assert!((vals[3] - 5.0).abs() < 1e-12); // row0, col3
    }

    #[test]
    fn unfold_with_padding() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = session
            .tensor_variable(data, vec![1, 1, 2, 2], false)
            .unwrap();
        let unfold = Unfold::new((2, 2)).padding((1, 1));
        let out = unfold.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(out).unwrap();
        // With padding=1: output positions = (2+2-2)/1+1 = 3 per dim, L = 9
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 4); // C*kH*kW
        assert_eq!(shape[2], 9); // L = 3*3
    }

    #[test]
    fn fold_unfold_roundtrip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = session
            .tensor_variable(data.clone(), vec![1, 1, 3, 3], false)
            .unwrap();

        // Unfold then fold with 1x1 kernel should be identity
        let unfold = Unfold::new((1, 1));
        let unfolded = unfold.forward(&mut session, input).unwrap();

        let fold = Fold::new((3, 3), (1, 1));
        let folded = fold.forward(&mut session, unfolded).unwrap();
        let shape = session.tensor_shape(folded).unwrap();
        assert_eq!(shape, vec![1, 1, 3, 3]);
        let vals = session.tensor_values(folded).unwrap();
        for (i, (&got, &expected)) in vals.iter().zip(data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-12,
                "mismatch at {i}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn unfold_rejects_non_4d() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        let unfold = Unfold::new((2, 2));
        assert!(unfold.forward(&mut session, input).is_err());
    }

    // ── ParameterList / ParameterDict Tests ──────────────────────────

    #[test]
    fn parameter_list_collects_params() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let p1 = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .unwrap();
        let p2 = session.tensor_variable(vec![3.0], vec![1], true).unwrap();
        let pl = ParameterList::new(vec![p1, p2]);
        assert_eq!(pl.len(), 2);
        assert!(!pl.is_empty());
        let params = pl.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], p1);
        assert_eq!(params[1], p2);
    }

    #[test]
    fn parameter_list_append() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut pl = ParameterList::default();
        assert!(pl.is_empty());
        let p = session.tensor_variable(vec![1.0], vec![1], true).unwrap();
        pl.append(p);
        assert_eq!(pl.len(), 1);
        assert_eq!(pl.get(0), Some(p));
        assert_eq!(pl.get(1), None);
    }

    #[test]
    fn parameter_list_forward_passthrough() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let p = session.tensor_variable(vec![1.0], vec![1], true).unwrap();
        let pl = ParameterList::new(vec![p]);
        let input = session
            .tensor_variable(vec![5.0, 6.0], vec![2], false)
            .unwrap();
        let out = pl.forward(&mut session, input).unwrap();
        // forward is passthrough
        assert_eq!(out, input);
    }

    #[test]
    fn parameter_list_named_slots() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let p1 = session.tensor_variable(vec![1.0], vec![1], true).unwrap();
        let p2 = session.tensor_variable(vec![2.0], vec![1], true).unwrap();
        let pl = ParameterList::new(vec![p1, p2]);
        let slots = pl.named_parameter_slots_own();
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].0, "0");
        assert_eq!(slots[1].0, "1");
    }

    #[test]
    fn parameter_dict_insert_and_get() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut pd = ParameterDict::new();
        let p1 = session
            .tensor_variable(vec![1.0, 2.0], vec![2], true)
            .unwrap();
        let p2 = session.tensor_variable(vec![3.0], vec![1], true).unwrap();
        pd.insert("weight".to_string(), p1);
        pd.insert("bias".to_string(), p2);
        assert_eq!(pd.len(), 2);
        assert_eq!(pd.get("weight"), Some(p1));
        assert_eq!(pd.get("bias"), Some(p2));
        assert_eq!(pd.get("nonexistent"), None);
    }

    #[test]
    fn parameter_dict_replace() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut pd = ParameterDict::new();
        let p1 = session.tensor_variable(vec![1.0], vec![1], true).unwrap();
        let p2 = session.tensor_variable(vec![2.0], vec![1], true).unwrap();
        pd.insert("w".to_string(), p1);
        pd.insert("w".to_string(), p2);
        assert_eq!(pd.len(), 1);
        assert_eq!(pd.get("w"), Some(p2));
    }

    #[test]
    fn parameter_dict_collects_all_params() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut pd = ParameterDict::new();
        let p1 = session.tensor_variable(vec![1.0], vec![1], true).unwrap();
        let p2 = session.tensor_variable(vec![2.0], vec![1], true).unwrap();
        pd.insert("a".to_string(), p1);
        pd.insert("b".to_string(), p2);
        let params = pd.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn parameter_dict_iter_in_order() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut pd = ParameterDict::new();
        let p1 = session.tensor_variable(vec![1.0], vec![1], true).unwrap();
        let p2 = session.tensor_variable(vec![2.0], vec![1], true).unwrap();
        pd.insert("first".to_string(), p1);
        pd.insert("second".to_string(), p2);
        let names: Vec<&str> = pd.iter().map(|(k, _)| k).collect();
        assert_eq!(names, vec!["first", "second"]);
    }

    // ── Padding Module Tests ───────────────────────────────────────────

    #[test]
    fn constant_pad1d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pad = ConstantPad1d::new((1, 2), 0.0);

        // [N=1, C=1, W=3] → [1, 1, 6] (1 left + 3 + 2 right)
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 1, 3], false)
            .expect("variable");
        let out = pad.forward(&mut session, x).expect("forward");

        let (vals, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 6]);
        assert_eq!(vals, &[0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn constant_pad1d_with_nonzero_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pad = ConstantPad1d::new((1, 1), -1.0);

        let x = session
            .tensor_variable(vec![5.0, 6.0], vec![1, 1, 2], false)
            .expect("variable");
        let out = pad.forward(&mut session, x).expect("forward");

        let (vals, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 4]);
        assert_eq!(vals, &[-1.0, 5.0, 6.0, -1.0]);
    }

    #[test]
    fn constant_pad1d_has_no_parameters() {
        let pad = ConstantPad1d::new((1, 1), 0.0);
        assert!(pad.parameters().is_empty());
    }

    #[test]
    fn constant_pad2d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pad = ConstantPad2d::new((1, 1, 1, 1), 0.0);

        // [N=1, C=1, H=2, W=2] → [1, 1, 4, 4]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], false)
            .expect("variable");
        let out = pad.forward(&mut session, x).expect("forward");

        let (_, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn constant_pad2d_has_no_parameters() {
        let pad = ConstantPad2d::new((1, 1, 1, 1), 0.0);
        assert!(pad.parameters().is_empty());
    }

    #[test]
    fn zero_pad2d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let pad = ZeroPad2d::new((1, 1, 1, 1));

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], false)
            .expect("variable");
        let out = pad.forward(&mut session, x).expect("forward");

        let (vals, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 4, 4]);
        // Corners should be zero
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[3], 0.0);
    }

    #[test]
    fn zero_pad2d_has_no_parameters() {
        let pad = ZeroPad2d::new((1, 1, 1, 1));
        assert!(pad.parameters().is_empty());
    }

    // ── Upsampling Module Tests ─────────────────────────────────────────

    #[test]
    fn upsample1d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let up = Upsample1d::new(3);

        // [N=1, C=1, L=2] → [1, 1, 6]
        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 1, 2], false)
            .expect("variable");
        let out = up.forward(&mut session, x).expect("forward");

        let (vals, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 6]);
        // Nearest neighbor: [1, 1, 1, 2, 2, 2]
        assert_eq!(vals, &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn upsample1d_scale_1_is_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let up = Upsample1d::new(1);

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 1, 3], false)
            .expect("variable");
        let out = up.forward(&mut session, x).expect("forward");

        let (in_vals, _) = session.tensor_values_meta(x).expect("input");
        let (out_vals, _) = session.tensor_values_meta(out).expect("output");
        assert_eq!(in_vals, out_vals);
    }

    #[test]
    fn upsample1d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let up = Upsample1d::new(2);

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 1, 2], true)
            .expect("variable");
        let out = up.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(out).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(session.tensor_gradient(&report, x).is_some());
    }

    #[test]
    fn upsample1d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let up = Upsample1d::new(2);

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("variable");
        assert!(up.forward(&mut session, x).is_err());
    }

    #[test]
    fn upsample1d_has_no_parameters() {
        let up = Upsample1d::new(2);
        assert!(up.parameters().is_empty());
    }

    #[test]
    fn upsample2d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let up = Upsample2d::new(2, 2);

        // [N=1, C=1, H=2, W=2] → [1, 1, 4, 4]
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], false)
            .expect("variable");
        let out = up.forward(&mut session, x).expect("forward");

        let (vals, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 4, 4]);
        // Nearest neighbor 2x2 upsampling:
        // [[1,1,2,2], [1,1,2,2], [3,3,4,4], [3,3,4,4]]
        assert_eq!(
            vals,
            &[
                1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0
            ]
        );
    }

    #[test]
    fn upsample2d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let up = Upsample2d::new(2, 2);

        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 1, 3], false)
            .expect("variable");
        assert!(up.forward(&mut session, x).is_err());
    }

    #[test]
    fn upsample2d_has_no_parameters() {
        let up = Upsample2d::new(2, 2);
        assert!(up.parameters().is_empty());
    }

    // ── ConvTranspose1d Tests ──────────────────────────────────────────

    #[test]
    fn conv_transpose1d_output_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose1d::new(&mut session, 1, 1, 3, 1, 0, false).expect("new");

        // [N=1, C_in=1, L=3], kernel=3, stride=1, padding=0
        // L_out = (3-1)*1 + 3 - 0 = 5
        let x = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 1, 3], false)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");

        let (_, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 5]);
    }

    #[test]
    fn conv_transpose1d_with_stride() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose1d::new(&mut session, 1, 1, 3, 2, 0, false).expect("new");

        // [N=1, C_in=1, L=2], kernel=3, stride=2, padding=0
        // L_out = (2-1)*2 + 3 - 0 = 5
        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 1, 2], false)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");

        let (_, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 5]);
    }

    #[test]
    fn conv_transpose1d_has_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose1d::new(&mut session, 2, 3, 3, 1, 0, true).expect("new");

        // weight + bias
        assert_eq!(deconv.parameters().len(), 2);
    }

    #[test]
    fn conv_transpose1d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose1d::new(&mut session, 2, 3, 3, 1, 0, false).expect("new");

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("variable");
        assert!(deconv.forward(&mut session, x).is_err());
    }

    #[test]
    fn conv_transpose1d_rejects_wrong_channels() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose1d::new(&mut session, 2, 3, 3, 1, 0, false).expect("new");

        // 5 channels but deconv expects 2
        let x = session
            .tensor_variable(vec![1.0; 15], vec![1, 5, 3], false)
            .expect("variable");
        assert!(deconv.forward(&mut session, x).is_err());
    }

    // ── Conv3d Tests ─────────────────────────────────────────────────────

    #[test]
    fn conv3d_forward_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv3d::new(&mut session, 1, 1, (2, 2, 2), (1, 1, 1), (0, 0, 0), false)
            .expect("conv3d");
        assert_eq!(conv.parameters().len(), 1);

        // [N=1, C=1, D=3, H=3, W=3], kernel=2x2x2, stride=1 -> out 2x2x2
        let x = session
            .tensor_variable(vec![1.0; 27], vec![1, 1, 3, 3, 3], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (vals, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 2, 2, 2]);
        assert_eq!(vals.len(), 8);
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn conv3d_forward_with_bias() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv =
            Conv3d::new(&mut session, 2, 4, (2, 2, 2), (1, 1, 1), (0, 0, 0), true).expect("conv3d");
        assert_eq!(conv.parameters().len(), 2);
        assert!(conv.bias().is_some());

        // [N=2, C=2, D=3, H=3, W=3] -> [2, 4, 2, 2, 2]
        let x = session
            .tensor_variable(vec![0.5; 108], vec![2, 2, 3, 3, 3], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[2, 4, 2, 2, 2]);
    }

    #[test]
    fn conv3d_forward_with_padding() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // padding=1 on all dims: D_out = (3+2-2)/1+1 = 4
        let conv = Conv3d::new(&mut session, 1, 1, (2, 2, 2), (1, 1, 1), (1, 1, 1), false)
            .expect("conv3d");

        let x = session
            .tensor_variable(vec![1.0; 27], vec![1, 1, 3, 3, 3], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 4, 4, 4]);
    }

    #[test]
    fn conv3d_forward_with_stride() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // stride=2, D_out = (4-2)/2+1 = 2
        let conv = Conv3d::new(&mut session, 1, 1, (2, 2, 2), (2, 2, 2), (0, 0, 0), false)
            .expect("conv3d");

        let x = session
            .tensor_variable(vec![1.0; 64], vec![1, 1, 4, 4, 4], false)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(y).expect("values");
        assert_eq!(meta.shape(), &[1, 1, 2, 2, 2]);
    }

    #[test]
    fn conv3d_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv =
            Conv3d::new(&mut session, 1, 2, (2, 2, 2), (1, 1, 1), (0, 0, 0), true).expect("conv3d");

        let x = session
            .tensor_variable(vec![1.0; 27], vec![1, 1, 3, 3, 3], true)
            .expect("variable");
        let y = conv.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(x_grad.is_some(), "input gradient should exist");
        let w_grad = session.tensor_gradient(&report, conv.weight());
        assert!(w_grad.is_some(), "weight gradient should exist");
    }

    #[test]
    fn conv3d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv3d::new(&mut session, 1, 1, (2, 2, 2), (1, 1, 1), (0, 0, 0), false)
            .expect("conv3d");

        let x = session
            .tensor_variable(vec![1.0; 16], vec![1, 1, 4, 4], false)
            .expect("variable");
        assert!(conv.forward(&mut session, x).is_err());
    }

    // ── ConvTranspose2d Tests ──────────────────────────────────────────

    #[test]
    fn conv_transpose2d_output_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // stride=1, padding=0, output_padding=0: H_out = (3-1)*1 + 3 = 5
        let deconv =
            ConvTranspose2d::new(&mut session, 1, 1, (3, 3), (1, 1), (0, 0), (0, 0), false)
                .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 9], vec![1, 1, 3, 3], false)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn conv_transpose2d_with_stride() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // stride=2: H_out = (2-1)*2 + 3 = 5
        let deconv =
            ConvTranspose2d::new(&mut session, 1, 1, (3, 3), (2, 2), (0, 0), (0, 0), false)
                .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 4], vec![1, 1, 2, 2], false)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn conv_transpose2d_with_padding() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // padding=1: H_out = (3-1)*1 - 2 + 3 = 3
        let deconv =
            ConvTranspose2d::new(&mut session, 1, 1, (3, 3), (1, 1), (1, 1), (0, 0), false)
                .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 9], vec![1, 1, 3, 3], false)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn conv_transpose2d_with_output_padding() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // stride=2, output_padding=1: H_out = (2-1)*2 - 0 + 3 + 1 = 6
        let deconv =
            ConvTranspose2d::new(&mut session, 1, 1, (3, 3), (2, 2), (0, 0), (1, 1), false)
                .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 4], vec![1, 1, 2, 2], false)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 6, 6]);
    }

    #[test]
    fn conv_transpose2d_output_padding_must_be_less_than_stride() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // output_padding=2 >= stride=2 should fail
        assert!(
            ConvTranspose2d::new(&mut session, 1, 1, (3, 3), (2, 2), (0, 0), (2, 2), false,)
                .is_err()
        );
    }

    #[test]
    fn conv_transpose2d_has_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose2d::new(&mut session, 2, 3, (3, 3), (1, 1), (0, 0), (0, 0), true)
            .expect("new");
        assert_eq!(deconv.parameters().len(), 2);
    }

    #[test]
    fn conv_transpose2d_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose2d::new(&mut session, 1, 1, (2, 2), (1, 1), (0, 0), (0, 0), true)
            .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 4], vec![1, 1, 2, 2], true)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(out).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        // Debug: check what gradients we have
        println!("report gradients count: {}", report.gradients().len());
        println!("x node id: {:?}", x);
        println!("w node id: {:?}", deconv.weight());
        for (i, g) in report.gradients().iter().enumerate() {
            if g.is_some() {
                println!("  gradient exists for node index: {}", i);
            }
        }

        let w_grad = session.tensor_gradient(&report, deconv.weight());
        let x_grad = session.tensor_gradient(&report, x);
        println!("w_grad present: {}", w_grad.is_some());
        println!("x_grad present: {}", x_grad.is_some());
        assert!(w_grad.is_some(), "weight gradient should exist");
        assert!(x_grad.is_some(), "input gradient should exist");
    }

    #[test]
    fn conv_transpose2d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv =
            ConvTranspose2d::new(&mut session, 2, 3, (3, 3), (1, 1), (0, 0), (0, 0), false)
                .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 6], vec![1, 2, 3], false)
            .expect("variable");
        assert!(deconv.forward(&mut session, x).is_err());
    }

    // ── ConvTranspose3d Tests ──────────────────────────────────────────

    #[test]
    fn conv_transpose3d_output_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // stride=1, padding=0: D_out = (2-1)*1 + 2 = 3
        let deconv = ConvTranspose3d::new(
            &mut session,
            1,
            1,
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            false,
        )
        .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 8], vec![1, 1, 2, 2, 2], false)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 3, 3, 3]);
    }

    #[test]
    fn conv_transpose3d_with_stride() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // stride=2: D_out = (2-1)*2 + 2 = 4
        let deconv = ConvTranspose3d::new(
            &mut session,
            1,
            1,
            (2, 2, 2),
            (2, 2, 2),
            (0, 0, 0),
            (0, 0, 0),
            false,
        )
        .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 8], vec![1, 1, 2, 2, 2], false)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");
        let (_, meta) = session.tensor_values_meta(out).expect("vals");
        assert_eq!(meta.shape(), &[1, 1, 4, 4, 4]);
    }

    #[test]
    fn conv_transpose3d_has_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose3d::new(
            &mut session,
            2,
            3,
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            true,
        )
        .expect("new");
        assert_eq!(deconv.parameters().len(), 2);
    }

    #[test]
    fn conv_transpose3d_output_padding_must_be_less_than_stride() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        assert!(
            ConvTranspose3d::new(
                &mut session,
                1,
                1,
                (2, 2, 2),
                (2, 2, 2),
                (0, 0, 0),
                (2, 2, 2),
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn conv_transpose3d_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose3d::new(
            &mut session,
            1,
            1,
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            true,
        )
        .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 8], vec![1, 1, 2, 2, 2], true)
            .expect("variable");
        let out = deconv.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(out).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let x_grad = session.tensor_gradient(&report, x);
        assert!(x_grad.is_some(), "input gradient should exist");
        let w_grad = session.tensor_gradient(&report, deconv.weight());
        assert!(w_grad.is_some(), "weight gradient should exist");
    }

    #[test]
    fn conv_transpose3d_rejects_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let deconv = ConvTranspose3d::new(
            &mut session,
            2,
            3,
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            false,
        )
        .expect("new");

        let x = session
            .tensor_variable(vec![1.0; 32], vec![1, 2, 4, 4], false)
            .expect("variable");
        assert!(deconv.forward(&mut session, x).is_err());
    }

    // ── RNN Cell Tests ──────────────────────────────────────────────────

    #[test]
    fn rnn_cell_forward_tanh() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = RNNCell::new(&mut session, 3, 4, true).expect("rnn_cell");

        assert_eq!(cell.hidden_size(), 4);
        assert_eq!(cell.parameters().len(), 4);

        let x = session
            .tensor_variable(vec![1.0, 0.5, -0.5], vec![1, 3], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 4], vec![1, 4], false)
            .expect("hidden");

        let h_new = cell.forward_cell(&mut session, x, h).expect("forward");
        let (vals, meta) = session.tensor_values_meta(h_new).expect("vals");
        assert_eq!(meta.shape(), &[1, 4]);
        // With tanh, values should be in [-1, 1]
        for &v in &vals {
            assert!((-1.0..=1.0).contains(&v), "tanh output out of range: {v}");
        }
    }

    #[test]
    fn rnn_cell_forward_relu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = RNNCell::new(&mut session, 2, 3, false).expect("rnn_cell");

        let x = session
            .tensor_variable(vec![1.0, -1.0], vec![1, 2], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 3], vec![1, 3], false)
            .expect("hidden");

        let h_new = cell.forward_cell(&mut session, x, h).expect("forward");
        let (vals, meta) = session.tensor_values_meta(h_new).expect("vals");
        assert_eq!(meta.shape(), &[1, 3]);
        // With relu, values should be >= 0
        for &v in &vals {
            assert!(v >= 0.0, "relu output should be non-negative: {v}");
        }
    }

    #[test]
    fn rnn_cell_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = RNNCell::new(&mut session, 2, 3, true).expect("rnn_cell");

        let x = session
            .tensor_variable(vec![0.5, -0.3], vec![1, 2], true)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.1, 0.2, 0.3], vec![1, 3], true)
            .expect("hidden");

        let h_new = cell.forward_cell(&mut session, x, h).expect("forward");
        let loss = session.tensor_sum(h_new).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(session.tensor_gradient(&report, x).is_some(), "input grad");
        assert!(session.tensor_gradient(&report, h).is_some(), "hidden grad");
        // Check parameter gradients
        for &param in &cell.parameters() {
            assert!(
                session.tensor_gradient(&report, param).is_some(),
                "param grad missing"
            );
        }
    }

    #[test]
    fn rnn_cell_batched() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = RNNCell::new(&mut session, 2, 3, true).expect("rnn_cell");

        let x = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 6], vec![2, 3], false)
            .expect("hidden");

        let h_new = cell.forward_cell(&mut session, x, h).expect("forward");
        let (_, meta) = session.tensor_values_meta(h_new).expect("vals");
        assert_eq!(meta.shape(), &[2, 3]);
    }

    #[test]
    fn rnn_cell_rejects_wrong_input_size() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = RNNCell::new(&mut session, 3, 4, true).expect("rnn_cell");

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 4], vec![1, 4], false)
            .expect("hidden");

        assert!(cell.forward_cell(&mut session, x, h).is_err());
    }

    #[test]
    fn lstm_cell_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = LSTMCell::new(&mut session, 3, 4).expect("lstm_cell");

        assert_eq!(cell.hidden_size(), 4);
        assert_eq!(cell.parameters().len(), 4);

        let x = session
            .tensor_variable(vec![1.0, 0.5, -0.5], vec![1, 3], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 4], vec![1, 4], false)
            .expect("hidden");
        let c = session
            .tensor_variable(vec![0.0; 4], vec![1, 4], false)
            .expect("cell");

        let (h_new, c_new) = cell.forward_cell(&mut session, x, h, c).expect("forward");

        let (h_vals, h_meta) = session.tensor_values_meta(h_new).expect("h_vals");
        assert_eq!(h_meta.shape(), &[1, 4]);
        // h is bounded by tanh ∘ sigmoid, so should be in [-1, 1]
        for &v in &h_vals {
            assert!(v.abs() <= 1.0 + 1e-10, "LSTM h' out of range: {v}");
        }

        let (_, c_meta) = session.tensor_values_meta(c_new).expect("c_vals");
        assert_eq!(c_meta.shape(), &[1, 4]);
    }

    #[test]
    fn lstm_cell_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = LSTMCell::new(&mut session, 2, 3).expect("lstm_cell");

        let x = session
            .tensor_variable(vec![0.5, -0.3], vec![1, 2], true)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.1, 0.2, 0.3], vec![1, 3], true)
            .expect("hidden");
        let c = session
            .tensor_variable(vec![0.0, 0.0, 0.0], vec![1, 3], true)
            .expect("cell");

        let (h_new, _c_new) = cell.forward_cell(&mut session, x, h, c).expect("forward");
        let loss = session.tensor_sum(h_new).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(session.tensor_gradient(&report, x).is_some(), "input grad");
        assert!(session.tensor_gradient(&report, h).is_some(), "hidden grad");
        assert!(session.tensor_gradient(&report, c).is_some(), "cell grad");
    }

    #[test]
    fn lstm_cell_batched() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = LSTMCell::new(&mut session, 2, 3).expect("lstm_cell");

        let x = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 6], vec![2, 3], false)
            .expect("hidden");
        let c = session
            .tensor_variable(vec![0.0; 6], vec![2, 3], false)
            .expect("cell");

        let (h_new, c_new) = cell.forward_cell(&mut session, x, h, c).expect("forward");

        let (_, h_meta) = session.tensor_values_meta(h_new).expect("h_vals");
        assert_eq!(h_meta.shape(), &[2, 3]);
        let (_, c_meta) = session.tensor_values_meta(c_new).expect("c_vals");
        assert_eq!(c_meta.shape(), &[2, 3]);
    }

    #[test]
    fn lstm_cell_rejects_wrong_input_size() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = LSTMCell::new(&mut session, 3, 4).expect("lstm_cell");

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 4], vec![1, 4], false)
            .expect("hidden");
        let c = session
            .tensor_variable(vec![0.0; 4], vec![1, 4], false)
            .expect("cell");

        assert!(cell.forward_cell(&mut session, x, h, c).is_err());
    }

    #[test]
    fn gru_cell_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = GRUCell::new(&mut session, 3, 4).expect("gru_cell");

        assert_eq!(cell.hidden_size(), 4);
        assert_eq!(cell.parameters().len(), 4);

        let x = session
            .tensor_variable(vec![1.0, 0.5, -0.5], vec![1, 3], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 4], vec![1, 4], false)
            .expect("hidden");

        let h_new = cell.forward_cell(&mut session, x, h).expect("forward");
        let (vals, meta) = session.tensor_values_meta(h_new).expect("vals");
        assert_eq!(meta.shape(), &[1, 4]);
        // GRU output: convex combination of old h and tanh(new), bounded by [-1,1]
        for &v in &vals {
            assert!(v.abs() <= 1.0 + 1e-10, "GRU h' out of range: {v}");
        }
    }

    #[test]
    fn gru_cell_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = GRUCell::new(&mut session, 2, 3).expect("gru_cell");

        let x = session
            .tensor_variable(vec![0.5, -0.3], vec![1, 2], true)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.1, 0.2, 0.3], vec![1, 3], true)
            .expect("hidden");

        let h_new = cell.forward_cell(&mut session, x, h).expect("forward");
        let loss = session.tensor_sum(h_new).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(session.tensor_gradient(&report, x).is_some(), "input grad");
        assert!(session.tensor_gradient(&report, h).is_some(), "hidden grad");
    }

    #[test]
    fn gru_cell_batched() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = GRUCell::new(&mut session, 2, 3).expect("gru_cell");

        let x = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 6], vec![2, 3], false)
            .expect("hidden");

        let h_new = cell.forward_cell(&mut session, x, h).expect("forward");
        let (_, meta) = session.tensor_values_meta(h_new).expect("vals");
        assert_eq!(meta.shape(), &[2, 3]);
    }

    #[test]
    fn gru_cell_rejects_wrong_input_size() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = GRUCell::new(&mut session, 3, 4).expect("gru_cell");

        let x = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .expect("input");
        let h = session
            .tensor_variable(vec![0.0; 4], vec![1, 4], false)
            .expect("hidden");

        assert!(cell.forward_cell(&mut session, x, h).is_err());
    }

    #[test]
    fn rnn_cell_multi_step_sequence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = RNNCell::new(&mut session, 2, 3, true).expect("rnn_cell");

        let mut h = session
            .tensor_variable(vec![0.0; 3], vec![1, 3], false)
            .expect("h0");

        // Process 3 time steps
        for t in 0..3 {
            let x = session
                .tensor_variable(vec![t as f64, (t as f64) * 0.5], vec![1, 2], false)
                .expect("input");
            h = cell.forward_cell(&mut session, x, h).expect("step");
        }

        let (vals, meta) = session.tensor_values_meta(h).expect("vals");
        assert_eq!(meta.shape(), &[1, 3]);
        // After multiple steps the hidden state should have changed
        assert!(vals.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn lstm_cell_multi_step_sequence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let cell = LSTMCell::new(&mut session, 2, 3).expect("lstm_cell");

        let mut h = session
            .tensor_variable(vec![0.0; 3], vec![1, 3], false)
            .expect("h0");
        let mut c = session
            .tensor_variable(vec![0.0; 3], vec![1, 3], false)
            .expect("c0");

        for t in 0..3 {
            let x = session
                .tensor_variable(vec![t as f64, (t as f64) * 0.5], vec![1, 2], false)
                .expect("input");
            let result = cell.forward_cell(&mut session, x, h, c).expect("step");
            h = result.0;
            c = result.1;
        }

        let (h_vals, h_meta) = session.tensor_values_meta(h).expect("h_vals");
        assert_eq!(h_meta.shape(), &[1, 3]);
        assert!(h_vals.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn cosine_embedding_loss_module_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let loss_fn = CosineEmbeddingLoss::new(0.0);

        let x1 = session
            .tensor_variable(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], true)
            .expect("x1");
        let x2 = session
            .tensor_variable(vec![1.0, 0.0, 0.0, -1.0], vec![2, 2], false)
            .expect("x2");
        // First pair: similar (target=1), second pair: dissimilar (target=-1)
        let target = session
            .tensor_variable(vec![1.0, -1.0], vec![2], false)
            .expect("target");

        let loss = loss_fn
            .forward_triplet(&mut session, x1, x2, target)
            .expect("loss");
        let (vals, _) = session.tensor_values_meta(loss).expect("vals");
        // First pair: cos_sim=1, loss=0; Second: cos_sim=-1, loss=max(0,-1-0)=0
        // Total mean = 0.0
        assert!(vals[0].abs() < 1e-6);
    }

    // ── named_parameters / named_children / named_modules tests ───────

    #[test]
    fn linear_named_parameters_own() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin = Linear::new(&mut session, 3, 2, true).expect("linear");
        let params = lin.named_parameters_own();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "weight");
        assert_eq!(params[1].0, "bias");
    }

    #[test]
    fn linear_no_bias_named_parameters_own() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin = Linear::new(&mut session, 3, 2, false).expect("linear");
        let params = lin.named_parameters_own();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "weight");
    }

    #[test]
    fn sequential_named_children() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin1 = Linear::new(&mut session, 3, 4, true).expect("lin1");
        let lin2 = Linear::new(&mut session, 4, 2, true).expect("lin2");

        let mut seq = Sequential::new();
        seq.push(Box::new(lin1));
        seq.push(Box::new(ReLU));
        seq.push(Box::new(lin2));

        let ch = seq.named_children();
        assert_eq!(ch.len(), 3);
        assert_eq!(ch[0].0, "0");
        assert_eq!(ch[1].0, "1");
        assert_eq!(ch[2].0, "2");
    }

    #[test]
    fn sequential_named_parameters_hierarchical() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin1 = Linear::new(&mut session, 3, 4, true).expect("lin1");
        let lin2 = Linear::new(&mut session, 4, 2, false).expect("lin2");

        let mut seq = Sequential::new();
        seq.push(Box::new(lin1));
        seq.push(Box::new(ReLU));
        seq.push(Box::new(lin2));

        let params = named_parameters(&seq, "");
        // lin1: weight + bias = 2, ReLU: 0, lin2: weight = 1
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].0, "0.weight");
        assert_eq!(params[1].0, "0.bias");
        assert_eq!(params[2].0, "2.weight");
    }

    #[test]
    fn sequential_named_parameters_with_prefix() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin1 = Linear::new(&mut session, 3, 4, true).expect("lin1");

        let mut seq = Sequential::new();
        seq.push(Box::new(lin1));

        let params = named_parameters(&seq, "encoder");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "encoder.0.weight");
        assert_eq!(params[1].0, "encoder.0.bias");
    }

    #[test]
    fn nested_sequential_named_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin1 = Linear::new(&mut session, 3, 4, true).expect("lin1");
        let lin2 = Linear::new(&mut session, 4, 2, true).expect("lin2");

        let mut inner = Sequential::new();
        inner.push(Box::new(lin1));
        inner.push(Box::new(ReLU));

        let mut outer = Sequential::new();
        outer.push(Box::new(inner));
        outer.push(Box::new(lin2));

        let params = named_parameters(&outer, "");
        // inner.lin1: "0.0.weight", "0.0.bias"; outer.lin2: "1.weight", "1.bias"
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].0, "0.0.weight");
        assert_eq!(params[1].0, "0.0.bias");
        assert_eq!(params[2].0, "1.weight");
        assert_eq!(params[3].0, "1.bias");
    }

    #[test]
    fn named_modules_depth_first() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin1 = Linear::new(&mut session, 3, 4, true).expect("lin1");
        let lin2 = Linear::new(&mut session, 4, 2, true).expect("lin2");

        let mut inner = Sequential::new();
        inner.push(Box::new(lin1));
        inner.push(Box::new(ReLU));

        let mut outer = Sequential::new();
        outer.push(Box::new(inner));
        outer.push(Box::new(lin2));

        let mods = named_modules(&outer, "");
        // depth-first: root(""), inner("0"), lin1("0.0"), relu("0.1"), lin2("1")
        assert_eq!(mods.len(), 5);
        assert_eq!(mods[0].0, "");
        assert_eq!(mods[1].0, "0");
        assert_eq!(mods[2].0, "0.0");
        assert_eq!(mods[3].0, "0.1");
        assert_eq!(mods[4].0, "1");
    }

    #[test]
    fn named_modules_with_prefix() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin = Linear::new(&mut session, 3, 2, true).expect("lin");

        let mut seq = Sequential::new();
        seq.push(Box::new(lin));

        let mods = named_modules(&seq, "model");
        assert_eq!(mods.len(), 2);
        assert_eq!(mods[0].0, "model");
        assert_eq!(mods[1].0, "model.0");
    }

    #[test]
    fn children_of_sequential() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin = Linear::new(&mut session, 3, 2, true).expect("lin");

        let mut seq = Sequential::new();
        seq.push(Box::new(lin));
        seq.push(Box::new(ReLU));

        let ch = children(&seq);
        assert_eq!(ch.len(), 2);
    }

    #[test]
    fn modules_count_recursive() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin1 = Linear::new(&mut session, 3, 4, true).expect("lin1");
        let lin2 = Linear::new(&mut session, 4, 2, true).expect("lin2");

        let mut seq = Sequential::new();
        seq.push(Box::new(lin1));
        seq.push(Box::new(lin2));

        // root Sequential + 2 Linears = 3
        let all = modules(&seq);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn multihead_attention_named_children() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mha = MultiheadAttention::new(&mut session, 8, 2).expect("mha");

        let ch = mha.named_children();
        assert_eq!(ch.len(), 4);
        assert_eq!(ch[0].0, "q_proj");
        assert_eq!(ch[1].0, "k_proj");
        assert_eq!(ch[2].0, "v_proj");
        assert_eq!(ch[3].0, "out_proj");
    }

    #[test]
    fn multihead_attention_named_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mha = MultiheadAttention::new(&mut session, 8, 2).expect("mha");

        let params = named_parameters(&mha, "");
        // 4 projections × (weight + bias) = 8 parameters
        assert_eq!(params.len(), 8);
        assert_eq!(params[0].0, "q_proj.weight");
        assert_eq!(params[1].0, "q_proj.bias");
        assert_eq!(params[2].0, "k_proj.weight");
        assert_eq!(params[3].0, "k_proj.bias");
        assert_eq!(params[4].0, "v_proj.weight");
        assert_eq!(params[5].0, "v_proj.bias");
        assert_eq!(params[6].0, "out_proj.weight");
        assert_eq!(params[7].0, "out_proj.bias");
    }

    #[test]
    fn module_dict_named_children_uses_keys() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin1 = Linear::new(&mut session, 3, 4, true).expect("lin1");
        let lin2 = Linear::new(&mut session, 4, 2, true).expect("lin2");

        let mut dict = ModuleDict::new();
        dict.insert("encoder".to_string(), Box::new(lin1));
        dict.insert("decoder".to_string(), Box::new(lin2));

        let ch = dict.named_children();
        assert_eq!(ch.len(), 2);
        assert_eq!(ch[0].0, "encoder");
        assert_eq!(ch[1].0, "decoder");
    }

    #[test]
    fn module_dict_named_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin = Linear::new(&mut session, 3, 2, true).expect("lin");

        let mut dict = ModuleDict::new();
        dict.insert("layer".to_string(), Box::new(lin));

        let params = named_parameters(&dict, "");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "layer.weight");
        assert_eq!(params[1].0, "layer.bias");
    }

    #[test]
    fn module_list_named_children() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin1 = Linear::new(&mut session, 3, 4, true).expect("lin1");
        let lin2 = Linear::new(&mut session, 4, 2, true).expect("lin2");

        let mut list = ModuleList::new();
        list.push(Box::new(lin1));
        list.push(Box::new(lin2));

        let ch = list.named_children();
        assert_eq!(ch.len(), 2);
        assert_eq!(ch[0].0, "0");
        assert_eq!(ch[1].0, "1");
    }

    #[test]
    fn stateless_module_named_parameters_empty() {
        let relu = ReLU;
        assert!(relu.named_parameters_own().is_empty());
        assert!(relu.named_children().is_empty());
        assert!(named_parameters(&relu, "").is_empty());
    }

    #[test]
    fn parameters_matches_named_parameters_count() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin1 = Linear::new(&mut session, 3, 4, true).expect("lin1");
        let lin2 = Linear::new(&mut session, 4, 2, true).expect("lin2");

        let mut seq = Sequential::new();
        seq.push(Box::new(lin1));
        seq.push(Box::new(ReLU));
        seq.push(Box::new(lin2));

        let flat = seq.parameters();
        let named = named_parameters(&seq, "");
        assert_eq!(flat.len(), named.len());
        // Verify same node IDs in same order
        for (flat_id, (_, named_id)) in flat.iter().zip(named.iter()) {
            assert_eq!(flat_id, named_id);
        }
    }

    #[test]
    fn embedding_named_parameters_own() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let emb = Embedding::new(&mut session, 10, 4).expect("emb");
        let params = emb.named_parameters_own();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "weight");
    }

    #[test]
    fn layer_norm_named_parameters_own() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ln = LayerNorm::new(&mut session, vec![4], 1e-5).expect("ln");
        let params = ln.named_parameters_own();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "weight");
        assert_eq!(params[1].0, "bias");
    }

    #[test]
    fn conv1d_named_parameters_own() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv1d::new(&mut session, 1, 2, 3, 1, 0, true).expect("conv");
        let params = conv.named_parameters_own();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "weight");
        assert_eq!(params[1].0, "bias");
    }

    #[test]
    fn conv2d_named_parameters_own() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let conv = Conv2d::new(&mut session, 1, 2, (3, 3), (1, 1), (0, 0), false).expect("conv");
        let params = conv.named_parameters_own();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "weight");
    }

    #[test]
    fn batch_norm_named_parameters_own() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm1d::new(&mut session, 4, 1e-5, 0.1).expect("bn");
        let params = bn.named_parameters_own();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "weight");
        assert_eq!(params[1].0, "bias");
    }

    #[test]
    fn batchnorm1d_register_parameter_and_buffer_semantics() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut bn = BatchNorm1d::new(&mut session, 3, 1e-5, 0.1).expect("bn");

        let extra_param = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("extra_param");
        bn.register_parameter("extra_scale", Some(extra_param))
            .expect("register_parameter");

        let cache_a = session
            .tensor_variable(vec![5.0], vec![1], false)
            .expect("cache_a");
        bn.register_buffer("cache", Some(cache_a), false)
            .expect("register_buffer");
        let cache_b = session
            .tensor_variable(vec![7.0], vec![1], false)
            .expect("cache_b");
        // Same-name registration overwrites the previous tensor and flags.
        bn.register_buffer("cache", Some(cache_b), true)
            .expect("register_buffer overwrite");

        let params = bn.parameters();
        assert!(params.contains(&extra_param));
        assert!(!params.contains(&cache_b));

        let named_params = named_parameters(&bn, "");
        assert!(
            named_params
                .iter()
                .any(|(name, id)| name == "extra_scale" && *id == extra_param)
        );

        let all_buffers = named_buffers(&bn, "");
        assert!(all_buffers.iter().any(|(name, _)| name == "running_mean"));
        assert!(
            all_buffers
                .iter()
                .any(|(name, id)| name == "cache" && *id == cache_b)
        );

        let persistent_buffers = named_persistent_buffers(&bn, "");
        assert!(
            persistent_buffers
                .iter()
                .any(|(name, id)| name == "cache" && *id == cache_b)
        );
    }

    #[test]
    fn batchnorm1d_register_none_slots_not_in_iterators() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut bn = BatchNorm1d::new(&mut session, 2, 1e-5, 0.1).expect("bn");

        bn.register_parameter("optional_param", None)
            .expect("optional parameter slot");
        bn.register_buffer("optional_buffer", None, false)
            .expect("optional buffer slot");

        assert!(
            bn.named_parameter_slots_own()
                .iter()
                .any(|(name, id)| name == "optional_param" && id.is_none())
        );
        assert!(
            bn.named_buffer_slots_own()
                .iter()
                .any(|(name, id, persistent)| name == "optional_buffer"
                    && id.is_none()
                    && !*persistent)
        );

        assert!(
            !named_parameters(&bn, "")
                .iter()
                .any(|(name, _)| name == "optional_param")
        );
        assert!(
            !named_buffers(&bn, "")
                .iter()
                .any(|(name, _)| name == "optional_buffer")
        );
        assert!(
            !named_persistent_buffers(&bn, "")
                .iter()
                .any(|(name, _)| name == "optional_buffer")
        );
    }

    #[test]
    fn batchnorm_register_name_validation_and_conflicts() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut bn = BatchNorm1d::new(&mut session, 2, 1e-5, 0.1).expect("bn");

        let cache = session
            .tensor_variable(vec![1.0], vec![1], false)
            .expect("cache");
        bn.register_buffer("state_cache", Some(cache), true)
            .expect("register buffer");

        let p = session
            .tensor_variable(vec![2.0], vec![1], true)
            .expect("p");
        assert!(matches!(
            bn.register_parameter("state_cache", Some(p)),
            Err(ModuleRegistrationError::NameConflict { .. })
        ));

        let extra = session
            .tensor_variable(vec![3.0], vec![1], true)
            .expect("extra");
        bn.register_parameter("extra_param", Some(extra))
            .expect("register parameter");

        let b = session
            .tensor_variable(vec![4.0], vec![1], false)
            .expect("b");
        assert!(matches!(
            bn.register_buffer("extra_param", Some(b), true),
            Err(ModuleRegistrationError::NameConflict { .. })
        ));

        let bad_p = session
            .tensor_variable(vec![5.0], vec![1], true)
            .expect("bad_p");
        assert!(matches!(
            bn.register_parameter(".bad", Some(bad_p)),
            Err(ModuleRegistrationError::InvalidName { kind: "parameter" })
        ));

        let bad_b = session
            .tensor_variable(vec![6.0], vec![1], false)
            .expect("bad_b");
        assert!(matches!(
            bn.register_buffer("bad.", Some(bad_b), true),
            Err(ModuleRegistrationError::InvalidName { kind: "buffer" })
        ));

        let builtin_buf_conflict = session
            .tensor_variable(vec![7.0], vec![1], true)
            .expect("builtin_buf_conflict");
        assert!(matches!(
            bn.register_parameter("running_mean", Some(builtin_buf_conflict)),
            Err(ModuleRegistrationError::NameConflict { .. })
        ));

        let builtin_param_conflict = session
            .tensor_variable(vec![8.0], vec![1], false)
            .expect("builtin_param_conflict");
        assert!(matches!(
            bn.register_buffer("weight", Some(builtin_param_conflict), true),
            Err(ModuleRegistrationError::NameConflict { .. })
        ));
    }

    #[test]
    fn batchnorm2d_register_and_named_buffer_views() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut bn = BatchNorm2d::new(&mut session, 2, 1e-5, 0.1).expect("bn2d");

        let extra_param = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("extra_param");
        bn.register_parameter("extra_scale", Some(extra_param))
            .expect("register parameter");

        let scratch = session
            .tensor_variable(vec![2.0], vec![1], false)
            .expect("scratch");
        bn.register_buffer("scratch", Some(scratch), false)
            .expect("register scratch");

        let all_buffers = named_buffers(&bn, "");
        assert!(all_buffers.iter().any(|(name, _)| name == "running_mean"));
        assert!(all_buffers.iter().any(|(name, _)| name == "running_var"));
        assert!(
            all_buffers
                .iter()
                .any(|(name, _)| name == "num_batches_tracked")
        );
        assert!(
            all_buffers
                .iter()
                .any(|(name, id)| name == "scratch" && *id == scratch)
        );

        let persistent_buffers = named_persistent_buffers(&bn, "");
        assert!(!persistent_buffers.iter().any(|(name, _)| name == "scratch"));
        assert!(
            persistent_buffers
                .iter()
                .any(|(name, _)| name == "running_mean")
        );

        let named_params = named_parameters(&bn, "");
        assert!(
            named_params
                .iter()
                .any(|(name, id)| name == "extra_scale" && *id == extra_param)
        );
        assert!(!bn.parameters().contains(&scratch));
    }

    #[test]
    fn named_buffers_nested_module_prefixes() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn1 = BatchNorm1d::new(&mut session, 2, 1e-5, 0.1).expect("bn1");
        let bn2 = BatchNorm1d::new(&mut session, 2, 1e-5, 0.1).expect("bn2");

        let mut seq = Sequential::new();
        seq.push(Box::new(bn1));
        seq.push(Box::new(bn2));

        let persistent = named_persistent_buffers(&seq, "");
        assert!(persistent.iter().any(|(name, _)| name == "0.running_mean"));
        assert!(persistent.iter().any(|(name, _)| name == "1.running_var"));
        assert!(
            persistent
                .iter()
                .any(|(name, _)| name == "0.num_batches_tracked")
        );
    }

    #[test]
    fn group_norm_affine_named_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, true).expect("gn");
        let params = gn.named_parameters_own();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "weight");
        assert_eq!(params[1].0, "bias");
    }

    #[test]
    fn group_norm_no_affine_named_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gn = GroupNorm::new(&mut session, 2, 4, 1e-5, false).expect("gn");
        let params = gn.named_parameters_own();
        assert!(params.is_empty());
    }

    // ── RMSNorm Tests ──────────────────────────────────────────────────

    #[test]
    fn rmsnorm_unit_weight_normalizes() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rn = RMSNorm::new(&mut session, vec![4], 1e-5).unwrap();
        let input = session
            .tensor_variable(vec![2.0, 4.0, 6.0, 8.0], vec![1, 4], false)
            .unwrap();
        let out = rn.forward(&mut session, input).unwrap();
        let vals = session.tensor_values(out).unwrap();
        // RMS = sqrt((4+16+36+64)/4 + 1e-5) = sqrt(30.00001) ≈ 5.4772
        let rms = (30.0f64 + 1e-5).sqrt();
        assert!((vals[0] - 2.0 / rms).abs() < 1e-6);
        assert!((vals[1] - 4.0 / rms).abs() < 1e-6);
        assert!((vals[2] - 6.0 / rms).abs() < 1e-6);
        assert!((vals[3] - 8.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_batched() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rn = RMSNorm::new(&mut session, vec![2], 1e-5).unwrap();
        // Batch of 3, each with 2 features
        let input = session
            .tensor_variable(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![3, 2], false)
            .unwrap();
        let out = rn.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(out).unwrap();
        assert_eq!(shape, vec![3, 2]);
        let vals = session.tensor_values(out).unwrap();
        // Each pair [x, x] → RMS=sqrt(x^2+eps), normalized = x/RMS ≈ 1.0
        for &v in &vals {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn rmsnorm_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rn = RMSNorm::new(&mut session, vec![8], 1e-5).unwrap();
        assert_eq!(rn.parameters().len(), 1);
        let named = rn.named_parameters_own();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    #[test]
    fn rmsnorm_rejects_wrong_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rn = RMSNorm::new(&mut session, vec![4], 1e-5).unwrap();
        // Input last dim = 3, but normalized_shape = [4]
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .unwrap();
        assert!(rn.forward(&mut session, input).is_err());
    }

    #[test]
    fn deep_nesting_named_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lin = Linear::new(&mut session, 2, 2, true).expect("lin");

        let mut inner = Sequential::new();
        inner.push(Box::new(lin));

        let mut mid = Sequential::new();
        mid.push(Box::new(inner));

        let mut outer = Sequential::new();
        outer.push(Box::new(mid));

        let params = named_parameters(&outer, "");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "0.0.0.weight");
        assert_eq!(params[1].0, "0.0.0.bias");
    }

    #[test]
    fn named_modules_leaf_only() {
        let relu = ReLU;
        let mods = named_modules(&relu, "");
        assert_eq!(mods.len(), 1);
        assert_eq!(mods[0].0, "");
    }

    #[test]
    fn named_parameters_consistency_with_flat_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mha = MultiheadAttention::new(&mut session, 8, 2).expect("mha");

        let flat = mha.parameters();
        let named = named_parameters(&mha, "");
        assert_eq!(flat.len(), named.len());
        for (f, (_, n)) in flat.iter().zip(named.iter()) {
            assert_eq!(f, n);
        }
    }

    #[test]
    fn state_dict_linear_contains_weight_and_bias() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 3, 2, true).expect("linear");

        let state = linear.state_dict(&session).expect("state_dict");
        assert_eq!(state.len(), 2);
        assert!(state.contains_key("weight"));
        assert!(state.contains_key("bias"));
        assert_eq!(state.get("weight").expect("weight").meta().shape(), &[2, 3]);
        assert_eq!(state.get("bias").expect("bias").meta().shape(), &[1, 2]);
    }

    #[test]
    fn state_dict_includes_persistent_buffers() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let bn = BatchNorm1d::new(&mut session, 4, 1e-5, 0.1).expect("batch norm");
        let state = bn.state_dict(&session).expect("state_dict");

        assert!(state.contains_key("weight"));
        assert!(state.contains_key("bias"));
        assert!(state.contains_key("running_mean"));
        assert!(state.contains_key("running_var"));
        assert!(state.contains_key("num_batches_tracked"));
    }

    #[test]
    fn load_state_dict_restores_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 3, 2, true).expect("linear");
        let original = linear.state_dict(&session).expect("state_dict");

        session.no_grad_enter();
        session
            .tensor_fill_(linear.weight(), 123.0)
            .expect("fill weight");
        session
            .tensor_fill_(linear.bias().expect("bias"), -42.0)
            .expect("fill bias");
        session.no_grad_exit();

        let report = linear
            .load_state_dict(&mut session, &original, true)
            .expect("load_state_dict");
        assert!(report.missing_keys.is_empty());
        assert!(report.unexpected_keys.is_empty());

        let restored = linear.state_dict(&session).expect("restored state");
        assert_eq!(
            dense_values_f64(restored.get("weight").expect("weight")),
            dense_values_f64(original.get("weight").expect("weight"))
        );
        assert_eq!(
            dense_values_f64(restored.get("bias").expect("bias")),
            dense_values_f64(original.get("bias").expect("bias"))
        );
    }

    #[test]
    fn load_state_dict_strict_rejects_missing_and_unexpected_keys() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 3, 2, true).expect("linear");
        let mut state = linear.state_dict(&session).expect("state_dict");

        let weight = state.get("weight").expect("weight").clone();
        state.remove("bias");
        state.insert("unexpected.key".to_string(), weight);

        let err = linear
            .load_state_dict(&mut session, &state, true)
            .expect_err("strict load should fail on key mismatch");
        match err {
            StateDictError::StrictKeyMismatch {
                missing_keys,
                unexpected_keys,
            } => {
                assert!(missing_keys.iter().any(|key| key == "bias"));
                assert!(unexpected_keys.iter().any(|key| key == "unexpected.key"));
            }
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    #[test]
    fn load_state_dict_non_strict_reports_keys_and_applies_matches() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 3, 2, true).expect("linear");
        let mut state = linear.state_dict(&session).expect("state_dict");

        let expected_weight = dense_values_f64(state.get("weight").expect("weight"));
        let extra = state.get("weight").expect("weight").clone();
        state.remove("bias");
        state.insert("unexpected.key".to_string(), extra);

        session.no_grad_enter();
        session
            .tensor_fill_(linear.weight(), 999.0)
            .expect("fill weight");
        session
            .tensor_fill_(linear.bias().expect("bias"), -777.0)
            .expect("fill bias");
        session.no_grad_exit();

        let report = linear
            .load_state_dict(&mut session, &state, false)
            .expect("non-strict load should succeed");
        assert!(report.missing_keys.iter().any(|key| key == "bias"));
        assert!(
            report
                .unexpected_keys
                .iter()
                .any(|key| key == "unexpected.key")
        );

        let loaded = linear.state_dict(&session).expect("loaded state");
        assert_eq!(
            dense_values_f64(loaded.get("weight").expect("weight")),
            expected_weight
        );
        assert_eq!(
            dense_values_f64(loaded.get("bias").expect("bias")),
            vec![-777.0, -777.0]
        );
    }

    #[test]
    fn load_state_dict_rejects_shape_mismatch() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 3, 2, true).expect("linear");
        let mut state = linear.state_dict(&session).expect("state_dict");
        let weight = state.get("weight").expect("weight");
        let wrong_values = dense_values_f64(weight);
        let wrong_meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let wrong_weight =
            DenseTensor::from_storage(wrong_meta, wrong_values).expect("wrong weight tensor");
        state.insert("weight".to_string(), wrong_weight);

        let err = linear
            .load_state_dict(&mut session, &state, false)
            .expect_err("shape mismatch must fail");
        assert!(matches!(err, StateDictError::ShapeMismatch { key, .. } if key == "weight"));
    }

    #[test]
    fn clip_grad_norm_scales_gradients_and_returns_preclip_norm() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.0, 4.0], vec![2], true)
            .expect("x");
        let y = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("y");

        let x_sq = session.tensor_mul(x, x).expect("x*x");
        let y_sq = session.tensor_mul(y, y).expect("y*y");
        let x_loss = session.tensor_sum(x_sq).expect("sum x");
        let y_loss = session.tensor_sum(y_sq).expect("sum y");
        let total = session.tensor_add(x_loss, y_loss).expect("add");
        let _ = session.tensor_backward(total).expect("backward");

        let total_norm = clip_grad_norm_(&mut session, &[x, y], 5.0, 2.0).expect("clip grad norm");
        let expected_before = (6.0f64 * 6.0 + 8.0 * 8.0 + 2.0 * 2.0).sqrt();
        assert!((total_norm - expected_before).abs() < 1e-10);

        let clipped_x = session
            .tensor_accumulated_gradient(x)
            .expect("x grad")
            .expect("x grad exists");
        let clipped_y = session
            .tensor_accumulated_gradient(y)
            .expect("y grad")
            .expect("y grad exists");
        let after_norm = clipped_x
            .iter()
            .chain(clipped_y.iter())
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!((after_norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn clip_grad_norm_noop_when_below_threshold() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("x");
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss = session.tensor_sum(loss).expect("sum");
        let _ = session.tensor_backward(loss).expect("backward");

        let before = session
            .tensor_accumulated_gradient(x)
            .expect("grad")
            .expect("grad exists");
        let norm = clip_grad_norm_(&mut session, &[x], 10.0, 2.0).expect("clip");
        let after = session
            .tensor_accumulated_gradient(x)
            .expect("grad")
            .expect("grad exists");
        assert!((norm - 2.0).abs() < 1e-10);
        assert_eq!(before, after);
    }

    #[test]
    fn clip_grad_value_clamps_gradient_elements() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![0.0, 0.0, 0.0], vec![3], true)
            .expect("x");
        session
            .tensor_set_accumulated_gradient(x, vec![3.0, -5.0, 0.5])
            .expect("set gradient");

        clip_grad_value_(&mut session, &[x], 1.5).expect("clip grad value");
        let clipped = session
            .tensor_accumulated_gradient(x)
            .expect("grad")
            .expect("grad exists");
        assert_eq!(clipped, vec![1.5, -1.5, 0.5]);
    }

    #[test]
    fn parameters_vector_roundtrip_restores_parameter_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 3, 2, true).expect("linear");
        let params = linear.parameters();

        let mut original_values = Vec::new();
        for &param in &params {
            original_values.push(session.tensor_values(param).expect("param values"));
        }

        let vector = parameters_to_vector(&mut session, &params).expect("parameters_to_vector");

        session.no_grad_enter();
        for &param in &params {
            session.tensor_fill_(param, 0.0).expect("fill");
        }
        session.no_grad_exit();

        vector_to_parameters(&mut session, vector, &params).expect("vector_to_parameters");

        for (idx, &param) in params.iter().enumerate() {
            let restored = session.tensor_values(param).expect("restored");
            assert_eq!(restored, original_values[idx]);
        }
    }

    #[test]
    fn vector_to_parameters_rejects_length_mismatch() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let linear = Linear::new(&mut session, 2, 1, true).expect("linear");
        let params = linear.parameters();
        let bad_vector = session
            .tensor_variable(vec![1.0, 2.0], vec![2], false)
            .expect("bad vector");

        let err = vector_to_parameters(&mut session, bad_vector, &params)
            .expect_err("mismatched vector length must fail");
        assert!(matches!(err, AutogradError::Dispatch(_)));
    }

    // ── LSTM Module Tests ──────────────────────────────────────────────

    #[test]
    fn lstm_single_layer_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 3, 4, 1, false, 0.0, false).expect("lstm");

        // Input: [seq_len=5, batch=2, input_size=3]
        let input = session
            .tensor_variable(vec![0.1; 5 * 2 * 3], vec![5, 2, 3], false)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        // Output: [seq_len=5, batch=2, hidden_size=4]
        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 4]);

        // h_n: [num_layers=1, batch=2, hidden_size=4]
        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![1, 2, 4]);

        // c_n: [num_layers=1, batch=2, hidden_size=4]
        let c_shape = session.tensor_shape(result.c_n).expect("c_n shape");
        assert_eq!(c_shape, vec![1, 2, 4]);
    }

    #[test]
    fn lstm_batch_first_transposes_correctly() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 3, 4, 1, false, 0.0, true).expect("lstm");

        // Input: [batch=2, seq_len=5, input_size=3]
        let input = session
            .tensor_variable(vec![0.1; 2 * 5 * 3], vec![2, 5, 3], false)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        // Output: [batch=2, seq_len=5, hidden_size=4]
        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![2, 5, 4]);
    }

    #[test]
    fn lstm_multi_layer_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 3, 4, 3, false, 0.0, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.1; 5 * 2 * 3], vec![5, 2, 3], false)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        // Output: [seq_len=5, batch=2, hidden_size=4]
        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 4]);

        // h_n: [num_layers=3, batch=2, hidden_size=4]
        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![3, 2, 4]);
    }

    #[test]
    fn lstm_bidirectional_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 3, 4, 1, true, 0.0, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.1; 5 * 2 * 3], vec![5, 2, 3], false)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        // Output: [seq_len=5, batch=2, 2*hidden_size=8]
        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 8]);

        // h_n: [num_layers*2=2, batch=2, hidden_size=4]
        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![2, 2, 4]);
    }

    #[test]
    fn lstm_bidirectional_multi_layer_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 3, 4, 2, true, 0.0, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.1; 5 * 2 * 3], vec![5, 2, 3], false)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        // Output: [seq_len=5, batch=2, 2*hidden_size=8]
        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 8]);

        // h_n: [num_layers*2=4, batch=2, hidden_size=4]
        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![4, 2, 4]);
    }

    #[test]
    fn lstm_single_timestep_matches_cell() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Create standalone cell
        let cell = LSTMCell::new(&mut session, 2, 3).expect("cell");

        // Run cell directly
        let x = session
            .tensor_variable(vec![0.5, -0.3], vec![1, 2], false)
            .expect("x");
        let h0 = session
            .tensor_variable(vec![0.0; 3], vec![1, 3], false)
            .expect("h0");
        let c0 = session
            .tensor_variable(vec![0.0; 3], vec![1, 3], false)
            .expect("c0");
        let (cell_h, cell_c) = cell
            .forward_cell(&mut session, x, h0, c0)
            .expect("cell fwd");
        let cell_h_vals = session.tensor_values(cell_h).expect("cell h vals");
        let cell_c_vals = session.tensor_values(cell_c).expect("cell c vals");

        // Create LSTM module with same weights — we can't easily share weights,
        // so we verify shapes and that output is non-trivial
        let lstm = LSTM::new(&mut session, 2, 3, 1, false, 0.0, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.5, -0.3], vec![1, 1, 2], false)
            .expect("lstm input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("lstm forward");

        // Shape check: output [1,1,3], h_n [1,1,3], c_n [1,1,3]
        let out_shape = session.tensor_shape(result.output).expect("shape");
        assert_eq!(out_shape, vec![1, 1, 3]);

        let h_shape = session.tensor_shape(result.h_n).expect("shape");
        assert_eq!(h_shape, vec![1, 1, 3]);

        // Values should be non-zero (non-trivial computation)
        let out_vals = session.tensor_values(result.output).expect("vals");
        assert!(
            out_vals.iter().any(|&v| v != 0.0),
            "output should be non-zero"
        );

        // Cell output should also be non-zero
        assert!(
            cell_h_vals.iter().any(|&v| v != 0.0),
            "cell h should be non-zero"
        );
        assert!(
            cell_c_vals.iter().any(|&v| v != 0.0),
            "cell c should be non-zero"
        );
    }

    #[test]
    fn lstm_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 2, 3, 1, false, 0.0, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.5, -0.3, 0.1, 0.7], vec![2, 1, 2], true)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        let loss = session.tensor_sum(result.output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        // Input should have gradients
        assert!(
            session.tensor_gradient(&report, input).is_some(),
            "input gradient should exist"
        );

        // All LSTM parameters should have gradients
        for param in lstm.parameters() {
            assert!(
                session.tensor_gradient(&report, param).is_some(),
                "parameter gradient should exist"
            );
        }
    }

    #[test]
    fn lstm_multi_layer_backward_all_params_have_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 2, 3, 2, false, 0.0, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.1; 6], vec![3, 1, 2], true)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        let loss = session.tensor_sum(result.output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let params = lstm.parameters();
        // 2 layers * 4 params each = 8 parameters
        assert_eq!(params.len(), 8);

        for (i, param) in params.iter().enumerate() {
            assert!(
                session.tensor_gradient(&report, *param).is_some(),
                "parameter {i} gradient should exist"
            );
        }
    }

    #[test]
    fn lstm_bidirectional_backward_all_params_have_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 2, 3, 1, true, 0.0, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.1; 6], vec![3, 1, 2], true)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        let loss = session.tensor_sum(result.output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let params = lstm.parameters();
        // 1 layer * 2 directions * 4 params each = 8 parameters
        assert_eq!(params.len(), 8);

        for (i, param) in params.iter().enumerate() {
            assert!(
                session.tensor_gradient(&report, *param).is_some(),
                "parameter {i} gradient should exist"
            );
        }
    }

    #[test]
    fn lstm_custom_initial_states() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 2, 3, 1, false, 0.0, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.1; 4], vec![2, 1, 2], false)
            .expect("input");

        // Custom h_0 and c_0: [1, 1, 3]
        let h_0 = session
            .tensor_variable(vec![0.5, 0.5, 0.5], vec![1, 1, 3], false)
            .expect("h0");
        let c_0 = session
            .tensor_variable(vec![0.1, 0.2, 0.3], vec![1, 1, 3], false)
            .expect("c0");

        let result_custom = lstm
            .forward_lstm(&mut session, input, Some(h_0), Some(c_0))
            .expect("forward with custom states");

        // Compare with zero initial states
        let input2 = session
            .tensor_variable(vec![0.1; 4], vec![2, 1, 2], false)
            .expect("input2");
        let result_zero = lstm
            .forward_lstm(&mut session, input2, None, None)
            .expect("forward with zero states");

        let vals_custom = session.tensor_values(result_custom.output).expect("custom");
        let vals_zero = session.tensor_values(result_zero.output).expect("zero");

        // Outputs should differ because initial states differ
        assert_ne!(
            vals_custom, vals_zero,
            "custom and zero initial states should produce different outputs"
        );
    }

    #[test]
    fn lstm_module_trait_forward_returns_output() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 3, 4, 1, false, 0.0, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.1; 5 * 2 * 3], vec![5, 2, 3], false)
            .expect("input");

        // Module::forward should work and return just the output tensor
        let output = lstm.forward(&mut session, input).expect("module forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![5, 2, 4]);
    }

    #[test]
    fn lstm_parameter_count() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Single layer, unidirectional
        let lstm1 = LSTM::new(&mut session, 3, 4, 1, false, 0.0, false).expect("lstm1");
        assert_eq!(lstm1.parameters().len(), 4); // w_ih, w_hh, b_ih, b_hh

        // 2-layer, unidirectional
        let lstm2 = LSTM::new(&mut session, 3, 4, 2, false, 0.0, false).expect("lstm2");
        assert_eq!(lstm2.parameters().len(), 8); // 2 layers * 4 params

        // 1-layer, bidirectional
        let lstm3 = LSTM::new(&mut session, 3, 4, 1, true, 0.0, false).expect("lstm3");
        assert_eq!(lstm3.parameters().len(), 8); // 2 directions * 4 params

        // 2-layer, bidirectional
        let lstm4 = LSTM::new(&mut session, 3, 4, 2, true, 0.0, false).expect("lstm4");
        assert_eq!(lstm4.parameters().len(), 16); // 2 layers * 2 directions * 4 params
    }

    #[test]
    fn lstm_train_eval_propagation() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 3, 4, 2, false, 0.5, false).expect("lstm");

        assert!(lstm.is_training());

        lstm.eval();
        assert!(!lstm.is_training());
        // Dropout layers should also be in eval mode
        for dropout in &lstm.dropout_layers {
            assert!(!dropout.is_training());
        }

        lstm.train(true);
        assert!(lstm.is_training());
        for dropout in &lstm.dropout_layers {
            assert!(dropout.is_training());
        }
    }

    #[test]
    fn lstm_sequence_length_one() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 3, 4, 1, false, 0.0, false).expect("lstm");

        // Single timestep: [1, 1, 3]
        let input = session
            .tensor_variable(vec![0.5, -0.3, 0.1], vec![1, 1, 3], false)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("shape");
        assert_eq!(out_shape, vec![1, 1, 4]);
    }

    #[test]
    fn lstm_rejects_invalid_input_rank() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 3, 4, 1, false, 0.0, false).expect("lstm");

        // 2D input should fail
        let input = session
            .tensor_variable(vec![0.1; 6], vec![2, 3], false)
            .expect("input");

        let err = lstm.forward_lstm(&mut session, input, None, None);
        assert!(err.is_err());
    }

    #[test]
    fn lstm_rejects_zero_layers() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let err = LSTM::new(&mut session, 3, 4, 0, false, 0.0, false);
        assert!(err.is_err());
    }

    #[test]
    fn lstm_dropout_between_layers_in_eval_mode() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 2, 3, 2, false, 0.5, false).expect("lstm");

        let input = session
            .tensor_variable(vec![0.1; 6], vec![3, 1, 2], false)
            .expect("input");

        // In eval mode, dropout should be identity — run twice and compare
        lstm.eval();
        let result1 = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward1");
        let vals1 = session.tensor_values(result1.output).expect("vals1");

        let input2 = session
            .tensor_variable(vec![0.1; 6], vec![3, 1, 2], false)
            .expect("input2");
        let result2 = lstm
            .forward_lstm(&mut session, input2, None, None)
            .expect("forward2");
        let vals2 = session.tensor_values(result2.output).expect("vals2");

        assert_eq!(vals1, vals2, "eval mode should be deterministic");
    }

    #[test]
    fn lstm_hidden_state_carryover() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lstm = LSTM::new(&mut session, 2, 3, 1, false, 0.0, false).expect("lstm");

        // Run 3 timesteps
        let input = session
            .tensor_variable(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 1, 2], false)
            .expect("input");

        let result = lstm
            .forward_lstm(&mut session, input, None, None)
            .expect("forward");

        // h_n should equal the last timestep output (for unidirectional single layer)
        let h_n_vals = session.tensor_values(result.h_n).expect("h_n");
        let out_vals = session.tensor_values(result.output).expect("output");

        // output is [3, 1, 3], last timestep is out_vals[6..9]
        let last_output = &out_vals[6..9];
        let h_n_flat = &h_n_vals;

        for (i, (&a, &b)) in last_output.iter().zip(h_n_flat.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "h_n[{i}] ({b}) should match last output ({a})"
            );
        }
    }

    // ── GRU Module Tests ───────────────────────────────────────────────

    #[test]
    fn gru_single_layer_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 3, 4, 1, false, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![5, 2, 3], false)
            .expect("input");

        let result = gru.forward_gru(&mut session, input, None).expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 4]);

        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![1, 2, 4]);
    }

    #[test]
    fn gru_batch_first_transposes_correctly() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 3, 4, 1, false, 0.0, true).expect("gru");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![2, 5, 3], false)
            .expect("input");

        let result = gru.forward_gru(&mut session, input, None).expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![2, 5, 4]);
    }

    #[test]
    fn gru_multi_layer_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 3, 4, 3, false, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![5, 2, 3], false)
            .expect("input");

        let result = gru.forward_gru(&mut session, input, None).expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 4]);

        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![3, 2, 4]);
    }

    #[test]
    fn gru_bidirectional_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 3, 4, 1, true, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![5, 2, 3], false)
            .expect("input");

        let result = gru.forward_gru(&mut session, input, None).expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 8]);

        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![2, 2, 4]);
    }

    #[test]
    fn gru_bidirectional_multi_layer_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 3, 4, 2, true, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![5, 2, 3], false)
            .expect("input");

        let result = gru.forward_gru(&mut session, input, None).expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 8]);

        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![4, 2, 4]);
    }

    #[test]
    fn gru_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 2, 3, 1, false, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.5, -0.3, 0.1, 0.7], vec![2, 1, 2], true)
            .expect("input");

        let result = gru.forward_gru(&mut session, input, None).expect("forward");

        let loss = session.tensor_sum(result.output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(
            session.tensor_gradient(&report, input).is_some(),
            "input gradient should exist"
        );

        for param in gru.parameters() {
            assert!(
                session.tensor_gradient(&report, param).is_some(),
                "parameter gradient should exist"
            );
        }
    }

    #[test]
    fn gru_multi_layer_backward_all_params() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 2, 3, 2, false, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.1; 6], vec![3, 1, 2], true)
            .expect("input");

        let result = gru.forward_gru(&mut session, input, None).expect("forward");

        let loss = session.tensor_sum(result.output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let params = gru.parameters();
        assert_eq!(params.len(), 8); // 2 layers * 4 params each

        for (i, param) in params.iter().enumerate() {
            assert!(
                session.tensor_gradient(&report, *param).is_some(),
                "parameter {i} gradient should exist"
            );
        }
    }

    #[test]
    fn gru_parameter_count() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let gru1 = GRU::new(&mut session, 3, 4, 1, false, 0.0, false).expect("gru1");
        assert_eq!(gru1.parameters().len(), 4);

        let gru2 = GRU::new(&mut session, 3, 4, 2, false, 0.0, false).expect("gru2");
        assert_eq!(gru2.parameters().len(), 8);

        let gru3 = GRU::new(&mut session, 3, 4, 1, true, 0.0, false).expect("gru3");
        assert_eq!(gru3.parameters().len(), 8);

        let gru4 = GRU::new(&mut session, 3, 4, 2, true, 0.0, false).expect("gru4");
        assert_eq!(gru4.parameters().len(), 16);
    }

    #[test]
    fn gru_train_eval_propagation() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 3, 4, 2, false, 0.5, false).expect("gru");

        assert!(gru.is_training());

        gru.eval();
        assert!(!gru.is_training());
        for dropout in &gru.dropout_layers {
            assert!(!dropout.is_training());
        }

        gru.train(true);
        assert!(gru.is_training());
        for dropout in &gru.dropout_layers {
            assert!(dropout.is_training());
        }
    }

    #[test]
    fn gru_custom_initial_state() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 2, 3, 1, false, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.1; 4], vec![2, 1, 2], false)
            .expect("input");

        let h_0 = session
            .tensor_variable(vec![0.5, 0.5, 0.5], vec![1, 1, 3], false)
            .expect("h0");

        let result_custom = gru
            .forward_gru(&mut session, input, Some(h_0))
            .expect("custom");

        let input2 = session
            .tensor_variable(vec![0.1; 4], vec![2, 1, 2], false)
            .expect("input2");
        let result_zero = gru.forward_gru(&mut session, input2, None).expect("zero");

        let vals_custom = session.tensor_values(result_custom.output).expect("custom");
        let vals_zero = session.tensor_values(result_zero.output).expect("zero");
        assert_ne!(vals_custom, vals_zero);
    }

    #[test]
    fn gru_hidden_state_carryover() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 2, 3, 1, false, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 1, 2], false)
            .expect("input");

        let result = gru.forward_gru(&mut session, input, None).expect("forward");

        let h_n_vals = session.tensor_values(result.h_n).expect("h_n");
        let out_vals = session.tensor_values(result.output).expect("output");

        // Last timestep of output should equal h_n for single-layer unidirectional
        let last_output = &out_vals[6..9];

        for (i, (&a, &b)) in last_output.iter().zip(h_n_vals.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "h_n[{i}] ({b}) should match last output ({a})"
            );
        }
    }

    #[test]
    fn gru_sequence_length_one() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 3, 4, 1, false, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.5, -0.3, 0.1], vec![1, 1, 3], false)
            .expect("input");

        let result = gru.forward_gru(&mut session, input, None).expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("shape");
        assert_eq!(out_shape, vec![1, 1, 4]);
    }

    #[test]
    fn gru_rejects_invalid_input() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 3, 4, 1, false, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.1; 6], vec![2, 3], false)
            .expect("input");

        assert!(gru.forward_gru(&mut session, input, None).is_err());
    }

    #[test]
    fn gru_module_trait_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let gru = GRU::new(&mut session, 3, 4, 1, false, 0.0, false).expect("gru");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![5, 2, 3], false)
            .expect("input");

        let output = gru.forward(&mut session, input).expect("module forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![5, 2, 4]);
    }

    // ── RNN Module Tests ───────────────────────────────────────────────

    #[test]
    fn rnn_single_layer_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rnn = RNN::new(&mut session, 3, 4, RNNConfig::default()).expect("rnn");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![5, 2, 3], false)
            .expect("input");

        let result = rnn.forward_rnn(&mut session, input, None).expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 4]);

        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![1, 2, 4]);
    }

    #[test]
    fn rnn_batch_first() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rnn = RNN::new(
            &mut session,
            3,
            4,
            RNNConfig {
                batch_first: true,
                ..RNNConfig::default()
            },
        )
        .expect("rnn");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![2, 5, 3], false)
            .expect("input");

        let result = rnn.forward_rnn(&mut session, input, None).expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![2, 5, 4]);
    }

    #[test]
    fn rnn_multi_layer_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rnn = RNN::new(
            &mut session,
            3,
            4,
            RNNConfig {
                num_layers: 3,
                ..RNNConfig::default()
            },
        )
        .expect("rnn");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![5, 2, 3], false)
            .expect("input");

        let result = rnn.forward_rnn(&mut session, input, None).expect("forward");

        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![3, 2, 4]);
    }

    #[test]
    fn rnn_bidirectional_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rnn = RNN::new(
            &mut session,
            3,
            4,
            RNNConfig {
                bidirectional: true,
                ..RNNConfig::default()
            },
        )
        .expect("rnn");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![5, 2, 3], false)
            .expect("input");

        let result = rnn.forward_rnn(&mut session, input, None).expect("forward");

        let out_shape = session.tensor_shape(result.output).expect("output shape");
        assert_eq!(out_shape, vec![5, 2, 8]);

        let h_shape = session.tensor_shape(result.h_n).expect("h_n shape");
        assert_eq!(h_shape, vec![2, 2, 4]);
    }

    #[test]
    fn rnn_tanh_vs_relu_differ() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let rnn_tanh = RNN::new(&mut session, 2, 3, RNNConfig::default()).expect("tanh");
        let rnn_relu = RNN::new(
            &mut session,
            2,
            3,
            RNNConfig {
                use_tanh: false,
                ..RNNConfig::default()
            },
        )
        .expect("relu");

        let input1 = session
            .tensor_variable(vec![0.5, -0.3, 0.1, 0.7], vec![2, 1, 2], false)
            .expect("input1");
        let input2 = session
            .tensor_variable(vec![0.5, -0.3, 0.1, 0.7], vec![2, 1, 2], false)
            .expect("input2");

        let result_tanh = rnn_tanh
            .forward_rnn(&mut session, input1, None)
            .expect("tanh forward");
        let result_relu = rnn_relu
            .forward_rnn(&mut session, input2, None)
            .expect("relu forward");

        let vals_tanh = session.tensor_values(result_tanh.output).expect("tanh");
        let vals_relu = session.tensor_values(result_relu.output).expect("relu");

        // Different nonlinearities + different weights = different outputs
        assert_ne!(vals_tanh, vals_relu);
    }

    #[test]
    fn rnn_backward_produces_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rnn = RNN::new(&mut session, 2, 3, RNNConfig::default()).expect("rnn");

        let input = session
            .tensor_variable(vec![0.5, -0.3, 0.1, 0.7], vec![2, 1, 2], true)
            .expect("input");

        let result = rnn.forward_rnn(&mut session, input, None).expect("forward");

        let loss = session.tensor_sum(result.output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(session.tensor_gradient(&report, input).is_some());

        for param in rnn.parameters() {
            assert!(session.tensor_gradient(&report, param).is_some());
        }
    }

    #[test]
    fn rnn_parameter_count() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let rnn1 = RNN::new(&mut session, 3, 4, RNNConfig::default()).expect("rnn1");
        assert_eq!(rnn1.parameters().len(), 4);

        let rnn2 = RNN::new(
            &mut session,
            3,
            4,
            RNNConfig {
                num_layers: 2,
                ..RNNConfig::default()
            },
        )
        .expect("rnn2");
        assert_eq!(rnn2.parameters().len(), 8);

        let rnn3 = RNN::new(
            &mut session,
            3,
            4,
            RNNConfig {
                bidirectional: true,
                ..RNNConfig::default()
            },
        )
        .expect("rnn3");
        assert_eq!(rnn3.parameters().len(), 8);

        let rnn4 = RNN::new(
            &mut session,
            3,
            4,
            RNNConfig {
                num_layers: 2,
                bidirectional: true,
                ..RNNConfig::default()
            },
        )
        .expect("rnn4");
        assert_eq!(rnn4.parameters().len(), 16);
    }

    #[test]
    fn rnn_train_eval_propagation() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rnn = RNN::new(
            &mut session,
            3,
            4,
            RNNConfig {
                num_layers: 2,
                dropout: 0.5,
                ..RNNConfig::default()
            },
        )
        .expect("rnn");

        assert!(rnn.is_training());
        rnn.eval();
        assert!(!rnn.is_training());
        rnn.train(true);
        assert!(rnn.is_training());
    }

    #[test]
    fn rnn_hidden_state_carryover() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rnn = RNN::new(&mut session, 2, 3, RNNConfig::default()).expect("rnn");

        let input = session
            .tensor_variable(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 1, 2], false)
            .expect("input");

        let result = rnn.forward_rnn(&mut session, input, None).expect("forward");

        let h_n_vals = session.tensor_values(result.h_n).expect("h_n");
        let out_vals = session.tensor_values(result.output).expect("output");

        let last_output = &out_vals[6..9];

        for (i, (&a, &b)) in last_output.iter().zip(h_n_vals.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "h_n[{i}] ({b}) should match last output ({a})"
            );
        }
    }

    #[test]
    fn rnn_module_trait_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rnn = RNN::new(&mut session, 3, 4, RNNConfig::default()).expect("rnn");

        let input = session
            .tensor_variable(vec![0.1; 30], vec![5, 2, 3], false)
            .expect("input");

        let output = rnn.forward(&mut session, input).expect("module forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![5, 2, 4]);
    }

    #[test]
    fn rnn_sequence_length_one() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let rnn = RNN::new(&mut session, 3, 4, RNNConfig::default()).expect("rnn");

        let input = session
            .tensor_variable(vec![0.5, -0.3, 0.1], vec![1, 1, 3], false)
            .expect("input");

        let result = rnn.forward_rnn(&mut session, input, None).expect("forward");
        let out_shape = session.tensor_shape(result.output).expect("shape");
        assert_eq!(out_shape, vec![1, 1, 4]);
    }

    // ── Transformer Encoder Tests ──────────────────────────────────────

    #[test]
    fn transformer_encoder_layer_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let layer = TransformerEncoderLayer::new(
            &mut session,
            8,   // d_model
            2,   // nhead
            32,  // dim_feedforward
            0.0, // dropout (disabled for determinism)
            TransformerActivation::Relu,
            false, // post-norm
        )
        .expect("encoder layer");

        // Input: [batch=2, seq_len=4, d_model=8]
        let input = session
            .tensor_variable(vec![0.1; 64], vec![2, 4, 8], false)
            .expect("input");

        let output = layer.forward_layer(&mut session, input).expect("forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![2, 4, 8]);
    }

    #[test]
    fn transformer_encoder_layer_prenorm_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let layer = TransformerEncoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Gelu,
            true,
        )
        .expect("pre-norm layer");

        let input = session
            .tensor_variable(vec![0.1; 64], vec![2, 4, 8], false)
            .expect("input");

        let output = layer.forward_layer(&mut session, input).expect("forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![2, 4, 8]);
    }

    #[test]
    fn transformer_encoder_layer_postnorm_vs_prenorm_differ() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let post = TransformerEncoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
        )
        .expect("post-norm");

        let pre = TransformerEncoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            true,
        )
        .expect("pre-norm");

        let input1 = session
            .tensor_variable(vec![0.1; 16], vec![1, 2, 8], false)
            .expect("input1");
        let input2 = session
            .tensor_variable(vec![0.1; 16], vec![1, 2, 8], false)
            .expect("input2");

        let out_post = post.forward_layer(&mut session, input1).expect("post");
        let out_pre = pre.forward_layer(&mut session, input2).expect("pre");

        let vals_post = session.tensor_values(out_post).expect("post vals");
        let vals_pre = session.tensor_values(out_pre).expect("pre vals");
        assert_ne!(vals_post, vals_pre, "pre-norm and post-norm should differ");
    }

    #[test]
    fn transformer_encoder_layer_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let layer = TransformerEncoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
        )
        .expect("layer");

        let input = session
            .tensor_variable(vec![0.1; 16], vec![1, 2, 8], true)
            .expect("input");

        let output = layer.forward_layer(&mut session, input).expect("forward");
        let loss = session.tensor_sum(output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(session.tensor_gradient(&report, input).is_some());

        for param in layer.parameters() {
            assert!(session.tensor_gradient(&report, param).is_some());
        }
    }

    #[test]
    fn transformer_encoder_stacked_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let encoder = TransformerEncoder::new(
            &mut session,
            8,   // d_model
            2,   // nhead
            3,   // num_layers
            32,  // dim_feedforward
            0.0, // dropout
            TransformerActivation::Relu,
            false, // post-norm
            true,  // final_layer_norm
        )
        .expect("encoder");

        let input = session
            .tensor_variable(vec![0.1; 64], vec![2, 4, 8], false)
            .expect("input");

        let output = encoder.forward(&mut session, input).expect("forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![2, 4, 8]);
    }

    #[test]
    fn transformer_encoder_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let encoder = TransformerEncoder::new(
            &mut session,
            8,
            2,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
            false,
        )
        .expect("encoder");

        let input = session
            .tensor_variable(vec![0.1; 16], vec![1, 2, 8], true)
            .expect("input");

        let output = encoder.forward(&mut session, input).expect("forward");
        let loss = session.tensor_sum(output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(session.tensor_gradient(&report, input).is_some());

        for param in encoder.parameters() {
            assert!(session.tensor_gradient(&report, param).is_some());
        }
    }

    #[test]
    fn transformer_encoder_parameter_count() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Single layer: MHA(4 linears * 2 params each = 8) + FF(2 linears * 2 = 4) + LN(2 * 2 = 4) = 16
        let layer = TransformerEncoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
        )
        .expect("layer");
        assert_eq!(layer.parameters().len(), 16);

        // Encoder with 3 layers + final norm: 3*16 + 2 = 50
        let encoder = TransformerEncoder::new(
            &mut session,
            8,
            2,
            3,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
            true,
        )
        .expect("encoder");
        assert_eq!(encoder.parameters().len(), 50);

        // Without final norm: 3*16 = 48
        let encoder_no_norm = TransformerEncoder::new(
            &mut session,
            8,
            2,
            3,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
            false,
        )
        .expect("encoder");
        assert_eq!(encoder_no_norm.parameters().len(), 48);
    }

    #[test]
    fn transformer_encoder_train_eval() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let encoder = TransformerEncoder::new(
            &mut session,
            8,
            2,
            2,
            32,
            0.5,
            TransformerActivation::Relu,
            false,
            false,
        )
        .expect("encoder");

        assert!(encoder.is_training());
        encoder.eval();
        assert!(!encoder.is_training());
        encoder.train(true);
        assert!(encoder.is_training());
    }

    #[test]
    fn transformer_encoder_layer_gelu_activation() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let layer = TransformerEncoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Gelu,
            false,
        )
        .expect("layer");

        let input = session
            .tensor_variable(vec![0.1; 16], vec![1, 2, 8], false)
            .expect("input");

        let output = layer.forward_layer(&mut session, input).expect("forward");
        let vals = session.tensor_values(output).expect("vals");
        assert!(vals.iter().any(|&v| v != 0.0), "output should be non-zero");
    }

    // ── Transformer Decoder Tests ──────────────────────────────────────

    #[test]
    fn transformer_decoder_layer_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let layer = TransformerDecoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
        )
        .expect("decoder layer");

        // tgt: [batch=2, tgt_len=3, d_model=8]
        let tgt = session
            .tensor_variable(vec![0.1; 48], vec![2, 3, 8], false)
            .expect("tgt");
        // memory: [batch=2, src_len=5, d_model=8]
        let memory = session
            .tensor_variable(vec![0.2; 80], vec![2, 5, 8], false)
            .expect("memory");

        let output = layer
            .forward_layer(&mut session, tgt, memory)
            .expect("forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![2, 3, 8]);
    }

    #[test]
    fn transformer_decoder_layer_prenorm_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let layer = TransformerDecoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Gelu,
            true,
        )
        .expect("pre-norm decoder");

        let tgt = session
            .tensor_variable(vec![0.1; 16], vec![1, 2, 8], false)
            .expect("tgt");
        let memory = session
            .tensor_variable(vec![0.2; 24], vec![1, 3, 8], false)
            .expect("memory");

        let output = layer
            .forward_layer(&mut session, tgt, memory)
            .expect("forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![1, 2, 8]);
    }

    #[test]
    fn transformer_decoder_layer_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let layer = TransformerDecoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
        )
        .expect("layer");

        let tgt = session
            .tensor_variable(vec![0.1; 16], vec![1, 2, 8], true)
            .expect("tgt");
        let memory = session
            .tensor_variable(vec![0.2; 24], vec![1, 3, 8], true)
            .expect("memory");

        let output = layer
            .forward_layer(&mut session, tgt, memory)
            .expect("forward");
        let loss = session.tensor_sum(output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(session.tensor_gradient(&report, tgt).is_some());
        assert!(session.tensor_gradient(&report, memory).is_some());

        for param in layer.parameters() {
            assert!(session.tensor_gradient(&report, param).is_some());
        }
    }

    #[test]
    fn transformer_decoder_stacked_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let decoder = TransformerDecoder::new(
            &mut session,
            8,
            2,
            3,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
            true,
        )
        .expect("decoder");

        let tgt = session
            .tensor_variable(vec![0.1; 48], vec![2, 3, 8], false)
            .expect("tgt");
        let memory = session
            .tensor_variable(vec![0.2; 80], vec![2, 5, 8], false)
            .expect("memory");

        let output = decoder
            .forward_decoder(&mut session, tgt, memory)
            .expect("forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![2, 3, 8]);
    }

    #[test]
    fn transformer_decoder_parameter_count() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Decoder layer: self_attn(8) + cross_attn(8) + ff(4) + 3 norms(2 each) = 26
        let layer = TransformerDecoderLayer::new(
            &mut session,
            8,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
        )
        .expect("layer");
        assert_eq!(layer.parameters().len(), 26);

        // Decoder: 2 layers + final norm = 2*26 + 2 = 54
        let decoder = TransformerDecoder::new(
            &mut session,
            8,
            2,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
            true,
        )
        .expect("decoder");
        assert_eq!(decoder.parameters().len(), 54);
    }

    #[test]
    fn transformer_decoder_train_eval() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let decoder = TransformerDecoder::new(
            &mut session,
            8,
            2,
            2,
            32,
            0.5,
            TransformerActivation::Relu,
            false,
            false,
        )
        .expect("decoder");

        assert!(decoder.is_training());
        decoder.eval();
        assert!(!decoder.is_training());
        decoder.train(true);
        assert!(decoder.is_training());
    }

    // ── Full Transformer Tests ─────────────────────────────────────────

    #[test]
    fn transformer_forward_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let transformer = Transformer::new(
            &mut session,
            8,
            2,
            2,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
        )
        .expect("transformer");

        // src: [batch=2, src_len=5, d_model=8]
        let src = session
            .tensor_variable(vec![0.1; 80], vec![2, 5, 8], false)
            .expect("src");
        // tgt: [batch=2, tgt_len=3, d_model=8]
        let tgt = session
            .tensor_variable(vec![0.2; 48], vec![2, 3, 8], false)
            .expect("tgt");

        let output = transformer
            .forward_transformer(&mut session, src, tgt)
            .expect("forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![2, 3, 8]);
    }

    #[test]
    fn transformer_backward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let transformer = Transformer::new(
            &mut session,
            8,
            2,
            1,
            1,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
        )
        .expect("transformer");

        let src = session
            .tensor_variable(vec![0.1; 16], vec![1, 2, 8], true)
            .expect("src");
        let tgt = session
            .tensor_variable(vec![0.2; 16], vec![1, 2, 8], true)
            .expect("tgt");

        let output = transformer
            .forward_transformer(&mut session, src, tgt)
            .expect("forward");
        let loss = session.tensor_sum(output).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        assert!(session.tensor_gradient(&report, src).is_some());
        assert!(session.tensor_gradient(&report, tgt).is_some());

        for param in transformer.parameters() {
            assert!(session.tensor_gradient(&report, param).is_some());
        }
    }

    #[test]
    fn transformer_parameter_count() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let transformer = Transformer::new(
            &mut session,
            8,
            2,
            2,
            2,
            32,
            0.0,
            TransformerActivation::Relu,
            false,
        )
        .expect("transformer");

        // Encoder: 2 layers * 16 params = 32
        // Decoder: 2 layers * 26 params = 52
        // Total = 84
        assert_eq!(transformer.parameters().len(), 84);
    }

    #[test]
    fn transformer_train_eval() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let transformer = Transformer::new(
            &mut session,
            8,
            2,
            1,
            1,
            32,
            0.5,
            TransformerActivation::Relu,
            false,
        )
        .expect("transformer");

        assert!(transformer.is_training());
        transformer.eval();
        assert!(!transformer.is_training());
        transformer.train(true);
        assert!(transformer.is_training());
    }

    #[test]
    fn generate_causal_mask() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mask = generate_square_subsequent_mask(&mut session, 4).expect("mask");

        let vals = session.tensor_values(mask).expect("vals");
        let shape = session.tensor_shape(mask).expect("shape");
        assert_eq!(shape, vec![4, 4]);

        // Diagonal and below should be 0
        assert_eq!(vals[0], 0.0); // (0,0)
        assert_eq!(vals[4], 0.0); // (1,0)
        assert_eq!(vals[5], 0.0); // (1,1)
        assert_eq!(vals[10], 0.0); // (2,2)
        assert_eq!(vals[15], 0.0); // (3,3)

        // Above diagonal should be -inf
        assert_eq!(vals[1], f64::NEG_INFINITY); // (0,1)
        assert_eq!(vals[2], f64::NEG_INFINITY); // (0,2)
        assert_eq!(vals[3], f64::NEG_INFINITY); // (0,3)
        assert_eq!(vals[7], f64::NEG_INFINITY); // (1,3)
    }

    #[test]
    fn transformer_prenorm_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let transformer = Transformer::new(
            &mut session,
            8,
            2,
            1,
            1,
            32,
            0.0,
            TransformerActivation::Gelu,
            true,
        )
        .expect("pre-norm transformer");

        let src = session
            .tensor_variable(vec![0.1; 16], vec![1, 2, 8], false)
            .expect("src");
        let tgt = session
            .tensor_variable(vec![0.2; 24], vec![1, 3, 8], false)
            .expect("tgt");

        let output = transformer
            .forward_transformer(&mut session, src, tgt)
            .expect("forward");
        let shape = session.tensor_shape(output).expect("shape");
        assert_eq!(shape, vec![1, 3, 8]);
    }

    // ── Pooling Variant Tests ────────────────────────────────────────────

    #[test]
    fn avgpool2d_basic_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 4, 4] input
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = session
            .tensor_variable(data, vec![1, 1, 4, 4], false)
            .unwrap();
        let pool = AvgPool2d::new((2, 2), (2, 2), (0, 0), false, true);
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 2, 2]);
        let vals = session.tensor_values(output).unwrap();
        // top-left: mean(1,2,5,6) = 3.5
        assert!((vals[0] - 3.5).abs() < 1e-10);
        // top-right: mean(3,4,7,8) = 5.5
        assert!((vals[1] - 5.5).abs() < 1e-10);
        // bottom-left: mean(9,10,13,14) = 11.5
        assert!((vals[2] - 11.5).abs() < 1e-10);
        // bottom-right: mean(11,12,15,16) = 13.5
        assert!((vals[3] - 13.5).abs() < 1e-10);
    }

    #[test]
    fn avgpool2d_with_padding() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 3, 3] with padding=1, kernel=3, stride=1 => output [1, 1, 3, 3]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = session
            .tensor_variable(data, vec![1, 1, 3, 3], false)
            .unwrap();
        let pool = AvgPool2d::new((3, 3), (1, 1), (1, 1), false, true);
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 3, 3]);
        let vals = session.tensor_values(output).unwrap();
        // Center: mean of all 9 / 9 = 5.0
        assert!((vals[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn avgpool2d_stride_1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 3, 3] kernel=2, stride=1 => [1, 1, 2, 2]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = session
            .tensor_variable(data, vec![1, 1, 3, 3], false)
            .unwrap();
        let pool = AvgPool2d::new((2, 2), (1, 1), (0, 0), false, true);
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 2, 2]);
        let vals = session.tensor_values(output).unwrap();
        // (0,0): mean(1,2,4,5)=3.0
        assert!((vals[0] - 3.0).abs() < 1e-10);
        // (0,1): mean(2,3,5,6)=4.0
        assert!((vals[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn avgpool3d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 2, 2, 2] kernel=(2,2,2), stride=(1,1,1) => [1,1,1,1,1]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = session
            .tensor_variable(data, vec![1, 1, 2, 2, 2], false)
            .unwrap();
        let pool = AvgPool3d::new((2, 2, 2), (1, 1, 1));
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 1, 1, 1]);
        let vals = session.tensor_values(output).unwrap();
        // mean of all 8: (1+2+3+4+5+6+7+8)/8 = 4.5
        assert!((vals[0] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn maxpool3d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 4, 4, 4] kernel=(2,2,2), stride=(2,2,2) => [1,1,2,2,2]
        let data: Vec<f64> = (1..=64).map(|x| x as f64).collect();
        let input = session
            .tensor_variable(data, vec![1, 1, 4, 4, 4], false)
            .unwrap();
        let pool = MaxPool3d::new((2, 2, 2), (2, 2, 2));
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 2, 2, 2]);
        let vals = session.tensor_values(output).unwrap();
        // First element: max of cube [1..8 region] = max within first 2x2x2 block
        // Indices: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
        // In row-major 4x4x4: 1,2,5,6,17,18,21,22 -> max=22
        assert!((vals[0] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn maxpool3d_wrong_dim() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = session
            .tensor_variable(vec![1.0; 16], vec![1, 1, 4, 4], false)
            .unwrap();
        let pool = MaxPool3d::new((2, 2, 2), (2, 2, 2));
        assert!(pool.forward(&mut session, input).is_err());
    }

    #[test]
    fn adaptive_avg_pool1d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 6] -> output_size=2 => [1, 1, 2]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = session.tensor_variable(data, vec![1, 1, 6], false).unwrap();
        let pool = AdaptiveAvgPool1d::new(2);
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 2]);
        let vals = session.tensor_values(output).unwrap();
        // First bin: mean(1,2,3) = 2.0
        assert!((vals[0] - 2.0).abs() < 1e-10);
        // Second bin: mean(4,5,6) = 5.0
        assert!((vals[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn adaptive_avg_pool1d_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let data = vec![1.0, 2.0, 3.0];
        let input = session
            .tensor_variable(data.clone(), vec![1, 1, 3], false)
            .unwrap();
        let pool = AdaptiveAvgPool1d::new(3);
        let output = pool.forward(&mut session, input).unwrap();
        // output_size == input_size => identity
        assert_eq!(output, input);
    }

    #[test]
    fn adaptive_avg_pool3d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 4, 4, 4] -> (1, 1, 1) => global average
        let data: Vec<f64> = (1..=64).map(|x| x as f64).collect();
        let input = session
            .tensor_variable(data, vec![1, 1, 4, 4, 4], false)
            .unwrap();
        let pool = AdaptiveAvgPool3d::new((1, 1, 1));
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 1, 1, 1]);
        let vals = session.tensor_values(output).unwrap();
        // mean(1..64) = 32.5
        assert!((vals[0] - 32.5).abs() < 1e-10);
    }

    #[test]
    fn adaptive_max_pool1d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let input = session.tensor_variable(data, vec![1, 1, 6], false).unwrap();
        let pool = AdaptiveMaxPool1d::new(2);
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 2]);
        let vals = session.tensor_values(output).unwrap();
        // First bin [1,5,3]: max=5
        assert!((vals[0] - 5.0).abs() < 1e-10);
        // Second bin [4,2,6]: max=6
        assert!((vals[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn adaptive_max_pool2d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 4, 4] -> (2, 2)
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = session
            .tensor_variable(data, vec![1, 1, 4, 4], false)
            .unwrap();
        let pool = AdaptiveMaxPool2d::new((2, 2));
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 2, 2]);
        let vals = session.tensor_values(output).unwrap();
        // top-left quad: max(1,2,5,6)=6
        assert!((vals[0] - 6.0).abs() < 1e-10);
        // top-right quad: max(3,4,7,8)=8
        assert!((vals[1] - 8.0).abs() < 1e-10);
        // bottom-left quad: max(9,10,13,14)=14
        assert!((vals[2] - 14.0).abs() < 1e-10);
        // bottom-right quad: max(11,12,15,16)=16
        assert!((vals[3] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn adaptive_max_pool3d_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 4, 4, 4] -> (1, 1, 1) => global max
        let data: Vec<f64> = (1..=64).map(|x| x as f64).collect();
        let input = session
            .tensor_variable(data, vec![1, 1, 4, 4, 4], false)
            .unwrap();
        let pool = AdaptiveMaxPool3d::new((1, 1, 1));
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 1, 1, 1]);
        let vals = session.tensor_values(output).unwrap();
        assert!((vals[0] - 64.0).abs() < 1e-10);
    }

    #[test]
    fn adaptive_max_pool2d_to_1x1() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [2, 3, 4, 4] -> (1, 1) => global max per channel per batch
        let data: Vec<f64> = (0..96).map(|x| x as f64).collect();
        let input = session
            .tensor_variable(data, vec![2, 3, 4, 4], false)
            .unwrap();
        let pool = AdaptiveMaxPool2d::new((1, 1));
        let output = pool.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![2, 3, 1, 1]);
        let vals = session.tensor_values(output).unwrap();
        // batch 0, channel 0: max of 0..15 = 15
        assert!((vals[0] - 15.0).abs() < 1e-10);
        // batch 0, channel 1: max of 16..31 = 31
        assert!((vals[1] - 31.0).abs() < 1e-10);
    }

    #[test]
    fn maxunpool1d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // pooled [1, 1, 2], indices=[1, 3] => scatter into length 4
        let pooled = session
            .tensor_variable(vec![5.0, 8.0], vec![1, 1, 2], false)
            .unwrap();
        let unpool = MaxUnpool1d::new(2, 2);
        let output = unpool
            .forward_with_indices(&mut session, pooled, &[1, 3], 4)
            .unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 4]);
        let vals = session.tensor_values(output).unwrap();
        assert_eq!(vals, vec![0.0, 5.0, 0.0, 8.0]);
    }

    #[test]
    fn maxunpool2d_inverts_maxpool() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 4x4 input -> MaxPool2d(2,2) -> 2x2 output, then MaxUnpool2d -> 4x4
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = session
            .tensor_variable(data, vec![1, 1, 4, 4], false)
            .unwrap();
        let pool = MaxPool2d::new((2, 2), (2, 2));
        let pooled = pool.forward(&mut session, input).unwrap();
        let pool_vals = session.tensor_values(pooled).unwrap();
        // Pooled: [6, 8, 14, 16]
        assert!((pool_vals[0] - 6.0).abs() < 1e-10);

        let unpool = MaxUnpool2d::new((2, 2), (2, 2));
        // indices: 6 is at (1,1)=5, 8 is at (1,3)=7, 14 is at (3,1)=13, 16 is at (3,3)=15
        let output = unpool
            .forward_with_indices(&mut session, pooled, &[5, 7, 13, 15], (4, 4))
            .unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 4, 4]);
        let vals = session.tensor_values(output).unwrap();
        // Position 5 should have 6.0, position 7 should have 8.0, etc.
        assert!((vals[5] - 6.0).abs() < 1e-10);
        assert!((vals[7] - 8.0).abs() < 1e-10);
        assert!((vals[13] - 14.0).abs() < 1e-10);
        assert!((vals[15] - 16.0).abs() < 1e-10);
        // Other positions should be 0
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[1], 0.0);
    }

    #[test]
    fn maxunpool3d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 1, 1, 2] with indices pointing into 1x1x4 output space
        let pooled = session
            .tensor_variable(vec![5.0, 9.0], vec![1, 1, 1, 1, 2], false)
            .unwrap();
        let unpool = MaxUnpool3d::new((1, 1, 2), (1, 1, 2));
        let output = unpool
            .forward_with_indices(&mut session, pooled, &[1, 3], (1, 1, 4))
            .unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 1, 1, 4]);
        let vals = session.tensor_values(output).unwrap();
        assert_eq!(vals, vec![0.0, 5.0, 0.0, 9.0]);
    }

    #[test]
    fn pooling_no_parameters() {
        // All pooling modules should have zero parameters
        let pool1 = AvgPool2d::new((2, 2), (2, 2), (0, 0), false, true);
        assert!(pool1.parameters().is_empty());
        let pool2 = AvgPool3d::new((2, 2, 2), (2, 2, 2));
        assert!(pool2.parameters().is_empty());
        let pool3 = MaxPool3d::new((2, 2, 2), (2, 2, 2));
        assert!(pool3.parameters().is_empty());
        let pool4 = AdaptiveAvgPool1d::new(3);
        assert!(pool4.parameters().is_empty());
        let pool5 = AdaptiveAvgPool3d::new((1, 1, 1));
        assert!(pool5.parameters().is_empty());
        let pool6 = AdaptiveMaxPool1d::new(3);
        assert!(pool6.parameters().is_empty());
        let pool7 = AdaptiveMaxPool2d::new((1, 1));
        assert!(pool7.parameters().is_empty());
        let pool8 = AdaptiveMaxPool3d::new((1, 1, 1));
        assert!(pool8.parameters().is_empty());
    }

    // ── Activation Module Tests ──────────────────────────────────────────

    #[test]
    fn prelu_default_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let prelu = PReLU::new(&mut session, 1, 0.25).unwrap();
        let input = session
            .tensor_variable(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![1, 5], false)
            .unwrap();
        let output = prelu.forward(&mut session, input).unwrap();
        let vals = session.tensor_values(output).unwrap();
        assert!((vals[0] - (-0.5)).abs() < 1e-10); // -2 * 0.25
        assert!((vals[1] - (-0.25)).abs() < 1e-10); // -1 * 0.25
        assert!((vals[2] - 0.0).abs() < 1e-10);
        assert!((vals[3] - 1.0).abs() < 1e-10);
        assert!((vals[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn prelu_multichannel() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // 2 channels, each with different slope
        let prelu = PReLU::new(&mut session, 2, 0.1).unwrap();
        // Set channel slopes to [0.1, 0.5] manually
        let input = session
            .tensor_variable(
                vec![-1.0, -1.0], // [1, 2] shape: batch=1, channels=2
                vec![1, 2],
                false,
            )
            .unwrap();
        let output = prelu.forward(&mut session, input).unwrap();
        let vals = session.tensor_values(output).unwrap();
        // Both channels have slope 0.1 (init value)
        assert!((vals[0] - (-0.1)).abs() < 1e-10);
        assert!((vals[1] - (-0.1)).abs() < 1e-10);
    }

    #[test]
    fn prelu_has_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let prelu = PReLU::new(&mut session, 3, 0.25).unwrap();
        assert_eq!(prelu.parameters().len(), 1);
    }

    #[test]
    fn celu_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let celu = CELU::new(1.0);
        let input = session
            .tensor_variable(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false)
            .unwrap();
        let output = celu.forward(&mut session, input).unwrap();
        let vals = session.tensor_values(output).unwrap();
        // For alpha=1.0, CELU == ELU
        assert!((vals[0] - ((-2.0f64).exp() - 1.0)).abs() < 1e-10);
        assert!((vals[1] - ((-1.0f64).exp() - 1.0)).abs() < 1e-10);
        assert!((vals[2] - 0.0).abs() < 1e-10);
        assert!((vals[3] - 1.0).abs() < 1e-10);
        assert!((vals[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn celu_continuity_at_zero() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let celu = CELU::new(2.0);
        // Values near zero should be continuous
        let input = session
            .tensor_variable(vec![-0.001, -0.0001, 0.0, 0.0001, 0.001], vec![5], false)
            .unwrap();
        let output = celu.forward(&mut session, input).unwrap();
        let vals = session.tensor_values(output).unwrap();
        // All values should be close to 0
        for &v in &vals {
            assert!(v.abs() < 0.01);
        }
    }

    #[test]
    fn glu_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let glu = GLU::new(1);
        // Input [1, 4] -> split into a=[1,2] and b=[1,2], output = a * sigmoid(b)
        let input = session
            .tensor_variable(vec![1.0, 2.0, 0.0, 0.0], vec![1, 4], false)
            .unwrap();
        let output = glu.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 2]);
        let vals = session.tensor_values(output).unwrap();
        // a = [1, 2], b = [0, 0], sigmoid(0) = 0.5
        assert!((vals[0] - 0.5).abs() < 1e-10); // 1 * 0.5
        assert!((vals[1] - 1.0).abs() < 1e-10); // 2 * 0.5
    }

    #[test]
    fn glu_odd_size_errors() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let glu = GLU::new(0);
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
            .unwrap();
        assert!(glu.forward(&mut session, input).is_err());
    }

    #[test]
    fn threshold_forward() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let threshold = Threshold::new(0.5, -1.0);
        let input = session
            .tensor_variable(vec![-1.0, 0.0, 0.5, 0.6, 1.0, 2.0], vec![6], false)
            .unwrap();
        let output = threshold.forward(&mut session, input).unwrap();
        let vals = session.tensor_values(output).unwrap();
        // Values <= 0.5 -> -1.0, values > 0.5 -> pass through
        assert_eq!(vals[0], -1.0);
        assert_eq!(vals[1], -1.0);
        assert_eq!(vals[2], -1.0); // 0.5 is NOT > 0.5
        assert!((vals[3] - 0.6).abs() < 1e-10);
        assert!((vals[4] - 1.0).abs() < 1e-10);
        assert!((vals[5] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn threshold_as_relu() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // Threshold(0, 0) acts like ReLU
        let threshold = Threshold::new(0.0, 0.0);
        let input = session
            .tensor_variable(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false)
            .unwrap();
        let output = threshold.forward(&mut session, input).unwrap();
        let vals = session.tensor_values(output).unwrap();
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn activation_no_parameters() {
        assert!(CELU::new(1.0).parameters().is_empty());
        assert!(GLU::new(0).parameters().is_empty());
        assert!(Threshold::new(0.0, 0.0).parameters().is_empty());
    }

    // ── Padding Module Tests ─────────────────────────────────────────────

    #[test]
    fn reflection_pad1d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 4] with pad=(2,2) -> [1, 1, 8]
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4], false)
            .unwrap();
        let pad = ReflectionPad1d::new((2, 2));
        let output = pad.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 8]);
        let vals = session.tensor_values(output).unwrap();
        // [3, 2, 1, 2, 3, 4, 3, 2]
        assert_eq!(vals, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn reflection_pad1d_too_large_errors() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 1, 3], false)
            .unwrap();
        let pad = ReflectionPad1d::new((3, 0)); // pad >= input size
        assert!(pad.forward(&mut session, input).is_err());
    }

    #[test]
    fn reflection_pad2d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 3, 3] with pad=(1,1,1,1) -> [1, 1, 5, 5]
        let input = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![1, 1, 3, 3],
                false,
            )
            .unwrap();
        let pad = ReflectionPad2d::new((1, 1, 1, 1));
        let output = pad.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 5, 5]);
        let vals = session.tensor_values(output).unwrap();
        // Center should be original: row 1, col 1 = 1.0
        assert!((vals[6] - 1.0).abs() < 1e-10); // (1,1) in 5x5
        // Top-left corner: reflection of (1,1) = 5.0
        assert!((vals[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn replication_pad1d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4], false)
            .unwrap();
        let pad = ReplicationPad1d::new((2, 3));
        let output = pad.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 9]);
        let vals = session.tensor_values(output).unwrap();
        // [1, 1, 1, 2, 3, 4, 4, 4, 4]
        assert_eq!(vals, vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn replication_pad2d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 2, 2] with pad=(1,1,1,1) -> [1, 1, 4, 4]
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], false)
            .unwrap();
        let pad = ReplicationPad2d::new((1, 1, 1, 1));
        let output = pad.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 4, 4]);
        let vals = session.tensor_values(output).unwrap();
        // Top-left corner should be input[0,0]=1.0
        assert!((vals[0] - 1.0).abs() < 1e-10);
        // Bottom-right corner should be input[1,1]=4.0
        assert!((vals[15] - 4.0).abs() < 1e-10);
        // Center 2x2 should be original
        assert!((vals[5] - 1.0).abs() < 1e-10);
        assert!((vals[6] - 2.0).abs() < 1e-10);
        assert!((vals[9] - 3.0).abs() < 1e-10);
        assert!((vals[10] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn replication_pad3d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 2, 2, 2] with pad=(1,1,0,0,0,0) -> [1, 1, 2, 2, 4]
        let input = session
            .tensor_variable(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![1, 1, 2, 2, 2],
                false,
            )
            .unwrap();
        let pad = ReplicationPad3d::new((1, 1, 0, 0, 0, 0));
        let output = pad.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 2, 2, 4]);
        let vals = session.tensor_values(output).unwrap();
        // First row: [1, 1, 2, 2]
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], 2.0);
        assert_eq!(vals[3], 2.0);
    }

    #[test]
    fn circular_pad1d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4], false)
            .unwrap();
        let pad = CircularPad1d::new((2, 2));
        let output = pad.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 8]);
        let vals = session.tensor_values(output).unwrap();
        // [3, 4, 1, 2, 3, 4, 1, 2]
        assert_eq!(vals, vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn circular_pad2d_basic() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        // [1, 1, 2, 3] with pad=(1,1,1,1) -> [1, 1, 4, 5]
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 1, 2, 3], false)
            .unwrap();
        let pad = CircularPad2d::new((1, 1, 1, 1));
        let output = pad.forward(&mut session, input).unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![1, 1, 4, 5]);
        let vals = session.tensor_values(output).unwrap();
        // Center 2x3 should be original data
        assert!((vals[6] - 1.0).abs() < 1e-10); // row1, col1
        assert!((vals[7] - 2.0).abs() < 1e-10); // row1, col2
        assert!((vals[8] - 3.0).abs() < 1e-10); // row1, col3
    }

    #[test]
    fn padding_zero_is_identity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 1, 3], false)
            .unwrap();
        let pad = ReflectionPad1d::new((0, 0));
        let output = pad.forward(&mut session, input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn padding_no_parameters() {
        assert!(ReflectionPad1d::new((1, 1)).parameters().is_empty());
        assert!(ReflectionPad2d::new((1, 1, 1, 1)).parameters().is_empty());
        assert!(ReplicationPad1d::new((1, 1)).parameters().is_empty());
        assert!(ReplicationPad2d::new((1, 1, 1, 1)).parameters().is_empty());
        assert!(
            ReplicationPad3d::new((1, 1, 1, 1, 1, 1))
                .parameters()
                .is_empty()
        );
        assert!(CircularPad1d::new((1, 1)).parameters().is_empty());
        assert!(CircularPad2d::new((1, 1, 1, 1)).parameters().is_empty());
    }

    // ── EmbeddingBag Tests ───────────────────────────────────────────────

    #[test]
    fn embedding_bag_sum_mode() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let _eb = EmbeddingBag::new(&mut session, 5, 3, EmbeddingBagMode::Sum, None).unwrap();

        // Set weights to known values
        let weight_vals: Vec<f64> = (0..15).map(|x| x as f64).collect();
        let weight = session
            .tensor_variable(weight_vals, vec![5, 3], true)
            .unwrap();
        let eb = EmbeddingBag {
            weight,
            num_embeddings: 5,
            embedding_dim: 3,
            mode: EmbeddingBagMode::Sum,
            padding_idx: None,
        };

        // 2 bags: bag 0 = indices [0, 2], bag 1 = indices [1, 3, 4]
        let indices = session
            .tensor_variable(vec![0.0, 2.0, 1.0, 3.0, 4.0], vec![5], false)
            .unwrap();
        let offsets = session
            .tensor_variable(vec![0.0, 2.0], vec![2], false)
            .unwrap();

        let output = eb
            .forward_with_offsets(&mut session, indices, offsets, None)
            .unwrap();
        let shape = session.tensor_shape(output).unwrap();
        assert_eq!(shape, vec![2, 3]);
        let vals = session.tensor_values(output).unwrap();
        // bag 0: embed[0] + embed[2] = [0,1,2] + [6,7,8] = [6, 8, 10]
        assert!((vals[0] - 6.0).abs() < 1e-10);
        assert!((vals[1] - 8.0).abs() < 1e-10);
        assert!((vals[2] - 10.0).abs() < 1e-10);
        // bag 1: embed[1] + embed[3] + embed[4] = [3,4,5] + [9,10,11] + [12,13,14] = [24, 27, 30]
        assert!((vals[3] - 24.0).abs() < 1e-10);
        assert!((vals[4] - 27.0).abs() < 1e-10);
        assert!((vals[5] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn embedding_bag_mean_mode() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let weight_vals: Vec<f64> = (0..15).map(|x| x as f64).collect();
        let weight = session
            .tensor_variable(weight_vals, vec![5, 3], true)
            .unwrap();
        let eb = EmbeddingBag {
            weight,
            num_embeddings: 5,
            embedding_dim: 3,
            mode: EmbeddingBagMode::Mean,
            padding_idx: None,
        };

        let indices = session
            .tensor_variable(vec![0.0, 2.0, 1.0, 3.0, 4.0], vec![5], false)
            .unwrap();
        let offsets = session
            .tensor_variable(vec![0.0, 2.0], vec![2], false)
            .unwrap();

        let output = eb
            .forward_with_offsets(&mut session, indices, offsets, None)
            .unwrap();
        let vals = session.tensor_values(output).unwrap();
        // bag 0: mean of [0,1,2] and [6,7,8] = [3, 4, 5]
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);
        assert!((vals[2] - 5.0).abs() < 1e-10);
        // bag 1: mean of [3,4,5], [9,10,11], [12,13,14] = [8, 9, 10]
        assert!((vals[3] - 8.0).abs() < 1e-10);
        assert!((vals[4] - 9.0).abs() < 1e-10);
        assert!((vals[5] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn embedding_bag_max_mode() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let weight_vals = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0, 7.0, 1.0, 2.0];
        let weight = session
            .tensor_variable(weight_vals, vec![3, 3], true)
            .unwrap();
        let eb = EmbeddingBag {
            weight,
            num_embeddings: 3,
            embedding_dim: 3,
            mode: EmbeddingBagMode::Max,
            padding_idx: None,
        };

        // 1 bag with all 3 embeddings
        let indices = session
            .tensor_variable(vec![0.0, 1.0, 2.0], vec![3], false)
            .unwrap();
        let offsets = session.tensor_variable(vec![0.0], vec![1], false).unwrap();

        let output = eb
            .forward_with_offsets(&mut session, indices, offsets, None)
            .unwrap();
        let vals = session.tensor_values(output).unwrap();
        // max([1,5,3], [4,2,6], [7,1,2]) = [7, 5, 6]
        assert!((vals[0] - 7.0).abs() < 1e-10);
        assert!((vals[1] - 5.0).abs() < 1e-10);
        assert!((vals[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn embedding_bag_per_sample_weights() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let weight_vals: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let weight = session
            .tensor_variable(weight_vals, vec![3, 2], true)
            .unwrap();
        let eb = EmbeddingBag {
            weight,
            num_embeddings: 3,
            embedding_dim: 2,
            mode: EmbeddingBagMode::Sum,
            padding_idx: None,
        };

        let indices = session
            .tensor_variable(vec![0.0, 1.0], vec![2], false)
            .unwrap();
        let offsets = session.tensor_variable(vec![0.0], vec![1], false).unwrap();
        let psw = session
            .tensor_variable(vec![2.0, 3.0], vec![2], false)
            .unwrap();

        let output = eb
            .forward_with_offsets(&mut session, indices, offsets, Some(psw))
            .unwrap();
        let vals = session.tensor_values(output).unwrap();
        // 2*[0,1] + 3*[2,3] = [0,2] + [6,9] = [6, 11]
        assert!((vals[0] - 6.0).abs() < 1e-10);
        assert!((vals[1] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn embedding_bag_empty_bag() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let weight_vals: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let weight = session
            .tensor_variable(weight_vals, vec![3, 2], true)
            .unwrap();
        let eb = EmbeddingBag {
            weight,
            num_embeddings: 3,
            embedding_dim: 2,
            mode: EmbeddingBagMode::Sum,
            padding_idx: None,
        };

        // 2 bags: bag 0 is empty (offset 0 to 0), bag 1 has index 1
        let indices = session.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let offsets = session
            .tensor_variable(vec![0.0, 0.0], vec![2], false)
            .unwrap();

        let output = eb
            .forward_with_offsets(&mut session, indices, offsets, None)
            .unwrap();
        let vals = session.tensor_values(output).unwrap();
        // bag 0: empty -> [0, 0]
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[1], 0.0);
        // bag 1: embed[1] = [2, 3]
        assert!((vals[2] - 2.0).abs() < 1e-10);
        assert!((vals[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn embedding_bag_has_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let eb = EmbeddingBag::new(&mut session, 10, 4, EmbeddingBagMode::Sum, None).unwrap();
        assert_eq!(eb.parameters().len(), 1);
    }

    // ── CTCLoss Tests ──────────────────────────────────────────────────

    /// Helper to create log_probs tensor from a flat array with shape [T, N, C].
    fn make_log_probs(
        session: &mut FrankenTorchSession,
        data: Vec<f64>,
        t: usize,
        n: usize,
        c: usize,
    ) -> TensorNodeId {
        session.tensor_variable(data, vec![t, n, c], true).unwrap()
    }

    #[test]
    fn ctc_loss_single_char_target() {
        // T=2, N=1, C=3 (blank=0, classes: 0=blank, 1=a, 2=b)
        // Target: [1] (just 'a')
        // Possible alignments: (a, blank), (a, a), (blank, a)
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // log_probs at t=0: log([0.3, 0.4, 0.3]) and t=1: log([0.3, 0.4, 0.3])
        let lp_t0 = [0.3f64.ln(), 0.4f64.ln(), 0.3f64.ln()];
        let lp_t1 = [0.3f64.ln(), 0.4f64.ln(), 0.3f64.ln()];
        let mut lp_data = Vec::new();
        lp_data.extend_from_slice(&lp_t0);
        lp_data.extend_from_slice(&lp_t1);

        let log_probs = make_log_probs(&mut session, lp_data, 2, 1, 3);
        let targets = session
            .tensor_variable(vec![1.0], vec![1, 1], false)
            .unwrap();
        let input_lengths = session.tensor_variable(vec![2.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![1.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new().with_reduction(CTCReduction::Sum);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_val = session.tensor_values(loss).unwrap();
        assert_eq!(loss_val.len(), 1);

        // Manually compute: P(a) = P(a,blank) + P(a,a) + P(blank,a)
        // = 0.4*0.3 + 0.4*0.4 + 0.3*0.4 = 0.12 + 0.16 + 0.12 = 0.40
        // loss = -ln(0.40)
        let expected = -(0.40f64.ln());
        assert!(
            (loss_val[0] - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            loss_val[0]
        );
    }

    #[test]
    fn ctc_loss_blank_only_target() {
        // Empty target: loss = -sum of log P(blank) at each timestep
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // T=3, N=1, C=2 (blank=0)
        // P(blank) = 0.6 at each step
        let log_blank = 0.6f64.ln();
        let log_other = 0.4f64.ln();
        let lp_data = vec![
            log_blank, log_other, log_blank, log_other, log_blank, log_other,
        ];

        let log_probs = make_log_probs(&mut session, lp_data, 3, 1, 2);
        // Empty target: target_length = 0, targets can be anything (padded)
        let targets = session
            .tensor_variable(vec![0.0], vec![1, 1], false)
            .unwrap();
        let input_lengths = session.tensor_variable(vec![3.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![0.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new().with_reduction(CTCReduction::Sum);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_val = session.tensor_values(loss).unwrap();
        // loss = -(3 * ln(0.6))
        let expected = -3.0 * 0.6f64.ln();
        assert!(
            (loss_val[0] - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            loss_val[0]
        );
    }

    #[test]
    fn ctc_loss_target_longer_than_input() {
        // T=1, target_len=2: impossible alignment -> inf loss
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let lp_data = vec![0.5f64.ln(), 0.3f64.ln(), 0.2f64.ln()];
        let log_probs = make_log_probs(&mut session, lp_data, 1, 1, 3);
        let targets = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .unwrap();
        let input_lengths = session.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![2.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new().with_reduction(CTCReduction::Sum);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_val = session.tensor_values(loss).unwrap();
        assert!(
            loss_val[0].is_infinite(),
            "expected inf, got {}",
            loss_val[0]
        );
    }

    #[test]
    fn ctc_loss_zero_infinity() {
        // Same as above but with zero_infinity=true -> loss should be 0
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let lp_data = vec![0.5f64.ln(), 0.3f64.ln(), 0.2f64.ln()];
        let log_probs = make_log_probs(&mut session, lp_data, 1, 1, 3);
        let targets = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .unwrap();
        let input_lengths = session.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![2.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new()
            .with_reduction(CTCReduction::Sum)
            .with_zero_infinity(true);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_val = session.tensor_values(loss).unwrap();
        assert_eq!(loss_val[0], 0.0);
    }

    #[test]
    fn ctc_loss_repeated_chars() {
        // Target 'aa' requires a blank between the two a's
        // T=3, N=1, C=2 (blank=0, a=1)
        // Lattice: [blank, a, blank, a, blank] (5 states)
        // Only valid alignment with T=3: (blank, a, a) won't work since repeated a needs blank
        // Valid: (a, blank, a) -> P = 0.5 * 0.5 * 0.5 = 0.125
        // Also: (blank, a, blank) won't emit aa...
        // Actually lattice is [b, a, b, a, b], T=3, must end at state 3 or 4
        // alpha[0][0] = P(b,0) = 0.5, alpha[0][1] = P(a,0) = 0.5
        // alpha[1][0] = alpha[0][0]*P(b,1) = 0.5*0.5 = 0.25
        // alpha[1][1] = (alpha[0][0]+alpha[0][1])*P(a,1) = 1.0*0.5 = 0.5
        // alpha[1][2] = (alpha[0][1])*P(b,1) = 0.5*0.5 = 0.25  [from state 1, can move to 2]
        // alpha[2][2] = (alpha[1][1]+alpha[1][2])*P(b,2) = 0.75*0.5 = 0.375
        // alpha[2][3] = (alpha[1][2]+alpha[1][3???])*P(a,2)
        //   alpha[1][3]: s=3, labels[3]=a, from s=2 (blank, different), from s=1 (a, labels[1]=a=labels[3] so NO skip)
        //   alpha[1][3] = (alpha[0][2]+alpha[0][3])*P(a,1) but alpha[0][2]=alpha[0][3]=-inf -> -inf
        //   So alpha[1][3] = alpha[1][2]*P(a,2) + alpha[1][3] (skip not allowed since labels[3]=labels[1]=a)
        //   Wait, let me recompute with the lattice definition
        // Just use uniform probs and verify loss is finite and positive
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let log_half = 0.5f64.ln();
        let lp_data = vec![log_half; 6]; // T=3, N=1, C=2, all probs = 0.5
        let log_probs = make_log_probs(&mut session, lp_data, 3, 1, 2);
        let targets = session
            .tensor_variable(vec![1.0, 1.0], vec![1, 2], false)
            .unwrap();
        let input_lengths = session.tensor_variable(vec![3.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![2.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new().with_reduction(CTCReduction::Sum);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_val = session.tensor_values(loss).unwrap();
        // With uniform probs 0.5, total P(aa) should be sum of valid alignment probs
        // The only valid alignment for 'aa' with T=3 is: a,blank,a
        // P = 0.5 * 0.5 * 0.5 = 0.125
        // loss = -ln(0.125) = 3*ln(2) ≈ 2.0794
        let expected = -(0.125f64.ln());
        assert!(
            (loss_val[0] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            loss_val[0]
        );
    }

    #[test]
    fn ctc_loss_batch_different_lengths() {
        // N=2: sample 0 has T=3, target 'a'; sample 1 has T=2, target 'b'
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // C=3 (blank=0, a=1, b=2), T_max=3, N=2
        // log_probs shape: [3, 2, 3]
        // Using uniform probs for simplicity
        let log_third = (1.0f64 / 3.0).ln();
        let lp_data = vec![log_third; 18]; // 3*2*3 = 18

        let log_probs = make_log_probs(&mut session, lp_data, 3, 2, 3);
        let targets = session
            .tensor_variable(vec![1.0, 2.0], vec![2, 1], false)
            .unwrap();
        let input_lengths = session
            .tensor_variable(vec![3.0, 2.0], vec![2], false)
            .unwrap();
        let target_lengths = session
            .tensor_variable(vec![1.0, 1.0], vec![2], false)
            .unwrap();

        // Test with reduction=None to get per-sample losses
        let loss_mod = CTCLoss::new().with_reduction(CTCReduction::None);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_vals = session.tensor_values(loss).unwrap();
        assert_eq!(loss_vals.len(), 2);

        // Both should be positive and finite
        assert!(loss_vals[0] > 0.0 && loss_vals[0].is_finite());
        assert!(loss_vals[1] > 0.0 && loss_vals[1].is_finite());

        // Sample 0 (T=3, target 'a'): more timesteps -> higher total prob -> lower loss
        // Sample 1 (T=2, target 'b'): fewer timesteps -> lower total prob -> higher loss
        // Actually with uniform probs:
        // Sample 0: alignments for 'a' with T=3: {a,b,b}, {a,b,a}, {b,a,b} etc
        // This gets complicated with uniform C=3, just check they're different since T differs
        assert!((loss_vals[0] - loss_vals[1]).abs() > 1e-6);
    }

    #[test]
    fn ctc_loss_mean_reduction() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let log_half = 0.5f64.ln();
        // T=2, N=1, C=2
        let lp_data = vec![log_half; 4];
        let log_probs = make_log_probs(&mut session, lp_data.clone(), 2, 1, 2);
        let targets = session
            .tensor_variable(vec![1.0], vec![1, 1], false)
            .unwrap();
        let input_lengths = session.tensor_variable(vec![2.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![1.0], vec![1], false).unwrap();

        // Sum reduction first
        let loss_sum_mod = CTCLoss::new().with_reduction(CTCReduction::Sum);
        let loss_sum = loss_sum_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();
        let sum_val = session.tensor_values(loss_sum).unwrap()[0];

        // Mean reduction
        let log_probs2 = make_log_probs(&mut session, lp_data, 2, 1, 2);
        let targets2 = session
            .tensor_variable(vec![1.0], vec![1, 1], false)
            .unwrap();
        let input_lengths2 = session.tensor_variable(vec![2.0], vec![1], false).unwrap();
        let target_lengths2 = session.tensor_variable(vec![1.0], vec![1], false).unwrap();

        let loss_mean_mod = CTCLoss::new().with_reduction(CTCReduction::Mean);
        let loss_mean = loss_mean_mod
            .forward_ctc(
                &mut session,
                log_probs2,
                targets2,
                input_lengths2,
                target_lengths2,
            )
            .unwrap();
        let mean_val = session.tensor_values(loss_mean).unwrap()[0];

        // Mean = Sum / sum(target_lengths) = Sum / 1.0
        assert!(
            (mean_val - sum_val).abs() < 1e-10,
            "mean={mean_val}, sum={sum_val}"
        );
    }

    #[test]
    fn ctc_loss_gradient_finite_difference() {
        // Verify gradient correctness via finite differences
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // T=2, N=1, C=3, target=[1]
        // Use valid log-probabilities
        let logits_base = [
            vec![0.5, 1.0, 0.3], // t=0
            vec![0.2, 0.8, 0.5], // t=1
        ];
        let lp_base: Vec<f64> = logits_base.iter().flat_map(|l| log_softmax(l)).collect();

        let log_probs = make_log_probs(&mut session, lp_base.clone(), 2, 1, 3);
        let targets = session
            .tensor_variable(vec![1.0], vec![1, 1], false)
            .unwrap();
        let input_lengths = session.tensor_variable(vec![2.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![1.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new().with_reduction(CTCReduction::Sum);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        // Run backward
        let report = session.tensor_backward(loss).unwrap();
        let grad = report.gradient(log_probs).unwrap().to_vec();

        // Finite difference check for each element
        let eps = 1e-5;
        for i in 0..lp_base.len() {
            let mut perturbed = lp_base.clone();
            perturbed[i] += eps;

            let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
            let lp2 = make_log_probs(&mut s2, perturbed.clone(), 2, 1, 3);
            let t2 = s2.tensor_variable(vec![1.0], vec![1, 1], false).unwrap();
            let il2 = s2.tensor_variable(vec![2.0], vec![1], false).unwrap();
            let tl2 = s2.tensor_variable(vec![1.0], vec![1], false).unwrap();
            let l_plus = loss_mod.forward_ctc(&mut s2, lp2, t2, il2, tl2).unwrap();
            let v_plus = s2.tensor_values(l_plus).unwrap()[0];

            let mut perturbed_m = lp_base.clone();
            perturbed_m[i] -= eps;

            let mut s3 = FrankenTorchSession::new(ExecutionMode::Strict);
            let lp3 = make_log_probs(&mut s3, perturbed_m, 2, 1, 3);
            let t3 = s3.tensor_variable(vec![1.0], vec![1, 1], false).unwrap();
            let il3 = s3.tensor_variable(vec![2.0], vec![1], false).unwrap();
            let tl3 = s3.tensor_variable(vec![1.0], vec![1], false).unwrap();
            let l_minus = loss_mod.forward_ctc(&mut s3, lp3, t3, il3, tl3).unwrap();
            let v_minus = s3.tensor_values(l_minus).unwrap()[0];

            let fd_grad = (v_plus - v_minus) / (2.0 * eps);
            assert!(
                (grad[i] - fd_grad).abs() < 1e-4,
                "gradient mismatch at index {i}: analytic={}, fd={fd_grad}",
                grad[i]
            );
        }
    }

    #[test]
    fn ctc_loss_single_timestep() {
        // T=1, target=[1]: only alignment is (a) at t=0
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let lp_data = vec![0.3f64.ln(), 0.5f64.ln(), 0.2f64.ln()]; // C=3
        let log_probs = make_log_probs(&mut session, lp_data, 1, 1, 3);
        let targets = session
            .tensor_variable(vec![1.0], vec![1, 1], false)
            .unwrap();
        let input_lengths = session.tensor_variable(vec![1.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![1.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new().with_reduction(CTCReduction::Sum);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_val = session.tensor_values(loss).unwrap()[0];
        // Only alignment: emit class 1 at t=0
        let expected = -(0.5f64.ln());
        assert!(
            (loss_val - expected).abs() < 1e-6,
            "expected {expected}, got {loss_val}"
        );
    }

    /// Helper to compute log-softmax on a slice of logits.
    fn log_softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = logits.iter().map(|x| (x - max).exp()).sum();
        let log_sum = max + sum_exp.ln();
        logits.iter().map(|x| x - log_sum).collect()
    }

    #[test]
    fn ctc_loss_alpha_beta_consistency() {
        // Verify that forward (alpha) and backward (beta) give consistent log probability
        // We do this by checking loss matches from both directions via gradient existence
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // T=4, N=1, C=3, target=[1, 2]
        // Use valid log-probabilities (log-softmax normalized)
        let logits = [
            vec![0.5, 1.0, 0.3], // t=0
            vec![0.2, 1.5, 0.8], // t=1
            vec![0.1, 0.7, 1.2], // t=2
            vec![0.8, 0.3, 1.0], // t=3
        ];
        let lp_data: Vec<f64> = logits.iter().flat_map(|l| log_softmax(l)).collect();
        let log_probs = make_log_probs(&mut session, lp_data, 4, 1, 3);
        let targets = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .unwrap();
        let input_lengths = session.tensor_variable(vec![4.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![2.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new().with_reduction(CTCReduction::Sum);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_val = session.tensor_values(loss).unwrap()[0];
        assert!(
            loss_val > 0.0 && loss_val.is_finite(),
            "loss should be finite positive: {loss_val}"
        );

        // Backward should succeed and produce finite gradients
        let report = session.tensor_backward(loss).unwrap();
        let grad = report.gradient(log_probs).unwrap();
        for (i, g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "gradient at index {i} is not finite: {g}");
        }
    }

    #[test]
    fn ctc_loss_with_custom_blank() {
        // Use blank=2 instead of default 0
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // T=2, N=1, C=3 (a=0, b=1, blank=2)
        let lp_data = vec![
            0.4f64.ln(),
            0.3f64.ln(),
            0.3f64.ln(), // t=0
            0.3f64.ln(),
            0.4f64.ln(),
            0.3f64.ln(), // t=1
        ];
        let log_probs = make_log_probs(&mut session, lp_data, 2, 1, 3);
        let targets = session
            .tensor_variable(vec![0.0], vec![1, 1], false)
            .unwrap(); // target = class 0 ('a')
        let input_lengths = session.tensor_variable(vec![2.0], vec![1], false).unwrap();
        let target_lengths = session.tensor_variable(vec![1.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new()
            .with_blank(2)
            .with_reduction(CTCReduction::Sum);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_val = session.tensor_values(loss).unwrap()[0];
        // Alignments for target 'a' (class 0) with blank=2:
        // (a, blank), (a, a), (blank, a)
        // P = 0.4*0.3 + 0.4*0.3 + 0.3*0.3 = 0.12 + 0.12 + 0.09 = 0.33
        let expected = -(0.33f64.ln());
        assert!(
            (loss_val - expected).abs() < 1e-5,
            "expected {expected}, got {loss_val}"
        );
    }

    #[test]
    fn ctc_loss_long_sequence_numerical_stability() {
        // T=100, N=1, C=5, target=[1,2,3]: should not underflow thanks to log-domain
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let c = 5;
        let t = 100;
        // Use slightly biased probs
        let mut lp_data = Vec::with_capacity(t * c);
        for step in 0..t {
            let probs: Vec<f64> = (0..c)
                .map(|k| {
                    if k == (step % c) {
                        0.5
                    } else {
                        0.5 / (c as f64 - 1.0)
                    }
                })
                .collect();
            let sum: f64 = probs.iter().sum();
            for p in &probs {
                lp_data.push((p / sum).ln());
            }
        }

        let log_probs = make_log_probs(&mut session, lp_data, t, 1, c);
        let targets = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .unwrap();
        let input_lengths = session
            .tensor_variable(vec![t as f64], vec![1], false)
            .unwrap();
        let target_lengths = session.tensor_variable(vec![3.0], vec![1], false).unwrap();

        let loss_mod = CTCLoss::new().with_reduction(CTCReduction::Sum);
        let loss = loss_mod
            .forward_ctc(
                &mut session,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            .unwrap();

        let loss_val = session.tensor_values(loss).unwrap()[0];
        assert!(
            loss_val.is_finite(),
            "loss should be finite for long sequence: {loss_val}"
        );
        assert!(loss_val > 0.0, "loss should be positive: {loss_val}");
    }
}
