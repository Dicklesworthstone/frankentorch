#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};
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

    for (target_node, source_tensor) in updates {
        let source_node = session.tensor_variable_from_storage(source_tensor, false);
        session.tensor_zero_(target_node)?;
        session.tensor_add_(target_node, source_node)?;
    }

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
    if !(norm_type.is_finite() && norm_type > 0.0) && !norm_type.is_infinite() {
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
    for &parameter in parameters {
        let (param_values, param_meta) = session.tensor_values_meta(parameter)?;
        let len = param_values.len();
        let Some(end) = offset.checked_add(len) else {
            return Err(gradient_utils_error("vector_to_parameters offset overflow"));
        };
        if end > flat_values.len() {
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
        let output = session.zeros(vec![batch_size, self.out_channels, l_out], true)?;

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

        session
            .tensor_fill_(linear.weight(), 123.0)
            .expect("fill weight");
        session
            .tensor_fill_(linear.bias().expect("bias"), -42.0)
            .expect("fill bias");

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

        session
            .tensor_fill_(linear.weight(), 999.0)
            .expect("fill weight");
        session
            .tensor_fill_(linear.bias().expect("bias"), -777.0)
            .expect("fill bias");

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

        for &param in &params {
            session.tensor_fill_(param, 0.0).expect("fill");
        }

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
}
