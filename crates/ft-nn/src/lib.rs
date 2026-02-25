#![forbid(unsafe_code)]

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};
use ft_dispatch::{DispatchError, DispatchKeyError};

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
}

/// Dropout module (stochastic regularization).
///
/// During training, randomly zeros elements with probability `p`.
/// During eval, passes through unchanged.
pub struct Dropout {
    p: f64,
    training: bool,
}

impl Dropout {
    /// Create a new Dropout module with the given drop probability.
    #[must_use]
    pub fn new(p: f64) -> Self {
        Self { p, training: true }
    }

    /// Set the module to training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set the module to evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Check if the module is in training mode.
    #[must_use]
    pub fn is_training(&self) -> bool {
        self.training
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
        if !self.training || self.p == 0.0 {
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
    num_features: usize,
    eps: f64,
    momentum: f64,
    training: bool,
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

        Ok(Self {
            weight,
            bias,
            running_mean: std::cell::RefCell::new(vec![0.0; num_features]),
            running_var: std::cell::RefCell::new(vec![1.0; num_features]),
            num_features,
            eps,
            momentum,
            training: true,
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

    /// Set the module to training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set the module to evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Check if the module is in training mode.
    #[must_use]
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get a copy of the current running mean.
    pub fn running_mean(&self) -> Vec<f64> {
        self.running_mean.borrow().clone()
    }

    /// Get a copy of the current running variance.
    pub fn running_var(&self) -> Vec<f64> {
        self.running_var.borrow().clone()
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

        // Create tensors from running stats (not tracked by autograd)
        let rm = self.running_mean.borrow().clone();
        let rv = self.running_var.borrow().clone();
        let mean_t = session.tensor_variable(rm, vec![c], false)?;
        let var_t = session.tensor_variable(rv, vec![c], false)?;

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

        if self.training {
            self.forward_train(session, input, &input_shape)
        } else {
            self.forward_eval(session, input, &input_shape)
        }
    }

    fn parameters(&self) -> Vec<TensorNodeId> {
        vec![self.weight, self.bias]
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
        let eps_t =
            session.full(vec![batch_size, self.num_groups, group_numel], self.eps, false)?;
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
}

/// Identity module that passes input through unchanged.
///
/// Useful as a placeholder in `Sequential` containers or for skip connections
/// where a no-op branch is needed.
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

#[cfg(test)]
mod tests {
    use ft_api::FrankenTorchSession;
    use ft_core::ExecutionMode;

    use super::*;

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

        let mut dropout = Dropout::new(0.5);
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
        let mut dropout = Dropout::new(0.5);
        assert!(dropout.is_training());
        dropout.eval();
        assert!(!dropout.is_training());
        dropout.train();
        assert!(dropout.is_training());
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
        let mut dropout = Dropout::new(0.5);
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
        let mut bn = BatchNorm1d::new(&mut session, 2, 1e-5, 0.1).expect("batchnorm");

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
        let mut bn = BatchNorm1d::new(&mut session, 4, 1e-5, 0.1).expect("batchnorm");
        assert!(bn.is_training());
        bn.eval();
        assert!(!bn.is_training());
        bn.train();
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
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 4, 2], true)
            .expect("var");
        let y = gn.forward(&mut session, x).expect("forward");
        let loss = session.tensor_sum(y).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");

        let grad = session.tensor_gradient(&report, x);
        assert!(grad.is_some(), "GroupNorm should produce gradients for input");
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
        assert!(vals[3].abs() < 1e-2, "constant channel should be ~0, got {}", vals[3]);
        assert!(vals[4].abs() < 1e-2, "constant channel should be ~0, got {}", vals[4]);
        assert!(vals[5].abs() < 1e-2, "constant channel should be ~0, got {}", vals[5]);
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
}
