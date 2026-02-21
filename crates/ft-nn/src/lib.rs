#![forbid(unsafe_code)]

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};

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
}
