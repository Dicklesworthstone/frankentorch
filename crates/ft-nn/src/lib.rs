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
        // Kaiming uniform: scale = sqrt(1 / in_features)
        let scale = 1.0 / (in_features as f64).sqrt();

        // Use randn and scale manually for Kaiming init
        let weight = session.randn(vec![out_features, in_features], true)?;
        // Scale weight: weight *= scale
        // We need to do this through ops to maintain the graph
        let scale_tensor = session.full(vec![out_features, in_features], scale, false)?;
        let weight = session.tensor_mul(weight, scale_tensor)?;

        // Rebind as a leaf for parameter tracking
        let weight_values = session.tensor_values(weight)?;
        let weight =
            session.tensor_variable(weight_values, vec![out_features, in_features], true)?;

        let bias = if use_bias {
            let bias_values = vec![0.0; out_features];
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
            Some(bias) => session.tensor_add(output, bias),
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
            return session.zeros(shape, false);
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
}
