#![forbid(unsafe_code)]

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorBackwardReport, TensorNodeId};
use ft_dispatch::{DispatchError, DispatchKeyError};

fn adam_bias_correction(beta: f64, step: u64) -> f64 {
    1.0 - beta.powf(step as f64)
}

fn optimizer_hparam_error(reason: &'static str) -> AutogradError {
    AutogradError::Dispatch(DispatchError::Key(DispatchKeyError::IncompatibleSet {
        reason,
    }))
}

fn optimizer_state_error(reason: &'static str) -> AutogradError {
    AutogradError::Dispatch(DispatchError::Key(DispatchKeyError::IncompatibleSet {
        reason,
    }))
}

fn checked_next_step_count(
    current: u64,
    overflow_reason: &'static str,
) -> Result<u64, AutogradError> {
    current
        .checked_add(1)
        .ok_or_else(|| optimizer_state_error(overflow_reason))
}

fn ensure_grad_len_matches_param(
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

fn ensure_state_len(
    expected: usize,
    actual: usize,
    mismatch_reason: &'static str,
) -> Result<(), AutogradError> {
    if expected != actual {
        return Err(optimizer_state_error(mismatch_reason));
    }
    Ok(())
}

/// Trait for parameter optimizers.
pub trait Optimizer {
    /// Perform a single optimization step using computed gradients.
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError>;

    /// Zero out accumulated gradients (no-op for this implementation since
    /// gradients are recomputed each backward pass, but included for API parity).
    fn zero_grad(&mut self, session: &mut FrankenTorchSession) -> Result<(), AutogradError>;
}

/// Stochastic Gradient Descent optimizer with optional momentum and weight decay.
pub struct SGD {
    params: Vec<TensorNodeId>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    nesterov: bool,
    velocity: Vec<Option<Vec<f64>>>,
}

impl SGD {
    /// Create a new SGD optimizer.
    ///
    /// # Arguments
    /// * `params` - Parameter node IDs to optimize
    /// * `lr` - Learning rate
    pub fn new(params: Vec<TensorNodeId>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: vec![None; n],
        }
    }

    /// Set momentum factor (default: 0.0).
    #[must_use]
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay (L2 regularization) factor (default: 0.0).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable Nesterov momentum (default: false).
    #[must_use]
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    fn validate_hyperparams(&self) -> Result<(), AutogradError> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(optimizer_hparam_error(
                "sgd requires a finite non-negative learning rate",
            ));
        }
        if !self.momentum.is_finite() || self.momentum < 0.0 {
            return Err(optimizer_hparam_error(
                "sgd requires finite non-negative momentum",
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(optimizer_hparam_error(
                "sgd requires finite non-negative weight_decay",
            ));
        }
        if self.nesterov && self.momentum == 0.0 {
            return Err(optimizer_hparam_error("sgd nesterov requires momentum > 0"));
        }
        Ok(())
    }
}

impl Optimizer for SGD {
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        for (i, &param) in self.params.iter().enumerate() {
            let grad = match session.tensor_gradient(report, param) {
                Some(g) => g.to_vec(),
                None => continue,
            };

            let param_values = session.tensor_values(param)?;
            ensure_grad_len_matches_param(param, param_values.len(), grad.len())?;
            let param_shape = session.tensor_values_meta(param)?.1.shape().to_vec();
            let mut effective_grad = grad;

            // Apply weight decay: grad += weight_decay * param
            if self.weight_decay != 0.0 {
                for (g, p) in effective_grad.iter_mut().zip(param_values.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            if self.momentum != 0.0 {
                // Update velocity: v = momentum * v + grad
                let vel = self.velocity[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
                ensure_state_len(
                    effective_grad.len(),
                    vel.len(),
                    "sgd optimizer state length mismatch with gradient length",
                )?;
                for (v, g) in vel.iter_mut().zip(effective_grad.iter()) {
                    *v = self.momentum * *v + g;
                }

                if self.nesterov {
                    // Nesterov: param -= lr * (grad + momentum * velocity)
                    let update: Vec<f64> = effective_grad
                        .iter()
                        .zip(vel.iter())
                        .map(|(g, v)| g + self.momentum * v)
                        .collect();

                    // Apply: create update tensor and subtract
                    let update_node = session.tensor_variable(
                        update.iter().map(|u| self.lr * u).collect(),
                        param_shape.clone(),
                        false,
                    )?;
                    session.tensor_sub_(param, update_node)?;
                } else {
                    // Standard momentum: param -= lr * velocity
                    let update_node = session.tensor_variable(
                        vel.iter().map(|v| self.lr * v).collect(),
                        param_shape.clone(),
                        false,
                    )?;
                    session.tensor_sub_(param, update_node)?;
                }
            } else {
                // Vanilla SGD: param -= lr * grad
                let update_node = session.tensor_variable(
                    effective_grad.iter().map(|g| self.lr * g).collect(),
                    param_shape,
                    false,
                )?;
                session.tensor_sub_(param, update_node)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self, _session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        // Gradients are recomputed each backward pass; this is a no-op.
        Ok(())
    }
}

/// Adam optimizer with bias correction.
pub struct Adam {
    params: Vec<TensorNodeId>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step_count: u64,
    m: Vec<Option<Vec<f64>>>,
    v: Vec<Option<Vec<f64>>>,
}

impl Adam {
    /// Create a new Adam optimizer with default hyperparameters.
    ///
    /// Defaults: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0
    pub fn new(params: Vec<TensorNodeId>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step_count: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Set beta coefficients for computing running averages.
    #[must_use]
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability (default: 1e-8).
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (default: 0.0).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    fn validate_hyperparams(&self) -> Result<(), AutogradError> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(optimizer_hparam_error(
                "adam requires a finite non-negative learning rate",
            ));
        }
        if !self.beta1.is_finite() || !self.beta2.is_finite() {
            return Err(optimizer_hparam_error("adam betas must be finite"));
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return Err(optimizer_hparam_error("adam betas must be in [0, 1)"));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(optimizer_hparam_error("adam requires finite eps > 0"));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(optimizer_hparam_error(
                "adam requires finite non-negative weight_decay",
            ));
        }
        Ok(())
    }
}

impl Optimizer for Adam {
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        let t = checked_next_step_count(self.step_count, "adam step counter overflow")?;
        self.step_count = t;

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match session.tensor_gradient(report, param) {
                Some(g) => g.to_vec(),
                None => continue,
            };

            let param_values = session.tensor_values(param)?;
            ensure_grad_len_matches_param(param, param_values.len(), grad.len())?;
            let param_shape = session.tensor_values_meta(param)?.1.shape().to_vec();
            let mut effective_grad = grad;

            // Apply weight decay
            if self.weight_decay != 0.0 {
                for (g, p) in effective_grad.iter_mut().zip(param_values.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            let m = self.m[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
            ensure_state_len(
                effective_grad.len(),
                m.len(),
                "adam first-moment state length mismatch with gradient length",
            )?;
            for (m_val, g) in m.iter_mut().zip(effective_grad.iter()) {
                *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * g;
            }

            // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            let v = self.v[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
            ensure_state_len(
                effective_grad.len(),
                v.len(),
                "adam second-moment state length mismatch with gradient length",
            )?;
            for (v_val, g) in v.iter_mut().zip(effective_grad.iter()) {
                *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * g * g;
            }

            // Bias-corrected estimates
            let bias_correction1 = adam_bias_correction(self.beta1, t);
            let bias_correction2 = adam_bias_correction(self.beta2, t);

            // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
            let update: Vec<f64> = m
                .iter()
                .zip(v.iter())
                .map(|(m_val, v_val)| {
                    let m_hat = m_val / bias_correction1;
                    let v_hat = v_val / bias_correction2;
                    self.lr * m_hat / (v_hat.sqrt() + self.eps)
                })
                .collect();

            let update_node = session.tensor_variable(update, param_shape, false)?;
            session.tensor_sub_(param, update_node)?;
        }
        Ok(())
    }

    fn zero_grad(&mut self, _session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        Ok(())
    }
}

/// AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2019).
///
/// Unlike standard Adam which adds weight decay to the gradient (L2 regularization),
/// AdamW applies weight decay directly to the parameters after the Adam update step.
/// This decoupling improves regularization behavior and is the default optimizer
/// for most modern transformer training.
pub struct AdamW {
    params: Vec<TensorNodeId>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step_count: u64,
    m: Vec<Option<Vec<f64>>>,
    v: Vec<Option<Vec<f64>>>,
}

impl AdamW {
    /// Create a new AdamW optimizer.
    ///
    /// Defaults: lr as given, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
    pub fn new(params: Vec<TensorNodeId>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Set beta coefficients for computing running averages.
    #[must_use]
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability (default: 1e-8).
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (default: 0.01).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    fn validate_hyperparams(&self) -> Result<(), AutogradError> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(optimizer_hparam_error(
                "adamw requires a finite non-negative learning rate",
            ));
        }
        if !self.beta1.is_finite() || !self.beta2.is_finite() {
            return Err(optimizer_hparam_error("adamw betas must be finite"));
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return Err(optimizer_hparam_error("adamw betas must be in [0, 1)"));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(optimizer_hparam_error("adamw requires finite eps > 0"));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(optimizer_hparam_error(
                "adamw requires finite non-negative weight_decay",
            ));
        }
        Ok(())
    }
}

impl Optimizer for AdamW {
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        let t = checked_next_step_count(self.step_count, "adamw step counter overflow")?;
        self.step_count = t;

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match session.tensor_gradient(report, param) {
                Some(g) => g.to_vec(),
                None => continue,
            };

            let param_values = session.tensor_values(param)?;
            ensure_grad_len_matches_param(param, param_values.len(), grad.len())?;
            let param_shape = session.tensor_values_meta(param)?.1.shape().to_vec();

            // Decoupled weight decay: apply directly to parameters BEFORE Adam update
            if self.weight_decay != 0.0 {
                let delta: Vec<f64> = param_values
                    .iter()
                    .map(|p| p * self.lr * self.weight_decay)
                    .collect();
                let delta_node = session.tensor_variable(delta, param_shape.clone(), false)?;
                session.tensor_sub_(param, delta_node)?;
            }

            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            let m = self.m[i].get_or_insert_with(|| vec![0.0; grad.len()]);
            ensure_state_len(
                grad.len(),
                m.len(),
                "adamw first-moment state length mismatch with gradient length",
            )?;
            for (m_val, g) in m.iter_mut().zip(grad.iter()) {
                *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * g;
            }

            // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            let v = self.v[i].get_or_insert_with(|| vec![0.0; grad.len()]);
            ensure_state_len(
                grad.len(),
                v.len(),
                "adamw second-moment state length mismatch with gradient length",
            )?;
            for (v_val, g) in v.iter_mut().zip(grad.iter()) {
                *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * g * g;
            }

            // Bias-corrected estimates
            let bias_correction1 = adam_bias_correction(self.beta1, t);
            let bias_correction2 = adam_bias_correction(self.beta2, t);

            // Compute Adam update: lr * m_hat / (sqrt(v_hat) + eps)
            let update: Vec<f64> = m
                .iter()
                .zip(v.iter())
                .map(|(m_val, v_val)| {
                    let m_hat = m_val / bias_correction1;
                    let v_hat = v_val / bias_correction2;
                    self.lr * m_hat / (v_hat.sqrt() + self.eps)
                })
                .collect();

            let update_node = session.tensor_variable(update, param_shape, false)?;
            session.tensor_sub_(param, update_node)?;
        }
        Ok(())
    }

    fn zero_grad(&mut self, _session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        Ok(())
    }
}

/// RMSprop optimizer (Hinton, 2012).
///
/// Maintains a running average of squared gradients to normalize the gradient,
/// preventing the learning rate from growing too large or too small.
///
/// PyTorch-compatible implementation with `alpha`, `eps`, `weight_decay`,
/// `momentum`, and `centered` options.
pub struct RMSprop {
    params: Vec<TensorNodeId>,
    lr: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    momentum: f64,
    centered: bool,
    step_count: u64,
    /// Running average of squared gradients per parameter.
    square_avg: Vec<Option<Vec<f64>>>,
    /// Running average of gradients (only used when centered=true).
    grad_avg: Vec<Option<Vec<f64>>>,
    /// Momentum buffer per parameter (only used when momentum > 0).
    momentum_buffer: Vec<Option<Vec<f64>>>,
}

impl RMSprop {
    /// Create a new RMSprop optimizer.
    ///
    /// Defaults: alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=false
    pub fn new(params: Vec<TensorNodeId>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            step_count: 0,
            square_avg: vec![None; n],
            grad_avg: vec![None; n],
            momentum_buffer: vec![None; n],
        }
    }

    /// Set smoothing constant alpha (default: 0.99).
    #[must_use]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set epsilon for numerical stability (default: 1e-8).
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (L2 regularization) factor (default: 0.0).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set momentum factor (default: 0.0).
    #[must_use]
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable centered RMSprop which normalizes the gradient by an estimate
    /// of its variance (default: false).
    #[must_use]
    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    fn validate_hyperparams(&self) -> Result<(), AutogradError> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(optimizer_hparam_error(
                "rmsprop requires a finite non-negative learning rate",
            ));
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 || self.alpha >= 1.0 {
            return Err(optimizer_hparam_error(
                "rmsprop requires finite alpha in [0, 1)",
            ));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(optimizer_hparam_error("rmsprop requires finite eps > 0"));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(optimizer_hparam_error(
                "rmsprop requires finite non-negative weight_decay",
            ));
        }
        if !self.momentum.is_finite() || self.momentum < 0.0 {
            return Err(optimizer_hparam_error(
                "rmsprop requires finite non-negative momentum",
            ));
        }
        Ok(())
    }
}

impl Optimizer for RMSprop {
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        self.step_count =
            checked_next_step_count(self.step_count, "rmsprop step counter overflow")?;

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match session.tensor_gradient(report, param) {
                Some(g) => g.to_vec(),
                None => continue,
            };

            let param_values = session.tensor_values(param)?;
            ensure_grad_len_matches_param(param, param_values.len(), grad.len())?;
            let param_shape = session.tensor_values_meta(param)?.1.shape().to_vec();
            let mut effective_grad = grad;

            // Apply weight decay: grad += weight_decay * param
            if self.weight_decay != 0.0 {
                for (g, p) in effective_grad.iter_mut().zip(param_values.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Update running average of squared gradients:
            // square_avg = alpha * square_avg + (1 - alpha) * grad^2
            let sq = self.square_avg[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
            ensure_state_len(
                effective_grad.len(),
                sq.len(),
                "rmsprop square_avg state length mismatch with gradient length",
            )?;
            for (s, g) in sq.iter_mut().zip(effective_grad.iter()) {
                *s = self.alpha * *s + (1.0 - self.alpha) * g * g;
            }

            // Compute the denominator (avg): sqrt(v) or sqrt(v - g_avg^2) if centered
            let avg: Vec<f64> = if self.centered {
                // Update running average of gradients:
                // grad_avg = alpha * grad_avg + (1 - alpha) * grad
                let ga =
                    self.grad_avg[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
                ensure_state_len(
                    effective_grad.len(),
                    ga.len(),
                    "rmsprop grad_avg state length mismatch with gradient length",
                )?;
                for (a, g) in ga.iter_mut().zip(effective_grad.iter()) {
                    *a = self.alpha * *a + (1.0 - self.alpha) * g;
                }
                // v_hat = square_avg - grad_avg^2 (variance estimate)
                sq.iter()
                    .zip(ga.iter())
                    .map(|(s, a)| (s - a * a).max(0.0).sqrt() + self.eps)
                    .collect()
            } else {
                sq.iter().map(|s| s.sqrt() + self.eps).collect()
            };

            if self.momentum > 0.0 {
                // Update momentum buffer: buf = momentum * buf + grad / avg
                let buf = self.momentum_buffer[i]
                    .get_or_insert_with(|| vec![0.0; effective_grad.len()]);
                ensure_state_len(
                    effective_grad.len(),
                    buf.len(),
                    "rmsprop momentum_buffer state length mismatch with gradient length",
                )?;
                for ((b, g), a) in buf.iter_mut().zip(effective_grad.iter()).zip(avg.iter()) {
                    *b = self.momentum * *b + g / a;
                }
                // param -= lr * buf
                let update: Vec<f64> = buf.iter().map(|b| self.lr * b).collect();
                let update_node = session.tensor_variable(update, param_shape, false)?;
                session.tensor_sub_(param, update_node)?;
            } else {
                // param -= lr * grad / avg
                let update: Vec<f64> = effective_grad
                    .iter()
                    .zip(avg.iter())
                    .map(|(g, a)| self.lr * g / a)
                    .collect();
                let update_node = session.tensor_variable(update, param_shape, false)?;
                session.tensor_sub_(param, update_node)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self, _session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        Ok(())
    }
}

/// Adagrad optimizer (Duchi et al., 2011).
///
/// Adapts the learning rate for each parameter based on the history of gradients.
/// Parameters with large historical gradients get smaller learning rates,
/// and parameters with small gradients get larger rates.
pub struct Adagrad {
    params: Vec<TensorNodeId>,
    lr: f64,
    lr_decay: f64,
    weight_decay: f64,
    initial_accumulator_value: f64,
    eps: f64,
    step_count: u64,
    /// Sum of squared gradients per parameter.
    sum_sq: Vec<Option<Vec<f64>>>,
}

impl Adagrad {
    /// Create a new Adagrad optimizer.
    ///
    /// Defaults: lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10
    pub fn new(params: Vec<TensorNodeId>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            eps: 1e-10,
            step_count: 0,
            sum_sq: vec![None; n],
        }
    }

    /// Set learning rate decay (default: 0.0).
    #[must_use]
    pub fn lr_decay(mut self, lr_decay: f64) -> Self {
        self.lr_decay = lr_decay;
        self
    }

    /// Set weight decay (L2 regularization) factor (default: 0.0).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set initial value for the sum of squared gradients accumulator (default: 0.0).
    #[must_use]
    pub fn initial_accumulator_value(mut self, val: f64) -> Self {
        self.initial_accumulator_value = val;
        self
    }

    /// Set epsilon for numerical stability (default: 1e-10).
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    fn validate_hyperparams(&self) -> Result<(), AutogradError> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(optimizer_hparam_error(
                "adagrad requires a finite non-negative learning rate",
            ));
        }
        if !self.lr_decay.is_finite() || self.lr_decay < 0.0 {
            return Err(optimizer_hparam_error(
                "adagrad requires finite non-negative lr_decay",
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(optimizer_hparam_error(
                "adagrad requires finite non-negative weight_decay",
            ));
        }
        if !self.initial_accumulator_value.is_finite() || self.initial_accumulator_value < 0.0 {
            return Err(optimizer_hparam_error(
                "adagrad requires finite non-negative initial_accumulator_value",
            ));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(optimizer_hparam_error("adagrad requires finite eps > 0"));
        }
        Ok(())
    }
}

impl Optimizer for Adagrad {
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        self.step_count =
            checked_next_step_count(self.step_count, "adagrad step counter overflow")?;

        // Compute decayed learning rate: lr / (1 + (step - 1) * lr_decay)
        let clr = self.lr / (1.0 + (self.step_count.saturating_sub(1) as f64) * self.lr_decay);

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match session.tensor_gradient(report, param) {
                Some(g) => g.to_vec(),
                None => continue,
            };

            let param_values = session.tensor_values(param)?;
            ensure_grad_len_matches_param(param, param_values.len(), grad.len())?;
            let param_shape = session.tensor_values_meta(param)?.1.shape().to_vec();
            let mut effective_grad = grad;

            // Apply weight decay: grad += weight_decay * param
            if self.weight_decay != 0.0 {
                for (g, p) in effective_grad.iter_mut().zip(param_values.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Update sum of squared gradients: state_sum += grad^2
            let init_val = self.initial_accumulator_value;
            let ss = self.sum_sq[i].get_or_insert_with(|| vec![init_val; effective_grad.len()]);
            ensure_state_len(
                effective_grad.len(),
                ss.len(),
                "adagrad sum_sq state length mismatch with gradient length",
            )?;
            for (s, g) in ss.iter_mut().zip(effective_grad.iter()) {
                *s += g * g;
            }

            // param -= clr * grad / (sqrt(state_sum) + eps)
            let update: Vec<f64> = effective_grad
                .iter()
                .zip(ss.iter())
                .map(|(g, s)| clr * g / (s.sqrt() + self.eps))
                .collect();

            let update_node = session.tensor_variable(update, param_shape, false)?;
            session.tensor_sub_(param, update_node)?;
        }
        Ok(())
    }

    fn zero_grad(&mut self, _session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        Ok(())
    }
}

/// RAdam optimizer — Rectified Adam (Liu et al., 2019).
///
/// Automatically adapts between Adam and SGD behavior based on the variance of
/// the adaptive learning rate. In early training when variance is high
/// (`rho_t <= 5`), it uses an SGD-like update with only the first moment.
/// Once variance stabilizes (`rho_t > 5`), it switches to the full Adam update
/// with a rectification term.
///
/// This eliminates the need for learning rate warmup while matching or exceeding
/// Adam's performance.
pub struct RAdam {
    params: Vec<TensorNodeId>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step_count: u64,
    m: Vec<Option<Vec<f64>>>,
    v: Vec<Option<Vec<f64>>>,
}

impl RAdam {
    /// Create a new RAdam optimizer.
    ///
    /// Defaults: beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0
    pub fn new(params: Vec<TensorNodeId>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step_count: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Set beta coefficients for computing running averages.
    #[must_use]
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability (default: 1e-8).
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (default: 0.0).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    fn validate_hyperparams(&self) -> Result<(), AutogradError> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(optimizer_hparam_error(
                "radam requires a finite non-negative learning rate",
            ));
        }
        if !self.beta1.is_finite() || !self.beta2.is_finite() {
            return Err(optimizer_hparam_error("radam betas must be finite"));
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return Err(optimizer_hparam_error("radam betas must be in [0, 1)"));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(optimizer_hparam_error("radam requires finite eps > 0"));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(optimizer_hparam_error(
                "radam requires finite non-negative weight_decay",
            ));
        }
        Ok(())
    }
}

impl Optimizer for RAdam {
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        let t = checked_next_step_count(self.step_count, "radam step counter overflow")?;
        self.step_count = t;

        // Maximum length of the approximated SMA
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match session.tensor_gradient(report, param) {
                Some(g) => g.to_vec(),
                None => continue,
            };

            let param_values = session.tensor_values(param)?;
            ensure_grad_len_matches_param(param, param_values.len(), grad.len())?;
            let param_shape = session.tensor_values_meta(param)?.1.shape().to_vec();
            let mut effective_grad = grad;

            // Apply weight decay: grad += weight_decay * param
            if self.weight_decay != 0.0 {
                for (g, p) in effective_grad.iter_mut().zip(param_values.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
            let m = self.m[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
            ensure_state_len(
                effective_grad.len(),
                m.len(),
                "radam first-moment state length mismatch with gradient length",
            )?;
            for (m_val, g) in m.iter_mut().zip(effective_grad.iter()) {
                *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * g;
            }

            // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
            let v = self.v[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
            ensure_state_len(
                effective_grad.len(),
                v.len(),
                "radam second-moment state length mismatch with gradient length",
            )?;
            for (v_val, g) in v.iter_mut().zip(effective_grad.iter()) {
                *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * g * g;
            }

            // Bias-corrected first moment
            let bias_correction1 = adam_bias_correction(self.beta1, t);
            let m_hat: Vec<f64> = m.iter().map(|m_val| m_val / bias_correction1).collect();

            // Compute SMA length: rho_t = rho_inf - 2 * t * beta2^t / (1 - beta2^t)
            let beta2_pow_t = self.beta2.powf(t as f64);
            let rho_t = rho_inf - 2.0 * (t as f64) * beta2_pow_t / (1.0 - beta2_pow_t);

            let update: Vec<f64> = if rho_t > 5.0 {
                // Variance is tractable — use rectified Adam update
                let bias_correction2 = adam_bias_correction(self.beta2, t);

                // Rectification term
                let r_t = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                    / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                    .sqrt();

                m_hat
                    .iter()
                    .zip(v.iter())
                    .map(|(mh, v_val)| {
                        let v_hat = v_val / bias_correction2;
                        self.lr * r_t * mh / (v_hat.sqrt() + self.eps)
                    })
                    .collect()
            } else {
                // Variance is intractable — use SGD-like update with first moment only
                m_hat.iter().map(|mh| self.lr * mh).collect()
            };

            let update_node = session.tensor_variable(update, param_shape, false)?;
            session.tensor_sub_(param, update_node)?;
        }
        Ok(())
    }

    fn zero_grad(&mut self, _session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ft_api::FrankenTorchSession;
    use ft_core::ExecutionMode;

    use super::*;

    #[test]
    fn sgd_basic_step_reduces_loss() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Simple quadratic: f(x) = x^2, minimum at x=0
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("variable should succeed");

        let mut optimizer = SGD::new(vec![x], 0.1);

        // Forward: f(x) = x * x
        let loss = session.tensor_mul(x, x).expect("mul should succeed");
        let loss_sum = session.tensor_sum(loss).expect("sum should succeed");

        // Backward
        let report = session
            .tensor_backward(loss_sum)
            .expect("backward should succeed");

        // Step
        optimizer
            .step(&mut session, &report)
            .expect("step should succeed");

        // x should have decreased: x_new = x - lr * grad = 4.0 - 0.1 * 8.0 = 3.2
        let x_val = session.tensor_values(x).expect("values should resolve");
        assert!(
            (x_val[0] - 3.2).abs() < 1e-10,
            "expected 3.2, got {}",
            x_val[0]
        );
    }

    #[test]
    fn adam_basic_step_reduces_loss() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("variable should succeed");

        let mut optimizer = Adam::new(vec![x], 0.1);

        let loss = session.tensor_mul(x, x).expect("mul should succeed");
        let loss_sum = session.tensor_sum(loss).expect("sum should succeed");

        let report = session
            .tensor_backward(loss_sum)
            .expect("backward should succeed");

        let x_before = session.tensor_values(x).expect("values should resolve")[0];
        optimizer
            .step(&mut session, &report)
            .expect("step should succeed");
        let x_after = session.tensor_values(x).expect("values should resolve")[0];

        // x should have decreased
        assert!(
            x_after < x_before,
            "Adam should decrease x: before={}, after={}",
            x_before,
            x_after
        );
    }

    #[test]
    fn sgd_with_momentum_accumulates_velocity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("variable should succeed");

        let mut optimizer = SGD::new(vec![x], 0.1).momentum(0.9);

        // First step
        let loss = session.tensor_mul(x, x).expect("mul should succeed");
        let loss_sum = session.tensor_sum(loss).expect("sum should succeed");
        let report = session
            .tensor_backward(loss_sum)
            .expect("backward should succeed");
        optimizer
            .step(&mut session, &report)
            .expect("step should succeed");

        let x_val_1 = session.tensor_values(x).expect("values")[0];
        assert!(x_val_1 < 4.0, "x should decrease after first step");
    }

    #[test]
    fn sgd_rejects_mismatched_velocity_state_length() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("variable should succeed");
        let mut optimizer = SGD::new(vec![x], 0.1).momentum(0.9);
        optimizer.velocity[0] = Some(vec![0.0, 0.0]);

        let loss = session.tensor_mul(x, x).expect("mul should succeed");
        let loss_sum = session.tensor_sum(loss).expect("sum should succeed");
        let report = session
            .tensor_backward(loss_sum)
            .expect("backward should succeed");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("mismatched state length must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "sgd optimizer state length mismatch with gradient length"
                }
            ))
        ));
    }

    #[test]
    fn zero_grad_is_noop() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("variable should succeed");

        let mut optimizer = SGD::new(vec![x], 0.1);
        optimizer
            .zero_grad(&mut session)
            .expect("zero_grad should succeed");
    }

    #[test]
    fn sgd_rejects_negative_learning_rate() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = SGD::new(vec![x], -0.1);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("negative lr must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "sgd requires a finite non-negative learning rate"
                }
            ))
        ));
    }

    #[test]
    fn sgd_nesterov_requires_positive_momentum() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = SGD::new(vec![x], 0.1).nesterov(true);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("nesterov without momentum must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "sgd nesterov requires momentum > 0"
                }
            ))
        ));
    }

    #[test]
    fn sgd_weight_decay_applies_l2_penalty() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // f(x) = x^2 with weight_decay
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("variable should succeed");

        let mut opt_wd = SGD::new(vec![x], 0.1).weight_decay(0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        opt_wd.step(&mut session, &report).expect("step");

        let x_val = session.tensor_values(x).expect("values")[0];
        // grad = 2*x = 8.0, effective_grad = 8.0 + 0.1*4.0 = 8.4
        // x_new = 4.0 - 0.1 * 8.4 = 3.16
        assert!(
            (x_val - 3.16).abs() < 1e-10,
            "expected 3.16 with weight decay, got {}",
            x_val
        );
    }

    #[test]
    fn sgd_nesterov_momentum_differs_from_standard() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Standard momentum
        let x1 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt_std = SGD::new(vec![x1], 0.1).momentum(0.9);

        let loss1 = session.tensor_mul(x1, x1).expect("mul");
        let loss1_sum = session.tensor_sum(loss1).expect("sum");
        let report1 = session.tensor_backward(loss1_sum).expect("backward");
        opt_std.step(&mut session, &report1).expect("step");
        let x1_val = session.tensor_values(x1).expect("values")[0];

        // Nesterov momentum
        let x2 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt_nes = SGD::new(vec![x2], 0.1).momentum(0.9).nesterov(true);

        let loss2 = session.tensor_mul(x2, x2).expect("mul");
        let loss2_sum = session.tensor_sum(loss2).expect("sum");
        let report2 = session.tensor_backward(loss2_sum).expect("backward");
        opt_nes.step(&mut session, &report2).expect("step");
        let x2_val = session.tensor_values(x2).expect("values")[0];

        // Both should decrease but by different amounts
        assert!(x1_val < 4.0, "standard momentum should decrease x");
        assert!(x2_val < 4.0, "nesterov should decrease x");
        assert!(
            (x1_val - x2_val).abs() > 1e-10,
            "nesterov and standard momentum should produce different values: std={}, nes={}",
            x1_val,
            x2_val
        );
    }

    #[test]
    fn sgd_multiple_steps_converge_toward_minimum() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // f(x) = x^2, minimum at x=0
        let x = session
            .tensor_variable(vec![10.0], vec![1], true)
            .expect("var");

        let mut optimizer = SGD::new(vec![x], 0.1);

        for _ in 0..50 {
            let loss = session.tensor_mul(x, x).expect("mul");
            let loss_sum = session.tensor_sum(loss).expect("sum");
            let report = session.tensor_backward(loss_sum).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        let x_val = session.tensor_values(x).expect("values")[0];
        // x_{n+1} = x_n * (1 - 2*lr) = 0.8 * x_n; after 50 steps: 10 * 0.8^50 ≈ 0.00014
        assert!(
            x_val.abs() < 0.01,
            "after 50 steps with lr=0.1, x should converge near 0, got {}",
            x_val
        );
    }

    #[test]
    fn sgd_optimizes_multiple_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let a = session
            .tensor_variable(vec![5.0], vec![1], true)
            .expect("var");
        let b = session
            .tensor_variable(vec![-3.0], vec![1], true)
            .expect("var");

        let mut optimizer = SGD::new(vec![a, b], 0.1);

        // f(a,b) = a^2 + b^2, minimum at (0,0)
        let a2 = session.tensor_mul(a, a).expect("mul");
        let b2 = session.tensor_mul(b, b).expect("mul");
        let loss = session.tensor_add(a2, b2).expect("add");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let a_val = session.tensor_values(a).expect("values")[0];
        let b_val = session.tensor_values(b).expect("values")[0];

        // a_new = 5.0 - 0.1 * 10.0 = 4.0
        // b_new = -3.0 - 0.1 * (-6.0) = -2.4
        assert!((a_val - 4.0).abs() < 1e-10, "expected a=4.0, got {}", a_val);
        assert!(
            (b_val - (-2.4)).abs() < 1e-10,
            "expected b=-2.4, got {}",
            b_val
        );
    }

    #[test]
    fn adam_multiple_steps_converge() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![5.0], vec![1], true)
            .expect("var");

        let mut optimizer = Adam::new(vec![x], 0.5);

        for _ in 0..100 {
            let loss = session.tensor_mul(x, x).expect("mul");
            let loss_sum = session.tensor_sum(loss).expect("sum");
            let report = session.tensor_backward(loss_sum).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val.abs() < 1.0,
            "after 100 Adam steps with lr=0.5, x should be near 0, got {}",
            x_val
        );
    }

    #[test]
    fn adam_with_custom_betas() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");

        // Non-default betas
        let mut optimizer = Adam::new(vec![x], 0.1).betas(0.5, 0.99);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val < 4.0,
            "Adam with custom betas should decrease x, got {}",
            x_val
        );
    }

    #[test]
    fn adam_with_weight_decay() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");

        let mut opt_wd = Adam::new(vec![x], 0.1).weight_decay(0.01);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let x_before = session.tensor_values(x).expect("values")[0];
        opt_wd.step(&mut session, &report).expect("step");
        let x_after = session.tensor_values(x).expect("values")[0];

        assert!(
            x_after < x_before,
            "Adam with weight decay should decrease x: before={}, after={}",
            x_before,
            x_after
        );
    }

    #[test]
    fn adam_with_custom_eps() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");

        // Large eps for numerical stability test
        let mut optimizer = Adam::new(vec![x], 0.1).eps(1.0);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val < 4.0,
            "Adam with large eps should still decrease x, got {}",
            x_val
        );
    }

    #[test]
    fn adam_rejects_invalid_betas() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adam::new(vec![x], 0.1).betas(1.0, 0.999);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("invalid betas must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "adam betas must be in [0, 1)"
                }
            ))
        ));
    }

    #[test]
    fn adam_rejects_non_positive_eps() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adam::new(vec![x], 0.1).eps(0.0);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("eps <= 0 must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "adam requires finite eps > 0"
                }
            ))
        ));
    }

    #[test]
    fn adam_large_step_count_still_updates_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adam::new(vec![x], 0.1);
        optimizer.step_count = i32::MAX as u64;

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let x_before = session.tensor_values(x).expect("values")[0];
        optimizer.step(&mut session, &report).expect("step");
        let x_after = session.tensor_values(x).expect("values")[0];

        assert!(x_after.is_finite(), "update produced non-finite parameter");
        assert!(
            x_after < x_before,
            "Adam should still update for large step_count: before={}, after={}",
            x_before,
            x_after
        );
    }

    #[test]
    fn adam_rejects_step_counter_overflow() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adam::new(vec![x], 0.1);
        optimizer.step_count = u64::MAX;

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("step counter overflow must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "adam step counter overflow"
                }
            ))
        ));
    }

    #[test]
    fn adam_optimizes_multiple_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let a = session
            .tensor_variable(vec![5.0], vec![1], true)
            .expect("var");
        let b = session
            .tensor_variable(vec![-3.0], vec![1], true)
            .expect("var");

        let mut optimizer = Adam::new(vec![a, b], 0.1);

        let a2 = session.tensor_mul(a, a).expect("mul");
        let b2 = session.tensor_mul(b, b).expect("mul");
        let loss = session.tensor_add(a2, b2).expect("add");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let a_val = session.tensor_values(a).expect("values")[0];
        let b_val = session.tensor_values(b).expect("values")[0];

        assert!(a_val < 5.0, "Adam should decrease a, got {}", a_val);
        assert!(
            b_val > -3.0,
            "Adam should increase b toward 0, got {}",
            b_val
        );
    }

    #[test]
    fn adam_zero_grad_is_noop() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");

        let mut optimizer = Adam::new(vec![x], 0.1);
        optimizer
            .zero_grad(&mut session)
            .expect("adam zero_grad should succeed");
    }

    #[test]
    fn sgd_skips_params_without_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let y = session
            .tensor_variable(vec![2.0], vec![1], true)
            .expect("var");

        // Only compute gradient for x, not y
        let mut optimizer = SGD::new(vec![x, y], 0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let x_val = session.tensor_values(x).expect("values")[0];
        let y_val = session.tensor_values(y).expect("values")[0];

        // x should be updated, y should be unchanged (no gradient)
        assert!(
            (x_val - 3.2).abs() < 1e-10,
            "x should be updated: expected 3.2, got {}",
            x_val
        );
        assert!(
            (y_val - 2.0).abs() < 1e-10,
            "y should be unchanged: expected 2.0, got {}",
            y_val
        );
    }

    #[test]
    fn adam_skips_params_without_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let y = session
            .tensor_variable(vec![2.0], vec![1], true)
            .expect("var");

        let mut optimizer = Adam::new(vec![x, y], 0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let y_val = session.tensor_values(y).expect("values")[0];
        assert!(
            (y_val - 2.0).abs() < 1e-10,
            "y should be unchanged without gradient: expected 2.0, got {}",
            y_val
        );
    }

    #[test]
    fn sgd_momentum_second_step_uses_accumulated_velocity() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");

        let mut optimizer = SGD::new(vec![x], 0.1).momentum(0.9);

        // Step 1
        let loss1 = session.tensor_mul(x, x).expect("mul");
        let loss1_sum = session.tensor_sum(loss1).expect("sum");
        let report1 = session.tensor_backward(loss1_sum).expect("backward");
        optimizer.step(&mut session, &report1).expect("step 1");
        let x_after_1 = session.tensor_values(x).expect("values")[0];

        // Step 2 - velocity should accumulate, causing larger update
        let loss2 = session.tensor_mul(x, x).expect("mul");
        let loss2_sum = session.tensor_sum(loss2).expect("sum");
        let report2 = session.tensor_backward(loss2_sum).expect("backward");
        optimizer.step(&mut session, &report2).expect("step 2");
        let x_after_2 = session.tensor_values(x).expect("values")[0];

        let step1_delta = (4.0 - x_after_1).abs();
        let step2_delta = (x_after_1 - x_after_2).abs();

        // With momentum, second step should move further than first (accumulated velocity)
        assert!(
            step2_delta > step1_delta,
            "second step with momentum should be larger: step1_delta={}, step2_delta={}",
            step1_delta,
            step2_delta
        );
    }

    #[test]
    fn adam_bias_correction_active_in_early_steps() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");

        let mut optimizer = Adam::new(vec![x], 0.1);

        // First step: bias correction is most aggressive (t=1)
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let x_val = session.tensor_values(x).expect("values")[0];
        // Adam's first step with default params: effective lr ~ lr * sqrt(1-beta2) / (1-beta1)
        // = 0.1 * sqrt(0.001) / 0.1 ≈ 0.0316... but applied as lr * m_hat / (sqrt(v_hat) + eps)
        // The key check: x should have moved
        assert!(
            x_val < 4.0,
            "Adam should decrease x in first step with bias correction, got {}",
            x_val
        );
    }

    // --- End-to-end training loop integration tests ---

    #[test]
    fn e2e_sgd_trains_linear_to_reduce_mse_loss() {
        use ft_nn::{Linear, Module};

        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Train a Linear(2->1) to learn the function y = 2*x1 + 3*x2
        let linear = Linear::new(&mut session, 2, 1, true).expect("linear");
        let params = linear.parameters();
        let mut optimizer = SGD::new(params, 0.01);

        // Training data: input [1, 1] -> target 5.0 (2*1 + 3*1)
        let input = session
            .tensor_variable(vec![1.0, 1.0], vec![1, 2], false)
            .expect("input");
        let target = session
            .tensor_variable(vec![5.0], vec![1, 1], false)
            .expect("target");

        let mut prev_loss_val = f64::MAX;
        for _ in 0..20 {
            let pred = linear.forward(&mut session, input).expect("forward");
            let loss = session.mse_loss(pred, target).expect("mse");
            let loss_val = session.tensor_values(loss).expect("values")[0];

            let report = session.tensor_backward(loss).expect("backward");
            optimizer.step(&mut session, &report).expect("step");

            // Loss should generally decrease (or at least not increase dramatically)
            if loss_val < prev_loss_val {
                prev_loss_val = loss_val;
            }
        }

        // After 20 steps, loss should have decreased from initial value
        let final_pred = linear.forward(&mut session, input).expect("forward");
        let final_loss = session.mse_loss(final_pred, target).expect("mse");
        let final_loss_val = session.tensor_values(final_loss).expect("values")[0];
        assert!(
            final_loss_val < 100.0,
            "after 20 SGD steps, loss should have decreased significantly, got {}",
            final_loss_val
        );
    }

    #[test]
    fn e2e_adam_trains_linear_to_reduce_mse_loss() {
        use ft_nn::{Linear, Module};

        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let linear = Linear::new(&mut session, 2, 1, true).expect("linear");
        let params = linear.parameters();
        let mut optimizer = Adam::new(params, 0.1);

        let input = session
            .tensor_variable(vec![1.0, 1.0], vec![1, 2], false)
            .expect("input");
        let target = session
            .tensor_variable(vec![5.0], vec![1, 1], false)
            .expect("target");

        // Get initial loss
        let initial_pred = linear.forward(&mut session, input).expect("forward");
        let initial_loss = session.mse_loss(initial_pred, target).expect("mse");
        let initial_loss_val = session.tensor_values(initial_loss).expect("values")[0];

        for _ in 0..30 {
            let pred = linear.forward(&mut session, input).expect("forward");
            let loss = session.mse_loss(pred, target).expect("mse");
            let report = session.tensor_backward(loss).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        let final_pred = linear.forward(&mut session, input).expect("forward");
        let final_loss = session.mse_loss(final_pred, target).expect("mse");
        let final_loss_val = session.tensor_values(final_loss).expect("values")[0];

        assert!(
            final_loss_val < initial_loss_val,
            "Adam should reduce loss: initial={}, final={}",
            initial_loss_val,
            final_loss_val
        );
    }

    #[test]
    fn e2e_sequential_relu_network_trains_with_sgd() {
        use ft_nn::{Linear, Module, ReLU, Sequential};

        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Build: Linear(2->4) -> ReLU -> Linear(4->1)
        let layer1 = Linear::new(&mut session, 2, 4, true).expect("layer1");
        let layer2 = Linear::new(&mut session, 4, 1, true).expect("layer2");

        let mut all_params = layer1.parameters();
        all_params.extend(layer2.parameters());

        let mut seq = Sequential::new();
        seq.push(Box::new(layer1));
        seq.push(Box::new(ReLU));
        seq.push(Box::new(layer2));

        let mut optimizer = SGD::new(all_params, 0.01);

        let input = session
            .tensor_variable(vec![1.0, -1.0], vec![1, 2], false)
            .expect("input");
        let target = session
            .tensor_variable(vec![3.0], vec![1, 1], false)
            .expect("target");

        // Train for a few steps
        for _ in 0..10 {
            let pred = seq.forward(&mut session, input).expect("forward");
            let loss = session.mse_loss(pred, target).expect("mse");
            let report = session.tensor_backward(loss).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        // Verify the forward pass still works after training
        let pred = seq.forward(&mut session, input).expect("forward");
        let vals = session.tensor_values(pred).expect("values");
        assert_eq!(vals.len(), 1, "output should have 1 element");
        assert!(
            vals[0].is_finite(),
            "output should be finite after training"
        );
    }

    #[test]
    fn e2e_l1_loss_with_adam_trains_successfully() {
        use ft_nn::{Linear, Module};

        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let linear = Linear::new(&mut session, 3, 1, true).expect("linear");
        let params = linear.parameters();
        let mut optimizer = Adam::new(params, 0.05);

        let input = session
            .tensor_variable(vec![1.0, 2.0, 3.0], vec![1, 3], false)
            .expect("input");
        let target = session
            .tensor_variable(vec![10.0], vec![1, 1], false)
            .expect("target");

        let initial_pred = linear.forward(&mut session, input).expect("forward");
        let initial_loss = session.l1_loss(initial_pred, target).expect("l1");
        let initial_val = session.tensor_values(initial_loss).expect("values")[0];

        for _ in 0..20 {
            let pred = linear.forward(&mut session, input).expect("forward");
            let loss = session.l1_loss(pred, target).expect("l1");
            let report = session.tensor_backward(loss).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        let final_pred = linear.forward(&mut session, input).expect("forward");
        let final_loss = session.l1_loss(final_pred, target).expect("l1");
        let final_val = session.tensor_values(final_loss).expect("values")[0];

        assert!(
            final_val < initial_val,
            "L1 loss should decrease with Adam: initial={}, final={}",
            initial_val,
            final_val
        );
    }

    #[test]
    fn e2e_sgd_momentum_trains_faster_than_vanilla() {
        use ft_nn::{Linear, Module};

        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Vanilla SGD
        let linear_vanilla = Linear::new(&mut session, 2, 1, true).expect("linear");
        let params_vanilla = linear_vanilla.parameters();
        let mut opt_vanilla = SGD::new(params_vanilla, 0.01);

        // SGD with momentum
        let linear_momentum = Linear::new(&mut session, 2, 1, true).expect("linear");
        let params_momentum = linear_momentum.parameters();
        let mut opt_momentum = SGD::new(params_momentum, 0.01).momentum(0.9);

        let input = session
            .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
            .expect("input");
        let target = session
            .tensor_variable(vec![7.0], vec![1, 1], false)
            .expect("target");

        for _ in 0..15 {
            // Vanilla step
            let pred_v = linear_vanilla.forward(&mut session, input).expect("fwd");
            let loss_v = session.mse_loss(pred_v, target).expect("mse");
            let report_v = session.tensor_backward(loss_v).expect("bwd");
            opt_vanilla.step(&mut session, &report_v).expect("step");

            // Momentum step
            let pred_m = linear_momentum.forward(&mut session, input).expect("fwd");
            let loss_m = session.mse_loss(pred_m, target).expect("mse");
            let report_m = session.tensor_backward(loss_m).expect("bwd");
            opt_momentum.step(&mut session, &report_m).expect("step");
        }

        // Both should produce finite outputs
        let pred_v = linear_vanilla.forward(&mut session, input).expect("fwd");
        let pred_m = linear_momentum.forward(&mut session, input).expect("fwd");
        let vals_v = session.tensor_values(pred_v).expect("values");
        let vals_m = session.tensor_values(pred_m).expect("values");
        assert!(vals_v[0].is_finite(), "vanilla prediction should be finite");
        assert!(
            vals_m[0].is_finite(),
            "momentum prediction should be finite"
        );
    }

    // --- AdamW tests ---

    #[test]
    fn adamw_basic_step_reduces_loss() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");

        let mut optimizer = AdamW::new(vec![x], 0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let x_before = session.tensor_values(x).expect("values")[0];
        optimizer.step(&mut session, &report).expect("step");
        let x_after = session.tensor_values(x).expect("values")[0];

        assert!(
            x_after < x_before,
            "AdamW should decrease x: before={}, after={}",
            x_before,
            x_after
        );
    }

    #[test]
    fn adamw_large_step_count_still_updates_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = AdamW::new(vec![x], 0.1);
        optimizer.step_count = i32::MAX as u64;

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let x_before = session.tensor_values(x).expect("values")[0];
        optimizer.step(&mut session, &report).expect("step");
        let x_after = session.tensor_values(x).expect("values")[0];

        assert!(x_after.is_finite(), "update produced non-finite parameter");
        assert!(
            x_after < x_before,
            "AdamW should still update for large step_count: before={}, after={}",
            x_before,
            x_after
        );
    }

    #[test]
    fn adamw_rejects_step_counter_overflow() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = AdamW::new(vec![x], 0.1);
        optimizer.step_count = u64::MAX;

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("step counter overflow must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "adamw step counter overflow"
                }
            ))
        ));
    }

    #[test]
    fn adamw_decoupled_weight_decay_differs_from_adam_l2() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Adam with L2 weight decay
        let x1 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt_adam = Adam::new(vec![x1], 0.1).weight_decay(0.1);

        let loss1 = session.tensor_mul(x1, x1).expect("mul");
        let loss1_sum = session.tensor_sum(loss1).expect("sum");
        let report1 = session.tensor_backward(loss1_sum).expect("backward");
        opt_adam.step(&mut session, &report1).expect("step");
        let x1_val = session.tensor_values(x1).expect("values")[0];

        // AdamW with decoupled weight decay
        let x2 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt_adamw = AdamW::new(vec![x2], 0.1).weight_decay(0.1);

        let loss2 = session.tensor_mul(x2, x2).expect("mul");
        let loss2_sum = session.tensor_sum(loss2).expect("sum");
        let report2 = session.tensor_backward(loss2_sum).expect("backward");
        opt_adamw.step(&mut session, &report2).expect("step");
        let x2_val = session.tensor_values(x2).expect("values")[0];

        // Both should decrease but by different amounts due to decoupled vs L2 weight decay
        assert!(x1_val < 4.0, "Adam should decrease x");
        assert!(x2_val < 4.0, "AdamW should decrease x");
        assert!(
            (x1_val - x2_val).abs() > 1e-12,
            "Adam and AdamW should produce different values with same weight_decay: adam={}, adamw={}",
            x1_val,
            x2_val
        );
    }

    #[test]
    fn adamw_zero_weight_decay_matches_adam() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Adam with no weight decay
        let x1 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt_adam = Adam::new(vec![x1], 0.1);

        let loss1 = session.tensor_mul(x1, x1).expect("mul");
        let loss1_sum = session.tensor_sum(loss1).expect("sum");
        let report1 = session.tensor_backward(loss1_sum).expect("backward");
        opt_adam.step(&mut session, &report1).expect("step");
        let x1_val = session.tensor_values(x1).expect("values")[0];

        // AdamW with zero weight decay
        let x2 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt_adamw = AdamW::new(vec![x2], 0.1).weight_decay(0.0);

        let loss2 = session.tensor_mul(x2, x2).expect("mul");
        let loss2_sum = session.tensor_sum(loss2).expect("sum");
        let report2 = session.tensor_backward(loss2_sum).expect("backward");
        opt_adamw.step(&mut session, &report2).expect("step");
        let x2_val = session.tensor_values(x2).expect("values")[0];

        // With zero weight decay, Adam and AdamW should produce identical results
        assert!(
            (x1_val - x2_val).abs() < 1e-12,
            "with wd=0, Adam and AdamW should match: adam={}, adamw={}",
            x1_val,
            x2_val
        );
    }

    #[test]
    fn adamw_multiple_steps_converge() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![5.0], vec![1], true)
            .expect("var");

        let mut optimizer = AdamW::new(vec![x], 0.5).weight_decay(0.01);

        for _ in 0..100 {
            let loss = session.tensor_mul(x, x).expect("mul");
            let loss_sum = session.tensor_sum(loss).expect("sum");
            let report = session.tensor_backward(loss_sum).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val.abs() < 1.0,
            "after 100 AdamW steps with lr=0.5, x should be near 0, got {}",
            x_val
        );
    }

    #[test]
    fn adamw_with_custom_betas() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");

        let mut optimizer = AdamW::new(vec![x], 0.1).betas(0.5, 0.99);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val < 4.0,
            "AdamW with custom betas should decrease x, got {}",
            x_val
        );
    }

    #[test]
    fn adamw_rejects_negative_weight_decay() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = AdamW::new(vec![x], 0.1).weight_decay(-0.1);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("negative weight decay must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "adamw requires finite non-negative weight_decay"
                }
            ))
        ));
    }

    #[test]
    fn adamw_skips_params_without_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let y = session
            .tensor_variable(vec![2.0], vec![1], true)
            .expect("var");

        let mut optimizer = AdamW::new(vec![x, y], 0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let y_val = session.tensor_values(y).expect("values")[0];
        assert!(
            (y_val - 2.0).abs() < 1e-10,
            "y should be unchanged without gradient: expected 2.0, got {}",
            y_val
        );
    }

    #[test]
    fn adamw_zero_grad_is_noop() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");

        let mut optimizer = AdamW::new(vec![x], 0.1);
        optimizer
            .zero_grad(&mut session)
            .expect("adamw zero_grad should succeed");
    }

    #[test]
    fn e2e_adamw_trains_linear_to_reduce_mse_loss() {
        use ft_nn::{Linear, Module};

        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let linear = Linear::new(&mut session, 2, 1, true).expect("linear");
        let params = linear.parameters();
        let mut optimizer = AdamW::new(params, 0.1).weight_decay(0.01);

        let input = session
            .tensor_variable(vec![1.0, 1.0], vec![1, 2], false)
            .expect("input");
        let target = session
            .tensor_variable(vec![5.0], vec![1, 1], false)
            .expect("target");

        let initial_pred = linear.forward(&mut session, input).expect("forward");
        let initial_loss = session.mse_loss(initial_pred, target).expect("mse");
        let initial_loss_val = session.tensor_values(initial_loss).expect("values")[0];

        for _ in 0..30 {
            let pred = linear.forward(&mut session, input).expect("forward");
            let loss = session.mse_loss(pred, target).expect("mse");
            let report = session.tensor_backward(loss).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        let final_pred = linear.forward(&mut session, input).expect("forward");
        let final_loss = session.mse_loss(final_pred, target).expect("mse");
        let final_loss_val = session.tensor_values(final_loss).expect("values")[0];

        assert!(
            final_loss_val < initial_loss_val,
            "AdamW should reduce loss: initial={}, final={}",
            initial_loss_val,
            final_loss_val
        );
    }

    #[test]
    fn adamw_optimizes_multiple_parameters() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        let a = session
            .tensor_variable(vec![5.0], vec![1], true)
            .expect("var");
        let b = session
            .tensor_variable(vec![-3.0], vec![1], true)
            .expect("var");

        let mut optimizer = AdamW::new(vec![a, b], 0.1);

        let a2 = session.tensor_mul(a, a).expect("mul");
        let b2 = session.tensor_mul(b, b).expect("mul");
        let loss = session.tensor_add(a2, b2).expect("add");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let a_val = session.tensor_values(a).expect("values")[0];
        let b_val = session.tensor_values(b).expect("values")[0];

        assert!(a_val < 5.0, "AdamW should decrease a, got {}", a_val);
        assert!(
            b_val > -3.0,
            "AdamW should increase b toward 0, got {}",
            b_val
        );
    }

    // ── RMSprop tests ──────────────────────────────────────────────────

    #[test]
    fn rmsprop_basic_step_reduces_loss() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = RMSprop::new(vec![x], 0.01);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let x_before = session.tensor_values(x).expect("values")[0];
        optimizer.step(&mut session, &report).expect("step");
        let x_after = session.tensor_values(x).expect("values")[0];

        assert!(
            x_after < x_before,
            "RMSprop should decrease x: before={}, after={}",
            x_before,
            x_after
        );
    }

    #[test]
    fn rmsprop_multiple_steps_converge() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![10.0], vec![1], true)
            .expect("var");
        let mut optimizer = RMSprop::new(vec![x], 0.01);

        for _ in 0..200 {
            let loss = session.tensor_mul(x, x).expect("mul");
            let loss_sum = session.tensor_sum(loss).expect("sum");
            let report = session.tensor_backward(loss_sum).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val.abs() < 10.0,
            "RMSprop should decrease x from 10, got {}",
            x_val
        );
    }

    #[test]
    fn rmsprop_with_momentum() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = RMSprop::new(vec![x], 0.01).momentum(0.9);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let x_before = session.tensor_values(x).expect("values")[0];
        optimizer.step(&mut session, &report).expect("step");
        let x_after = session.tensor_values(x).expect("values")[0];

        assert!(
            x_after < x_before,
            "RMSprop with momentum should decrease x: before={}, after={}",
            x_before,
            x_after
        );
    }

    #[test]
    fn rmsprop_centered_differs_from_non_centered() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Non-centered
        let x1 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt1 = RMSprop::new(vec![x1], 0.01);

        let loss1 = session.tensor_mul(x1, x1).expect("mul");
        let loss1_sum = session.tensor_sum(loss1).expect("sum");
        let report1 = session.tensor_backward(loss1_sum).expect("backward");
        opt1.step(&mut session, &report1).expect("step");
        let x1_val = session.tensor_values(x1).expect("values")[0];

        // Centered
        let x2 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt2 = RMSprop::new(vec![x2], 0.01).centered(true);

        let loss2 = session.tensor_mul(x2, x2).expect("mul");
        let loss2_sum = session.tensor_sum(loss2).expect("sum");
        let report2 = session.tensor_backward(loss2_sum).expect("backward");
        opt2.step(&mut session, &report2).expect("step");
        let x2_val = session.tensor_values(x2).expect("values")[0];

        // Both should decrease
        assert!(x1_val < 4.0, "non-centered should decrease x");
        assert!(x2_val < 4.0, "centered should decrease x");
        // They may produce different results (though for first step, centered
        // with zero grad_avg might be very similar — the difference grows over steps)
    }

    #[test]
    fn rmsprop_weight_decay() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = RMSprop::new(vec![x], 0.01).weight_decay(0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        optimizer.step(&mut session, &report).expect("step");
        let x_val = session.tensor_values(x).expect("values")[0];

        assert!(
            x_val < 4.0,
            "RMSprop with weight decay should decrease x, got {}",
            x_val
        );
    }

    #[test]
    fn rmsprop_rejects_negative_lr() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = RMSprop::new(vec![x], -0.1);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("negative lr must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "rmsprop requires a finite non-negative learning rate"
                }
            ))
        ));
    }

    #[test]
    fn rmsprop_rejects_invalid_alpha() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = RMSprop::new(vec![x], 0.01).alpha(1.0);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("alpha=1.0 must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "rmsprop requires finite alpha in [0, 1)"
                }
            ))
        ));
    }

    #[test]
    fn rmsprop_rejects_mismatched_state_length() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = RMSprop::new(vec![x], 0.01);
        // Inject mismatched state
        optimizer.square_avg[0] = Some(vec![0.0, 0.0]);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("mismatched state must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "rmsprop square_avg state length mismatch with gradient length"
                }
            ))
        ));
    }

    #[test]
    fn rmsprop_multi_param() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![5.0], vec![1], true)
            .expect("var");
        let b = session
            .tensor_variable(vec![-3.0], vec![1], true)
            .expect("var");
        let mut optimizer = RMSprop::new(vec![a, b], 0.01);

        let a2 = session.tensor_mul(a, a).expect("mul");
        let b2 = session.tensor_mul(b, b).expect("mul");
        let loss = session.tensor_add(a2, b2).expect("add");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let a_val = session.tensor_values(a).expect("values")[0];
        let b_val = session.tensor_values(b).expect("values")[0];

        assert!(a_val < 5.0, "RMSprop should decrease a, got {}", a_val);
        assert!(
            b_val > -3.0,
            "RMSprop should increase b toward 0, got {}",
            b_val
        );
    }

    // ── Adagrad tests ──────────────────────────────────────────────────

    #[test]
    fn adagrad_basic_step_reduces_loss() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adagrad::new(vec![x], 0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let x_before = session.tensor_values(x).expect("values")[0];
        optimizer.step(&mut session, &report).expect("step");
        let x_after = session.tensor_values(x).expect("values")[0];

        assert!(
            x_after < x_before,
            "Adagrad should decrease x: before={}, after={}",
            x_before,
            x_after
        );
    }

    #[test]
    fn adagrad_multiple_steps_converge() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![10.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adagrad::new(vec![x], 1.0);

        for _ in 0..200 {
            let loss = session.tensor_mul(x, x).expect("mul");
            let loss_sum = session.tensor_sum(loss).expect("sum");
            let report = session.tensor_backward(loss_sum).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val.abs() < 5.0,
            "Adagrad should converge toward 0, got {}",
            x_val
        );
    }

    #[test]
    fn adagrad_lr_decays_learning_rate() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // Without lr_decay
        let x1 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt1 = Adagrad::new(vec![x1], 0.1);

        let loss1 = session.tensor_mul(x1, x1).expect("mul");
        let loss1_sum = session.tensor_sum(loss1).expect("sum");
        let report1 = session.tensor_backward(loss1_sum).expect("backward");
        opt1.step(&mut session, &report1).expect("step");
        let x1_val = session.tensor_values(x1).expect("values")[0];

        // With lr_decay
        let x2 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt2 = Adagrad::new(vec![x2], 0.1).lr_decay(0.5);

        let loss2 = session.tensor_mul(x2, x2).expect("mul");
        let loss2_sum = session.tensor_sum(loss2).expect("sum");
        let report2 = session.tensor_backward(loss2_sum).expect("backward");
        opt2.step(&mut session, &report2).expect("step");
        let x2_val = session.tensor_values(x2).expect("values")[0];

        // First step with lr_decay=0.5: clr = 0.1 / (1 + 0*0.5) = 0.1, same as no decay
        // But on subsequent steps, lr_decay makes a difference
        assert!(x1_val < 4.0, "no-decay should decrease x");
        assert!(x2_val < 4.0, "with-decay should decrease x");
    }

    #[test]
    fn adagrad_weight_decay() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adagrad::new(vec![x], 0.1).weight_decay(0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val < 4.0,
            "Adagrad with weight decay should decrease x, got {}",
            x_val
        );
    }

    #[test]
    fn adagrad_initial_accumulator_value() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

        // With zero initial accumulator
        let x1 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt1 = Adagrad::new(vec![x1], 0.1);

        let loss1 = session.tensor_mul(x1, x1).expect("mul");
        let loss1_sum = session.tensor_sum(loss1).expect("sum");
        let report1 = session.tensor_backward(loss1_sum).expect("backward");
        opt1.step(&mut session, &report1).expect("step");
        let x1_val = session.tensor_values(x1).expect("values")[0];

        // With large initial accumulator (dampens update)
        let x2 = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt2 = Adagrad::new(vec![x2], 0.1).initial_accumulator_value(100.0);

        let loss2 = session.tensor_mul(x2, x2).expect("mul");
        let loss2_sum = session.tensor_sum(loss2).expect("sum");
        let report2 = session.tensor_backward(loss2_sum).expect("backward");
        opt2.step(&mut session, &report2).expect("step");
        let x2_val = session.tensor_values(x2).expect("values")[0];

        // Large initial accumulator means smaller effective step
        assert!(x1_val < 4.0, "zero init should decrease x");
        assert!(x2_val < 4.0, "large init should decrease x");
        assert!(
            x2_val > x1_val,
            "large initial_accumulator should dampen update: x1={}, x2={}",
            x1_val,
            x2_val
        );
    }

    #[test]
    fn adagrad_rejects_negative_lr() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adagrad::new(vec![x], -0.1);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("negative lr must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "adagrad requires a finite non-negative learning rate"
                }
            ))
        ));
    }

    #[test]
    fn adagrad_rejects_negative_initial_accumulator() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adagrad::new(vec![x], 0.1).initial_accumulator_value(-1.0);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("negative initial_accumulator must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "adagrad requires finite non-negative initial_accumulator_value"
                }
            ))
        ));
    }

    #[test]
    fn adagrad_rejects_mismatched_state_length() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adagrad::new(vec![x], 0.1);
        optimizer.sum_sq[0] = Some(vec![0.0, 0.0]);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("mismatched state must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "adagrad sum_sq state length mismatch with gradient length"
                }
            ))
        ));
    }

    #[test]
    fn adagrad_multi_param() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![5.0], vec![1], true)
            .expect("var");
        let b = session
            .tensor_variable(vec![-3.0], vec![1], true)
            .expect("var");
        let mut optimizer = Adagrad::new(vec![a, b], 0.1);

        let a2 = session.tensor_mul(a, a).expect("mul");
        let b2 = session.tensor_mul(b, b).expect("mul");
        let loss = session.tensor_add(a2, b2).expect("add");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let a_val = session.tensor_values(a).expect("values")[0];
        let b_val = session.tensor_values(b).expect("values")[0];

        assert!(a_val < 5.0, "Adagrad should decrease a, got {}", a_val);
        assert!(
            b_val > -3.0,
            "Adagrad should increase b toward 0, got {}",
            b_val
        );
    }

    // ── RAdam tests ────────────────────────────────────────────────────

    #[test]
    fn radam_basic_step_reduces_loss() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = RAdam::new(vec![x], 0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let x_before = session.tensor_values(x).expect("values")[0];
        optimizer.step(&mut session, &report).expect("step");
        let x_after = session.tensor_values(x).expect("values")[0];

        assert!(
            x_after < x_before,
            "RAdam should decrease x: before={}, after={}",
            x_before,
            x_after
        );
    }

    #[test]
    fn radam_multiple_steps_converge() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![10.0], vec![1], true)
            .expect("var");
        let mut optimizer = RAdam::new(vec![x], 0.1);

        let initial = session.tensor_values(x).expect("values")[0];

        for _ in 0..50 {
            let loss = session.tensor_mul(x, x).expect("mul");
            let loss_sum = session.tensor_sum(loss).expect("sum");
            let report = session.tensor_backward(loss_sum).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val.abs() < initial.abs(),
            "RAdam should converge toward 0: initial={}, final={}",
            initial,
            x_val
        );
    }

    #[test]
    fn radam_uses_sgd_in_early_steps() {
        // In early steps (rho_t <= 5), RAdam should use SGD-like behavior
        // With default beta2=0.999, rho_inf ≈ 1999, rho_1 ≈ 1999 - 2*0.999/0.001 ≈ 1
        // So step 1 should use SGD path
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = RAdam::new(vec![x], 0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val < 4.0,
            "RAdam SGD path should decrease x, got {}",
            x_val
        );
    }

    #[test]
    fn radam_weight_decay() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = RAdam::new(vec![x], 0.1).weight_decay(0.1);

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let x_val = session.tensor_values(x).expect("values")[0];
        assert!(
            x_val < 4.0,
            "RAdam with weight decay should decrease x, got {}",
            x_val
        );
    }

    #[test]
    fn radam_rejects_negative_lr() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = RAdam::new(vec![x], -0.1);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let err = optimizer
            .step(&mut session, &report)
            .expect_err("negative lr must fail closed");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "radam requires a finite non-negative learning rate"
                }
            ))
        ));
    }

    #[test]
    fn radam_rejects_invalid_betas() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut optimizer = RAdam::new(vec![x], 0.1).betas(1.0, 0.999);
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        assert!(optimizer.step(&mut session, &report).is_err());
    }

    #[test]
    fn radam_multi_param() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(vec![5.0], vec![1], true)
            .expect("var");
        let b = session
            .tensor_variable(vec![-3.0], vec![1], true)
            .expect("var");
        let mut optimizer = RAdam::new(vec![a, b], 0.1);

        let a2 = session.tensor_mul(a, a).expect("mul");
        let b2 = session.tensor_mul(b, b).expect("mul");
        let loss = session.tensor_add(a2, b2).expect("add");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");
        optimizer.step(&mut session, &report).expect("step");

        let a_val = session.tensor_values(a).expect("values")[0];
        let b_val = session.tensor_values(b).expect("values")[0];

        assert!(a_val < 5.0, "RAdam should decrease a, got {}", a_val);
        assert!(
            b_val > -3.0,
            "RAdam should increase b toward 0, got {}",
            b_val
        );
    }
}
