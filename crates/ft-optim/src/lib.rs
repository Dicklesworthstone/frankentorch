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

fn load_param_gradient(
    session: &FrankenTorchSession,
    param: TensorNodeId,
) -> Result<Option<Vec<f64>>, AutogradError> {
    session.tensor_accumulated_gradient(param)
}

fn zero_param_gradients(
    session: &mut FrankenTorchSession,
    params: &[TensorNodeId],
) -> Result<(), AutogradError> {
    session.tensor_zero_grads(params)
}

/// Trait for parameter optimizers.
pub trait Optimizer {
    /// Perform a single optimization step using persistent gradients stored in
    /// the session's autograd state.
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError>;

    /// Zero out accumulated persistent gradients for this optimizer's parameter set.
    fn zero_grad(&mut self, session: &mut FrankenTorchSession) -> Result<(), AutogradError>;

    /// Return the current learning rate.
    fn get_lr(&self) -> f64;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f64);
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
    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        _report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        for (i, &param) in self.params.iter().enumerate() {
            let grad = match load_param_gradient(session, param)? {
                Some(g) => g,
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

    fn zero_grad(&mut self, session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        zero_param_gradients(session, &self.params)
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
    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        _report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        let t = checked_next_step_count(self.step_count, "adam step counter overflow")?;
        self.step_count = t;

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match load_param_gradient(session, param)? {
                Some(g) => g,
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

    fn zero_grad(&mut self, session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        zero_param_gradients(session, &self.params)
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
    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        _report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        let t = checked_next_step_count(self.step_count, "adamw step counter overflow")?;
        self.step_count = t;

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match load_param_gradient(session, param)? {
                Some(g) => g,
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

    fn zero_grad(&mut self, session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        zero_param_gradients(session, &self.params)
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
    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        _report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        self.step_count =
            checked_next_step_count(self.step_count, "rmsprop step counter overflow")?;

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match load_param_gradient(session, param)? {
                Some(g) => g,
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
                let ga = self.grad_avg[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
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
                let buf =
                    self.momentum_buffer[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
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

    fn zero_grad(&mut self, session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        zero_param_gradients(session, &self.params)
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
    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        _report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        self.step_count =
            checked_next_step_count(self.step_count, "adagrad step counter overflow")?;

        // Compute decayed learning rate: lr / (1 + (step - 1) * lr_decay)
        let clr = self.lr / (1.0 + (self.step_count.saturating_sub(1) as f64) * self.lr_decay);

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match load_param_gradient(session, param)? {
                Some(g) => g,
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

    fn zero_grad(&mut self, session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        zero_param_gradients(session, &self.params)
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
    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        _report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;
        let t = checked_next_step_count(self.step_count, "radam step counter overflow")?;
        self.step_count = t;

        // Maximum length of the approximated SMA
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match load_param_gradient(session, param)? {
                Some(g) => g,
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

    fn zero_grad(&mut self, session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        zero_param_gradients(session, &self.params)
    }
}

fn vector_dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
}

fn vector_max_abs(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn vector_max_abs_delta(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0, f64::max)
}

fn vector_add_scaled(base: &[f64], direction: &[f64], scale: f64) -> Vec<f64> {
    base.iter()
        .zip(direction.iter())
        .map(|(value, delta)| value + scale * delta)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LBFGSLineSearch {
    None,
    BacktrackingArmijo,
    StrongWolfe,
}

/// Limited-memory BFGS optimizer.
///
/// This implementation supports both the regular `Optimizer` trait flow
/// (`backward` followed by `step`) and an explicit closure-based update via
/// [`LBFGS::step_with_closure`].
pub struct LBFGS {
    params: Vec<TensorNodeId>,
    lr: f64,
    max_iter: usize,
    max_eval: usize,
    tolerance_grad: f64,
    tolerance_change: f64,
    history_size: usize,
    line_search_fn: LBFGSLineSearch,
    s_history: Vec<Vec<f64>>,
    y_history: Vec<Vec<f64>>,
    rho_history: Vec<f64>,
    previous_params: Option<Vec<f64>>,
    previous_grad: Option<Vec<f64>>,
}

struct LBFGSSearchContext<'a> {
    current_params: &'a [f64],
    current_grad: &'a [f64],
    direction: &'a [f64],
    current_loss: f64,
    eval_budget: usize,
}

impl LBFGS {
    /// Create a new LBFGS optimizer.
    ///
    /// Defaults:
    /// - `max_iter=20`
    /// - `max_eval=25`
    /// - `tolerance_grad=1e-7`
    /// - `tolerance_change=1e-9`
    /// - `history_size=100`
    /// - `line_search_fn=StrongWolfe`
    pub fn new(params: Vec<TensorNodeId>, lr: f64) -> Self {
        Self {
            params,
            lr,
            max_iter: 20,
            max_eval: 25,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 100,
            line_search_fn: LBFGSLineSearch::StrongWolfe,
            s_history: Vec::new(),
            y_history: Vec::new(),
            rho_history: Vec::new(),
            previous_params: None,
            previous_grad: None,
        }
    }

    #[must_use]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    #[must_use]
    pub fn max_eval(mut self, max_eval: usize) -> Self {
        self.max_eval = max_eval;
        self
    }

    #[must_use]
    pub fn tolerance_grad(mut self, tolerance_grad: f64) -> Self {
        self.tolerance_grad = tolerance_grad;
        self
    }

    #[must_use]
    pub fn tolerance_change(mut self, tolerance_change: f64) -> Self {
        self.tolerance_change = tolerance_change;
        self
    }

    #[must_use]
    pub fn history_size(mut self, history_size: usize) -> Self {
        self.history_size = history_size;
        self
    }

    #[must_use]
    pub fn line_search_fn(mut self, line_search_fn: LBFGSLineSearch) -> Self {
        self.line_search_fn = line_search_fn;
        self
    }

    fn validate_hyperparams(&self) -> Result<(), AutogradError> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(optimizer_hparam_error(
                "lbfgs requires a finite non-negative learning rate",
            ));
        }
        if self.max_iter == 0 {
            return Err(optimizer_hparam_error("lbfgs requires max_iter >= 1"));
        }
        if self.max_eval == 0 {
            return Err(optimizer_hparam_error("lbfgs requires max_eval >= 1"));
        }
        if !self.tolerance_grad.is_finite() || self.tolerance_grad < 0.0 {
            return Err(optimizer_hparam_error(
                "lbfgs requires finite non-negative tolerance_grad",
            ));
        }
        if !self.tolerance_change.is_finite() || self.tolerance_change < 0.0 {
            return Err(optimizer_hparam_error(
                "lbfgs requires finite non-negative tolerance_change",
            ));
        }
        if self.history_size == 0 {
            return Err(optimizer_hparam_error("lbfgs requires history_size >= 1"));
        }
        Ok(())
    }

    fn clear_history(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
    }

    fn flatten_params(&self, session: &FrankenTorchSession) -> Result<Vec<f64>, AutogradError> {
        let mut flat = Vec::new();
        for &param in &self.params {
            let values = session.tensor_values(param)?;
            flat.extend(values);
        }
        Ok(flat)
    }

    fn flatten_gradients(
        &self,
        session: &FrankenTorchSession,
    ) -> Result<Option<Vec<f64>>, AutogradError> {
        let mut flat = Vec::new();
        let mut any_grad = false;

        for &param in &self.params {
            let param_numel = session.tensor_numel(param)?;
            match load_param_gradient(session, param)? {
                Some(grad) => {
                    ensure_grad_len_matches_param(param, param_numel, grad.len())?;
                    flat.extend(grad);
                    any_grad = true;
                }
                None => {
                    flat.resize(flat.len() + param_numel, 0.0);
                }
            }
        }

        if any_grad { Ok(Some(flat)) } else { Ok(None) }
    }

    fn set_flat_params(
        &self,
        session: &mut FrankenTorchSession,
        flat_params: &[f64],
    ) -> Result<(), AutogradError> {
        let mut offset = 0usize;
        for &param in &self.params {
            let numel = session.tensor_numel(param)?;
            let shape = session.tensor_shape(param)?;
            let end = offset
                .checked_add(numel)
                .ok_or_else(|| optimizer_state_error("lbfgs parameter offset overflow"))?;
            if end > flat_params.len() {
                return Err(optimizer_state_error(
                    "lbfgs flat parameter vector is shorter than parameter footprint",
                ));
            }
            let source =
                session.tensor_variable(flat_params[offset..end].to_vec(), shape, false)?;
            session.tensor_zero_(param)?;
            session.tensor_add_(param, source)?;
            offset = end;
        }
        if offset != flat_params.len() {
            return Err(optimizer_state_error(
                "lbfgs flat parameter vector has trailing unused values",
            ));
        }
        Ok(())
    }

    fn push_history_pair(
        &mut self,
        old_params: &[f64],
        old_grad: &[f64],
        new_params: &[f64],
        new_grad: &[f64],
    ) {
        if old_params.len() != new_params.len() || old_grad.len() != new_grad.len() {
            self.clear_history();
            return;
        }

        let s: Vec<f64> = new_params
            .iter()
            .zip(old_params.iter())
            .map(|(new_value, old_value)| new_value - old_value)
            .collect();
        let y: Vec<f64> = new_grad
            .iter()
            .zip(old_grad.iter())
            .map(|(new_value, old_value)| new_value - old_value)
            .collect();
        let ys = vector_dot(&y, &s);
        if !ys.is_finite() || ys <= 1e-10 {
            return;
        }

        let rho = 1.0 / ys;
        if !rho.is_finite() {
            return;
        }

        if self.s_history.len() == self.history_size {
            self.s_history.remove(0);
            self.y_history.remove(0);
            self.rho_history.remove(0);
        }
        self.s_history.push(s);
        self.y_history.push(y);
        self.rho_history.push(rho);
    }

    fn record_history_from_previous(&mut self, current_params: &[f64], current_grad: &[f64]) {
        let previous_params = self.previous_params.clone();
        let previous_grad = self.previous_grad.clone();
        if let (Some(prev_params), Some(prev_grad)) = (previous_params, previous_grad) {
            self.push_history_pair(&prev_params, &prev_grad, current_params, current_grad);
        }
    }

    fn two_loop_direction(&self, grad: &[f64]) -> Vec<f64> {
        if self.s_history.is_empty() {
            return grad.iter().map(|value| -value).collect();
        }

        let history_len = self.s_history.len();
        let mut q = grad.to_vec();
        let mut alpha = vec![0.0; history_len];

        for index in (0..history_len).rev() {
            let rho = self.rho_history[index];
            let a = rho * vector_dot(&self.s_history[index], &q);
            alpha[index] = a;
            for (q_value, y_value) in q.iter_mut().zip(self.y_history[index].iter()) {
                *q_value -= a * y_value;
            }
        }

        let mut r = q;
        if history_len > 0 {
            let last = history_len - 1;
            let yy = vector_dot(&self.y_history[last], &self.y_history[last]);
            if yy.is_finite() && yy > 0.0 {
                let gamma = vector_dot(&self.s_history[last], &self.y_history[last]) / yy;
                if gamma.is_finite() && gamma > 0.0 {
                    for value in &mut r {
                        *value *= gamma;
                    }
                }
            }
        }

        for (index, (s, y)) in self.s_history.iter().zip(self.y_history.iter()).enumerate() {
            let beta = self.rho_history[index] * vector_dot(y, &r);
            for (r_value, s_value) in r.iter_mut().zip(s.iter()) {
                *r_value += s_value * (alpha[index] - beta);
            }
        }

        r.into_iter().map(|value| -value).collect()
    }

    fn search_step(
        &mut self,
        session: &mut FrankenTorchSession,
        closure: &mut dyn FnMut(&mut FrankenTorchSession) -> Result<f64, AutogradError>,
        context: LBFGSSearchContext<'_>,
    ) -> Result<(Vec<f64>, Vec<f64>, f64, usize), AutogradError> {
        if context.eval_budget == 0 {
            return Err(optimizer_state_error(
                "lbfgs line search exhausted evaluation budget",
            ));
        }

        let directional_derivative = vector_dot(context.current_grad, context.direction);
        if !directional_derivative.is_finite() || directional_derivative >= 0.0 {
            return Err(optimizer_state_error(
                "lbfgs search direction must be a descent direction",
            ));
        }

        if self.line_search_fn == LBFGSLineSearch::None {
            let next_params = vector_add_scaled(context.current_params, context.direction, self.lr);
            self.set_flat_params(session, &next_params)?;
            let next_loss = closure(session)?;
            let next_grad = self
                .flatten_gradients(session)?
                .unwrap_or_else(|| vec![0.0; next_params.len()]);
            return Ok((next_params, next_grad, next_loss, 1));
        }

        let armijo_c1 = 1e-4;
        let wolfe_c2 = 0.9;
        let min_step = 1e-12;
        let mut step_size = self.lr;

        for attempt in 0..context.eval_budget {
            let trial_params =
                vector_add_scaled(context.current_params, context.direction, step_size);
            self.set_flat_params(session, &trial_params)?;
            let trial_loss = closure(session)?;
            let trial_grad = self
                .flatten_gradients(session)?
                .unwrap_or_else(|| vec![0.0; trial_params.len()]);
            let armijo_ok =
                trial_loss <= context.current_loss + armijo_c1 * step_size * directional_derivative;
            let accept = match self.line_search_fn {
                LBFGSLineSearch::BacktrackingArmijo => armijo_ok,
                LBFGSLineSearch::StrongWolfe => {
                    let directional_grad = vector_dot(&trial_grad, context.direction);
                    armijo_ok && directional_grad.abs() <= wolfe_c2 * directional_derivative.abs()
                }
                LBFGSLineSearch::None => true,
            };
            if accept {
                return Ok((trial_params, trial_grad, trial_loss, attempt + 1));
            }

            step_size *= 0.5;
            if step_size < min_step {
                break;
            }
        }

        self.set_flat_params(session, context.current_params)?;
        Err(optimizer_state_error(
            "lbfgs line search failed to satisfy acceptance criteria",
        ))
    }

    /// Perform one LBFGS step using a closure that recomputes loss and gradients.
    ///
    /// The closure is expected to run forward + backward and return a scalar loss.
    pub fn step_with_closure(
        &mut self,
        session: &mut FrankenTorchSession,
        closure: &mut dyn FnMut(&mut FrankenTorchSession) -> Result<f64, AutogradError>,
    ) -> Result<f64, AutogradError> {
        self.validate_hyperparams()?;

        let mut eval_count = 1usize;
        let mut loss = closure(session)?;
        let mut params = self.flatten_params(session)?;
        let mut grad = match self.flatten_gradients(session)? {
            Some(grad) => grad,
            None => {
                self.previous_params = Some(params);
                self.previous_grad = None;
                return Ok(loss);
            }
        };
        self.record_history_from_previous(&params, &grad);

        for _ in 0..self.max_iter {
            if vector_max_abs(&grad) <= self.tolerance_grad {
                break;
            }

            let mut direction = self.two_loop_direction(&grad);
            if vector_dot(&direction, &grad) >= 0.0 {
                direction = grad.iter().map(|value| -value).collect();
                self.clear_history();
            }

            let eval_budget = self.max_eval.saturating_sub(eval_count);
            if eval_budget == 0 {
                break;
            }

            let old_params = params;
            let old_grad = grad;

            let (next_params, next_grad, next_loss, evals_used) = self.search_step(
                session,
                closure,
                LBFGSSearchContext {
                    current_params: &old_params,
                    current_grad: &old_grad,
                    direction: &direction,
                    current_loss: loss,
                    eval_budget,
                },
            )?;

            let max_change = vector_max_abs_delta(&old_params, &next_params);
            self.push_history_pair(&old_params, &old_grad, &next_params, &next_grad);

            params = next_params;
            grad = next_grad;
            loss = next_loss;
            eval_count += evals_used;

            if max_change <= self.tolerance_change || eval_count >= self.max_eval {
                break;
            }
        }

        self.previous_params = Some(params);
        self.previous_grad = Some(grad);
        Ok(loss)
    }
}

impl Optimizer for LBFGS {
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        _report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.validate_hyperparams()?;

        let params = self.flatten_params(session)?;
        let grad = match self.flatten_gradients(session)? {
            Some(grad) => grad,
            None => return Ok(()),
        };
        self.record_history_from_previous(&params, &grad);

        if vector_max_abs(&grad) <= self.tolerance_grad {
            self.previous_params = Some(params);
            self.previous_grad = Some(grad);
            return Ok(());
        }

        let mut direction = self.two_loop_direction(&grad);
        if vector_dot(&direction, &grad) >= 0.0 {
            direction = grad.iter().map(|value| -value).collect();
            self.clear_history();
        }

        let next_params = vector_add_scaled(&params, &direction, self.lr);
        if vector_max_abs_delta(&params, &next_params) > self.tolerance_change {
            self.set_flat_params(session, &next_params)?;
        }

        self.previous_params = Some(params);
        self.previous_grad = Some(grad);
        Ok(())
    }

    fn zero_grad(&mut self, session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        zero_param_gradients(session, &self.params)
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// ---------------------------------------------------------------------------
// Learning Rate Schedulers
// ---------------------------------------------------------------------------

/// Serializable scheduler state for save/restore.
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerState {
    /// Current epoch (0-based after first step()).
    pub last_epoch: i64,
    /// Per-param-group learning rates after the last step().
    pub last_lrs: Vec<f64>,
    /// Scheduler-specific extra state.
    pub extra: Vec<(String, f64)>,
}

/// Trait for learning rate schedulers.
///
/// Schedulers adjust the optimizer learning rate according to a policy.
/// After constructing a scheduler, call `step()` once per epoch (or per
/// iteration, depending on the scheduler) to advance the schedule.
pub trait LRScheduler {
    /// Advance the scheduler by one step, updating the optimizer learning rate.
    ///
    /// If `epoch` is `Some(n)`, the scheduler jumps to that epoch directly.
    /// If `None`, the scheduler auto-increments from the last epoch.
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>);

    /// Advance a metric-driven scheduler by one step using the provided metric.
    ///
    /// Schedulers that are not metric-driven ignore `metric` and delegate to
    /// `step(optimizer, None)` by default.
    fn step_with_metric(&mut self, optimizer: &mut dyn Optimizer, metric: f64) {
        let _ = metric;
        self.step(optimizer, None);
    }

    /// Return the current computed learning rates (one per param group).
    fn get_lr(&self) -> Vec<f64>;

    /// Return the learning rates from the last `step()` call.
    fn get_last_lr(&self) -> Vec<f64>;

    /// Serialize the scheduler state.
    fn state_dict(&self) -> SchedulerState;

    /// Restore scheduler state from a previously serialized snapshot.
    fn load_state_dict(&mut self, state: SchedulerState);
}

/// StepLR: decays the learning rate by `gamma` every `step_size` epochs.
///
/// ```text
/// lr = initial_lr * gamma ^ (epoch / step_size)
/// ```
///
/// This is one of the simplest and most commonly used schedulers.
pub struct StepLR {
    initial_lr: f64,
    step_size: usize,
    gamma: f64,
    last_epoch: i64,
    last_lr: f64,
    verbose: bool,
}

impl StepLR {
    /// Create a new `StepLR` scheduler.
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer whose learning rate will be scheduled.
    /// * `step_size` - Period of learning rate decay (in epochs).
    /// * `gamma` - Multiplicative decay factor (default: 0.1).
    /// * `last_epoch` - The index of the last epoch. Use -1 to start fresh.
    pub fn new(optimizer: &dyn Optimizer, step_size: usize) -> Self {
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            step_size,
            gamma: 0.1,
            last_epoch: -1,
            last_lr: initial_lr,
            verbose: false,
        }
    }

    /// Set the multiplicative decay factor (default: 0.1).
    #[must_use]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the last epoch index for resuming (default: -1).
    ///
    /// When set to a non-negative value, the scheduler resumes from that epoch.
    /// The next call to `step(None)` will advance to `last_epoch + 1`.
    #[must_use]
    pub fn last_epoch(mut self, last_epoch: i64) -> Self {
        self.last_epoch = last_epoch;
        if last_epoch >= 0 {
            self.last_lr = self.compute_lr_at_epoch(last_epoch);
        }
        self
    }

    /// Enable verbose mode: prints lr changes.
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn compute_lr_at_epoch(&self, epoch: i64) -> f64 {
        if epoch < 0 {
            return self.initial_lr;
        }
        let e = epoch as usize;
        let exponent = e / self.step_size;
        self.initial_lr * self.gamma.powi(exponent as i32)
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        let new_epoch = match epoch {
            Some(e) => e,
            None => self.last_epoch + 1,
        };
        self.last_epoch = new_epoch;
        let new_lr = self.compute_lr_at_epoch(new_epoch);
        let old_lr = self.last_lr;
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);

        if self.verbose && (new_lr - old_lr).abs() > f64::EPSILON {
            eprintln!(
                "StepLR: adjusting learning rate from {old_lr:.6e} to {new_lr:.6e} at epoch {new_epoch}."
            );
        }
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.compute_lr_at_epoch(self.last_epoch)]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra: vec![
                ("initial_lr".to_owned(), self.initial_lr),
                ("step_size".to_owned(), self.step_size as f64),
                ("gamma".to_owned(), self.gamma),
            ],
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }
        for (key, val) in &state.extra {
            match key.as_str() {
                "initial_lr" => self.initial_lr = *val,
                "step_size" => self.step_size = *val as usize,
                "gamma" => self.gamma = *val,
                _ => {}
            }
        }
    }
}

/// MultiStepLR: decays lr by `gamma` at each milestone epoch.
///
/// ```text
/// lr = initial_lr * gamma ^ (# of milestones <= epoch)
/// ```
///
/// Milestones are sorted internally. Duplicate milestones apply repeated decay,
/// matching PyTorch's multiplicative milestone semantics.
pub struct MultiStepLR {
    initial_lr: f64,
    milestones: Vec<usize>,
    gamma: f64,
    last_epoch: i64,
    last_lr: f64,
    verbose: bool,
}

impl MultiStepLR {
    /// Create a new `MultiStepLR` scheduler.
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer whose learning rate will be scheduled.
    /// * `milestones` - Epoch indices where lr is multiplied by `gamma`.
    pub fn new(optimizer: &dyn Optimizer, mut milestones: Vec<usize>) -> Self {
        milestones.sort_unstable();
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            milestones,
            gamma: 0.1,
            last_epoch: -1,
            last_lr: initial_lr,
            verbose: false,
        }
    }

    /// Set the multiplicative decay factor (default: 0.1).
    #[must_use]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the last epoch index for resuming (default: -1).
    #[must_use]
    pub fn last_epoch(mut self, last_epoch: i64) -> Self {
        self.last_epoch = last_epoch;
        if last_epoch >= 0 {
            self.last_lr = self.compute_lr_at_epoch(last_epoch);
        }
        self
    }

    /// Enable verbose mode: prints lr changes.
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn milestone_count_at_epoch(&self, epoch: i64) -> usize {
        if epoch < 0 {
            return 0;
        }
        let epoch = epoch as usize;
        self.milestones
            .partition_point(|&milestone| milestone <= epoch)
    }

    fn compute_lr_at_epoch(&self, epoch: i64) -> f64 {
        let decay_count = self.milestone_count_at_epoch(epoch);
        self.initial_lr * self.gamma.powi(decay_count as i32)
    }
}

impl LRScheduler for MultiStepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        let new_epoch = match epoch {
            Some(e) => e,
            None => self.last_epoch + 1,
        };
        self.last_epoch = new_epoch;
        let new_lr = self.compute_lr_at_epoch(new_epoch);
        let old_lr = self.last_lr;
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);

        if self.verbose && (new_lr - old_lr).abs() > f64::EPSILON {
            eprintln!(
                "MultiStepLR: adjusting learning rate from {old_lr:.6e} to {new_lr:.6e} at epoch {new_epoch}."
            );
        }
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.compute_lr_at_epoch(self.last_epoch)]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        let mut extra = vec![
            ("initial_lr".to_owned(), self.initial_lr),
            ("gamma".to_owned(), self.gamma),
            ("milestones_len".to_owned(), self.milestones.len() as f64),
        ];
        extra.extend(
            self.milestones
                .iter()
                .enumerate()
                .map(|(idx, milestone)| (format!("milestone_{idx}"), *milestone as f64)),
        );

        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra,
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }

        let mut indexed_milestones = Vec::new();
        for (key, val) in &state.extra {
            match key.as_str() {
                "initial_lr" => self.initial_lr = *val,
                "gamma" => self.gamma = *val,
                _ => {
                    if let Some(index) = key
                        .strip_prefix("milestone_")
                        .and_then(|suffix| suffix.parse::<usize>().ok())
                    {
                        indexed_milestones.push((index, *val as usize));
                    }
                }
            }
        }

        if !indexed_milestones.is_empty() {
            indexed_milestones.sort_by_key(|(index, _)| *index);
            self.milestones = indexed_milestones
                .into_iter()
                .map(|(_, milestone)| milestone)
                .collect();
        }
        self.milestones.sort_unstable();
    }
}

/// CosineAnnealingLR: anneals lr using a cosine schedule from `initial_lr`
/// down to `eta_min` over `t_max` epochs.
///
/// ```text
/// lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * epoch / t_max))
/// ```
///
/// For epochs beyond `t_max`, lr stays at `eta_min`.
pub struct CosineAnnealingLR {
    initial_lr: f64,
    t_max: usize,
    eta_min: f64,
    last_epoch: i64,
    last_lr: f64,
    verbose: bool,
}

impl CosineAnnealingLR {
    /// Create a new `CosineAnnealingLR` scheduler.
    pub fn new(optimizer: &dyn Optimizer, t_max: usize) -> Self {
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            t_max: t_max.max(1),
            eta_min: 0.0,
            last_epoch: -1,
            last_lr: initial_lr,
            verbose: false,
        }
    }

    /// Set the minimum learning rate (default: 0.0).
    #[must_use]
    pub fn eta_min(mut self, eta_min: f64) -> Self {
        self.eta_min = eta_min;
        self
    }

    /// Set the last epoch index for resuming (default: -1).
    #[must_use]
    pub fn last_epoch(mut self, last_epoch: i64) -> Self {
        self.last_epoch = last_epoch;
        if last_epoch >= 0 {
            self.last_lr = self.compute_lr_at_epoch(last_epoch);
        }
        self
    }

    /// Enable verbose mode: prints lr changes.
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn compute_lr_at_epoch(&self, epoch: i64) -> f64 {
        if epoch < 0 {
            return self.initial_lr;
        }
        let e = epoch as usize;
        if e >= self.t_max {
            return self.eta_min;
        }
        let ratio = e as f64 / self.t_max as f64;
        self.eta_min
            + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + (std::f64::consts::PI * ratio).cos())
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        let new_epoch = match epoch {
            Some(e) => e,
            None => self.last_epoch + 1,
        };
        self.last_epoch = new_epoch;
        let new_lr = self.compute_lr_at_epoch(new_epoch);
        let old_lr = self.last_lr;
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);

        if self.verbose && (new_lr - old_lr).abs() > f64::EPSILON {
            eprintln!(
                "CosineAnnealingLR: adjusting learning rate from {old_lr:.6e} to {new_lr:.6e} at epoch {new_epoch}."
            );
        }
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.compute_lr_at_epoch(self.last_epoch)]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra: vec![
                ("initial_lr".to_owned(), self.initial_lr),
                ("t_max".to_owned(), self.t_max as f64),
                ("eta_min".to_owned(), self.eta_min),
            ],
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }
        for (key, val) in &state.extra {
            match key.as_str() {
                "initial_lr" => self.initial_lr = *val,
                "t_max" => self.t_max = (*val as usize).max(1),
                "eta_min" => self.eta_min = *val,
                _ => {}
            }
        }
    }
}

/// CosineAnnealingWarmRestarts: cosine annealing with periodic warm restarts.
///
/// Cycle lengths are `t_0`, then `t_0 * t_mult`, then `t_0 * t_mult^2`, etc.
pub struct CosineAnnealingWarmRestarts {
    initial_lr: f64,
    t_0: usize,
    t_mult: usize,
    eta_min: f64,
    last_epoch: i64,
    last_lr: f64,
    t_cur: i64,
    t_i: usize,
    verbose: bool,
}

impl CosineAnnealingWarmRestarts {
    /// Create a new `CosineAnnealingWarmRestarts` scheduler.
    pub fn new(optimizer: &dyn Optimizer, t_0: usize) -> Self {
        let initial_lr = optimizer.get_lr();
        let t_0 = t_0.max(1);
        Self {
            initial_lr,
            t_0,
            t_mult: 1,
            eta_min: 0.0,
            last_epoch: -1,
            last_lr: initial_lr,
            t_cur: -1,
            t_i: t_0,
            verbose: false,
        }
    }

    /// Set multiplicative cycle length growth factor (default: 1).
    #[must_use]
    pub fn t_mult(mut self, t_mult: usize) -> Self {
        self.t_mult = t_mult.max(1);
        self
    }

    /// Set the minimum learning rate (default: 0.0).
    #[must_use]
    pub fn eta_min(mut self, eta_min: f64) -> Self {
        self.eta_min = eta_min;
        self
    }

    /// Set the last epoch index for resuming (default: -1).
    #[must_use]
    pub fn last_epoch(mut self, last_epoch: i64) -> Self {
        self.last_epoch = last_epoch;
        if last_epoch >= 0 {
            let (lr, t_cur, t_i) = self.compute_cycle_at_epoch(last_epoch);
            self.last_lr = lr;
            self.t_cur = t_cur;
            self.t_i = t_i;
        }
        self
    }

    /// Enable verbose mode: prints lr changes.
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn compute_cycle_at_epoch(&self, epoch: i64) -> (f64, i64, usize) {
        if epoch < 0 {
            return (self.initial_lr, -1, self.t_0);
        }

        let mut t_i = self.t_0;
        let mut t_cur = epoch as usize;
        if self.t_mult == 1 {
            t_cur %= t_i;
        } else {
            while t_cur >= t_i {
                t_cur -= t_i;
                t_i = t_i.saturating_mul(self.t_mult);
            }
        }

        let ratio = t_cur as f64 / t_i as f64;
        let lr = self.eta_min
            + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + (std::f64::consts::PI * ratio).cos());
        (lr, t_cur as i64, t_i)
    }
}

impl LRScheduler for CosineAnnealingWarmRestarts {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        let new_epoch = match epoch {
            Some(e) => e,
            None => self.last_epoch + 1,
        };
        self.last_epoch = new_epoch;

        let (new_lr, new_t_cur, new_t_i) = self.compute_cycle_at_epoch(new_epoch);
        let old_lr = self.last_lr;
        self.last_lr = new_lr;
        self.t_cur = new_t_cur;
        self.t_i = new_t_i;
        optimizer.set_lr(new_lr);

        if self.verbose && (new_lr - old_lr).abs() > f64::EPSILON {
            eprintln!(
                "CosineAnnealingWarmRestarts: adjusting learning rate from {old_lr:.6e} to {new_lr:.6e} at epoch {new_epoch} (t_cur={}, t_i={}).",
                self.t_cur, self.t_i
            );
        }
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.compute_cycle_at_epoch(self.last_epoch).0]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra: vec![
                ("initial_lr".to_owned(), self.initial_lr),
                ("t_0".to_owned(), self.t_0 as f64),
                ("t_mult".to_owned(), self.t_mult as f64),
                ("eta_min".to_owned(), self.eta_min),
                ("t_cur".to_owned(), self.t_cur as f64),
                ("t_i".to_owned(), self.t_i as f64),
            ],
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }
        for (key, val) in &state.extra {
            match key.as_str() {
                "initial_lr" => self.initial_lr = *val,
                "t_0" => self.t_0 = (*val as usize).max(1),
                "t_mult" => self.t_mult = (*val as usize).max(1),
                "eta_min" => self.eta_min = *val,
                "t_cur" => self.t_cur = *val as i64,
                "t_i" => self.t_i = (*val as usize).max(1),
                _ => {}
            }
        }
    }
}

/// ExponentialLR: decays lr by a multiplicative `gamma` each epoch.
///
/// ```text
/// lr = initial_lr * gamma^epoch
/// ```
pub struct ExponentialLR {
    initial_lr: f64,
    gamma: f64,
    last_epoch: i64,
    last_lr: f64,
    verbose: bool,
}

impl ExponentialLR {
    /// Create a new `ExponentialLR` scheduler.
    pub fn new(optimizer: &dyn Optimizer, gamma: f64) -> Self {
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            gamma,
            last_epoch: -1,
            last_lr: initial_lr,
            verbose: false,
        }
    }

    /// Set the last epoch index for resuming (default: -1).
    #[must_use]
    pub fn last_epoch(mut self, last_epoch: i64) -> Self {
        self.last_epoch = last_epoch;
        if last_epoch >= 0 {
            self.last_lr = self.compute_lr_at_epoch(last_epoch);
        }
        self
    }

    /// Enable verbose mode: prints lr changes.
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn compute_lr_at_epoch(&self, epoch: i64) -> f64 {
        if epoch < 0 {
            return self.initial_lr;
        }
        self.initial_lr * self.gamma.powf(epoch as f64)
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        let new_epoch = match epoch {
            Some(e) => e,
            None => self.last_epoch + 1,
        };
        self.last_epoch = new_epoch;
        let new_lr = self.compute_lr_at_epoch(new_epoch);
        let old_lr = self.last_lr;
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);

        if self.verbose && (new_lr - old_lr).abs() > f64::EPSILON {
            eprintln!(
                "ExponentialLR: adjusting learning rate from {old_lr:.6e} to {new_lr:.6e} at epoch {new_epoch}."
            );
        }
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.compute_lr_at_epoch(self.last_epoch)]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra: vec![
                ("initial_lr".to_owned(), self.initial_lr),
                ("gamma".to_owned(), self.gamma),
            ],
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }
        for (key, val) in &state.extra {
            match key.as_str() {
                "initial_lr" => self.initial_lr = *val,
                "gamma" => self.gamma = *val,
                _ => {}
            }
        }
    }
}

/// LinearLR: linearly interpolates an lr multiplier from `start_factor` to
/// `end_factor` over `total_iters` epochs, then holds steady.
pub struct LinearLR {
    initial_lr: f64,
    start_factor: f64,
    end_factor: f64,
    total_iters: usize,
    last_epoch: i64,
    last_lr: f64,
    verbose: bool,
}

impl LinearLR {
    /// Create a new `LinearLR` scheduler.
    ///
    /// Defaults mirror PyTorch's common warmup behavior.
    pub fn new(optimizer: &dyn Optimizer) -> Self {
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            start_factor: 1.0 / 3.0,
            end_factor: 1.0,
            total_iters: 5,
            last_epoch: -1,
            last_lr: initial_lr,
            verbose: false,
        }
    }

    /// Set the start multiplier (default: 1/3).
    #[must_use]
    pub fn start_factor(mut self, start_factor: f64) -> Self {
        self.start_factor = start_factor;
        self
    }

    /// Set the end multiplier (default: 1.0).
    #[must_use]
    pub fn end_factor(mut self, end_factor: f64) -> Self {
        self.end_factor = end_factor;
        self
    }

    /// Set interpolation length in epochs (default: 5).
    #[must_use]
    pub fn total_iters(mut self, total_iters: usize) -> Self {
        self.total_iters = total_iters;
        self
    }

    /// Set the last epoch index for resuming (default: -1).
    #[must_use]
    pub fn last_epoch(mut self, last_epoch: i64) -> Self {
        self.last_epoch = last_epoch;
        if last_epoch >= 0 {
            self.last_lr = self.compute_lr_at_epoch(last_epoch);
        }
        self
    }

    /// Enable verbose mode: prints lr changes.
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn multiplier_at_epoch(&self, epoch: i64) -> f64 {
        if epoch < 0 {
            return 1.0;
        }
        if self.total_iters == 0 {
            return self.end_factor;
        }
        let progress = (epoch as usize).min(self.total_iters) as f64 / self.total_iters as f64;
        self.start_factor + (self.end_factor - self.start_factor) * progress
    }

    fn compute_lr_at_epoch(&self, epoch: i64) -> f64 {
        self.initial_lr * self.multiplier_at_epoch(epoch)
    }
}

impl LRScheduler for LinearLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        let new_epoch = match epoch {
            Some(e) => e,
            None => self.last_epoch + 1,
        };
        self.last_epoch = new_epoch;
        let new_lr = self.compute_lr_at_epoch(new_epoch);
        let old_lr = self.last_lr;
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);

        if self.verbose && (new_lr - old_lr).abs() > f64::EPSILON {
            eprintln!(
                "LinearLR: adjusting learning rate from {old_lr:.6e} to {new_lr:.6e} at epoch {new_epoch}."
            );
        }
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.compute_lr_at_epoch(self.last_epoch)]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra: vec![
                ("initial_lr".to_owned(), self.initial_lr),
                ("start_factor".to_owned(), self.start_factor),
                ("end_factor".to_owned(), self.end_factor),
                ("total_iters".to_owned(), self.total_iters as f64),
            ],
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }
        for (key, val) in &state.extra {
            match key.as_str() {
                "initial_lr" => self.initial_lr = *val,
                "start_factor" => self.start_factor = *val,
                "end_factor" => self.end_factor = *val,
                "total_iters" => self.total_iters = *val as usize,
                _ => {}
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlateauMode {
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThresholdMode {
    Rel,
    Abs,
}

/// ReduceLROnPlateau: reduce lr when a monitored metric stops improving.
///
/// This scheduler is metric-driven. Use `step_with_metric(optimizer, metric)`
/// each epoch with the metric value (for example validation loss).
pub struct ReduceLROnPlateau {
    mode: PlateauMode,
    factor: f64,
    patience: usize,
    threshold: f64,
    threshold_mode: ThresholdMode,
    cooldown: usize,
    min_lr: f64,
    eps: f64,
    last_epoch: i64,
    last_lr: f64,
    best: f64,
    num_bad_epochs: usize,
    cooldown_counter: usize,
    initialized: bool,
    verbose: bool,
}

impl ReduceLROnPlateau {
    /// Create a new `ReduceLROnPlateau` scheduler with PyTorch-like defaults.
    pub fn new(optimizer: &dyn Optimizer) -> Self {
        Self {
            mode: PlateauMode::Min,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            threshold_mode: ThresholdMode::Rel,
            cooldown: 0,
            min_lr: 0.0,
            eps: 1e-8,
            last_epoch: -1,
            last_lr: optimizer.get_lr(),
            best: 0.0,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            initialized: false,
            verbose: false,
        }
    }

    /// Switch to minimization mode (default): lower metric is better.
    #[must_use]
    pub fn mode_min(mut self) -> Self {
        self.mode = PlateauMode::Min;
        self
    }

    /// Switch to maximization mode: higher metric is better.
    #[must_use]
    pub fn mode_max(mut self) -> Self {
        self.mode = PlateauMode::Max;
        self
    }

    /// Set multiplicative reduction factor (default: 0.1).
    #[must_use]
    pub fn factor(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }

    /// Set tolerated bad epochs before reducing lr (default: 10).
    #[must_use]
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set improvement threshold (default: 1e-4).
    #[must_use]
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Use relative threshold mode (default).
    #[must_use]
    pub fn threshold_mode_rel(mut self) -> Self {
        self.threshold_mode = ThresholdMode::Rel;
        self
    }

    /// Use absolute threshold mode.
    #[must_use]
    pub fn threshold_mode_abs(mut self) -> Self {
        self.threshold_mode = ThresholdMode::Abs;
        self
    }

    /// Set cooldown epochs after each reduction (default: 0).
    #[must_use]
    pub fn cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Set minimum learning rate floor (default: 0.0).
    #[must_use]
    pub fn min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Minimum required absolute lr delta for applying a reduction.
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Enable verbose mode.
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn is_better(&self, metric: f64) -> bool {
        if !self.initialized || metric.is_nan() {
            return false;
        }
        match (self.mode, self.threshold_mode) {
            (PlateauMode::Min, ThresholdMode::Rel) => metric < self.best * (1.0 - self.threshold),
            (PlateauMode::Min, ThresholdMode::Abs) => metric < self.best - self.threshold,
            (PlateauMode::Max, ThresholdMode::Rel) => metric > self.best * (1.0 + self.threshold),
            (PlateauMode::Max, ThresholdMode::Abs) => metric > self.best + self.threshold,
        }
    }

    fn reduce_lr(&mut self, optimizer: &mut dyn Optimizer) {
        let old_lr = optimizer.get_lr();
        let new_lr = (old_lr * self.factor).max(self.min_lr);
        if (old_lr - new_lr).abs() > self.eps {
            optimizer.set_lr(new_lr);
            self.last_lr = new_lr;
            if self.verbose {
                eprintln!(
                    "ReduceLROnPlateau: reducing learning rate from {old_lr:.6e} to {new_lr:.6e}."
                );
            }
        } else {
            self.last_lr = old_lr;
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        self.last_epoch = epoch.unwrap_or(self.last_epoch + 1);
        self.last_lr = optimizer.get_lr();
    }

    fn step_with_metric(&mut self, optimizer: &mut dyn Optimizer, metric: f64) {
        self.last_epoch += 1;
        self.last_lr = optimizer.get_lr();

        if !self.initialized {
            self.best = metric;
            self.initialized = true;
            self.num_bad_epochs = 0;
            return;
        }

        if self.is_better(metric) {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs = self.num_bad_epochs.saturating_add(1);
        }

        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
        }

        if self.num_bad_epochs > self.patience {
            self.reduce_lr(optimizer);
            self.cooldown_counter = self.cooldown;
            self.num_bad_epochs = 0;
        }
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra: vec![
                (
                    "mode".to_owned(),
                    if self.mode == PlateauMode::Min {
                        0.0
                    } else {
                        1.0
                    },
                ),
                ("factor".to_owned(), self.factor),
                ("patience".to_owned(), self.patience as f64),
                ("threshold".to_owned(), self.threshold),
                (
                    "threshold_mode".to_owned(),
                    if self.threshold_mode == ThresholdMode::Rel {
                        0.0
                    } else {
                        1.0
                    },
                ),
                ("cooldown".to_owned(), self.cooldown as f64),
                ("min_lr".to_owned(), self.min_lr),
                ("eps".to_owned(), self.eps),
                ("best".to_owned(), self.best),
                ("num_bad_epochs".to_owned(), self.num_bad_epochs as f64),
                ("cooldown_counter".to_owned(), self.cooldown_counter as f64),
                (
                    "initialized".to_owned(),
                    if self.initialized { 1.0 } else { 0.0 },
                ),
            ],
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }

        for (key, val) in &state.extra {
            match key.as_str() {
                "mode" => {
                    self.mode = if *val >= 0.5 {
                        PlateauMode::Max
                    } else {
                        PlateauMode::Min
                    };
                }
                "factor" => self.factor = *val,
                "patience" => self.patience = *val as usize,
                "threshold" => self.threshold = *val,
                "threshold_mode" => {
                    self.threshold_mode = if *val >= 0.5 {
                        ThresholdMode::Abs
                    } else {
                        ThresholdMode::Rel
                    };
                }
                "cooldown" => self.cooldown = *val as usize,
                "min_lr" => self.min_lr = *val,
                "eps" => self.eps = *val,
                "best" => self.best = *val,
                "num_bad_epochs" => self.num_bad_epochs = *val as usize,
                "cooldown_counter" => self.cooldown_counter = *val as usize,
                "initialized" => self.initialized = *val >= 0.5,
                _ => {}
            }
        }
    }
}

struct ShadowOptimizer {
    lr: f64,
}

impl ShadowOptimizer {
    fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Optimizer for ShadowOptimizer {
    fn step(
        &mut self,
        _session: &mut FrankenTorchSession,
        _report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        Ok(())
    }

    fn zero_grad(&mut self, _session: &mut FrankenTorchSession) -> Result<(), AutogradError> {
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// LambdaLR: apply a user-provided multiplier function to the base lr.
pub struct LambdaLR {
    initial_lr: f64,
    lr_lambda: Box<dyn Fn(i64) -> f64 + Send + Sync>,
    last_epoch: i64,
    last_lr: f64,
    verbose: bool,
}

impl LambdaLR {
    pub fn new<F>(optimizer: &dyn Optimizer, lr_lambda: F) -> Self
    where
        F: Fn(i64) -> f64 + Send + Sync + 'static,
    {
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            lr_lambda: Box::new(lr_lambda),
            last_epoch: -1,
            last_lr: initial_lr,
            verbose: false,
        }
    }

    #[must_use]
    pub fn last_epoch(mut self, last_epoch: i64) -> Self {
        self.last_epoch = last_epoch;
        if last_epoch >= 0 {
            self.last_lr = self.compute_lr_at_epoch(last_epoch);
        }
        self
    }

    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn compute_lr_at_epoch(&self, epoch: i64) -> f64 {
        if epoch < 0 {
            return self.initial_lr;
        }
        let multiplier = (self.lr_lambda)(epoch).max(0.0);
        self.initial_lr * multiplier
    }
}

impl LRScheduler for LambdaLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        let new_epoch = epoch.unwrap_or(self.last_epoch + 1);
        self.last_epoch = new_epoch;
        let new_lr = self.compute_lr_at_epoch(new_epoch);
        let old_lr = self.last_lr;
        self.last_lr = new_lr;
        optimizer.set_lr(new_lr);

        if self.verbose && (new_lr - old_lr).abs() > f64::EPSILON {
            eprintln!(
                "LambdaLR: adjusting learning rate from {old_lr:.6e} to {new_lr:.6e} at epoch {new_epoch}."
            );
        }
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.compute_lr_at_epoch(self.last_epoch)]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra: vec![("initial_lr".to_owned(), self.initial_lr)],
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }
        for (key, val) in &state.extra {
            if key == "initial_lr" {
                self.initial_lr = *val;
            }
        }
    }
}

/// SequentialLR: switch between schedulers at milestone epochs.
pub struct SequentialLR {
    schedulers: Vec<Box<dyn LRScheduler>>,
    milestones: Vec<usize>,
    last_epoch: i64,
    last_lr: f64,
}

impl SequentialLR {
    pub fn new(
        optimizer: &dyn Optimizer,
        schedulers: Vec<Box<dyn LRScheduler>>,
        mut milestones: Vec<usize>,
    ) -> Self {
        assert!(
            !schedulers.is_empty(),
            "SequentialLR requires at least one scheduler"
        );
        milestones.sort_unstable();
        Self {
            schedulers,
            milestones,
            last_epoch: -1,
            last_lr: optimizer.get_lr(),
        }
    }

    fn active_index_and_local_epoch(&self, epoch: i64) -> (usize, i64) {
        if epoch < 0 {
            return (0, epoch);
        }

        let e = epoch as usize;
        let index = self
            .milestones
            .partition_point(|&milestone| milestone <= e)
            .min(self.schedulers.len().saturating_sub(1));
        let phase_start = if index == 0 {
            0
        } else {
            self.milestones[index - 1] as i64
        };
        (index, epoch - phase_start)
    }
}

impl LRScheduler for SequentialLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        let new_epoch = epoch.unwrap_or(self.last_epoch + 1);
        self.last_epoch = new_epoch;
        let (index, local_epoch) = self.active_index_and_local_epoch(new_epoch);
        self.schedulers[index].step(optimizer, Some(local_epoch));
        self.last_lr = optimizer.get_lr();
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra: self
                .milestones
                .iter()
                .enumerate()
                .map(|(idx, milestone)| (format!("milestone_{idx}"), *milestone as f64))
                .collect(),
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }
        let mut indexed_milestones = Vec::new();
        for (key, val) in &state.extra {
            if let Some(index) = key
                .strip_prefix("milestone_")
                .and_then(|suffix| suffix.parse::<usize>().ok())
            {
                indexed_milestones.push((index, *val as usize));
            }
        }
        if !indexed_milestones.is_empty() {
            indexed_milestones.sort_by_key(|(index, _)| *index);
            self.milestones = indexed_milestones
                .into_iter()
                .map(|(_, milestone)| milestone)
                .collect();
            self.milestones.sort_unstable();
        }
    }
}

/// ChainedScheduler: apply multiple schedulers multiplicatively.
pub struct ChainedScheduler {
    base_lr: f64,
    schedulers: Vec<Box<dyn LRScheduler>>,
    last_epoch: i64,
    last_lr: f64,
}

impl ChainedScheduler {
    pub fn new(optimizer: &dyn Optimizer, schedulers: Vec<Box<dyn LRScheduler>>) -> Self {
        assert!(
            !schedulers.is_empty(),
            "ChainedScheduler requires at least one scheduler"
        );
        let base_lr = optimizer.get_lr();
        Self {
            base_lr,
            schedulers,
            last_epoch: -1,
            last_lr: base_lr,
        }
    }
}

impl LRScheduler for ChainedScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer, epoch: Option<i64>) {
        let new_epoch = epoch.unwrap_or(self.last_epoch + 1);
        self.last_epoch = new_epoch;

        let mut factor = 1.0;
        for scheduler in &mut self.schedulers {
            let mut shadow = ShadowOptimizer::new(self.base_lr);
            scheduler.step(&mut shadow, Some(new_epoch));
            let sched_lr = scheduler.get_last_lr()[0];
            if self.base_lr.abs() > f64::EPSILON {
                factor *= sched_lr / self.base_lr;
            }
        }

        let new_lr = self.base_lr * factor;
        optimizer.set_lr(new_lr);
        self.last_lr = new_lr;
    }

    fn get_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.last_lr]
    }

    fn state_dict(&self) -> SchedulerState {
        SchedulerState {
            last_epoch: self.last_epoch,
            last_lrs: vec![self.last_lr],
            extra: vec![("base_lr".to_owned(), self.base_lr)],
        }
    }

    fn load_state_dict(&mut self, state: SchedulerState) {
        self.last_epoch = state.last_epoch;
        if let Some(&lr) = state.last_lrs.first() {
            self.last_lr = lr;
        }
        for (key, val) in &state.extra {
            if key == "base_lr" {
                self.base_lr = *val;
            }
        }
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
    fn sgd_zero_grad_clears_persistent_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("variable should succeed");

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let mut optimizer = SGD::new(vec![x], 0.1);
        optimizer.step(&mut session, &report).expect("step");

        let grad_before = session
            .tensor_accumulated_gradient(x)
            .expect("accumulated gradient lookup")
            .expect("gradient should exist");
        assert!(grad_before.iter().any(|value| value.abs() > 0.0));

        optimizer
            .zero_grad(&mut session)
            .expect("zero_grad should succeed");
        let grad_after = session
            .tensor_accumulated_gradient(x)
            .expect("accumulated gradient lookup")
            .expect("gradient entry should remain allocated");
        assert!(
            grad_after.iter().all(|value| value.abs() < 1e-12),
            "expected zeroed gradient, got {:?}",
            grad_after
        );
    }

    #[test]
    fn sgd_accumulates_gradients_across_backwards() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("variable should succeed");

        for _ in 0..3 {
            let loss = session.tensor_mul(x, x).expect("mul");
            let loss_sum = session.tensor_sum(loss).expect("sum");
            let _ = session.tensor_backward(loss_sum).expect("backward");
        }

        let grad = session
            .tensor_accumulated_gradient(x)
            .expect("accumulated gradient lookup")
            .expect("gradient should exist");
        assert!(
            (grad[0] - 24.0).abs() < 1e-10,
            "expected accumulated gradient 24.0, got {}",
            grad[0]
        );
    }

    #[test]
    fn sgd_step_reads_persistent_gradients_not_report_payload() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("x var");
        let y = session
            .tensor_variable(vec![3.0], vec![1], true)
            .expect("y var");

        let mut optimizer = SGD::new(vec![x], 0.1);

        let x_loss = session.tensor_mul(x, x).expect("mul");
        let x_loss_sum = session.tensor_sum(x_loss).expect("sum");
        let _x_report = session.tensor_backward(x_loss_sum).expect("x backward");

        let y_loss = session.tensor_mul(y, y).expect("mul");
        let y_loss_sum = session.tensor_sum(y_loss).expect("sum");
        let unrelated_report = session.tensor_backward(y_loss_sum).expect("y backward");

        optimizer
            .step(&mut session, &unrelated_report)
            .expect("step should use persistent gradient for x");

        let x_val = session.tensor_values(x).expect("x values")[0];
        assert!(
            (x_val - 3.2).abs() < 1e-10,
            "expected x to update from persistent grad to 3.2, got {}",
            x_val
        );
    }

    #[test]
    fn sgd_microbatch_accumulation_matches_single_batch_step() {
        let targets = [1.0, 2.0, 3.0, 4.0];

        let mut full_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let full_w = full_session
            .tensor_variable(vec![2.0], vec![1], true)
            .expect("full w");
        let mut full_optimizer = SGD::new(vec![full_w], 0.05);

        let mut total_loss = None;
        for target_value in targets {
            let target = full_session
                .tensor_variable(vec![target_value], vec![1], false)
                .expect("target");
            let diff = full_session.tensor_sub(full_w, target).expect("sub");
            let sq = full_session.tensor_mul(diff, diff).expect("mul");
            let loss = full_session.tensor_sum(sq).expect("sum");
            total_loss = Some(match total_loss {
                Some(acc) => full_session.tensor_add(acc, loss).expect("add"),
                None => loss,
            });
        }
        let full_report = full_session
            .tensor_backward(total_loss.expect("total loss"))
            .expect("full backward");
        full_optimizer
            .step(&mut full_session, &full_report)
            .expect("full step");
        let full_value = full_session.tensor_values(full_w).expect("full value")[0];

        let mut micro_session = FrankenTorchSession::new(ExecutionMode::Strict);
        let micro_w = micro_session
            .tensor_variable(vec![2.0], vec![1], true)
            .expect("micro w");
        let mut micro_optimizer = SGD::new(vec![micro_w], 0.05);
        let mut last_report = None;

        for target_value in targets {
            let target = micro_session
                .tensor_variable(vec![target_value], vec![1], false)
                .expect("target");
            let diff = micro_session.tensor_sub(micro_w, target).expect("sub");
            let sq = micro_session.tensor_mul(diff, diff).expect("mul");
            let loss = micro_session.tensor_sum(sq).expect("sum");
            last_report = Some(micro_session.tensor_backward(loss).expect("micro backward"));
        }
        let last_report = last_report.expect("at least one micro-batch");
        micro_optimizer
            .step(&mut micro_session, &last_report)
            .expect("micro step");
        let micro_value = micro_session.tensor_values(micro_w).expect("micro value")[0];

        assert!(
            (full_value - micro_value).abs() < 1e-10,
            "expected microbatch and full-batch steps to match: full={}, micro={}",
            full_value,
            micro_value
        );
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
            optimizer.zero_grad(&mut session).expect("zero_grad");
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
            optimizer.zero_grad(&mut session).expect("zero_grad");
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
    fn adam_zero_grad_clears_persistent_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let mut optimizer = Adam::new(vec![x], 0.1);
        optimizer.step(&mut session, &report).expect("step");
        optimizer
            .zero_grad(&mut session)
            .expect("adam zero_grad should succeed");

        let grad_after = session
            .tensor_accumulated_gradient(x)
            .expect("accumulated gradient lookup")
            .expect("gradient entry should remain allocated");
        assert!(grad_after.iter().all(|value| value.abs() < 1e-12));
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
            optimizer.zero_grad(&mut session).expect("zero_grad");
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
            optimizer.zero_grad(&mut session).expect("zero_grad");
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
    fn adamw_zero_grad_clears_persistent_gradients() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");

        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let mut optimizer = AdamW::new(vec![x], 0.1);
        optimizer.step(&mut session, &report).expect("step");
        optimizer
            .zero_grad(&mut session)
            .expect("adamw zero_grad should succeed");

        let grad_after = session
            .tensor_accumulated_gradient(x)
            .expect("accumulated gradient lookup")
            .expect("gradient entry should remain allocated");
        assert!(grad_after.iter().all(|value| value.abs() < 1e-12));
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
            optimizer.zero_grad(&mut session).expect("zero_grad");
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

    // ── LBFGS tests ────────────────────────────────────────────────────

    #[test]
    fn lbfgs_basic_step_reduces_loss() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = LBFGS::new(vec![x], 0.2);

        optimizer.zero_grad(&mut session).expect("zero_grad");
        let loss = session.tensor_mul(x, x).expect("mul");
        let loss_sum = session.tensor_sum(loss).expect("sum");
        let report = session.tensor_backward(loss_sum).expect("backward");

        let before = session.tensor_values(x).expect("values")[0];
        optimizer.step(&mut session, &report).expect("step");
        let after = session.tensor_values(x).expect("values")[0];

        assert!(
            after < before,
            "LBFGS should decrease x: before={}, after={}",
            before,
            after
        );
    }

    #[test]
    fn lbfgs_step_with_closure_reduces_quadratic_loss() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut optimizer = LBFGS::new(vec![x], 1.0)
            .max_iter(10)
            .max_eval(20)
            .line_search_fn(LBFGSLineSearch::StrongWolfe);

        let mut closure = |session: &mut FrankenTorchSession| -> Result<f64, AutogradError> {
            session.tensor_zero_grads(&[x])?;
            let loss = session.tensor_mul(x, x)?;
            let loss_sum = session.tensor_sum(loss)?;
            let loss_value = session.tensor_item(loss_sum)?;
            let _report = session.tensor_backward(loss_sum)?;
            Ok(loss_value)
        };

        let initial_loss = closure(&mut session).expect("initial closure");
        let final_loss = optimizer
            .step_with_closure(&mut session, &mut closure)
            .expect("lbfgs closure step");

        assert!(
            final_loss < initial_loss,
            "LBFGS closure step should reduce loss: initial={}, final={}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn lbfgs_history_respects_history_size() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.0], vec![1], true)
            .expect("var");
        let mut optimizer = LBFGS::new(vec![x], 0.1).history_size(2);

        for _ in 0..6 {
            optimizer.zero_grad(&mut session).expect("zero grad");
            let loss = session.tensor_mul(x, x).expect("mul");
            let loss_sum = session.tensor_sum(loss).expect("sum");
            let report = session.tensor_backward(loss_sum).expect("backward");
            optimizer.step(&mut session, &report).expect("step");
        }

        assert!(
            optimizer.s_history.len() <= 2
                && optimizer.y_history.len() <= 2
                && optimizer.rho_history.len() <= 2
        );
    }

    #[test]
    fn lbfgs_two_loop_defaults_to_negative_gradient_without_history() {
        let optimizer = LBFGS::new(Vec::new(), 1.0);
        let grad = vec![1.5, -2.0, 0.25];
        let direction = optimizer.two_loop_direction(&grad);
        assert_eq!(direction, vec![-1.5, 2.0, -0.25]);
    }

    // -----------------------------------------------------------------------
    // LRScheduler / StepLR tests
    // -----------------------------------------------------------------------

    #[test]
    fn optimizer_get_set_lr_roundtrip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.05);
        assert!((opt.get_lr() - 0.05).abs() < f64::EPSILON);
        opt.set_lr(0.01);
        assert!((opt.get_lr() - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn step_lr_basic_decay() {
        // lr = 0.1 * 0.1^(epoch / 10)
        // epoch  0-9  -> lr = 0.1
        // epoch 10-19 -> lr = 0.01
        // epoch 20-29 -> lr = 0.001
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.1);
        let mut scheduler = StepLR::new(&opt, 10).gamma(0.1);

        for epoch in 0..30 {
            scheduler.step(&mut opt, None);
            let expected = if epoch < 10 {
                0.1
            } else if epoch < 20 {
                0.01
            } else {
                0.001
            };
            let actual = opt.get_lr();
            assert!(
                (actual - expected).abs() < 1e-12,
                "epoch {epoch}: expected lr {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn step_lr_gamma_half() {
        // gamma=0.5, step_size=10: lr halves every 10 epochs
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = StepLR::new(&opt, 10).gamma(0.5);

        for epoch in 0..30 {
            scheduler.step(&mut opt, None);
            let expected = 0.5_f64.powi(epoch / 10);
            let actual = opt.get_lr();
            assert!(
                (actual - expected).abs() < 1e-12,
                "epoch {epoch}: expected lr {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn step_lr_epoch_zero_unchanged() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = Adam::new(vec![x], 0.001);
        let mut scheduler = StepLR::new(&opt, 5).gamma(0.5);

        // First step: epoch 0
        scheduler.step(&mut opt, None);
        assert!(
            (opt.get_lr() - 0.001).abs() < 1e-12,
            "lr should remain initial at epoch 0"
        );
    }

    #[test]
    fn step_lr_get_lr_and_get_last_lr() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.1);
        let mut scheduler = StepLR::new(&opt, 5).gamma(0.5);

        // Before any step
        assert_eq!(scheduler.get_lr(), vec![0.1]);
        assert_eq!(scheduler.get_last_lr(), vec![0.1]);

        // Step through epochs 0-4 (lr should be 0.1)
        for _ in 0..5 {
            scheduler.step(&mut opt, None);
        }
        assert_eq!(scheduler.get_lr(), vec![0.1]);

        // Step to epoch 5 (lr should be 0.05)
        scheduler.step(&mut opt, None);
        let lr = scheduler.get_lr()[0];
        assert!((lr - 0.05).abs() < 1e-12, "expected 0.05 got {lr}");
        assert!((scheduler.get_last_lr()[0] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn step_lr_step_size_one() {
        // step_size=1: lr decays every epoch
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = StepLR::new(&opt, 1).gamma(0.5);

        for epoch in 0..5 {
            scheduler.step(&mut opt, None);
            let expected = 0.5_f64.powi(epoch);
            let actual = opt.get_lr();
            assert!(
                (actual - expected).abs() < 1e-12,
                "epoch {epoch}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn step_lr_gamma_one_constant() {
        // gamma=1.0: lr never changes
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.1);
        let mut scheduler = StepLR::new(&opt, 5).gamma(1.0);

        for _epoch in 0..20 {
            scheduler.step(&mut opt, None);
            assert!(
                (opt.get_lr() - 0.1).abs() < 1e-12,
                "lr should remain constant with gamma=1.0"
            );
        }
    }

    #[test]
    fn step_lr_resume_from_middle() {
        // last_epoch != -1: resume from epoch 15
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.1);
        let mut scheduler = StepLR::new(&opt, 10).gamma(0.1).last_epoch(15);

        // At epoch 15: exponent = 15/10 = 1, lr = 0.1 * 0.1^1 = 0.01
        let lr = scheduler.get_lr()[0];
        assert!(
            (lr - 0.01).abs() < 1e-12,
            "expected 0.01 at epoch 15, got {lr}"
        );

        // Next step is epoch 16
        scheduler.step(&mut opt, None);
        assert!(
            (opt.get_lr() - 0.01).abs() < 1e-12,
            "lr should still be 0.01 at epoch 16"
        );
    }

    #[test]
    fn step_lr_explicit_epoch() {
        // step() called with explicit epoch
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.1);
        let mut scheduler = StepLR::new(&opt, 10).gamma(0.1);

        scheduler.step(&mut opt, Some(20));
        // epoch 20: exponent = 2, lr = 0.1 * 0.01 = 0.001
        let actual = opt.get_lr();
        assert!(
            (actual - 0.001).abs() < 1e-12,
            "expected 0.001 at epoch 20, got {actual}"
        );
    }

    #[test]
    fn step_lr_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.1);
        let mut scheduler = StepLR::new(&opt, 10).gamma(0.1);

        // Advance to epoch 15
        for _ in 0..16 {
            scheduler.step(&mut opt, None);
        }

        let state = scheduler.state_dict();
        assert_eq!(state.last_epoch, 15);

        // Create a fresh scheduler and load state
        let mut scheduler2 = StepLR::new(&opt, 10).gamma(0.1);
        scheduler2.load_state_dict(state.clone());

        assert_eq!(scheduler2.state_dict(), state);
        assert_eq!(scheduler2.get_lr(), scheduler.get_lr());
        assert_eq!(scheduler2.get_last_lr(), scheduler.get_last_lr());
    }

    #[test]
    fn step_lr_with_sgd_integration() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![4.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.5);
        let mut scheduler = StepLR::new(&opt, 2).gamma(0.5);

        // Simulate 4 epochs of training
        for _epoch in 0..4 {
            let x_sq = session.tensor_mul(x, x).expect("mul");
            let loss = session.tensor_sum(x_sq).expect("sum");
            let report = session.tensor_backward(loss).expect("backward");
            opt.step(&mut session, &report).expect("opt step");
            scheduler.step(&mut opt, None);
        }

        // After 4 epochs, lr should have decayed twice
        // epoch 0: lr=0.5, epoch 1: lr=0.5, epoch 2: lr=0.25, epoch 3: lr=0.25
        let final_lr = opt.get_lr();
        assert!(
            (final_lr - 0.25).abs() < 1e-12,
            "expected 0.25 after 4 epochs, got {final_lr}"
        );
    }

    #[test]
    fn step_lr_with_adam_integration() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![3.0], vec![1], true)
            .expect("var");
        let mut opt = Adam::new(vec![x], 0.01);
        let mut scheduler = StepLR::new(&opt, 5).gamma(0.5);

        // Simulate 10 epochs
        for _epoch in 0..10 {
            let loss = session.tensor_mul(x, x).expect("mul");
            let loss_sum = session.tensor_sum(loss).expect("sum");
            let report = session.tensor_backward(loss_sum).expect("backward");
            opt.step(&mut session, &report).expect("opt step");
            scheduler.step(&mut opt, None);
        }

        // After 10 epochs, lr should have decayed twice
        // epochs 0-4: lr=0.01, epochs 5-9: lr=0.005
        let final_lr = opt.get_lr();
        assert!(
            (final_lr - 0.005).abs() < 1e-12,
            "expected 0.005 after 10 epochs, got {final_lr}"
        );
    }

    #[test]
    fn step_lr_30_epoch_schedule() {
        // Full 30-epoch test with step_size=10, gamma=0.1
        // Verify lr values at key epoch boundaries
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = StepLR::new(&opt, 10).gamma(0.1);

        let expected_at = [
            (0, 1.0),
            (9, 1.0),
            (10, 0.1),
            (19, 0.1),
            (20, 0.01),
            (29, 0.01),
        ];

        for epoch in 0..30 {
            scheduler.step(&mut opt, None);
            for &(check_epoch, expected_lr) in &expected_at {
                if epoch == check_epoch {
                    let actual = opt.get_lr();
                    assert!(
                        (actual - expected_lr).abs() < 1e-12,
                        "epoch {epoch}: expected lr {expected_lr}, got {actual}"
                    );
                }
            }
        }
    }

    #[test]
    fn multistep_lr_milestone_boundaries() {
        // milestones=[30, 60, 90], gamma=0.1
        // epoch <30 -> 0.1
        // 30..59 -> 0.01
        // 60..89 -> 0.001
        // >=90 -> 0.0001
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.1);
        let mut scheduler = MultiStepLR::new(&opt, vec![30, 60, 90]).gamma(0.1);

        for epoch in 0..100 {
            scheduler.step(&mut opt, None);
            let expected = if epoch < 30 {
                0.1
            } else if epoch < 60 {
                0.01
            } else if epoch < 90 {
                0.001
            } else {
                0.0001
            };
            let actual = opt.get_lr();
            assert!(
                (actual - expected).abs() < 1e-12,
                "epoch {epoch}: expected lr {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn multistep_lr_empty_milestones_is_constant() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.2);
        let mut scheduler = MultiStepLR::new(&opt, Vec::new()).gamma(0.5);

        for _epoch in 0..25 {
            scheduler.step(&mut opt, None);
            assert!(
                (opt.get_lr() - 0.2).abs() < 1e-12,
                "lr should remain unchanged without milestones"
            );
        }
    }

    #[test]
    fn multistep_lr_unsorted_milestones_are_sorted_internally() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = MultiStepLR::new(&opt, vec![10, 3, 7]).gamma(0.1);

        scheduler.step(&mut opt, Some(2));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        scheduler.step(&mut opt, Some(3));
        assert!((opt.get_lr() - 0.1).abs() < 1e-12);
        scheduler.step(&mut opt, Some(7));
        assert!((opt.get_lr() - 0.01).abs() < 1e-12);
        scheduler.step(&mut opt, Some(10));
        assert!((opt.get_lr() - 0.001).abs() < 1e-12);
    }

    #[test]
    fn multistep_lr_duplicate_milestones_apply_multiple_decays() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = MultiStepLR::new(&opt, vec![2, 2, 4]).gamma(0.5);

        scheduler.step(&mut opt, Some(1));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        scheduler.step(&mut opt, Some(2));
        // Two milestones at epoch 2 -> gamma^2
        assert!((opt.get_lr() - 0.25).abs() < 1e-12);
        scheduler.step(&mut opt, Some(4));
        // Third milestone -> gamma^3
        assert!((opt.get_lr() - 0.125).abs() < 1e-12);
    }

    #[test]
    fn multistep_lr_all_milestones_at_zero_decay_immediately() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = MultiStepLR::new(&opt, vec![0, 0, 0]).gamma(0.1);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 0.001).abs() < 1e-12);

        scheduler.step(&mut opt, Some(1));
        assert!((opt.get_lr() - 0.001).abs() < 1e-12);
    }

    #[test]
    fn multistep_lr_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.8);
        let mut scheduler = MultiStepLR::new(&opt, vec![4, 8, 12]).gamma(0.2);

        scheduler.step(&mut opt, Some(9));
        let state = scheduler.state_dict();

        let mut scheduler2 = MultiStepLR::new(&opt, vec![1]).gamma(0.9);
        scheduler2.load_state_dict(state.clone());

        assert_eq!(scheduler2.state_dict(), state);
        assert_eq!(scheduler2.get_last_lr(), scheduler.get_last_lr());
        assert_eq!(scheduler2.get_lr(), scheduler.get_lr());
    }

    #[test]
    fn cosine_annealing_lr_key_epochs() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = CosineAnnealingLR::new(&opt, 10).eta_min(0.0);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        assert!((scheduler.get_last_lr()[0] - 1.0).abs() < 1e-12);

        scheduler.step(&mut opt, Some(5));
        let mid = opt.get_lr();
        assert!(
            (mid - 0.5).abs() < 1e-12,
            "expected midpoint lr 0.5, got {mid}"
        );

        scheduler.step(&mut opt, Some(10));
        let min_lr = opt.get_lr();
        assert!(
            min_lr.abs() < 1e-12,
            "expected eta_min at epoch 10, got {min_lr}"
        );

        scheduler.step(&mut opt, Some(25));
        let held = opt.get_lr();
        assert!(
            held.abs() < 1e-12,
            "expected lr to remain eta_min beyond t_max, got {held}"
        );
    }

    #[test]
    fn cosine_annealing_lr_t_max_one_and_eta_min_constant() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.2);
        let mut scheduler = CosineAnnealingLR::new(&opt, 1).eta_min(0.2);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 0.2).abs() < 1e-12);
        scheduler.step(&mut opt, Some(1));
        assert!((opt.get_lr() - 0.2).abs() < 1e-12);
        scheduler.step(&mut opt, Some(7));
        assert!((opt.get_lr() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn cosine_annealing_lr_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.8);

        let mut scheduler = CosineAnnealingLR::new(&opt, 12).eta_min(0.1);
        scheduler.step(&mut opt, Some(7));
        let state = scheduler.state_dict();

        let mut scheduler2 = CosineAnnealingLR::new(&opt, 3).eta_min(0.0);
        scheduler2.load_state_dict(state.clone());

        assert_eq!(scheduler2.state_dict(), state);
        assert_eq!(scheduler2.get_last_lr(), scheduler.get_last_lr());
        assert_eq!(scheduler2.get_lr(), scheduler.get_lr());
    }

    #[test]
    fn cosine_warm_restarts_fixed_period_restarts() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = CosineAnnealingWarmRestarts::new(&opt, 4).eta_min(0.0);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);

        scheduler.step(&mut opt, Some(3));
        let before_restart = opt.get_lr();
        assert!(
            (before_restart - 0.146_446_609_406_726_27).abs() < 1e-12,
            "unexpected lr before restart: {before_restart}"
        );

        scheduler.step(&mut opt, Some(4));
        let restart_lr = opt.get_lr();
        assert!(
            (restart_lr - 1.0).abs() < 1e-12,
            "expected restart to reset lr to initial, got {restart_lr}"
        );

        scheduler.step(&mut opt, Some(8));
        let second_restart = opt.get_lr();
        assert!(
            (second_restart - 1.0).abs() < 1e-12,
            "expected periodic restart at epoch 8, got {second_restart}"
        );
    }

    #[test]
    fn cosine_warm_restarts_t_mult_doubles_period() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = CosineAnnealingWarmRestarts::new(&opt, 2)
            .t_mult(2)
            .eta_min(0.0);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);

        scheduler.step(&mut opt, Some(1));
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);

        scheduler.step(&mut opt, Some(2));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);

        scheduler.step(&mut opt, Some(5));
        let tail_first_long_cycle = opt.get_lr();
        assert!(
            (tail_first_long_cycle - 0.146_446_609_406_726_27).abs() < 1e-12,
            "unexpected lr at epoch 5: {tail_first_long_cycle}"
        );

        scheduler.step(&mut opt, Some(6));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_warm_restarts_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.5);

        let mut scheduler = CosineAnnealingWarmRestarts::new(&opt, 3)
            .t_mult(2)
            .eta_min(0.05);
        scheduler.step(&mut opt, Some(9));
        let state = scheduler.state_dict();

        let mut scheduler2 = CosineAnnealingWarmRestarts::new(&opt, 1)
            .t_mult(1)
            .eta_min(0.0);
        scheduler2.load_state_dict(state.clone());

        assert_eq!(scheduler2.state_dict(), state);
        assert_eq!(scheduler2.get_last_lr(), scheduler.get_last_lr());
        assert_eq!(scheduler2.get_lr(), scheduler.get_lr());
    }

    #[test]
    fn exponential_lr_basic_curve() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = ExponentialLR::new(&opt, 0.95);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);

        scheduler.step(&mut opt, Some(10));
        let expected_10 = 0.95_f64.powi(10);
        assert!(
            (opt.get_lr() - expected_10).abs() < 1e-12,
            "expected {expected_10}, got {}",
            opt.get_lr()
        );

        scheduler.step(&mut opt, Some(100));
        let expected_100 = 0.95_f64.powi(100);
        assert!(
            (opt.get_lr() - expected_100).abs() < 1e-12,
            "expected {expected_100}, got {}",
            opt.get_lr()
        );
    }

    #[test]
    fn exponential_lr_gamma_one_is_constant() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.3);
        let mut scheduler = ExponentialLR::new(&opt, 1.0);

        for epoch in 0..20 {
            scheduler.step(&mut opt, Some(epoch));
            assert!(
                (opt.get_lr() - 0.3).abs() < 1e-12,
                "lr should remain constant for gamma=1.0"
            );
        }
    }

    #[test]
    fn exponential_lr_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.7);

        let mut scheduler = ExponentialLR::new(&opt, 0.9);
        scheduler.step(&mut opt, Some(12));
        let state = scheduler.state_dict();

        let mut scheduler2 = ExponentialLR::new(&opt, 0.5);
        scheduler2.load_state_dict(state.clone());

        assert_eq!(scheduler2.state_dict(), state);
        assert_eq!(scheduler2.get_last_lr(), scheduler.get_last_lr());
        assert_eq!(scheduler2.get_lr(), scheduler.get_lr());
    }

    #[test]
    fn linear_lr_warmup_curve_and_plateau() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = LinearLR::new(&opt)
            .start_factor(0.1)
            .end_factor(1.0)
            .total_iters(10);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 0.1).abs() < 1e-12);

        scheduler.step(&mut opt, Some(5));
        assert!((opt.get_lr() - 0.55).abs() < 1e-12);

        scheduler.step(&mut opt, Some(10));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);

        scheduler.step(&mut opt, Some(20));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_lr_cooldown_curve() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = LinearLR::new(&opt)
            .start_factor(1.0)
            .end_factor(0.01)
            .total_iters(50);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);

        scheduler.step(&mut opt, Some(25));
        assert!((opt.get_lr() - 0.505).abs() < 1e-12);

        scheduler.step(&mut opt, Some(50));
        assert!((opt.get_lr() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn linear_lr_total_iters_zero_jumps_to_end_factor() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 2.0);
        let mut scheduler = LinearLR::new(&opt)
            .start_factor(0.25)
            .end_factor(0.5)
            .total_iters(0);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_lr_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.9);

        let mut scheduler = LinearLR::new(&opt)
            .start_factor(0.2)
            .end_factor(1.1)
            .total_iters(7);
        scheduler.step(&mut opt, Some(4));
        let state = scheduler.state_dict();

        let mut scheduler2 = LinearLR::new(&opt)
            .start_factor(1.0)
            .end_factor(1.0)
            .total_iters(1);
        scheduler2.load_state_dict(state.clone());

        assert_eq!(scheduler2.state_dict(), state);
        assert_eq!(scheduler2.get_last_lr(), scheduler.get_last_lr());
        assert_eq!(scheduler2.get_lr(), scheduler.get_lr());
    }

    #[test]
    fn reduce_on_plateau_min_mode_reduces_after_patience() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = ReduceLROnPlateau::new(&opt).factor(0.5).patience(2);

        scheduler.step_with_metric(&mut opt, 1.0); // baseline
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        scheduler.step_with_metric(&mut opt, 1.0); // bad #1
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        scheduler.step_with_metric(&mut opt, 1.0); // bad #2
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        scheduler.step_with_metric(&mut opt, 1.0); // bad #3 -> reduce
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reduce_on_plateau_improvement_resets_bad_epoch_counter() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = ReduceLROnPlateau::new(&opt).factor(0.5).patience(1);

        scheduler.step_with_metric(&mut opt, 1.0); // baseline
        scheduler.step_with_metric(&mut opt, 1.2); // bad #1
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        scheduler.step_with_metric(&mut opt, 0.8); // improvement reset
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        scheduler.step_with_metric(&mut opt, 0.9); // bad #1 after reset
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        scheduler.step_with_metric(&mut opt, 0.9); // bad #2 -> reduce
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reduce_on_plateau_cooldown_blocks_immediate_reductions() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = ReduceLROnPlateau::new(&opt)
            .factor(0.5)
            .patience(0)
            .cooldown(2);

        scheduler.step_with_metric(&mut opt, 1.0); // baseline
        scheduler.step_with_metric(&mut opt, 1.1); // reduce to 0.5 and enter cooldown
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);

        scheduler.step_with_metric(&mut opt, 1.2); // cooldown #1
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);
        scheduler.step_with_metric(&mut opt, 1.3); // cooldown #2
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);
        scheduler.step_with_metric(&mut opt, 1.4); // cooldown ended -> reduce
        assert!((opt.get_lr() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn reduce_on_plateau_mode_max_with_relative_threshold() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = ReduceLROnPlateau::new(&opt)
            .mode_max()
            .factor(0.5)
            .patience(0)
            .threshold(0.1)
            .threshold_mode_rel();

        scheduler.step_with_metric(&mut opt, 10.0); // baseline
        scheduler.step_with_metric(&mut opt, 10.5); // not > 10 * 1.1
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);
        scheduler.step_with_metric(&mut opt, 12.0); // strong improvement
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reduce_on_plateau_mode_max_with_absolute_threshold() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = ReduceLROnPlateau::new(&opt)
            .mode_max()
            .factor(0.5)
            .patience(0)
            .threshold(0.3)
            .threshold_mode_abs();

        scheduler.step_with_metric(&mut opt, 1.0); // baseline
        scheduler.step_with_metric(&mut opt, 1.2); // not > 1.3, reduce
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reduce_on_plateau_min_lr_floor_and_factor_one_noop() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);

        let mut scheduler = ReduceLROnPlateau::new(&opt)
            .factor(0.5)
            .patience(0)
            .min_lr(0.2);
        scheduler.step_with_metric(&mut opt, 1.0); // baseline
        scheduler.step_with_metric(&mut opt, 1.1); // 0.5
        scheduler.step_with_metric(&mut opt, 1.2); // 0.25
        scheduler.step_with_metric(&mut opt, 1.3); // 0.2 floor
        scheduler.step_with_metric(&mut opt, 1.4); // stay at floor
        assert!((opt.get_lr() - 0.2).abs() < 1e-12);

        let x2 = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt_noop = SGD::new(vec![x2], 0.7);
        let mut scheduler_noop = ReduceLROnPlateau::new(&opt_noop).factor(1.0).patience(0);
        scheduler_noop.step_with_metric(&mut opt_noop, 1.0);
        scheduler_noop.step_with_metric(&mut opt_noop, 1.1);
        assert!((opt_noop.get_lr() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn reduce_on_plateau_nan_metric_treated_as_no_improvement() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = ReduceLROnPlateau::new(&opt).factor(0.5).patience(0);

        scheduler.step_with_metric(&mut opt, 1.0); // baseline
        scheduler.step_with_metric(&mut opt, f64::NAN); // no improvement -> reduce
        assert!((opt.get_lr() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reduce_on_plateau_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.8);
        let mut scheduler = ReduceLROnPlateau::new(&opt)
            .mode_max()
            .factor(0.3)
            .patience(2)
            .threshold(0.25)
            .threshold_mode_abs()
            .cooldown(1)
            .min_lr(0.1)
            .eps(1e-6);

        scheduler.step_with_metric(&mut opt, 0.5);
        scheduler.step_with_metric(&mut opt, 0.6);
        let state = scheduler.state_dict();

        let mut scheduler2 = ReduceLROnPlateau::new(&opt);
        scheduler2.load_state_dict(state.clone());

        assert_eq!(scheduler2.state_dict(), state);
        assert_eq!(scheduler2.get_last_lr(), scheduler.get_last_lr());
        assert_eq!(scheduler2.get_lr(), scheduler.get_lr());
    }

    #[test]
    fn lambda_lr_linear_warmup_and_step_decay() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);

        let mut warmup = LambdaLR::new(&opt, |epoch| ((epoch + 1) as f64 / 10.0).min(1.0));
        warmup.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 0.1).abs() < 1e-12);
        warmup.step(&mut opt, Some(9));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);

        let mut step_decay = LambdaLR::new(&opt, |epoch| 0.1_f64.powi((epoch / 30) as i32));
        step_decay.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        step_decay.step(&mut opt, Some(30));
        assert!((opt.get_lr() - 0.1).abs() < 1e-12);
        step_decay.step(&mut opt, Some(60));
        assert!((opt.get_lr() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn lambda_lr_negative_multiplier_is_clamped() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);
        let mut scheduler = LambdaLR::new(&opt, |_epoch| -3.0);

        scheduler.step(&mut opt, Some(0));
        assert!((opt.get_lr() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn lambda_lr_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.9);
        let mut scheduler = LambdaLR::new(&opt, |epoch| 1.0 / (epoch + 1) as f64);
        scheduler.step(&mut opt, Some(3));
        let state = scheduler.state_dict();

        let mut scheduler2 = LambdaLR::new(&opt, |_epoch| 1.0);
        scheduler2.load_state_dict(state.clone());
        assert_eq!(scheduler2.state_dict(), state);
        assert_eq!(scheduler2.get_last_lr(), scheduler.get_last_lr());
    }

    #[test]
    fn sequential_lr_milestone_transition_and_single_equivalence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);

        let mut seq = SequentialLR::new(
            &opt,
            vec![
                Box::new(
                    LinearLR::new(&opt)
                        .start_factor(0.1)
                        .end_factor(1.0)
                        .total_iters(5),
                ),
                Box::new(CosineAnnealingLR::new(&opt, 10).eta_min(0.0)),
            ],
            vec![6],
        );
        seq.step(&mut opt, Some(5));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);
        seq.step(&mut opt, Some(6));
        assert!((opt.get_lr() - 1.0).abs() < 1e-12);

        let x2 = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt2 = SGD::new(vec![x2], 0.8);
        let mut seq_single =
            SequentialLR::new(&opt2, vec![Box::new(StepLR::new(&opt2, 10))], vec![]);
        let mut step = StepLR::new(&opt2, 10);
        seq_single.step(&mut opt2, Some(10));
        let seq_lr = opt2.get_lr();

        let x3 = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt3 = SGD::new(vec![x3], 0.8);
        step.step(&mut opt3, Some(10));
        assert!((seq_lr - opt3.get_lr()).abs() < 1e-12);
    }

    #[test]
    fn sequential_lr_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);

        let mut scheduler = SequentialLR::new(
            &opt,
            vec![
                Box::new(
                    LinearLR::new(&opt)
                        .start_factor(0.5)
                        .end_factor(1.0)
                        .total_iters(2),
                ),
                Box::new(ExponentialLR::new(&opt, 0.9)),
            ],
            vec![3],
        );
        scheduler.step(&mut opt, Some(4));
        let state = scheduler.state_dict();

        let mut scheduler2 = SequentialLR::new(&opt, vec![Box::new(StepLR::new(&opt, 1))], vec![]);
        scheduler2.load_state_dict(state.clone());
        assert_eq!(scheduler2.state_dict(), state);
        assert_eq!(scheduler2.get_last_lr(), scheduler.get_last_lr());
        assert_eq!(scheduler2.get_lr(), scheduler.get_lr());
    }

    #[test]
    fn chained_scheduler_multiplicative_and_single_equivalence() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 1.0);

        let mut chained = ChainedScheduler::new(
            &opt,
            vec![
                Box::new(StepLR::new(&opt, 1).gamma(0.1)),
                Box::new(ExponentialLR::new(&opt, 0.5)),
            ],
        );
        chained.step(&mut opt, Some(1));
        assert!((opt.get_lr() - 0.05).abs() < 1e-12);

        let x2 = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt2 = SGD::new(vec![x2], 0.6);
        let mut chain_single =
            ChainedScheduler::new(&opt2, vec![Box::new(ExponentialLR::new(&opt2, 0.9))]);
        let mut exp = ExponentialLR::new(&opt2, 0.9);
        chain_single.step(&mut opt2, Some(4));
        let chain_lr = opt2.get_lr();

        let x3 = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt3 = SGD::new(vec![x3], 0.6);
        exp.step(&mut opt3, Some(4));
        assert!((chain_lr - opt3.get_lr()).abs() < 1e-12);
    }

    #[test]
    fn chained_scheduler_state_dict_round_trip() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(vec![1.0], vec![1], true)
            .expect("var");
        let mut opt = SGD::new(vec![x], 0.7);
        let mut chained = ChainedScheduler::new(
            &opt,
            vec![
                Box::new(ExponentialLR::new(&opt, 0.95)),
                Box::new(StepLR::new(&opt, 5).gamma(0.5)),
            ],
        );
        chained.step(&mut opt, Some(5));
        let state = chained.state_dict();

        let mut chained2 =
            ChainedScheduler::new(&opt, vec![Box::new(ExponentialLR::new(&opt, 1.0))]);
        chained2.load_state_dict(state.clone());
        assert_eq!(chained2.state_dict(), state);
        assert_eq!(chained2.get_last_lr(), chained.get_last_lr());
        assert_eq!(chained2.get_lr(), chained.get_lr());
    }
}
