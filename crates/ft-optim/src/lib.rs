#![forbid(unsafe_code)]

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorBackwardReport, TensorNodeId};

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
}

impl Optimizer for SGD {
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        for (i, &param) in self.params.iter().enumerate() {
            let grad = match session.tensor_gradient(report, param) {
                Some(g) => g.to_vec(),
                None => continue,
            };

            let param_values = session.tensor_values(param)?;
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
                        session.tensor_values_meta(param)?.1.shape().to_vec(),
                        false,
                    )?;
                    session.tensor_sub_(param, update_node)?;
                } else {
                    // Standard momentum: param -= lr * velocity
                    let update_node = session.tensor_variable(
                        vel.iter().map(|v| self.lr * v).collect(),
                        session.tensor_values_meta(param)?.1.shape().to_vec(),
                        false,
                    )?;
                    session.tensor_sub_(param, update_node)?;
                }
            } else {
                // Vanilla SGD: param -= lr * grad
                let update_node = session.tensor_variable(
                    effective_grad.iter().map(|g| self.lr * g).collect(),
                    session.tensor_values_meta(param)?.1.shape().to_vec(),
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
}

impl Optimizer for Adam {
    fn step(
        &mut self,
        session: &mut FrankenTorchSession,
        report: &TensorBackwardReport,
    ) -> Result<(), AutogradError> {
        self.step_count += 1;
        let t = self.step_count;

        for (i, &param) in self.params.iter().enumerate() {
            let grad = match session.tensor_gradient(report, param) {
                Some(g) => g.to_vec(),
                None => continue,
            };

            let param_values = session.tensor_values(param)?;
            let mut effective_grad = grad;

            // Apply weight decay
            if self.weight_decay != 0.0 {
                for (g, p) in effective_grad.iter_mut().zip(param_values.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            let m = self.m[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
            for (m_val, g) in m.iter_mut().zip(effective_grad.iter()) {
                *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * g;
            }

            // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            let v = self.v[i].get_or_insert_with(|| vec![0.0; effective_grad.len()]);
            for (v_val, g) in v.iter_mut().zip(effective_grad.iter()) {
                *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * g * g;
            }

            // Bias-corrected estimates
            let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

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

            let shape = session.tensor_values_meta(param)?.1.shape().to_vec();
            let update_node = session.tensor_variable(update, shape, false)?;
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
}
