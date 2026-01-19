//! Cross-Modal Knowledge Distillation for WiFi-to-Vision transfer.
//!
//! The distillation framework enables training the WiFi "Student" model
//! to match outputs from a camera-based "Teacher" (pre-trained DensePose).
//!
//! Loss function: L_total = λ₁·L_distill + λ₂·L_keypoint + λ₃·L_uv
//!
//! where L_distill = KL(P_teacher || P_student) for probability distributions

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops::{log_softmax, softmax};

/// Configuration for knowledge distillation
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Temperature for softening probability distributions
    pub temperature: f64,
    /// Weight for distillation loss
    pub lambda_distill: f64,
    /// Weight for keypoint loss
    pub lambda_keypoint: f64,
    /// Weight for UV regression loss
    pub lambda_uv: f64,
    /// Weight for body part classification loss
    pub lambda_part: f64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            lambda_distill: 1.0,
            lambda_keypoint: 1.0,
            lambda_uv: 0.5,
            lambda_part: 1.0,
        }
    }
}

/// Knowledge distillation loss calculator
pub struct DistillationLoss {
    config: DistillationConfig,
}

impl DistillationLoss {
    pub fn new(config: DistillationConfig) -> Self {
        Self { config }
    }

    /// Compute KL divergence between teacher and student distributions
    ///
    /// KL(P_teacher || P_student) = Σ P_teacher * log(P_teacher / P_student)
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher model logits [batch, classes, ...]
    /// * `student_logits` - Student model logits [batch, classes, ...]
    ///
    /// # Returns
    /// KL divergence loss scalar
    pub fn kl_divergence(&self, teacher_logits: &Tensor, student_logits: &Tensor) -> Result<Tensor> {
        let t = self.config.temperature;

        // Soft targets from teacher
        let teacher_soft = softmax(&(teacher_logits / t)?, 1)?;

        // Log softmax from student
        let student_log_soft = log_softmax(&(student_logits / t)?, 1)?;

        // KL divergence: sum over classes, mean over batch
        let kl = (&teacher_soft * (teacher_soft.log()? - &student_log_soft)?)?;
        let kl = kl.sum(1)?; // Sum over classes
        let kl = kl.mean_all()?; // Mean over batch and spatial dims

        // Scale by T² as per Hinton et al.
        kl * (t * t)?
    }

    /// Compute keypoint heatmap loss (MSE + focal weighting)
    ///
    /// # Arguments
    /// * `pred_heatmaps` - Predicted heatmaps [batch, n_keypoints, size]
    /// * `target_heatmaps` - Target heatmaps [batch, n_keypoints, size]
    pub fn keypoint_loss(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // Basic MSE loss
        let diff = (pred - target)?;
        let mse = (&diff * &diff)?;

        // Focal weighting: emphasize hard examples (where target > 0)
        let focal_weight = (target * 4.0)?.clamp(0.1, 1.0)?;
        let weighted_mse = (mse * focal_weight)?;

        weighted_mse.mean_all()
    }

    /// Compute UV regression loss (Smooth L1)
    ///
    /// # Arguments
    /// * `pred_u` - Predicted U coordinates [batch, n_parts, size]
    /// * `pred_v` - Predicted V coordinates [batch, n_parts, size]
    /// * `target_u` - Target U coordinates
    /// * `target_v` - Target V coordinates
    /// * `mask` - Valid region mask [batch, size]
    pub fn uv_loss(
        &self,
        pred_u: &Tensor,
        pred_v: &Tensor,
        target_u: &Tensor,
        target_v: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let u_loss = smooth_l1_loss(pred_u, target_u)?;
        let v_loss = smooth_l1_loss(pred_v, target_v)?;

        // Apply mask (only compute loss where we have valid targets)
        let mask_expanded = mask.unsqueeze(1)?;
        let u_loss = (u_loss * &mask_expanded)?;
        let v_loss = (v_loss * &mask_expanded)?;

        // Average over valid regions
        let mask_sum = mask.sum_all()?;
        let n_valid = (mask_sum + 1e-6)?; // Avoid division by zero

        let u_mean = (u_loss.sum_all()? / &n_valid)?;
        let v_mean = (v_loss.sum_all()? / &n_valid)?;

        (u_mean + v_mean)? / 2.0
    }

    /// Compute body part classification loss (Cross-entropy)
    ///
    /// # Arguments
    /// * `pred_logits` - Predicted part logits [batch, n_parts, size]
    /// * `target_parts` - Target part indices [batch, size]
    pub fn part_loss(&self, pred_logits: &Tensor, target_parts: &Tensor) -> Result<Tensor> {
        // Cross-entropy loss
        let log_probs = log_softmax(pred_logits, 1)?;

        // Gather log probs at target indices
        let (batch, n_parts, size) = log_probs.dims3()?;

        // One-hot encode targets
        let target_one_hot = one_hot(target_parts, n_parts)?;

        // Compute cross-entropy
        let ce = (&log_probs * &target_one_hot)?.sum(1)?;
        let ce = ce.neg()?;

        ce.mean_all()
    }

    /// Compute total distillation loss
    pub fn total_loss(&self, losses: &LossComponents) -> Result<Tensor> {
        let total = (&losses.distill * self.config.lambda_distill)?;
        let total = (total + &losses.keypoint * self.config.lambda_keypoint)?;
        let total = (total + &losses.uv * self.config.lambda_uv)?;
        let total = (total + &losses.part * self.config.lambda_part)?;

        Ok(total)
    }

    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }
}

/// Individual loss components for logging
pub struct LossComponents {
    pub distill: Tensor,
    pub keypoint: Tensor,
    pub uv: Tensor,
    pub part: Tensor,
    pub total: Tensor,
}

/// Smooth L1 loss (Huber loss)
fn smooth_l1_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let diff = (pred - target)?;
    let abs_diff = diff.abs()?;

    // Smooth L1: 0.5*x² if |x| < 1, else |x| - 0.5
    let ones = Tensor::ones_like(&abs_diff)?;
    let half = Tensor::new(0.5f32, abs_diff.device())?;

    let squared_loss = (&diff * &diff)? * &half;
    let linear_loss = (&abs_diff - &half)?;

    let mask = abs_diff.lt(&ones)?;
    let loss = mask.where_cond(&squared_loss, &linear_loss)?;

    Ok(loss)
}

/// One-hot encoding
fn one_hot(indices: &Tensor, n_classes: usize) -> Result<Tensor> {
    let (batch, size) = indices.dims2()?;
    let device = indices.device();

    let mut one_hot_data = vec![0.0f32; batch * n_classes * size];

    let indices_data: Vec<u32> = indices.to_vec2::<u32>()?.into_iter().flatten().collect();

    for b in 0..batch {
        for s in 0..size {
            let idx = indices_data[b * size + s] as usize;
            if idx < n_classes {
                one_hot_data[b * n_classes * size + idx * size + s] = 1.0;
            }
        }
    }

    Tensor::from_vec(one_hot_data, (batch, n_classes, size), device)
}

/// Teacher model wrapper for inference-only usage
pub struct TeacherModel {
    /// Whether to detach gradients (always true for teacher)
    detach: bool,
}

impl TeacherModel {
    pub fn new() -> Self {
        Self { detach: true }
    }

    /// Process teacher outputs (detach gradients)
    pub fn process_output(&self, tensor: &Tensor) -> Result<Tensor> {
        if self.detach {
            tensor.detach()
        } else {
            Ok(tensor.clone())
        }
    }
}

impl Default for TeacherModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Training step result
#[derive(Debug, Clone)]
pub struct TrainingStep {
    pub loss: f32,
    pub distill_loss: f32,
    pub keypoint_loss: f32,
    pub uv_loss: f32,
    pub part_loss: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kl_divergence() -> Result<()> {
        let device = Device::Cpu;
        let distill = DistillationLoss::new(DistillationConfig::default());

        // Create test logits
        let teacher = Tensor::randn(0f32, 1.0, (2, 10, 16), &device)?;
        let student = Tensor::randn(0f32, 1.0, (2, 10, 16), &device)?;

        let kl = distill.kl_divergence(&teacher, &student)?;

        // KL divergence should be non-negative
        let kl_val: f32 = kl.to_scalar()?;
        assert!(kl_val >= 0.0);

        Ok(())
    }

    #[test]
    fn test_smooth_l1() -> Result<()> {
        let device = Device::Cpu;

        let pred = Tensor::new(&[[0.0f32, 0.5, 2.0]], &device)?;
        let target = Tensor::new(&[[0.0f32, 0.0, 0.0]], &device)?;

        let loss = smooth_l1_loss(&pred, &target)?;
        let loss_vals: Vec<f32> = loss.to_vec2::<f32>()?.into_iter().flatten().collect();

        // At x=0: loss=0
        assert!((loss_vals[0]).abs() < 0.01);

        // At x=0.5: loss=0.5*0.25=0.125 (quadratic region)
        assert!((loss_vals[1] - 0.125).abs() < 0.01);

        // At x=2: loss=2-0.5=1.5 (linear region)
        assert!((loss_vals[2] - 1.5).abs() < 0.01);

        Ok(())
    }
}
