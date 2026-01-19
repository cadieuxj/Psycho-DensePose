//! Output heads for keypoint and DensePose prediction.
//!
//! Two parallel heads process backbone features:
//! 1. Keypoint Head: Predicts 17 skeletal joint locations
//! 2. DensePose Head: Predicts 24 body parts + UV coordinates

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, VarBuilder};

use psycho_core::{BodyPart, Keypoint};

/// Configuration for the keypoint detection head
#[derive(Debug, Clone)]
pub struct KeypointHeadConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of keypoints (17 for COCO format)
    pub n_keypoints: usize,
    /// Heatmap resolution
    pub heatmap_size: usize,
}

impl Default for KeypointHeadConfig {
    fn default() -> Self {
        Self {
            input_dim: 2048,
            hidden_dim: 256,
            n_keypoints: Keypoint::COUNT,
            heatmap_size: 64,
        }
    }
}

/// Keypoint detection head producing heatmaps for 17 joints
pub struct KeypointHead {
    deconv1: Conv1d,
    deconv2: Conv1d,
    deconv3: Conv1d,
    heatmap_conv: Conv1d,
    config: KeypointHeadConfig,
}

impl KeypointHead {
    pub fn new(config: KeypointHeadConfig, vb: VarBuilder) -> Result<Self> {
        // Deconvolutional layers for upsampling
        let deconv1 = conv1d(
            config.input_dim,
            config.hidden_dim,
            4,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("deconv1"),
        )?;

        let deconv2 = conv1d(
            config.hidden_dim,
            config.hidden_dim,
            4,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("deconv2"),
        )?;

        let deconv3 = conv1d(
            config.hidden_dim,
            config.hidden_dim,
            4,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("deconv3"),
        )?;

        // Final heatmap prediction
        let heatmap_conv = conv1d(
            config.hidden_dim,
            config.n_keypoints,
            1,
            Conv1dConfig::default(),
            vb.pp("heatmap"),
        )?;

        Ok(Self {
            deconv1,
            deconv2,
            deconv3,
            heatmap_conv,
            config,
        })
    }

    /// Forward pass producing keypoint heatmaps
    ///
    /// # Arguments
    /// * `features` - Backbone features [batch, channels, seq_len]
    ///
    /// # Returns
    /// Heatmaps tensor [batch, n_keypoints, heatmap_size]
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        // Upsample through deconv layers
        let x = self.deconv1.forward(features)?;
        let x = x.relu()?;
        let x = x.upsample_nearest1d(x.dim(2)? * 2)?;

        let x = self.deconv2.forward(&x)?;
        let x = x.relu()?;
        let x = x.upsample_nearest1d(x.dim(2)? * 2)?;

        let x = self.deconv3.forward(&x)?;
        let x = x.relu()?;
        let x = x.upsample_nearest1d(self.config.heatmap_size)?;

        // Generate heatmaps
        self.heatmap_conv.forward(&x)
    }

    /// Extract keypoint coordinates from heatmaps
    pub fn decode_keypoints(&self, heatmaps: &Tensor) -> Result<KeypointPredictions> {
        let (batch, n_kp, heatmap_len) = heatmaps.dims3()?;

        let mut predictions = Vec::with_capacity(batch);

        for b in 0..batch {
            let mut keypoints = Vec::with_capacity(n_kp);

            for k in 0..n_kp {
                let heatmap = heatmaps.i((b, k))?;
                let heatmap_data: Vec<f32> = heatmap.to_vec1()?;

                // Find argmax
                let (max_idx, &max_val) = heatmap_data
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                // Convert to normalized coordinate [0, 1]
                let position = max_idx as f32 / heatmap_len as f32;

                keypoints.push(KeypointPrediction {
                    keypoint: Keypoint::from_index(k as u8).unwrap_or(Keypoint::Nose),
                    position,
                    confidence: max_val.sigmoid(),
                });
            }

            predictions.push(keypoints);
        }

        Ok(KeypointPredictions { predictions })
    }

    pub fn config(&self) -> &KeypointHeadConfig {
        &self.config
    }
}

/// Single keypoint prediction
#[derive(Debug, Clone)]
pub struct KeypointPrediction {
    pub keypoint: Keypoint,
    /// Normalized position [0, 1]
    pub position: f32,
    /// Confidence score [0, 1]
    pub confidence: f32,
}

/// Batch of keypoint predictions
#[derive(Debug, Clone)]
pub struct KeypointPredictions {
    pub predictions: Vec<Vec<KeypointPrediction>>,
}

/// Configuration for the DensePose head
#[derive(Debug, Clone)]
pub struct DensePoseHeadConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of body parts (24 + background)
    pub n_parts: usize,
    /// UV output resolution
    pub uv_resolution: usize,
}

impl Default for DensePoseHeadConfig {
    fn default() -> Self {
        Self {
            input_dim: 2048,
            hidden_dim: 256,
            n_parts: 25, // 24 body parts + background
            uv_resolution: 112,
        }
    }
}

/// DensePose head for body part segmentation and UV regression
pub struct DensePoseHead {
    // Body part classification branch
    part_conv1: Conv1d,
    part_conv2: Conv1d,
    part_output: Conv1d,

    // UV coordinate regression branch
    uv_conv1: Conv1d,
    uv_conv2: Conv1d,
    u_output: Conv1d,
    v_output: Conv1d,

    config: DensePoseHeadConfig,
}

impl DensePoseHead {
    pub fn new(config: DensePoseHeadConfig, vb: VarBuilder) -> Result<Self> {
        // Body part classification branch
        let part_conv1 = conv1d(
            config.input_dim,
            config.hidden_dim,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("part_conv1"),
        )?;

        let part_conv2 = conv1d(
            config.hidden_dim,
            config.hidden_dim,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("part_conv2"),
        )?;

        let part_output = conv1d(
            config.hidden_dim,
            config.n_parts,
            1,
            Conv1dConfig::default(),
            vb.pp("part_output"),
        )?;

        // UV regression branch
        let uv_conv1 = conv1d(
            config.input_dim,
            config.hidden_dim,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("uv_conv1"),
        )?;

        let uv_conv2 = conv1d(
            config.hidden_dim,
            config.hidden_dim,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("uv_conv2"),
        )?;

        // Separate U and V outputs per body part
        let u_output = conv1d(
            config.hidden_dim,
            config.n_parts,
            1,
            Conv1dConfig::default(),
            vb.pp("u_output"),
        )?;

        let v_output = conv1d(
            config.hidden_dim,
            config.n_parts,
            1,
            Conv1dConfig::default(),
            vb.pp("v_output"),
        )?;

        Ok(Self {
            part_conv1,
            part_conv2,
            part_output,
            uv_conv1,
            uv_conv2,
            u_output,
            v_output,
            config,
        })
    }

    /// Forward pass producing body part logits and UV coordinates
    ///
    /// # Arguments
    /// * `features` - Backbone features [batch, channels, seq_len]
    ///
    /// # Returns
    /// Tuple of (part_logits, u_coords, v_coords)
    pub fn forward(&self, features: &Tensor) -> Result<DensePoseOutput> {
        // Body part classification
        let part_x = self.part_conv1.forward(features)?;
        let part_x = part_x.relu()?;
        let part_x = part_x.upsample_nearest1d(part_x.dim(2)? * 2)?;
        let part_x = self.part_conv2.forward(&part_x)?;
        let part_x = part_x.relu()?;
        let part_x = part_x.upsample_nearest1d(self.config.uv_resolution)?;
        let part_logits = self.part_output.forward(&part_x)?;

        // UV regression
        let uv_x = self.uv_conv1.forward(features)?;
        let uv_x = uv_x.relu()?;
        let uv_x = uv_x.upsample_nearest1d(uv_x.dim(2)? * 2)?;
        let uv_x = self.uv_conv2.forward(&uv_x)?;
        let uv_x = uv_x.relu()?;
        let uv_x = uv_x.upsample_nearest1d(self.config.uv_resolution)?;

        let u_coords = self.u_output.forward(&uv_x)?;
        let v_coords = self.v_output.forward(&uv_x)?;

        // Sigmoid for UV (bounded [0, 1])
        let u_coords = candle_nn::ops::sigmoid(&u_coords)?;
        let v_coords = candle_nn::ops::sigmoid(&v_coords)?;

        Ok(DensePoseOutput {
            part_logits,
            u_coords,
            v_coords,
        })
    }

    /// Decode DensePose output to predictions
    pub fn decode(&self, output: &DensePoseOutput) -> Result<DensePosePredictions> {
        let (batch, n_parts, resolution) = output.part_logits.dims3()?;

        let mut predictions = Vec::with_capacity(batch);

        for b in 0..batch {
            let part_probs = candle_nn::ops::softmax(&output.part_logits.i(b)?, 0)?;
            let part_data: Vec<Vec<f32>> = (0..n_parts)
                .map(|p| part_probs.i(p).unwrap().to_vec1().unwrap())
                .collect();

            let u_data: Vec<Vec<f32>> = (0..n_parts)
                .map(|p| output.u_coords.i((b, p)).unwrap().to_vec1().unwrap())
                .collect();

            let v_data: Vec<Vec<f32>> = (0..n_parts)
                .map(|p| output.v_coords.i((b, p)).unwrap().to_vec1().unwrap())
                .collect();

            let mut dense_points = Vec::new();

            for pos in 0..resolution {
                // Find most likely body part
                let mut max_prob = 0.0f32;
                let mut max_part = 0;

                for (p, probs) in part_data.iter().enumerate() {
                    if probs[pos] > max_prob {
                        max_prob = probs[pos];
                        max_part = p;
                    }
                }

                // Skip background
                if max_part == 0 || max_prob < 0.5 {
                    continue;
                }

                if let Some(body_part) = BodyPart::from_index(max_part as u8) {
                    dense_points.push(DensePointPrediction {
                        position: pos as f32 / resolution as f32,
                        body_part,
                        u: u_data[max_part][pos],
                        v: v_data[max_part][pos],
                        confidence: max_prob,
                    });
                }
            }

            predictions.push(dense_points);
        }

        Ok(DensePosePredictions { predictions })
    }

    pub fn config(&self) -> &DensePoseHeadConfig {
        &self.config
    }
}

/// Raw DensePose head output
#[derive(Debug)]
pub struct DensePoseOutput {
    /// Body part logits [batch, n_parts, resolution]
    pub part_logits: Tensor,
    /// U coordinates [batch, n_parts, resolution]
    pub u_coords: Tensor,
    /// V coordinates [batch, n_parts, resolution]
    pub v_coords: Tensor,
}

/// Single dense point prediction
#[derive(Debug, Clone)]
pub struct DensePointPrediction {
    /// Position along the CSI sequence
    pub position: f32,
    /// Predicted body part
    pub body_part: BodyPart,
    /// U coordinate [0, 1]
    pub u: f32,
    /// V coordinate [0, 1]
    pub v: f32,
    /// Confidence score
    pub confidence: f32,
}

/// Batch of DensePose predictions
#[derive(Debug, Clone)]
pub struct DensePosePredictions {
    pub predictions: Vec<Vec<DensePointPrediction>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_keypoint_head_shapes() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = KeypointHeadConfig {
            input_dim: 256,
            hidden_dim: 64,
            n_keypoints: 17,
            heatmap_size: 32,
        };

        let head = KeypointHead::new(config, vb)?;
        let features = Tensor::zeros((2, 256, 8), DType::F32, &device)?;
        let heatmaps = head.forward(&features)?;

        assert_eq!(heatmaps.dims(), &[2, 17, 32]);
        Ok(())
    }

    #[test]
    fn test_densepose_head_shapes() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = DensePoseHeadConfig {
            input_dim: 256,
            hidden_dim: 64,
            n_parts: 25,
            uv_resolution: 56,
        };

        let head = DensePoseHead::new(config, vb)?;
        let features = Tensor::zeros((2, 256, 8), DType::F32, &device)?;
        let output = head.forward(&features)?;

        assert_eq!(output.part_logits.dims(), &[2, 25, 56]);
        assert_eq!(output.u_coords.dims(), &[2, 25, 56]);
        assert_eq!(output.v_coords.dims(), &[2, 25, 56]);
        Ok(())
    }
}
