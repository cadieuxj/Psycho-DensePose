//! Complete WiFi-to-DensePose translation model.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use psycho_core::{DensePoseFrame, Keypoint, SkeletalPose, SubjectId, Timestamp};

use crate::backbone::{BackboneConfig, FeaturePyramid, ResNetBackbone};
use crate::encoder::{DualBranchEncoder, EncoderConfig, TransformerEncoderLayer};
use crate::heads::{
    DensePoseHead, DensePoseHeadConfig, DensePoseOutput, DensePosePredictions, KeypointHead,
    KeypointHeadConfig, KeypointPredictions,
};
use crate::roi::{nms_1d, Roi, RoiAlign1d, RoiAlignConfig, RoiProposer};

/// Complete model configuration
#[derive(Debug, Clone)]
pub struct WiFiDensePoseConfig {
    pub encoder: EncoderConfig,
    pub backbone: BackboneConfig,
    pub keypoint_head: KeypointHeadConfig,
    pub densepose_head: DensePoseHeadConfig,
    pub roi_align: RoiAlignConfig,
    /// Number of transformer layers
    pub n_transformer_layers: usize,
    /// Number of attention heads
    pub n_attention_heads: usize,
    /// NMS threshold for ROI filtering
    pub nms_threshold: f32,
}

impl Default for WiFiDensePoseConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            backbone: BackboneConfig::default(),
            keypoint_head: KeypointHeadConfig::default(),
            densepose_head: DensePoseHeadConfig::default(),
            roi_align: RoiAlignConfig::default(),
            n_transformer_layers: 4,
            n_attention_heads: 8,
            nms_threshold: 0.5,
        }
    }
}

/// WiFi-to-DensePose translation model
pub struct WiFiDensePoseModel {
    encoder: DualBranchEncoder,
    transformer_layers: Vec<TransformerEncoderLayer>,
    backbone: ResNetBackbone,
    fpn: FeaturePyramid,
    keypoint_head: KeypointHead,
    densepose_head: DensePoseHead,
    roi_align: RoiAlign1d,
    roi_proposer: RoiProposer,
    config: WiFiDensePoseConfig,
}

impl WiFiDensePoseModel {
    pub fn new(config: WiFiDensePoseConfig, vb: VarBuilder) -> Result<Self> {
        // Dual-branch encoder
        let encoder = DualBranchEncoder::new(config.encoder.clone(), vb.pp("encoder"))?;

        // Transformer layers
        let mut transformer_layers = Vec::new();
        for i in 0..config.n_transformer_layers {
            let layer = TransformerEncoderLayer::new(
                config.encoder.output_dim,
                config.n_attention_heads,
                config.encoder.output_dim * 4,
                vb.pp(format!("transformer_{}", i)),
            )?;
            transformer_layers.push(layer);
        }

        // ResNet backbone
        let backbone = ResNetBackbone::new(config.backbone.clone(), vb.pp("backbone"))?;

        // Feature Pyramid Network
        let fpn_channels = [256, 512, 1024]; // Intermediate channel sizes
        let fpn = FeaturePyramid::new(&fpn_channels, 256, vb.pp("fpn"))?;

        // Output heads
        let keypoint_head = KeypointHead::new(config.keypoint_head.clone(), vb.pp("keypoint"))?;
        let densepose_head = DensePoseHead::new(config.densepose_head.clone(), vb.pp("densepose"))?;

        // ROI components
        let roi_align = RoiAlign1d::new(config.roi_align.clone());
        let roi_proposer = RoiProposer::default();

        Ok(Self {
            encoder,
            transformer_layers,
            backbone,
            fpn,
            keypoint_head,
            densepose_head,
            roi_align,
            roi_proposer,
            config,
        })
    }

    /// Forward pass through the complete model
    ///
    /// # Arguments
    /// * `amplitude` - CSI amplitude tensor [batch, timesteps, subcarriers]
    /// * `phase` - CSI phase tensor [batch, timesteps, subcarriers]
    ///
    /// # Returns
    /// Model outputs including keypoints and DensePose predictions
    pub fn forward(&self, amplitude: &Tensor, phase: &Tensor) -> Result<ModelOutput> {
        // Encode CSI
        let encoded = self.encoder.forward(amplitude, phase)?;

        // Reshape for transformer [batch, 1, features]
        let encoded = encoded.unsqueeze(1)?;

        // Apply transformer layers
        let mut x = encoded;
        for layer in &self.transformer_layers {
            x = layer.forward(&x)?;
        }

        // Reshape for backbone [batch, features, 1]
        let x = x.transpose(1, 2)?;

        // Extract features through backbone
        let (backbone_features, intermediates) = self.backbone.forward(&x)?;

        // Build feature pyramid
        let pyramid_features = self.fpn.forward(&intermediates)?;

        // Generate ROI proposals from activity
        let activity = backbone_features.mean(1)?; // [batch, seq_len]
        let activity = candle_nn::ops::sigmoid(&activity)?;
        let rois = self.roi_proposer.propose(&activity)?;

        // Apply NMS
        let rois = nms_1d(&rois, self.config.nms_threshold);

        // ROI Align on FPN features
        let roi_features = if !pyramid_features.is_empty() && !rois.is_empty() {
            self.roi_align.forward(&pyramid_features[0], &rois)?
        } else {
            backbone_features.clone()
        };

        // Output heads
        let keypoint_heatmaps = self.keypoint_head.forward(&roi_features)?;
        let densepose_output = self.densepose_head.forward(&roi_features)?;

        Ok(ModelOutput {
            keypoint_heatmaps,
            densepose_output,
            rois,
            backbone_features,
        })
    }

    /// Decode model output to structured predictions
    pub fn decode(&self, output: &ModelOutput) -> Result<DecodedPredictions> {
        let keypoints = self.keypoint_head.decode_keypoints(&output.keypoint_heatmaps)?;
        let densepose = self.densepose_head.decode(&output.densepose_output)?;

        Ok(DecodedPredictions {
            keypoints,
            densepose,
            rois: output.rois.clone(),
        })
    }

    /// End-to-end inference from CSI to predictions
    pub fn predict(
        &self,
        amplitude: &Tensor,
        phase: &Tensor,
    ) -> Result<DecodedPredictions> {
        let output = self.forward(amplitude, phase)?;
        self.decode(&output)
    }

    pub fn config(&self) -> &WiFiDensePoseConfig {
        &self.config
    }
}

/// Raw model output tensors
pub struct ModelOutput {
    /// Keypoint heatmaps [batch, n_keypoints, heatmap_size]
    pub keypoint_heatmaps: Tensor,
    /// DensePose output (parts + UV)
    pub densepose_output: DensePoseOutput,
    /// Detected ROIs
    pub rois: Vec<Roi>,
    /// Backbone features (for visualization/debugging)
    pub backbone_features: Tensor,
}

/// Decoded predictions in structured format
#[derive(Debug, Clone)]
pub struct DecodedPredictions {
    pub keypoints: KeypointPredictions,
    pub densepose: DensePosePredictions,
    pub rois: Vec<Roi>,
}

impl DecodedPredictions {
    /// Convert to DensePoseFrame for downstream processing
    pub fn to_frame(&self, subject_id: SubjectId, timestamp: Timestamp) -> DensePoseFrame {
        use psycho_core::{
            BodyPart, DensePosePoint, KeypointDetection, Position3D, UvCoordinate,
        };

        // Convert keypoint predictions
        let mut keypoints = [None; Keypoint::COUNT];

        if let Some(batch_keypoints) = self.keypoints.predictions.first() {
            for kp in batch_keypoints {
                let idx = kp.keypoint as usize;
                if idx < Keypoint::COUNT {
                    keypoints[idx] = Some(KeypointDetection {
                        keypoint: kp.keypoint,
                        position: Position3D::new(kp.position as f64, 0.0, 0.0),
                        confidence: kp.confidence,
                    });
                }
            }
        }

        // Calculate overall confidence
        let valid_kps: Vec<_> = keypoints.iter().filter_map(|k| k.as_ref()).collect();
        let overall_confidence = if valid_kps.is_empty() {
            0.0
        } else {
            valid_kps.iter().map(|k| k.confidence).sum::<f32>() / valid_kps.len() as f32
        };

        let skeletal_pose = SkeletalPose {
            timestamp,
            subject_id,
            keypoints,
            overall_confidence,
        };

        // Convert DensePose predictions
        let surface_points: Vec<DensePosePoint> = self
            .densepose
            .predictions
            .first()
            .map(|pts| {
                pts.iter()
                    .map(|p| DensePosePoint {
                        body_part: p.body_part,
                        uv: UvCoordinate::new(p.u, p.v),
                        position_3d: Position3D::new(p.position as f64, 0.0, 0.0),
                        confidence: p.confidence,
                    })
                    .collect()
            })
            .unwrap_or_default();

        DensePoseFrame {
            timestamp,
            subject_id,
            skeletal_pose,
            surface_points,
            body_part_mask: Vec::new(),
            mask_width: 0,
            mask_height: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_model_creation() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Use smaller config for testing
        let config = WiFiDensePoseConfig {
            encoder: EncoderConfig {
                n_subcarriers: 64,
                n_timesteps: 16,
                hidden_dim: 32,
                output_dim: 64,
                n_conv_layers: 2,
                dropout: 0.0,
            },
            backbone: BackboneConfig {
                input_dim: 64,
                base_channels: 16,
                block_counts: [1, 1, 1, 1],
                output_dim: 256,
            },
            keypoint_head: KeypointHeadConfig {
                input_dim: 256,
                hidden_dim: 32,
                n_keypoints: 17,
                heatmap_size: 16,
            },
            densepose_head: DensePoseHeadConfig {
                input_dim: 256,
                hidden_dim: 32,
                n_parts: 25,
                uv_resolution: 28,
            },
            n_transformer_layers: 1,
            n_attention_heads: 4,
            ..Default::default()
        };

        let _model = WiFiDensePoseModel::new(config, vb)?;
        Ok(())
    }
}
