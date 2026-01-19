//! ResNet-50 backbone adapted for 1D CSI input.
//!
//! The backbone extracts hierarchical features from the encoded CSI signal,
//! providing multi-scale representations for the pose estimation heads.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    batch_norm, conv1d, linear, BatchNorm, Conv1d, Conv1dConfig, Linear, VarBuilder,
};

/// ResNet block with skip connection
struct ResNetBlock {
    conv1: Conv1d,
    bn1: BatchNorm,
    conv2: Conv1d,
    bn2: BatchNorm,
    conv3: Conv1d,
    bn3: BatchNorm,
    downsample: Option<(Conv1d, BatchNorm)>,
    stride: usize,
}

impl ResNetBlock {
    fn new(
        in_channels: usize,
        mid_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // 1x1 conv
        let conv1 = conv1d(
            in_channels,
            mid_channels,
            1,
            Conv1dConfig::default(),
            vb.pp("conv1"),
        )?;
        let bn1 = batch_norm(mid_channels, 1e-5, vb.pp("bn1"))?;

        // 3x3 conv with stride
        let conv2_config = Conv1dConfig {
            padding: 1,
            stride,
            ..Default::default()
        };
        let conv2 = conv1d(mid_channels, mid_channels, 3, conv2_config, vb.pp("conv2"))?;
        let bn2 = batch_norm(mid_channels, 1e-5, vb.pp("bn2"))?;

        // 1x1 conv
        let conv3 = conv1d(
            mid_channels,
            out_channels,
            1,
            Conv1dConfig::default(),
            vb.pp("conv3"),
        )?;
        let bn3 = batch_norm(out_channels, 1e-5, vb.pp("bn3"))?;

        // Downsample path for skip connection
        let downsample = if downsample || in_channels != out_channels {
            let ds_config = Conv1dConfig {
                stride,
                ..Default::default()
            };
            let ds_conv = conv1d(in_channels, out_channels, 1, ds_config, vb.pp("ds_conv"))?;
            let ds_bn = batch_norm(out_channels, 1e-5, vb.pp("ds_bn"))?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
            stride,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = match &self.downsample {
            Some((conv, bn)) => {
                let x = conv.forward(x)?;
                bn.forward_train(&x)?
            }
            None => x.clone(),
        };

        let out = self.conv1.forward(x)?;
        let out = self.bn1.forward_train(&out)?;
        let out = out.relu()?;

        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward_train(&out)?;
        let out = out.relu()?;

        let out = self.conv3.forward(&out)?;
        let out = self.bn3.forward_train(&out)?;

        let out = (out + identity)?;
        out.relu()
    }
}

/// ResNet-50 backbone configuration
#[derive(Debug, Clone)]
pub struct BackboneConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Base channel width
    pub base_channels: usize,
    /// Block counts for each stage [3, 4, 6, 3] for ResNet-50
    pub block_counts: [usize; 4],
    /// Output feature dimension
    pub output_dim: usize,
}

impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            input_dim: 512,
            base_channels: 64,
            block_counts: [3, 4, 6, 3],
            output_dim: 2048,
        }
    }
}

/// ResNet-50 backbone for feature extraction
pub struct ResNetBackbone {
    stem_conv: Conv1d,
    stem_bn: BatchNorm,
    stage1: Vec<ResNetBlock>,
    stage2: Vec<ResNetBlock>,
    stage3: Vec<ResNetBlock>,
    stage4: Vec<ResNetBlock>,
    config: BackboneConfig,
}

impl ResNetBackbone {
    pub fn new(config: BackboneConfig, vb: VarBuilder) -> Result<Self> {
        let c = config.base_channels;

        // Stem: input_dim -> base_channels
        let stem_config = Conv1dConfig {
            padding: 3,
            stride: 2,
            ..Default::default()
        };
        let stem_conv = conv1d(config.input_dim, c, 7, stem_config, vb.pp("stem_conv"))?;
        let stem_bn = batch_norm(c, 1e-5, vb.pp("stem_bn"))?;

        // Stage 1: 64 -> 256
        let stage1 = Self::make_stage(c, c, c * 4, config.block_counts[0], 1, vb.pp("stage1"))?;

        // Stage 2: 256 -> 512
        let stage2 =
            Self::make_stage(c * 4, c * 2, c * 8, config.block_counts[1], 2, vb.pp("stage2"))?;

        // Stage 3: 512 -> 1024
        let stage3 =
            Self::make_stage(c * 8, c * 4, c * 16, config.block_counts[2], 2, vb.pp("stage3"))?;

        // Stage 4: 1024 -> 2048
        let stage4 =
            Self::make_stage(c * 16, c * 8, c * 32, config.block_counts[3], 2, vb.pp("stage4"))?;

        Ok(Self {
            stem_conv,
            stem_bn,
            stage1,
            stage2,
            stage3,
            stage4,
            config,
        })
    }

    fn make_stage(
        in_channels: usize,
        mid_channels: usize,
        out_channels: usize,
        n_blocks: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Vec<ResNetBlock>> {
        let mut blocks = Vec::with_capacity(n_blocks);

        // First block may have stride and always has downsample
        blocks.push(ResNetBlock::new(
            in_channels,
            mid_channels,
            out_channels,
            stride,
            true,
            vb.pp("block_0"),
        )?);

        // Remaining blocks
        for i in 1..n_blocks {
            blocks.push(ResNetBlock::new(
                out_channels,
                mid_channels,
                out_channels,
                1,
                false,
                vb.pp(format!("block_{}", i)),
            )?);
        }

        Ok(blocks)
    }

    /// Forward pass returning multi-scale features
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, input_dim, seq_len]
    ///
    /// # Returns
    /// Tuple of (final_features, intermediate_features) for skip connections
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let mut intermediates = Vec::new();

        // Stem
        let mut x = self.stem_conv.forward(x)?;
        x = self.stem_bn.forward_train(&x)?;
        x = x.relu()?;

        // Stage 1
        for block in &self.stage1 {
            x = block.forward(&x)?;
        }
        intermediates.push(x.clone());

        // Stage 2
        for block in &self.stage2 {
            x = block.forward(&x)?;
        }
        intermediates.push(x.clone());

        // Stage 3
        for block in &self.stage3 {
            x = block.forward(&x)?;
        }
        intermediates.push(x.clone());

        // Stage 4
        for block in &self.stage4 {
            x = block.forward(&x)?;
        }

        Ok((x, intermediates))
    }

    /// Forward pass returning only final features
    pub fn forward_final(&self, x: &Tensor) -> Result<Tensor> {
        let (final_features, _) = self.forward(x)?;
        Ok(final_features)
    }

    pub fn config(&self) -> &BackboneConfig {
        &self.config
    }
}

/// Feature Pyramid Network for multi-scale feature fusion
pub struct FeaturePyramid {
    lateral_convs: Vec<Conv1d>,
    output_convs: Vec<Conv1d>,
    top_down_dim: usize,
}

impl FeaturePyramid {
    pub fn new(
        in_channels: &[usize],
        out_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut lateral_convs = Vec::new();
        let mut output_convs = Vec::new();

        for (i, &c) in in_channels.iter().enumerate() {
            // Lateral connection (1x1 conv to reduce channels)
            let lateral = conv1d(
                c,
                out_channels,
                1,
                Conv1dConfig::default(),
                vb.pp(format!("lateral_{}", i)),
            )?;
            lateral_convs.push(lateral);

            // Output conv (3x3 to smooth upsampled features)
            let output = conv1d(
                out_channels,
                out_channels,
                3,
                Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp(format!("output_{}", i)),
            )?;
            output_convs.push(output);
        }

        Ok(Self {
            lateral_convs,
            output_convs,
            top_down_dim: out_channels,
        })
    }

    /// Build feature pyramid from backbone outputs
    ///
    /// # Arguments
    /// * `features` - List of features from backbone stages (low to high resolution)
    ///
    /// # Returns
    /// List of pyramid features (all at `out_channels` dimension)
    pub fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        assert_eq!(features.len(), self.lateral_convs.len());

        let n = features.len();
        let mut pyramid = vec![Tensor::zeros((), DType::F32, features[0].device())?; n];

        // Start from highest level (smallest spatial size)
        let mut prev_features = self.lateral_convs[n - 1].forward(&features[n - 1])?;
        pyramid[n - 1] = self.output_convs[n - 1].forward(&prev_features)?;

        // Top-down pathway
        for i in (0..n - 1).rev() {
            let lateral = self.lateral_convs[i].forward(&features[i])?;

            // Upsample previous features
            let (_batch, _channels, prev_len) = prev_features.dims3()?;
            let (_, _, curr_len) = lateral.dims3()?;

            // Simple nearest-neighbor upsampling
            let upsampled = if curr_len > prev_len {
                prev_features.upsample_nearest1d(curr_len)?
            } else {
                prev_features.clone()
            };

            // Add lateral connection
            prev_features = (lateral + upsampled)?;
            pyramid[i] = self.output_convs[i].forward(&prev_features)?;
        }

        Ok(pyramid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_backbone_shapes() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = BackboneConfig {
            input_dim: 64,
            base_channels: 16,
            block_counts: [1, 1, 1, 1],
            output_dim: 512,
        };

        let backbone = ResNetBackbone::new(config, vb)?;

        let x = Tensor::zeros((2, 64, 128), DType::F32, &device)?;
        let (out, intermediates) = backbone.forward(&x)?;

        assert_eq!(intermediates.len(), 3);
        assert_eq!(out.dims()[0], 2); // Batch size preserved

        Ok(())
    }
}
