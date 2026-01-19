//! Dual-branch encoder for CSI amplitude and phase processing.
//!
//! The encoder processes the two components of CSI separately before fusion,
//! as they encode different aspects of the signal:
//! - Amplitude: Signal strength variations from body reflections
//! - Phase: Fine-grained motion information from Doppler shifts

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv1d, layer_norm, linear, Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};

/// Configuration for the dual-branch encoder
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Number of input subcarriers
    pub n_subcarriers: usize,
    /// Number of time steps in input window
    pub n_timesteps: usize,
    /// Hidden dimension for each branch
    pub hidden_dim: usize,
    /// Output feature dimension
    pub output_dim: usize,
    /// Number of convolutional layers per branch
    pub n_conv_layers: usize,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            n_subcarriers: 1992,
            n_timesteps: 64,
            hidden_dim: 256,
            output_dim: 512,
            n_conv_layers: 4,
            dropout: 0.1,
        }
    }
}

/// 1D Convolutional block with batch norm and ReLU
struct ConvBlock {
    conv: Conv1d,
    norm: LayerNorm,
}

impl ConvBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding: kernel_size / 2,
            stride: 1,
            dilation: 1,
            groups: 1,
        };

        let conv = conv1d(in_channels, out_channels, kernel_size, config, vb.pp("conv"))?;
        let norm = layer_norm(out_channels, 1e-5, vb.pp("norm"))?;

        Ok(Self { conv, norm })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        // Transpose for layer norm (expects last dim to be normalized)
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = x.transpose(1, 2)?;
        x.relu()
    }
}

/// Single branch of the dual-branch encoder (for amplitude or phase)
pub struct EncoderBranch {
    conv_layers: Vec<ConvBlock>,
    temporal_conv: Conv1d,
    output_proj: Linear,
    config: EncoderConfig,
}

impl EncoderBranch {
    pub fn new(config: EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let mut conv_layers = Vec::new();

        // First layer: subcarriers -> hidden_dim
        conv_layers.push(ConvBlock::new(
            config.n_timesteps,
            config.hidden_dim,
            7,
            vb.pp("conv_0"),
        )?);

        // Subsequent layers
        for i in 1..config.n_conv_layers {
            conv_layers.push(ConvBlock::new(
                config.hidden_dim,
                config.hidden_dim,
                5,
                vb.pp(format!("conv_{}", i)),
            )?);
        }

        // Temporal aggregation
        let temporal_config = Conv1dConfig {
            padding: 0,
            stride: 4,
            dilation: 1,
            groups: 1,
        };
        let temporal_conv = conv1d(
            config.hidden_dim,
            config.hidden_dim,
            8,
            temporal_config,
            vb.pp("temporal"),
        )?;

        // Output projection
        let output_proj = linear(config.hidden_dim, config.output_dim, vb.pp("output"))?;

        Ok(Self {
            conv_layers,
            temporal_conv,
            output_proj,
            config,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Input shape: [batch, timesteps, subcarriers]
        // Transpose to [batch, timesteps, subcarriers] for conv1d
        let mut x = x.clone();

        // Apply convolutional layers
        for conv in &self.conv_layers {
            x = conv.forward(&x)?;
        }

        // Temporal aggregation
        x = self.temporal_conv.forward(&x)?;

        // Global average pooling over remaining spatial dimension
        let x = x.mean(2)?;

        // Output projection
        self.output_proj.forward(&x)
    }
}

/// Dual-branch encoder combining amplitude and phase streams
pub struct DualBranchEncoder {
    amplitude_branch: EncoderBranch,
    phase_branch: EncoderBranch,
    fusion: Linear,
    config: EncoderConfig,
}

impl DualBranchEncoder {
    pub fn new(config: EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let amplitude_branch =
            EncoderBranch::new(config.clone(), vb.pp("amplitude"))?;
        let phase_branch = EncoderBranch::new(config.clone(), vb.pp("phase"))?;

        // Fusion layer: concatenated features -> output
        let fusion = linear(config.output_dim * 2, config.output_dim, vb.pp("fusion"))?;

        Ok(Self {
            amplitude_branch,
            phase_branch,
            fusion,
            config,
        })
    }

    /// Forward pass through dual-branch encoder
    ///
    /// # Arguments
    /// * `amplitude` - Tensor of shape [batch, timesteps, subcarriers]
    /// * `phase` - Tensor of shape [batch, timesteps, subcarriers]
    ///
    /// # Returns
    /// Fused features of shape [batch, output_dim]
    pub fn forward(&self, amplitude: &Tensor, phase: &Tensor) -> Result<Tensor> {
        let amp_features = self.amplitude_branch.forward(amplitude)?;
        let phase_features = self.phase_branch.forward(phase)?;

        // Concatenate and fuse
        let concat = Tensor::cat(&[&amp_features, &phase_features], 1)?;
        let fused = self.fusion.forward(&concat)?;
        fused.relu()
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

/// Temporal attention module for isolating human Doppler signatures
pub struct TemporalAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl TemporalAttention {
    pub fn new(dim: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        assert!(dim % n_heads == 0);
        let head_dim = dim / n_heads;

        let query = linear(dim, dim, vb.pp("query"))?;
        let key = linear(dim, dim, vb.pp("key"))?;
        let value = linear(dim, dim, vb.pp("value"))?;
        let output = linear(dim, dim, vb.pp("output"))?;

        Ok(Self {
            query,
            key,
            value,
            output,
            n_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;

        // Project to Q, K, V
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for multi-head attention
        let q = q.reshape((batch, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.n_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.n_heads, self.head_dim))?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores / scale)?;
        let attn = candle_nn::ops::softmax(&scores, 3)?;

        // Apply attention to values
        let out = attn.matmul(&v)?;

        // Reshape back
        let out = out.transpose(1, 2)?;
        let out = out.reshape((batch, seq_len, dim))?;

        self.output.forward(&out)
    }
}

/// Transformer encoder layer for CSI features
pub struct TransformerEncoderLayer {
    attention: TemporalAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
}

impl TransformerEncoderLayer {
    pub fn new(dim: usize, n_heads: usize, ff_dim: usize, vb: VarBuilder) -> Result<Self> {
        let attention = TemporalAttention::new(dim, n_heads, vb.pp("attn"))?;
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let ffn1 = linear(dim, ff_dim, vb.pp("ffn1"))?;
        let ffn2 = linear(ff_dim, dim, vb.pp("ffn2"))?;

        Ok(Self {
            attention,
            norm1,
            norm2,
            ffn1,
            ffn2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self-attention with residual
        let attn_out = self.attention.forward(x)?;
        let x = (x + attn_out)?;
        let x = self.norm1.forward(&x)?;

        // FFN with residual
        let ffn_out = self.ffn1.forward(&x)?;
        let ffn_out = ffn_out.gelu()?;
        let ffn_out = self.ffn2.forward(&ffn_out)?;
        let x = (&x + ffn_out)?;

        self.norm2.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_encoder_config() {
        let config = EncoderConfig::default();
        assert_eq!(config.n_subcarriers, 1992);
        assert_eq!(config.hidden_dim, 256);
    }

    #[test]
    fn test_temporal_attention_shapes() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let attn = TemporalAttention::new(256, 8, vb)?;

        let x = Tensor::zeros((2, 16, 256), DType::F32, &device)?;
        let out = attn.forward(&x)?;

        assert_eq!(out.dims(), &[2, 16, 256]);
        Ok(())
    }
}
