//! Inference engine for real-time WiFi-to-DensePose translation.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use parking_lot::RwLock;
use tokio::sync::mpsc;

use psycho_core::{DensePoseFrame, SubjectId, Timestamp};
use psycho_csi::pipeline::ProcessedCsi;

use crate::model::{DecodedPredictions, WiFiDensePoseConfig, WiFiDensePoseModel};

/// Inference engine configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Model configuration
    pub model: WiFiDensePoseConfig,
    /// Device to run inference on
    pub device: DeviceType,
    /// Batch size for inference
    pub batch_size: usize,
    /// Number of CSI frames to accumulate before inference
    pub accumulation_frames: usize,
    /// Minimum confidence threshold
    pub confidence_threshold: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum DeviceType {
    Cpu,
    Cuda(usize),
    Metal,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model: WiFiDensePoseConfig::default(),
            device: DeviceType::Cpu,
            batch_size: 1,
            accumulation_frames: 64,
            confidence_threshold: 0.3,
        }
    }
}

/// Real-time inference engine
pub struct InferenceEngine {
    model: WiFiDensePoseModel,
    device: Device,
    config: InferenceConfig,
    csi_buffer: RwLock<Vec<ProcessedCsi>>,
}

impl InferenceEngine {
    /// Create a new inference engine with random weights (for testing)
    pub fn new_random(config: InferenceConfig) -> Result<Self> {
        let device = Self::get_device(config.device)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = WiFiDensePoseModel::new(config.model.clone(), vb)?;

        Ok(Self {
            model,
            device,
            config,
            csi_buffer: RwLock::new(Vec::new()),
        })
    }

    /// Load model from checkpoint
    pub fn load<P: AsRef<Path>>(path: P, config: InferenceConfig) -> Result<Self> {
        let device = Self::get_device(config.device)?;

        // Load weights from safetensors file
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.as_ref()], DType::F32, &device)?
        };

        let model = WiFiDensePoseModel::new(config.model.clone(), vb)?;

        Ok(Self {
            model,
            device,
            config,
            csi_buffer: RwLock::new(Vec::new()),
        })
    }

    fn get_device(device_type: DeviceType) -> Result<Device> {
        match device_type {
            DeviceType::Cpu => Ok(Device::Cpu),
            DeviceType::Cuda(ordinal) => Device::new_cuda(ordinal),
            DeviceType::Metal => Device::new_metal(0),
        }
    }

    /// Process a single CSI frame, buffering until ready for inference
    pub fn process_csi(&self, csi: ProcessedCsi) -> Option<InferenceResult> {
        let mut buffer = self.csi_buffer.write();
        buffer.push(csi);

        if buffer.len() >= self.config.accumulation_frames {
            let frames: Vec<_> = buffer.drain(..).collect();
            drop(buffer);

            self.run_inference(&frames).ok()
        } else {
            None
        }
    }

    /// Run inference on accumulated CSI frames
    pub fn run_inference(&self, csi_frames: &[ProcessedCsi]) -> Result<InferenceResult> {
        if csi_frames.is_empty() {
            return Ok(InferenceResult::empty());
        }

        let timestamp = csi_frames.last().unwrap().timestamp;

        // Prepare input tensors
        let (amplitude, phase) = self.prepare_input(csi_frames)?;

        // Run model
        let predictions = self.model.predict(&amplitude, &phase)?;

        // Filter by confidence
        let predictions = self.filter_predictions(predictions);

        // Convert to frames
        let frames = self.predictions_to_frames(&predictions, timestamp);

        Ok(InferenceResult {
            timestamp,
            predictions,
            frames,
            latency_ms: 0.0, // TODO: measure actual latency
        })
    }

    /// Prepare input tensors from CSI frames
    fn prepare_input(&self, csi_frames: &[ProcessedCsi]) -> Result<(Tensor, Tensor)> {
        let n_frames = csi_frames.len();
        let n_subcarriers = csi_frames[0].frame.streams[0].amplitude.len();

        // Stack amplitude and phase data
        let mut amplitude_data = Vec::with_capacity(n_frames * n_subcarriers);
        let mut phase_data = Vec::with_capacity(n_frames * n_subcarriers);

        for frame in csi_frames {
            if let Some(stream) = frame.frame.streams.first() {
                amplitude_data.extend(&stream.amplitude);
                phase_data.extend(&stream.phase);
            }
        }

        // Convert to f32
        let amplitude_f32: Vec<f32> = amplitude_data.iter().map(|&x| x as f32).collect();
        let phase_f32: Vec<f32> = phase_data.iter().map(|&x| x as f32).collect();

        // Create tensors [batch=1, timesteps, subcarriers]
        let amplitude = Tensor::from_vec(
            amplitude_f32,
            (1, n_frames, n_subcarriers),
            &self.device,
        )?;

        let phase = Tensor::from_vec(
            phase_f32,
            (1, n_frames, n_subcarriers),
            &self.device,
        )?;

        Ok((amplitude, phase))
    }

    /// Filter predictions by confidence threshold
    fn filter_predictions(&self, mut predictions: DecodedPredictions) -> DecodedPredictions {
        // Filter keypoints
        for batch in &mut predictions.keypoints.predictions {
            batch.retain(|kp| kp.confidence >= self.config.confidence_threshold);
        }

        // Filter DensePose points
        for batch in &mut predictions.densepose.predictions {
            batch.retain(|p| p.confidence >= self.config.confidence_threshold);
        }

        predictions
    }

    /// Convert predictions to DensePoseFrame format
    fn predictions_to_frames(
        &self,
        predictions: &DecodedPredictions,
        timestamp: Timestamp,
    ) -> Vec<DensePoseFrame> {
        // For now, create one frame per detected ROI
        predictions
            .rois
            .iter()
            .enumerate()
            .map(|(i, _roi)| {
                let subject_id = SubjectId::new(); // TODO: track subjects across frames
                predictions.to_frame(subject_id, timestamp)
            })
            .collect()
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.csi_buffer.read().len()
    }

    /// Clear the CSI buffer
    pub fn clear_buffer(&self) {
        self.csi_buffer.write().clear();
    }

    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

/// Result from inference
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub timestamp: Timestamp,
    pub predictions: DecodedPredictions,
    pub frames: Vec<DensePoseFrame>,
    pub latency_ms: f64,
}

impl InferenceResult {
    pub fn empty() -> Self {
        Self {
            timestamp: Timestamp::now(),
            predictions: DecodedPredictions {
                keypoints: crate::heads::KeypointPredictions { predictions: Vec::new() },
                densepose: crate::heads::DensePosePredictions { predictions: Vec::new() },
                rois: Vec::new(),
            },
            frames: Vec::new(),
            latency_ms: 0.0,
        }
    }

    pub fn has_detections(&self) -> bool {
        !self.frames.is_empty()
    }

    pub fn n_subjects(&self) -> usize {
        self.frames.len()
    }
}

/// Async inference service
pub struct InferenceService {
    engine: Arc<InferenceEngine>,
    input_tx: mpsc::Sender<ProcessedCsi>,
    output_rx: mpsc::Receiver<InferenceResult>,
}

impl InferenceService {
    /// Create and start the inference service
    pub async fn start(config: InferenceConfig) -> Result<Self> {
        let engine = Arc::new(InferenceEngine::new_random(config)?);
        let (input_tx, mut input_rx) = mpsc::channel::<ProcessedCsi>(1000);
        let (output_tx, output_rx) = mpsc::channel::<InferenceResult>(100);

        let engine_clone = engine.clone();

        // Spawn inference task
        tokio::spawn(async move {
            while let Some(csi) = input_rx.recv().await {
                if let Some(result) = engine_clone.process_csi(csi) {
                    if output_tx.send(result).await.is_err() {
                        break;
                    }
                }
            }
        });

        Ok(Self {
            engine,
            input_tx,
            output_rx,
        })
    }

    /// Submit CSI data for processing
    pub async fn submit(&self, csi: ProcessedCsi) -> std::result::Result<(), mpsc::error::SendError<ProcessedCsi>> {
        self.input_tx.send(csi).await
    }

    /// Receive next inference result
    pub async fn recv(&mut self) -> Option<InferenceResult> {
        self.output_rx.recv().await
    }

    /// Get reference to underlying engine
    pub fn engine(&self) -> &InferenceEngine {
        &self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_csi::packet::{CsiFrame, SanitizedCsi};
    use psycho_core::{AntennaConfig, ChannelBandwidth, FrequencyBand};

    fn create_test_csi() -> ProcessedCsi {
        let timestamp = Timestamp::now();

        let sanitized = SanitizedCsi::new(
            timestamp,
            0,
            vec![1.0; 64],
            vec![0.0; 64],
            0.9,
        );

        let mut frame = CsiFrame::new(
            timestamp,
            AntennaConfig::new(2, 2),
            ChannelBandwidth::Bw20MHz,
            FrequencyBand::Band5GHz,
        );
        frame.add_stream(sanitized);

        ProcessedCsi {
            timestamp,
            frame,
            doppler: psycho_csi::DopplerSpectrogram::empty(),
            velocity: psycho_csi::VelocityProfile {
                velocities: vec![0.0; 64],
                powers: vec![1.0; 64],
                mean_velocity: 0.0,
                max_velocity: 0.0,
            },
            features: psycho_csi::CsiFeatures::default(),
        }
    }

    #[test]
    fn test_engine_creation() -> Result<()> {
        let config = InferenceConfig {
            model: WiFiDensePoseConfig {
                encoder: crate::encoder::EncoderConfig {
                    n_subcarriers: 64,
                    n_timesteps: 8,
                    hidden_dim: 16,
                    output_dim: 32,
                    n_conv_layers: 1,
                    dropout: 0.0,
                },
                backbone: crate::backbone::BackboneConfig {
                    input_dim: 32,
                    base_channels: 8,
                    block_counts: [1, 1, 1, 1],
                    output_dim: 64,
                },
                keypoint_head: crate::heads::KeypointHeadConfig {
                    input_dim: 64,
                    hidden_dim: 16,
                    n_keypoints: 17,
                    heatmap_size: 8,
                },
                densepose_head: crate::heads::DensePoseHeadConfig {
                    input_dim: 64,
                    hidden_dim: 16,
                    n_parts: 25,
                    uv_resolution: 14,
                },
                n_transformer_layers: 1,
                n_attention_heads: 2,
                ..Default::default()
            },
            accumulation_frames: 8,
            ..Default::default()
        };

        let _engine = InferenceEngine::new_random(config)?;
        Ok(())
    }
}
