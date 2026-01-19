//! Complete CSI processing pipeline.
//!
//! Integrates acquisition, sanitization, Doppler extraction, and feature computation
//! into a streaming processing pipeline.

use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};

use psycho_core::{Error, Result, Timestamp};

use crate::acquisition::CsiAcquisition;
use crate::doppler::{DopplerProcessor, DopplerSpectrogram, VelocityProfile};
use crate::filtering::{ButterworthFilter, KalmanFilter1D};
use crate::packet::{CsiFrame, CsiHardwareConfig, CsiPacket, SanitizedCsi};
use crate::sanitizer::{BatchSanitizer, CsiSanitizer};

/// Configuration for the CSI processing pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Hardware configuration
    pub hardware: CsiHardwareConfig,

    /// Sanitization parameters
    pub hampel_window: usize,
    pub hampel_threshold: f64,
    pub min_snr_db: f64,

    /// Doppler processing parameters
    pub doppler_fft_size: usize,

    /// Filtering parameters
    pub lowpass_cutoff_hz: f64,

    /// Buffering
    pub temporal_buffer_size: usize,
    pub doppler_window_size: usize,

    /// Output queue size
    pub output_queue_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            hardware: CsiHardwareConfig::default(),
            hampel_window: 5,
            hampel_threshold: 3.0,
            min_snr_db: 10.0,
            doppler_fft_size: 64,
            lowpass_cutoff_hz: 50.0,
            temporal_buffer_size: 5,
            doppler_window_size: 64,
            output_queue_size: 1000,
        }
    }
}

/// Processed CSI output containing all extracted features
#[derive(Debug, Clone)]
pub struct ProcessedCsi {
    pub timestamp: Timestamp,
    pub frame: CsiFrame,
    pub doppler: DopplerSpectrogram,
    pub velocity: VelocityProfile,
    pub features: CsiFeatures,
}

/// Statistical features extracted from CSI
#[derive(Debug, Clone, Default)]
pub struct CsiFeatures {
    /// Mean amplitude across subcarriers
    pub mean_amplitude: f64,
    /// Amplitude variance
    pub amplitude_variance: f64,
    /// Phase variance (indicator of motion)
    pub phase_variance: f64,
    /// Dominant Doppler velocity (m/s)
    pub dominant_velocity: f64,
    /// Motion activity level (0-1)
    pub motion_activity: f64,
    /// Number of detected targets (estimated)
    pub target_count: usize,
    /// Signal quality score
    pub quality_score: f64,
}

/// The main CSI processing pipeline
pub struct CsiPipeline {
    config: PipelineConfig,
    sanitizer: CsiSanitizer,
    doppler: DopplerProcessor,
    lowpass: ButterworthFilter,
    csi_buffer: Vec<SanitizedCsi>,
    is_running: Arc<RwLock<bool>>,
}

impl CsiPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let sanitizer = CsiSanitizer::new()
            .with_hampel(config.hampel_window, config.hampel_threshold);

        let wavelength = config.hardware.wavelength_m();
        let sample_rate = config.hardware.capture_rate as f64;

        let doppler = DopplerProcessor::new(config.doppler_fft_size, sample_rate, wavelength);

        let lowpass = ButterworthFilter::new(2, config.lowpass_cutoff_hz, sample_rate);

        Self {
            config,
            sanitizer,
            doppler,
            lowpass,
            csi_buffer: Vec::new(),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    /// Process a single CSI packet
    pub fn process_packet(&mut self, packet: &CsiPacket) -> Result<Option<ProcessedCsi>> {
        // Sanitize the packet
        let sanitized = self.sanitizer.sanitize(packet)?;

        // Add to buffer
        self.csi_buffer.push(sanitized);

        // Check if we have enough samples for Doppler processing
        if self.csi_buffer.len() < self.config.doppler_window_size {
            return Ok(None);
        }

        // Process the window
        let result = self.process_window()?;

        // Slide the window (keep half for overlap)
        let keep = self.config.doppler_window_size / 2;
        self.csi_buffer.drain(0..(self.csi_buffer.len() - keep));

        Ok(Some(result))
    }

    /// Process a buffered window of CSI samples
    fn process_window(&mut self) -> Result<ProcessedCsi> {
        let window = &self.csi_buffer;
        let timestamp = window.last().unwrap().timestamp;

        // Build CSI frame
        let mut frame = CsiFrame::new(
            timestamp,
            self.config.hardware.antenna_config,
            self.config.hardware.bandwidth,
            self.config.hardware.band,
        );

        // Add the latest sanitized CSI to the frame
        if let Some(latest) = window.last() {
            frame.add_stream(latest.clone());
        }

        // Compute Doppler spectrogram
        let doppler = self.doppler.compute_spectrogram(window);

        // Extract velocity profile
        let velocity = self.doppler.velocity_profile(window);

        // Compute statistical features
        let features = self.compute_features(window, &doppler, &velocity);

        Ok(ProcessedCsi {
            timestamp,
            frame,
            doppler,
            velocity,
            features,
        })
    }

    /// Compute statistical features from CSI window
    fn compute_features(
        &self,
        window: &[SanitizedCsi],
        doppler: &DopplerSpectrogram,
        velocity: &VelocityProfile,
    ) -> CsiFeatures {
        if window.is_empty() {
            return CsiFeatures::default();
        }

        // Amplitude statistics
        let all_amplitudes: Vec<f64> = window
            .iter()
            .flat_map(|csi| csi.amplitude.iter().copied())
            .collect();

        let mean_amplitude = all_amplitudes.iter().sum::<f64>() / all_amplitudes.len() as f64;
        let amplitude_variance = all_amplitudes
            .iter()
            .map(|a| (a - mean_amplitude).powi(2))
            .sum::<f64>()
            / all_amplitudes.len() as f64;

        // Phase variance (motion indicator)
        let all_phases: Vec<f64> = window
            .iter()
            .flat_map(|csi| csi.phase.iter().copied())
            .collect();

        let mean_phase = all_phases.iter().sum::<f64>() / all_phases.len() as f64;
        let phase_variance = all_phases
            .iter()
            .map(|p| (p - mean_phase).powi(2))
            .sum::<f64>()
            / all_phases.len() as f64;

        // Motion activity (normalized velocity magnitude)
        let motion_activity = (velocity.mean_velocity / self.doppler.max_velocity()).clamp(0.0, 1.0);

        // Target count estimation (based on spectral peaks)
        let target_count = self.estimate_target_count(doppler);

        // Quality score (average across window)
        let quality_score = window.iter().map(|csi| csi.quality_score).sum::<f64>()
            / window.len() as f64;

        CsiFeatures {
            mean_amplitude,
            amplitude_variance,
            phase_variance,
            dominant_velocity: velocity.max_velocity,
            motion_activity,
            target_count,
            quality_score,
        }
    }

    /// Estimate number of moving targets from Doppler spectrum
    fn estimate_target_count(&self, doppler: &DopplerSpectrogram) -> usize {
        let mean_spectrum = doppler.mean_spectrum();

        // Count significant peaks in the power spectrum
        let threshold = mean_spectrum.power.iter().sum::<f64>() / mean_spectrum.power.len() as f64
            * 3.0; // 3x mean

        let mut peak_count = 0;
        let n = mean_spectrum.power.len();

        for i in 2..(n / 2 - 2) {
            let is_peak = mean_spectrum.power[i] > mean_spectrum.power[i - 1]
                && mean_spectrum.power[i] > mean_spectrum.power[i + 1]
                && mean_spectrum.power[i] > threshold;

            if is_peak {
                peak_count += 1;
            }
        }

        // Clamp to reasonable range
        peak_count.min(10)
    }

    /// Start the pipeline with an acquisition source
    pub async fn start_streaming<A: CsiAcquisition + 'static>(
        &mut self,
        mut acquisition: A,
    ) -> Result<mpsc::Receiver<ProcessedCsi>> {
        let (tx, rx) = mpsc::channel(self.config.output_queue_size);

        // Start acquisition
        acquisition.start().await?;

        *self.is_running.write().await = true;
        let is_running = self.is_running.clone();

        // Clone necessary components for the task
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut pipeline = CsiPipeline::new(config);

            loop {
                if !*is_running.read().await {
                    break;
                }

                match acquisition.recv().await {
                    Ok(packet) => {
                        match pipeline.process_packet(&packet) {
                            Ok(Some(processed)) => {
                                if tx.send(processed).await.is_err() {
                                    break; // Receiver dropped
                                }
                            }
                            Ok(None) => continue, // Buffering
                            Err(e) => {
                                tracing::warn!("CSI processing error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Acquisition error: {}", e);
                        break;
                    }
                }
            }

            let _ = acquisition.stop().await;
        });

        Ok(rx)
    }

    /// Stop the pipeline
    pub async fn stop(&mut self) {
        *self.is_running.write().await = false;
    }

    /// Get current configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Reset the pipeline state
    pub fn reset(&mut self) {
        self.csi_buffer.clear();
        self.lowpass.reset();
    }
}

/// Real-time CSI stream processor with callback support
pub struct CsiStreamProcessor {
    pipeline: CsiPipeline,
    callbacks: Vec<Box<dyn Fn(&ProcessedCsi) + Send + Sync>>,
}

impl CsiStreamProcessor {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            pipeline: CsiPipeline::new(config),
            callbacks: Vec::new(),
        }
    }

    /// Add a callback for processed CSI data
    pub fn on_processed<F>(&mut self, callback: F)
    where
        F: Fn(&ProcessedCsi) + Send + Sync + 'static,
    {
        self.callbacks.push(Box::new(callback));
    }

    /// Process a packet and invoke callbacks
    pub fn process(&mut self, packet: &CsiPacket) -> Result<()> {
        if let Some(processed) = self.pipeline.process_packet(packet)? {
            for callback in &self.callbacks {
                callback(&processed);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn create_test_packet(seq: u32) -> CsiPacket {
        let subcarrier_count = 256u16;
        let csi_matrix: Vec<Complex<f64>> = (0..subcarrier_count)
            .map(|i| {
                let phase = (i as f64 * 0.01) + (seq as f64 * 0.05);
                Complex::from_polar(1.0 + 0.1 * (seq as f64).sin(), phase)
            })
            .collect();

        CsiPacket {
            timestamp: seq as i64 * 1_000_000,
            antenna_id: 0,
            subcarrier_count,
            csi_matrix,
            rssi: -45,
            noise_floor: -90,
            tx_mac_hash: 0,
            sequence_number: seq,
        }
    }

    #[test]
    fn test_pipeline_buffering() {
        let config = PipelineConfig {
            doppler_window_size: 16,
            ..Default::default()
        };

        let mut pipeline = CsiPipeline::new(config);

        // First 15 packets should return None (buffering)
        for i in 0..15 {
            let packet = create_test_packet(i);
            let result = pipeline.process_packet(&packet).unwrap();
            assert!(result.is_none(), "Should be buffering at packet {}", i);
        }

        // 16th packet should produce output
        let packet = create_test_packet(15);
        let result = pipeline.process_packet(&packet).unwrap();
        assert!(result.is_some(), "Should produce output after buffer full");
    }

    #[test]
    fn test_feature_extraction() {
        let config = PipelineConfig {
            doppler_window_size: 8,
            doppler_fft_size: 8,
            ..Default::default()
        };

        let mut pipeline = CsiPipeline::new(config);

        // Process enough packets
        let mut last_result = None;
        for i in 0..16 {
            let packet = create_test_packet(i);
            if let Ok(Some(result)) = pipeline.process_packet(&packet) {
                last_result = Some(result);
            }
        }

        let result = last_result.expect("Should have result");
        assert!(result.features.mean_amplitude > 0.0);
        assert!(result.features.quality_score > 0.0);
    }
}
