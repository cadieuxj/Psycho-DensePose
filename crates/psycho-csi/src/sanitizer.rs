//! CSI sanitization pipeline for removing hardware artifacts.
//!
//! Raw CSI data contains several artifacts that must be removed:
//!
//! 1. **Phase Wrapping**: Phase values wrap at ±π, creating discontinuities
//! 2. **Sampling Frequency Offset (SFO)**: Clock drift between Tx/Rx causes linear phase drift
//! 3. **Packet Detection Delay (PDD)**: Variable timing adds phase offset
//! 4. **Central Frequency Offset (CFO)**: Carrier frequency mismatch
//! 5. **Noise Outliers**: Hardware glitches and interference spikes

use num_complex::Complex;
use psycho_core::{Error, Result, Timestamp};

use crate::packet::{CsiPacket, SanitizedCsi};

/// CSI Sanitizer implementing the complete preprocessing pipeline
pub struct CsiSanitizer {
    /// Window size for Hampel filter outlier detection
    pub hampel_window: usize,

    /// Threshold for Hampel filter (in MAD units)
    pub hampel_threshold: f64,

    /// Whether to apply linear phase removal
    pub remove_linear_phase: bool,

    /// Minimum SNR threshold (dB) for valid packets
    pub min_snr_db: f64,
}

impl Default for CsiSanitizer {
    fn default() -> Self {
        Self {
            hampel_window: 5,
            hampel_threshold: 3.0,
            remove_linear_phase: true,
            min_snr_db: 10.0,
        }
    }
}

impl CsiSanitizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure Hampel filter parameters
    pub fn with_hampel(mut self, window: usize, threshold: f64) -> Self {
        self.hampel_window = window;
        self.hampel_threshold = threshold;
        self
    }

    /// Process a raw CSI packet through the sanitization pipeline
    pub fn sanitize(&self, packet: &CsiPacket) -> Result<SanitizedCsi> {
        if !packet.is_valid() {
            return Err(Error::CsiProcessing("Invalid CSI packet".into()));
        }

        if packet.snr_db() < self.min_snr_db {
            return Err(Error::CsiProcessing(format!(
                "SNR too low: {} dB < {} dB",
                packet.snr_db(),
                self.min_snr_db
            )));
        }

        // Extract amplitude and phase
        let amplitude = packet.amplitudes();
        let mut phase = packet.phases();

        // Step 1: Unwrap phase to remove 2π discontinuities
        self.unwrap_phase(&mut phase);

        // Step 2: Remove linear phase slope (SFO + PDD)
        if self.remove_linear_phase {
            self.remove_linear_phase_slope(&mut phase);
        }

        // Step 3: Apply Hampel filter to both amplitude and phase
        let amplitude = self.hampel_filter(&amplitude);
        let phase = self.hampel_filter(&phase);

        // Calculate quality score based on variance and SNR
        let quality_score = self.calculate_quality(&amplitude, packet.snr_db());

        Ok(SanitizedCsi::new(
            packet.to_timestamp(),
            packet.antenna_id,
            amplitude,
            phase,
            quality_score,
        ))
    }

    /// Unwrap phase to remove 2π discontinuities
    ///
    /// Phase values are wrapped to [-π, π], but continuous motion causes
    /// smooth phase changes. This function detects and corrects jumps.
    pub fn unwrap_phase(&self, phase: &mut [f64]) {
        if phase.len() < 2 {
            return;
        }

        let mut cumulative_offset = 0.0;
        let threshold = std::f64::consts::PI;

        for i in 1..phase.len() {
            let diff = phase[i] - phase[i - 1];

            // Detect phase wrap
            if diff > threshold {
                cumulative_offset -= 2.0 * std::f64::consts::PI;
            } else if diff < -threshold {
                cumulative_offset += 2.0 * std::f64::consts::PI;
            }

            phase[i] += cumulative_offset;
        }
    }

    /// Remove linear phase slope caused by SFO and PDD
    ///
    /// The raw phase across subcarriers has a linear trend due to:
    /// - Sampling Frequency Offset: clock drift between Tx and Rx
    /// - Packet Detection Delay: variable timing in packet detection
    ///
    /// We fit a line y = mx + b and subtract it.
    pub fn remove_linear_phase_slope(&self, phase: &mut [f64]) {
        if phase.len() < 2 {
            return;
        }

        let n = phase.len() as f64;

        // Calculate linear regression coefficients
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = phase.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in phase.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        let intercept = y_mean - slope * x_mean;

        // Subtract the linear fit
        for (i, p) in phase.iter_mut().enumerate() {
            *p -= slope * i as f64 + intercept;
        }
    }

    /// Hampel filter for outlier removal
    ///
    /// The Hampel filter uses the Median Absolute Deviation (MAD) to detect
    /// outliers in a sliding window. Outliers are replaced with the local median.
    pub fn hampel_filter(&self, data: &[f64]) -> Vec<f64> {
        if data.len() <= self.hampel_window {
            return data.to_vec();
        }

        let mut result = data.to_vec();
        let half_window = self.hampel_window / 2;

        // Scale factor for MAD to estimate standard deviation (for normal distribution)
        const MAD_SCALE: f64 = 1.4826;

        for i in half_window..(data.len() - half_window) {
            let start = i - half_window;
            let end = i + half_window + 1;
            let window: Vec<f64> = data[start..end].to_vec();

            let median = Self::median(&window);
            let mad = Self::mad(&window, median);

            let threshold = self.hampel_threshold * MAD_SCALE * mad;

            if (data[i] - median).abs() > threshold {
                result[i] = median;
            }
        }

        result
    }

    /// Calculate median of a slice
    fn median(data: &[f64]) -> f64 {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Calculate Median Absolute Deviation
    fn mad(data: &[f64], median: f64) -> f64 {
        let deviations: Vec<f64> = data.iter().map(|&x| (x - median).abs()).collect();
        Self::median(&deviations)
    }

    /// Calculate quality score for sanitized CSI
    fn calculate_quality(&self, amplitude: &[f64], snr_db: f64) -> f64 {
        // Normalized SNR component (0-1, assuming max 40dB is excellent)
        let snr_score = (snr_db / 40.0).clamp(0.0, 1.0);

        // Amplitude stability (inverse of coefficient of variation)
        let mean_amp: f64 = amplitude.iter().sum::<f64>() / amplitude.len() as f64;
        let var_amp: f64 = amplitude.iter().map(|&a| (a - mean_amp).powi(2)).sum::<f64>()
            / amplitude.len() as f64;
        let cv = if mean_amp > 1e-10 {
            var_amp.sqrt() / mean_amp
        } else {
            1.0
        };
        let stability_score = (1.0 / (1.0 + cv)).clamp(0.0, 1.0);

        // Combined score (weighted average)
        0.6 * snr_score + 0.4 * stability_score
    }
}

/// Batch CSI sanitizer for processing multiple packets
pub struct BatchSanitizer {
    sanitizer: CsiSanitizer,
    /// Buffer for temporal smoothing
    buffer: Vec<SanitizedCsi>,
    /// Buffer size for temporal processing
    buffer_size: usize,
}

impl BatchSanitizer {
    pub fn new(sanitizer: CsiSanitizer, buffer_size: usize) -> Self {
        Self {
            sanitizer,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }

    /// Process a packet and optionally return temporally smoothed result
    pub fn process(&mut self, packet: &CsiPacket) -> Result<Option<SanitizedCsi>> {
        let sanitized = self.sanitizer.sanitize(packet)?;

        self.buffer.push(sanitized);

        if self.buffer.len() >= self.buffer_size {
            // Apply temporal smoothing across buffer
            let smoothed = self.temporal_smooth();
            self.buffer.clear();
            Ok(Some(smoothed))
        } else {
            Ok(None)
        }
    }

    /// Apply temporal smoothing (moving average) across buffered samples
    fn temporal_smooth(&self) -> SanitizedCsi {
        let n = self.buffer.len() as f64;
        let n_subcarriers = self.buffer[0].amplitude.len();

        let mut avg_amplitude = vec![0.0; n_subcarriers];
        let mut avg_phase = vec![0.0; n_subcarriers];
        let mut avg_quality = 0.0;

        for sample in &self.buffer {
            for (i, &amp) in sample.amplitude.iter().enumerate() {
                avg_amplitude[i] += amp / n;
            }
            for (i, &ph) in sample.phase.iter().enumerate() {
                avg_phase[i] += ph / n;
            }
            avg_quality += sample.quality_score / n;
        }

        SanitizedCsi::new(
            self.buffer.last().unwrap().timestamp,
            self.buffer[0].antenna_id,
            avg_amplitude,
            avg_phase,
            avg_quality,
        )
    }

    /// Flush buffer and return any remaining smoothed data
    pub fn flush(&mut self) -> Option<SanitizedCsi> {
        if self.buffer.is_empty() {
            return None;
        }

        let smoothed = self.temporal_smooth();
        self.buffer.clear();
        Some(smoothed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn create_test_packet(phases: Vec<f64>) -> CsiPacket {
        let csi: Vec<Complex<f64>> = phases
            .into_iter()
            .map(|p| Complex::from_polar(1.0, p))
            .collect();
        let count = csi.len() as u16;
        CsiPacket::new(0, 0, count, csi)
    }

    #[test]
    fn test_phase_unwrap() {
        let sanitizer = CsiSanitizer::new();

        // Create phase with artificial wraps
        let mut phase = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -3.0, -2.5, -2.0];
        // The jump from 3.0 to -3.0 should be unwrapped

        sanitizer.unwrap_phase(&mut phase);

        // After unwrapping, the sequence should be monotonically increasing
        for i in 1..phase.len() {
            assert!(
                phase[i] >= phase[i - 1] - 0.1,
                "Phase should be continuous after unwrapping"
            );
        }
    }

    #[test]
    fn test_linear_phase_removal() {
        let sanitizer = CsiSanitizer::new();

        // Create phase with linear trend
        let mut phase: Vec<f64> = (0..100).map(|i| 0.1 * i as f64 + 0.5).collect();
        let original_slope = 0.1;

        sanitizer.remove_linear_phase_slope(&mut phase);

        // Calculate remaining slope
        let new_slope = (phase.last().unwrap() - phase.first().unwrap()) / (phase.len() - 1) as f64;

        assert!(
            new_slope.abs() < 0.01,
            "Linear phase should be removed, residual slope: {}",
            new_slope
        );
    }

    #[test]
    fn test_hampel_filter() {
        let sanitizer = CsiSanitizer::new().with_hampel(5, 3.0);

        // Create data with outlier
        let mut data: Vec<f64> = vec![1.0; 20];
        data[10] = 100.0; // Outlier

        let filtered = sanitizer.hampel_filter(&data);

        // Outlier should be replaced with median (1.0)
        assert!(
            (filtered[10] - 1.0).abs() < 0.1,
            "Outlier should be removed"
        );
    }

    #[test]
    fn test_full_sanitization() {
        let sanitizer = CsiSanitizer::new();

        // Create realistic CSI with linear phase and some noise
        let phases: Vec<f64> = (0..256)
            .map(|i| 0.02 * i as f64 + 0.3 * (i as f64 * 0.1).sin())
            .collect();

        let packet = create_test_packet(phases);
        let result = sanitizer.sanitize(&packet);

        assert!(result.is_ok());
        let sanitized = result.unwrap();
        assert_eq!(sanitized.amplitude.len(), 256);
        assert!(sanitized.quality_score > 0.0);
    }
}
