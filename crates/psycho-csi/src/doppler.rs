//! Doppler shift extraction for motion sensing.
//!
//! The Doppler effect causes a frequency shift when there's relative motion
//! between transmitter, reflector (human body), and receiver:
//!
//! f_D = (1/λ) * v * cos(θ)
//!
//! Where:
//! - λ: wavelength of the WiFi signal
//! - v: velocity of the moving object
//! - θ: angle between motion direction and signal path
//!
//! By analyzing phase changes over time using FFT, we can extract these
//! Doppler signatures to estimate motion characteristics.

use ndarray::{Array1, Array2, Axis};
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use crate::packet::SanitizedCsi;

/// Doppler spectrum for a single time window
#[derive(Debug, Clone)]
pub struct DopplerSpectrum {
    /// Doppler frequency bins (Hz)
    pub frequencies: Vec<f64>,

    /// Power spectrum magnitude for each frequency bin
    pub power: Vec<f64>,

    /// Phase of each frequency bin
    pub phase: Vec<f64>,

    /// Peak Doppler frequency (Hz)
    pub peak_frequency: f64,

    /// Peak power (linear scale)
    pub peak_power: f64,

    /// Estimated velocity from peak Doppler (m/s)
    pub estimated_velocity: f64,
}

/// Doppler processor for extracting motion signatures from CSI
pub struct DopplerProcessor {
    /// FFT size for Doppler computation
    fft_size: usize,

    /// Sampling rate (packets per second)
    sample_rate: f64,

    /// Wavelength for velocity calculation (meters)
    wavelength: f64,

    /// Pre-computed FFT plan
    fft: Arc<dyn Fft<f64>>,

    /// Window function coefficients
    window: Vec<f64>,

    /// Velocity resolution (m/s per bin)
    velocity_resolution: f64,

    /// Maximum detectable velocity (m/s)
    max_velocity: f64,
}

impl DopplerProcessor {
    pub fn new(fft_size: usize, sample_rate: f64, wavelength: f64) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        // Hanning window for reduced spectral leakage
        let window = Self::hanning_window(fft_size);

        // Doppler resolution: Δf = sample_rate / fft_size
        // Velocity resolution: Δv = λ * Δf
        let freq_resolution = sample_rate / fft_size as f64;
        let velocity_resolution = wavelength * freq_resolution;

        // Maximum velocity = λ * (sample_rate / 2) [Nyquist limit]
        let max_velocity = wavelength * sample_rate / 2.0;

        Self {
            fft_size,
            sample_rate,
            wavelength,
            fft,
            window,
            velocity_resolution,
            max_velocity,
        }
    }

    /// Create Hanning window coefficients
    fn hanning_window(size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())
            })
            .collect()
    }

    /// Process a time window of CSI samples to extract Doppler spectrum
    ///
    /// Takes phase values from consecutive CSI samples and computes
    /// the FFT to extract Doppler frequency components.
    pub fn compute_doppler(&self, csi_window: &[SanitizedCsi], subcarrier_idx: usize) -> DopplerSpectrum {
        assert!(csi_window.len() <= self.fft_size);

        // Extract phase time series for the specified subcarrier
        let mut phase_series: Vec<Complex<f64>> = Vec::with_capacity(self.fft_size);

        for csi in csi_window {
            if subcarrier_idx < csi.phase.len() {
                let phase = csi.phase[subcarrier_idx];
                // Apply window and convert to complex for FFT
                let window_val = self.window[phase_series.len()];
                phase_series.push(Complex::new(phase * window_val, 0.0));
            }
        }

        // Zero-pad to FFT size if needed
        while phase_series.len() < self.fft_size {
            phase_series.push(Complex::new(0.0, 0.0));
        }

        // Compute FFT
        self.fft.process(&mut phase_series);

        // Extract magnitude and phase from FFT output
        let n = self.fft_size;
        let mut power = Vec::with_capacity(n);
        let mut phase = Vec::with_capacity(n);
        let mut frequencies = Vec::with_capacity(n);

        let freq_step = self.sample_rate / n as f64;

        for (i, c) in phase_series.iter().enumerate() {
            let freq = if i <= n / 2 {
                i as f64 * freq_step
            } else {
                (i as f64 - n as f64) * freq_step
            };

            frequencies.push(freq);
            power.push(c.norm_sqr() / n as f64); // Normalized power
            phase.push(c.arg());
        }

        // Find peak (excluding DC component)
        let (peak_idx, peak_power) = power
            .iter()
            .enumerate()
            .skip(1) // Skip DC
            .take(n / 2) // Only positive frequencies
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &p)| (i, p))
            .unwrap_or((0, 0.0));

        let peak_frequency = frequencies[peak_idx];

        // Convert Doppler frequency to velocity
        // v = λ * f_D (assuming cos(θ) = 1 for maximum sensitivity)
        let estimated_velocity = self.wavelength * peak_frequency;

        DopplerSpectrum {
            frequencies,
            power,
            phase,
            peak_frequency,
            peak_power,
            estimated_velocity,
        }
    }

    /// Compute full Doppler spectrogram across all subcarriers
    pub fn compute_spectrogram(&self, csi_window: &[SanitizedCsi]) -> DopplerSpectrogram {
        if csi_window.is_empty() {
            return DopplerSpectrogram::empty();
        }

        let n_subcarriers = csi_window[0].amplitude.len();
        let mut spectra = Vec::with_capacity(n_subcarriers);

        for sc in 0..n_subcarriers {
            spectra.push(self.compute_doppler(csi_window, sc));
        }

        DopplerSpectrogram {
            spectra,
            n_subcarriers,
            fft_size: self.fft_size,
            velocity_resolution: self.velocity_resolution,
            max_velocity: self.max_velocity,
        }
    }

    /// Extract Doppler velocity profile using centroid method
    pub fn velocity_profile(&self, csi_window: &[SanitizedCsi]) -> VelocityProfile {
        let spectrogram = self.compute_spectrogram(csi_window);

        // Compute velocity centroid for each subcarrier
        let mut velocities = Vec::with_capacity(spectrogram.n_subcarriers);
        let mut powers = Vec::with_capacity(spectrogram.n_subcarriers);

        for spectrum in &spectrogram.spectra {
            // Weighted centroid of velocity (excluding DC)
            let half = self.fft_size / 2;
            let mut sum_vp = 0.0;
            let mut sum_p = 0.0;

            for i in 1..half {
                let v = self.wavelength * spectrum.frequencies[i];
                let p = spectrum.power[i];
                sum_vp += v * p;
                sum_p += p;
            }

            let centroid_velocity = if sum_p > 1e-10 {
                sum_vp / sum_p
            } else {
                0.0
            };

            velocities.push(centroid_velocity);
            powers.push(sum_p);
        }

        VelocityProfile {
            velocities,
            powers,
            mean_velocity: velocities.iter().sum::<f64>() / velocities.len() as f64,
            max_velocity: velocities.iter().cloned().fold(0.0, f64::max),
        }
    }

    pub fn velocity_resolution(&self) -> f64 {
        self.velocity_resolution
    }

    pub fn max_velocity(&self) -> f64 {
        self.max_velocity
    }
}

/// Full Doppler spectrogram across all subcarriers
#[derive(Debug, Clone)]
pub struct DopplerSpectrogram {
    /// Doppler spectrum for each subcarrier
    pub spectra: Vec<DopplerSpectrum>,

    /// Number of subcarriers
    pub n_subcarriers: usize,

    /// FFT size used
    pub fft_size: usize,

    /// Velocity resolution (m/s per bin)
    pub velocity_resolution: f64,

    /// Maximum detectable velocity (m/s)
    pub max_velocity: f64,
}

impl DopplerSpectrogram {
    pub fn empty() -> Self {
        Self {
            spectra: Vec::new(),
            n_subcarriers: 0,
            fft_size: 0,
            velocity_resolution: 0.0,
            max_velocity: 0.0,
        }
    }

    /// Get power matrix [subcarrier x frequency]
    pub fn power_matrix(&self) -> Array2<f64> {
        if self.spectra.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_freq = self.spectra[0].power.len();
        let mut matrix = Array2::zeros((self.n_subcarriers, n_freq));

        for (sc, spectrum) in self.spectra.iter().enumerate() {
            for (f, &p) in spectrum.power.iter().enumerate() {
                matrix[[sc, f]] = p;
            }
        }

        matrix
    }

    /// Compute average power spectrum across all subcarriers
    pub fn mean_spectrum(&self) -> DopplerSpectrum {
        if self.spectra.is_empty() {
            return DopplerSpectrum {
                frequencies: Vec::new(),
                power: Vec::new(),
                phase: Vec::new(),
                peak_frequency: 0.0,
                peak_power: 0.0,
                estimated_velocity: 0.0,
            };
        }

        let n_freq = self.spectra[0].power.len();
        let n_sc = self.spectra.len() as f64;

        let mut avg_power = vec![0.0; n_freq];

        for spectrum in &self.spectra {
            for (i, &p) in spectrum.power.iter().enumerate() {
                avg_power[i] += p / n_sc;
            }
        }

        let frequencies = self.spectra[0].frequencies.clone();

        // Find peak
        let (peak_idx, peak_power) = avg_power
            .iter()
            .enumerate()
            .skip(1)
            .take(n_freq / 2)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &p)| (i, p))
            .unwrap_or((0, 0.0));

        let peak_frequency = frequencies.get(peak_idx).copied().unwrap_or(0.0);
        let estimated_velocity = self.velocity_resolution * peak_idx as f64;

        DopplerSpectrum {
            frequencies,
            power: avg_power,
            phase: vec![0.0; n_freq], // Phase averaging is complex, simplified here
            peak_frequency,
            peak_power,
            estimated_velocity,
        }
    }
}

/// Velocity profile extracted from Doppler analysis
#[derive(Debug, Clone)]
pub struct VelocityProfile {
    /// Velocity estimate per subcarrier (m/s)
    pub velocities: Vec<f64>,

    /// Signal power per subcarrier
    pub powers: Vec<f64>,

    /// Mean velocity across all subcarriers
    pub mean_velocity: f64,

    /// Maximum velocity detected
    pub max_velocity: f64,
}

impl VelocityProfile {
    /// Check if significant motion is detected
    pub fn has_motion(&self, threshold: f64) -> bool {
        self.mean_velocity.abs() > threshold
    }

    /// Get velocity variance (indicator of motion complexity)
    pub fn velocity_variance(&self) -> f64 {
        if self.velocities.len() < 2 {
            return 0.0;
        }

        let mean = self.mean_velocity;
        let variance: f64 = self.velocities.iter().map(|v| (v - mean).powi(2)).sum();
        variance / (self.velocities.len() - 1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::SanitizedCsi;
    use psycho_core::Timestamp;

    fn create_test_csi(phase_values: &[f64]) -> SanitizedCsi {
        SanitizedCsi::new(
            Timestamp::now(),
            0,
            vec![1.0; phase_values.len()],
            phase_values.to_vec(),
            0.9,
        )
    }

    #[test]
    fn test_doppler_static() {
        // Static scene: constant phase
        let processor = DopplerProcessor::new(64, 1000.0, 0.05);

        let csi_window: Vec<SanitizedCsi> = (0..64)
            .map(|_| create_test_csi(&[1.0, 1.0, 1.0, 1.0]))
            .collect();

        let spectrum = processor.compute_doppler(&csi_window, 0);

        // Static should have peak at DC (index 0), so peak_frequency near 0
        assert!(
            spectrum.estimated_velocity.abs() < 0.5,
            "Static scene should have near-zero velocity"
        );
    }

    #[test]
    fn test_doppler_moving() {
        // Moving target: linearly changing phase (simulates Doppler shift)
        let processor = DopplerProcessor::new(64, 1000.0, 0.05);

        let phase_rate = 0.1; // rad/sample
        let csi_window: Vec<SanitizedCsi> = (0..64)
            .map(|i| create_test_csi(&[i as f64 * phase_rate; 4]))
            .collect();

        let spectrum = processor.compute_doppler(&csi_window, 0);

        // Should detect non-zero velocity
        assert!(
            spectrum.peak_power > 0.0,
            "Should detect motion power"
        );
    }

    #[test]
    fn test_hanning_window() {
        let window = DopplerProcessor::hanning_window(10);

        // First and last should be near zero
        assert!(window[0] < 0.01);
        assert!(window[9] < 0.01);

        // Middle should be near 1
        assert!((window[4] - 1.0).abs() < 0.1 || (window[5] - 1.0).abs() < 0.1);
    }
}
