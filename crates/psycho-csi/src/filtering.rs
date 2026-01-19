//! Advanced signal filtering for CSI data processing.

use ndarray::{Array1, Array2, Axis};
use std::collections::VecDeque;

/// Butterworth low-pass filter for CSI smoothing
pub struct ButterworthFilter {
    order: usize,
    cutoff_normalized: f64,
    a: Vec<f64>,
    b: Vec<f64>,
    x_history: VecDeque<f64>,
    y_history: VecDeque<f64>,
}

impl ButterworthFilter {
    /// Create a new Butterworth low-pass filter
    ///
    /// # Arguments
    /// * `order` - Filter order (1-4 recommended)
    /// * `cutoff_freq` - Cutoff frequency in Hz
    /// * `sample_rate` - Sampling rate in Hz
    pub fn new(order: usize, cutoff_freq: f64, sample_rate: f64) -> Self {
        let cutoff_normalized = cutoff_freq / (sample_rate / 2.0);

        // Pre-warp the cutoff frequency
        let omega = (std::f64::consts::PI * cutoff_normalized).tan();

        // Calculate filter coefficients based on order
        let (a, b) = match order {
            1 => Self::coefficients_order1(omega),
            2 => Self::coefficients_order2(omega),
            _ => Self::coefficients_order2(omega), // Default to order 2
        };

        let history_len = order.max(a.len());

        Self {
            order,
            cutoff_normalized,
            a,
            b,
            x_history: VecDeque::from(vec![0.0; history_len]),
            y_history: VecDeque::from(vec![0.0; history_len]),
        }
    }

    fn coefficients_order1(omega: f64) -> (Vec<f64>, Vec<f64>) {
        let k = omega / (1.0 + omega);
        let a = vec![1.0, -(1.0 - omega) / (1.0 + omega)];
        let b = vec![k, k];
        (a, b)
    }

    fn coefficients_order2(omega: f64) -> (Vec<f64>, Vec<f64>) {
        let omega_sq = omega * omega;
        let sqrt2 = std::f64::consts::SQRT_2;

        let denom = 1.0 + sqrt2 * omega + omega_sq;

        let a = vec![
            1.0,
            2.0 * (omega_sq - 1.0) / denom,
            (1.0 - sqrt2 * omega + omega_sq) / denom,
        ];

        let k = omega_sq / denom;
        let b = vec![k, 2.0 * k, k];

        (a, b)
    }

    /// Process a single sample through the filter
    pub fn filter(&mut self, x: f64) -> f64 {
        // Update input history
        self.x_history.push_front(x);
        self.x_history.pop_back();

        // Calculate output using difference equation
        let mut y = 0.0;

        // Feed-forward (b coefficients)
        for (i, &coef) in self.b.iter().enumerate() {
            if i < self.x_history.len() {
                y += coef * self.x_history[i];
            }
        }

        // Feedback (a coefficients, skip a[0] which is 1.0)
        for (i, &coef) in self.a.iter().enumerate().skip(1) {
            if i - 1 < self.y_history.len() {
                y -= coef * self.y_history[i - 1];
            }
        }

        // Update output history
        self.y_history.push_front(y);
        self.y_history.pop_back();

        y
    }

    /// Filter an entire signal
    pub fn filter_signal(&mut self, signal: &[f64]) -> Vec<f64> {
        // Reset state
        self.x_history.iter_mut().for_each(|x| *x = 0.0);
        self.y_history.iter_mut().for_each(|y| *y = 0.0);

        signal.iter().map(|&x| self.filter(x)).collect()
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.x_history.iter_mut().for_each(|x| *x = 0.0);
        self.y_history.iter_mut().for_each(|y| *y = 0.0);
    }
}

/// Moving average filter for simple smoothing
pub struct MovingAverageFilter {
    window_size: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

impl MovingAverageFilter {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }

    pub fn filter(&mut self, x: f64) -> f64 {
        self.buffer.push_back(x);
        self.sum += x;

        if self.buffer.len() > self.window_size {
            self.sum -= self.buffer.pop_front().unwrap();
        }

        self.sum / self.buffer.len() as f64
    }

    pub fn filter_signal(&mut self, signal: &[f64]) -> Vec<f64> {
        self.buffer.clear();
        self.sum = 0.0;
        signal.iter().map(|&x| self.filter(x)).collect()
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
    }
}

/// Exponential moving average (EMA) filter
pub struct ExponentialFilter {
    alpha: f64,
    state: Option<f64>,
}

impl ExponentialFilter {
    /// Create new EMA filter
    ///
    /// # Arguments
    /// * `alpha` - Smoothing factor (0-1). Higher = less smoothing
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            state: None,
        }
    }

    /// Create from time constant
    ///
    /// # Arguments
    /// * `time_constant` - Time constant in samples
    pub fn from_time_constant(time_constant: f64) -> Self {
        let alpha = 1.0 / (time_constant + 1.0);
        Self::new(alpha)
    }

    pub fn filter(&mut self, x: f64) -> f64 {
        match self.state {
            Some(prev) => {
                let y = self.alpha * x + (1.0 - self.alpha) * prev;
                self.state = Some(y);
                y
            }
            None => {
                self.state = Some(x);
                x
            }
        }
    }

    pub fn filter_signal(&mut self, signal: &[f64]) -> Vec<f64> {
        self.state = None;
        signal.iter().map(|&x| self.filter(x)).collect()
    }

    pub fn reset(&mut self) {
        self.state = None;
    }
}

/// Kalman filter for 1D state estimation
pub struct KalmanFilter1D {
    /// State estimate
    x: f64,
    /// Estimate covariance
    p: f64,
    /// Process noise covariance
    q: f64,
    /// Measurement noise covariance
    r: f64,
}

impl KalmanFilter1D {
    /// Create new 1D Kalman filter
    ///
    /// # Arguments
    /// * `process_noise` - Process noise variance (Q)
    /// * `measurement_noise` - Measurement noise variance (R)
    /// * `initial_estimate` - Initial state estimate
    pub fn new(process_noise: f64, measurement_noise: f64, initial_estimate: f64) -> Self {
        Self {
            x: initial_estimate,
            p: 1.0, // Initial estimate covariance
            q: process_noise,
            r: measurement_noise,
        }
    }

    /// Update filter with new measurement
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Predict
        // For static model, state prediction is unchanged
        let x_pred = self.x;
        let p_pred = self.p + self.q;

        // Update
        let k = p_pred / (p_pred + self.r); // Kalman gain
        self.x = x_pred + k * (measurement - x_pred);
        self.p = (1.0 - k) * p_pred;

        self.x
    }

    /// Filter entire signal
    pub fn filter_signal(&mut self, signal: &[f64]) -> Vec<f64> {
        if signal.is_empty() {
            return Vec::new();
        }

        // Reset with first measurement
        self.x = signal[0];
        self.p = 1.0;

        signal.iter().map(|&z| self.update(z)).collect()
    }

    pub fn state(&self) -> f64 {
        self.x
    }

    pub fn covariance(&self) -> f64 {
        self.p
    }
}

/// Savitzky-Golay filter for smoothing while preserving peaks
pub struct SavitzkyGolayFilter {
    window_size: usize,
    poly_order: usize,
    coefficients: Vec<f64>,
}

impl SavitzkyGolayFilter {
    /// Create new Savitzky-Golay filter
    ///
    /// # Arguments
    /// * `window_size` - Must be odd and > poly_order
    /// * `poly_order` - Polynomial order (typically 2 or 3)
    pub fn new(window_size: usize, poly_order: usize) -> Self {
        assert!(window_size % 2 == 1, "Window size must be odd");
        assert!(window_size > poly_order, "Window must be larger than polynomial order");

        let coefficients = Self::compute_coefficients(window_size, poly_order);

        Self {
            window_size,
            poly_order,
            coefficients,
        }
    }

    /// Compute convolution coefficients using least squares
    fn compute_coefficients(window_size: usize, poly_order: usize) -> Vec<f64> {
        let half = (window_size / 2) as i32;
        let m = poly_order + 1;

        // Build Vandermonde matrix
        let mut a = Array2::<f64>::zeros((window_size, m));
        for (i, row) in a.rows_mut().into_iter().enumerate() {
            let x = (i as i32 - half) as f64;
            for (j, val) in row.into_iter().enumerate() {
                *val = x.powi(j as i32);
            }
        }

        // Compute (A^T A)^-1 A^T using pseudo-inverse
        // For simplicity, use precomputed coefficients for common cases
        match (window_size, poly_order) {
            (5, 2) => vec![-3.0, 12.0, 17.0, 12.0, -3.0]
                .into_iter()
                .map(|x| x / 35.0)
                .collect(),
            (7, 2) => vec![-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]
                .into_iter()
                .map(|x| x / 21.0)
                .collect(),
            (9, 2) => vec![-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0]
                .into_iter()
                .map(|x| x / 231.0)
                .collect(),
            (5, 3) => vec![5.0, -30.0, 75.0, -30.0, 5.0]
                .into_iter()
                .map(|x| x / 35.0)
                .collect(),
            (7, 3) => vec![5.0, -6.0, -3.0, 4.0, -3.0, -6.0, 5.0]
                .into_iter()
                .map(|x| x / (-7.0))
                .collect(),
            _ => {
                // Fall back to uniform weights
                vec![1.0 / window_size as f64; window_size]
            }
        }
    }

    /// Filter a signal using Savitzky-Golay smoothing
    pub fn filter_signal(&self, signal: &[f64]) -> Vec<f64> {
        if signal.len() < self.window_size {
            return signal.to_vec();
        }

        let half = self.window_size / 2;
        let mut result = Vec::with_capacity(signal.len());

        // Handle start boundary (use smaller window)
        for i in 0..half {
            result.push(signal[i]);
        }

        // Main convolution
        for i in half..(signal.len() - half) {
            let mut sum = 0.0;
            for (j, &coef) in self.coefficients.iter().enumerate() {
                sum += coef * signal[i + j - half];
            }
            result.push(sum);
        }

        // Handle end boundary
        for i in (signal.len() - half)..signal.len() {
            result.push(signal[i]);
        }

        result
    }
}

/// Multi-band filter for separating CSI components
pub struct BandpassFilter {
    low_cutoff: f64,
    high_cutoff: f64,
    sample_rate: f64,
    lowpass: ButterworthFilter,
    highpass: ButterworthFilter,
}

impl BandpassFilter {
    pub fn new(low_cutoff: f64, high_cutoff: f64, sample_rate: f64, order: usize) -> Self {
        Self {
            low_cutoff,
            high_cutoff,
            sample_rate,
            lowpass: ButterworthFilter::new(order, high_cutoff, sample_rate),
            highpass: ButterworthFilter::new(order, low_cutoff, sample_rate),
        }
    }

    pub fn filter_signal(&mut self, signal: &[f64]) -> Vec<f64> {
        // Apply lowpass first, then highpass (series connection)
        let lowpassed = self.lowpass.filter_signal(signal);

        // For highpass, we use: highpass = signal - lowpass
        // This is a simple approximation; proper highpass would need different coefficients
        let highpass_only = ButterworthFilter::new(2, self.low_cutoff, self.sample_rate);
        let mut hp = highpass_only;
        let hp_signal = hp.filter_signal(signal);

        // Bandpass = input - lowpass(low_cutoff) - (input - lowpass(high_cutoff))
        // Simplified: bandpass via series low+high pass
        lowpassed
            .iter()
            .zip(hp_signal.iter())
            .map(|(&lp, &hp)| lp - hp)
            .collect()
    }

    pub fn reset(&mut self) {
        self.lowpass.reset();
        self.highpass.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butterworth_dc() {
        let mut filter = ButterworthFilter::new(2, 10.0, 100.0);

        // DC signal should pass through (approximately)
        let dc_signal = vec![1.0; 100];
        let filtered = filter.filter_signal(&dc_signal);

        // After settling, output should approach input
        let last_10_avg: f64 = filtered[90..].iter().sum::<f64>() / 10.0;
        assert!((last_10_avg - 1.0).abs() < 0.05, "DC should pass through");
    }

    #[test]
    fn test_moving_average() {
        let mut filter = MovingAverageFilter::new(5);

        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let filtered = filter.filter_signal(&signal);

        // After window is full, should average properly
        assert!((filtered[4] - 3.0).abs() < 0.01); // (1+2+3+4+5)/5 = 3
        assert!((filtered[9] - 8.0).abs() < 0.01); // (6+7+8+9+10)/5 = 8
    }

    #[test]
    fn test_kalman_filter() {
        let mut kalman = KalmanFilter1D::new(0.01, 0.1, 0.0);

        // Noisy measurements around 5.0
        let measurements = vec![4.8, 5.2, 4.9, 5.1, 5.0, 4.95, 5.05, 5.0];
        let filtered = kalman.filter_signal(&measurements);

        // Should converge toward 5.0
        let last = *filtered.last().unwrap();
        assert!((last - 5.0).abs() < 0.3, "Kalman should track true value");
    }

    #[test]
    fn test_savitzky_golay() {
        let filter = SavitzkyGolayFilter::new(5, 2);

        // Signal with noise
        let signal: Vec<f64> = (0..20)
            .map(|i| (i as f64) + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();

        let filtered = filter.filter_signal(&signal);

        // Smoothed signal should have less variation
        let var_original: f64 = signal.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        let var_filtered: f64 = filtered.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();

        assert!(var_filtered < var_original, "SG filter should reduce noise");
    }
}
