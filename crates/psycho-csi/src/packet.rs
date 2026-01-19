//! CSI packet data structures for WiFi 6E sensing.

use num_complex::Complex;
use psycho_core::{AntennaConfig, ChannelBandwidth, FrequencyBand, Timestamp};
use serde::{Deserialize, Serialize};

/// Raw CSI packet from WiFi hardware
///
/// Contains the complex-valued Channel State Information matrix H(f, t)
/// where each element represents how a signal is attenuated and phase-shifted
/// at a specific subcarrier frequency.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct CsiPacket {
    /// Nanosecond timestamp when packet was captured
    pub timestamp: i64,

    /// Antenna pair identifier (Tx * N_rx + Rx)
    pub antenna_id: u8,

    /// Number of OFDM subcarriers (1992 for 160MHz WiFi 6E)
    pub subcarrier_count: u16,

    /// Raw complex CSI matrix H(f, t) = |H(f,t)| * e^(j*∠H(f,t))
    pub csi_matrix: Vec<Complex<f64>>,

    /// Received Signal Strength Indicator (dBm)
    pub rssi: i8,

    /// Noise floor (dBm)
    pub noise_floor: i8,

    /// MAC address of transmitter (anonymized hash)
    pub tx_mac_hash: u64,

    /// Sequence number for packet ordering
    pub sequence_number: u32,
}

impl CsiPacket {
    /// Create a new CSI packet with the given parameters
    pub fn new(
        timestamp: i64,
        antenna_id: u8,
        subcarrier_count: u16,
        csi_matrix: Vec<Complex<f64>>,
    ) -> Self {
        Self {
            timestamp,
            antenna_id,
            subcarrier_count,
            csi_matrix,
            rssi: -50,
            noise_floor: -90,
            tx_mac_hash: 0,
            sequence_number: 0,
        }
    }

    /// Extract amplitude values from CSI matrix
    pub fn amplitudes(&self) -> Vec<f64> {
        self.csi_matrix.iter().map(|c| c.norm()).collect()
    }

    /// Extract phase values from CSI matrix (radians)
    pub fn phases(&self) -> Vec<f64> {
        self.csi_matrix.iter().map(|c| c.arg()).collect()
    }

    /// Get Signal-to-Noise Ratio in dB
    pub fn snr_db(&self) -> f64 {
        (self.rssi - self.noise_floor) as f64
    }

    /// Validate packet integrity
    pub fn is_valid(&self) -> bool {
        self.csi_matrix.len() == self.subcarrier_count as usize
            && self.subcarrier_count > 0
            && self.csi_matrix.iter().all(|c| c.norm().is_finite())
    }

    /// Convert to sanitized packet structure
    pub fn to_timestamp(&self) -> Timestamp {
        Timestamp::from_nanos(self.timestamp)
    }
}

/// Sanitized CSI data after preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizedCsi {
    pub timestamp: Timestamp,
    pub antenna_id: u8,

    /// Cleaned amplitude values
    pub amplitude: Vec<f64>,

    /// Unwrapped and detrended phase values
    pub phase: Vec<f64>,

    /// Quality score (0-1) based on SNR and artifact removal
    pub quality_score: f64,
}

impl SanitizedCsi {
    pub fn new(
        timestamp: Timestamp,
        antenna_id: u8,
        amplitude: Vec<f64>,
        phase: Vec<f64>,
        quality_score: f64,
    ) -> Self {
        Self {
            timestamp,
            antenna_id,
            amplitude,
            phase,
            quality_score,
        }
    }

    pub fn subcarrier_count(&self) -> usize {
        self.amplitude.len()
    }
}

/// CSI frame aggregating multiple antenna streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsiFrame {
    pub timestamp: Timestamp,
    pub antenna_config: AntennaConfig,
    pub bandwidth: ChannelBandwidth,
    pub band: FrequencyBand,

    /// CSI data per antenna pair
    pub streams: Vec<SanitizedCsi>,

    /// Combined quality score across all streams
    pub frame_quality: f64,
}

impl CsiFrame {
    pub fn new(
        timestamp: Timestamp,
        antenna_config: AntennaConfig,
        bandwidth: ChannelBandwidth,
        band: FrequencyBand,
    ) -> Self {
        Self {
            timestamp,
            antenna_config,
            bandwidth,
            band,
            streams: Vec::with_capacity(antenna_config.total_streams()),
            frame_quality: 0.0,
        }
    }

    pub fn add_stream(&mut self, stream: SanitizedCsi) {
        self.streams.push(stream);
        self.update_quality();
    }

    fn update_quality(&mut self) {
        if self.streams.is_empty() {
            self.frame_quality = 0.0;
        } else {
            self.frame_quality =
                self.streams.iter().map(|s| s.quality_score).sum::<f64>() / self.streams.len() as f64;
        }
    }

    /// Get averaged amplitude across all antenna streams
    pub fn mean_amplitude(&self) -> Vec<f64> {
        if self.streams.is_empty() {
            return Vec::new();
        }

        let n_subcarriers = self.streams[0].amplitude.len();
        let n_streams = self.streams.len() as f64;

        let mut result = vec![0.0; n_subcarriers];
        for stream in &self.streams {
            for (i, &amp) in stream.amplitude.iter().enumerate() {
                result[i] += amp / n_streams;
            }
        }
        result
    }
}

/// Hardware configuration for CSI capture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsiHardwareConfig {
    /// WiFi NIC identifier
    pub interface: String,

    /// Channel number
    pub channel: u8,

    /// Bandwidth setting
    pub bandwidth: ChannelBandwidth,

    /// Frequency band
    pub band: FrequencyBand,

    /// Antenna configuration
    pub antenna_config: AntennaConfig,

    /// Packet capture rate (packets per second)
    pub capture_rate: u32,

    /// Center frequency in MHz
    pub center_freq_mhz: u32,
}

impl Default for CsiHardwareConfig {
    fn default() -> Self {
        Self {
            interface: "wlan0".to_string(),
            channel: 149, // 5GHz channel
            bandwidth: ChannelBandwidth::Bw160MHz,
            band: FrequencyBand::Band5GHz,
            antenna_config: AntennaConfig::ax210_default(),
            capture_rate: 1000,
            center_freq_mhz: 5745,
        }
    }
}

impl CsiHardwareConfig {
    /// Configuration for Intel AX210 in 6GHz band
    pub fn ax210_6ghz() -> Self {
        Self {
            interface: "wlan0".to_string(),
            channel: 5, // 6GHz UNII-5
            bandwidth: ChannelBandwidth::Bw160MHz,
            band: FrequencyBand::Band6GHz,
            antenna_config: AntennaConfig::new(2, 2),
            capture_rate: 1000,
            center_freq_mhz: 5975,
        }
    }

    /// Calculate wavelength in meters for Doppler computation
    pub fn wavelength_m(&self) -> f64 {
        // c = λf → λ = c/f
        const SPEED_OF_LIGHT: f64 = 299_792_458.0; // m/s
        SPEED_OF_LIGHT / (self.center_freq_mhz as f64 * 1e6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csi_packet_amplitudes() {
        let csi = vec![
            Complex::new(3.0, 4.0),  // |z| = 5
            Complex::new(0.0, 1.0),  // |z| = 1
            Complex::new(1.0, 0.0),  // |z| = 1
        ];

        let packet = CsiPacket::new(0, 0, 3, csi);
        let amps = packet.amplitudes();

        assert!((amps[0] - 5.0).abs() < 1e-10);
        assert!((amps[1] - 1.0).abs() < 1e-10);
        assert!((amps[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wavelength_calculation() {
        let config = CsiHardwareConfig::default();
        let wavelength = config.wavelength_m();
        // 5745 MHz → ~5.2cm wavelength
        assert!(wavelength > 0.05 && wavelength < 0.06);
    }

    #[test]
    fn test_wifi6e_subcarriers() {
        assert_eq!(ChannelBandwidth::Bw160MHz.usable_subcarriers(), 1992);
    }
}
