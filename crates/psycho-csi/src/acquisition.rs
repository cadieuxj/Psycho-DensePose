//! CSI acquisition interfaces for WiFi hardware.
//!
//! This module provides abstractions for capturing CSI data from various
//! WiFi hardware platforms, with primary support for:
//!
//! - Intel AX210 (WiFi 6E) via PicoScenes middleware
//! - ESP32-S3 mesh nodes for distributed sensing

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};

use psycho_core::{Error, Result};

use crate::packet::{CsiHardwareConfig, CsiPacket};

/// Trait for CSI acquisition backends
#[async_trait]
pub trait CsiAcquisition: Send + Sync {
    /// Start CSI capture
    async fn start(&mut self) -> Result<()>;

    /// Stop CSI capture
    async fn stop(&mut self) -> Result<()>;

    /// Check if capture is active
    fn is_running(&self) -> bool;

    /// Get hardware configuration
    fn config(&self) -> &CsiHardwareConfig;

    /// Receive next CSI packet (blocking)
    async fn recv(&mut self) -> Result<CsiPacket>;

    /// Try to receive CSI packet (non-blocking)
    fn try_recv(&mut self) -> Option<CsiPacket>;
}

/// PicoScenes-based CSI acquisition for Intel AX210
///
/// PicoScenes is a middleware that enables CSI extraction from
/// commercial WiFi hardware. This implementation interfaces with
/// PicoScenes via Unix domain socket or shared memory.
pub struct PicoScenesAcquisition {
    config: CsiHardwareConfig,
    is_running: bool,
    rx: Option<mpsc::Receiver<CsiPacket>>,
    socket_path: String,
}

impl PicoScenesAcquisition {
    pub fn new(config: CsiHardwareConfig) -> Self {
        Self {
            config,
            is_running: false,
            rx: None,
            socket_path: "/tmp/picoscenes.sock".to_string(),
        }
    }

    pub fn with_socket_path(mut self, path: &str) -> Self {
        self.socket_path = path.to_string();
        self
    }

    /// Parse PicoScenes binary packet format
    fn parse_packet(data: &[u8]) -> Result<CsiPacket> {
        // PicoScenes packet format (simplified):
        // [0-7]:   timestamp (i64)
        // [8]:     antenna_id (u8)
        // [9-10]:  subcarrier_count (u16)
        // [11]:    rssi (i8)
        // [12]:    noise_floor (i8)
        // [13-20]: tx_mac_hash (u64)
        // [21-24]: sequence_number (u32)
        // [25..]:  CSI data (complex f64 pairs)

        if data.len() < 25 {
            return Err(Error::CsiProcessing("Packet too short".into()));
        }

        let timestamp = i64::from_le_bytes(data[0..8].try_into().unwrap());
        let antenna_id = data[8];
        let subcarrier_count = u16::from_le_bytes(data[9..11].try_into().unwrap());
        let rssi = data[11] as i8;
        let noise_floor = data[12] as i8;
        let tx_mac_hash = u64::from_le_bytes(data[13..21].try_into().unwrap());
        let sequence_number = u32::from_le_bytes(data[21..25].try_into().unwrap());

        let csi_data_start = 25;
        let expected_csi_bytes = subcarrier_count as usize * 16; // 2 f64 per subcarrier

        if data.len() < csi_data_start + expected_csi_bytes {
            return Err(Error::CsiProcessing("Insufficient CSI data".into()));
        }

        let mut csi_matrix = Vec::with_capacity(subcarrier_count as usize);
        for i in 0..subcarrier_count as usize {
            let offset = csi_data_start + i * 16;
            let real = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            let imag = f64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
            csi_matrix.push(num_complex::Complex::new(real, imag));
        }

        Ok(CsiPacket {
            timestamp,
            antenna_id,
            subcarrier_count,
            csi_matrix,
            rssi,
            noise_floor,
            tx_mac_hash,
            sequence_number,
        })
    }
}

#[async_trait]
impl CsiAcquisition for PicoScenesAcquisition {
    async fn start(&mut self) -> Result<()> {
        if self.is_running {
            return Ok(());
        }

        // In production, this would connect to PicoScenes socket
        // For now, we create a mock channel
        let (tx, rx) = mpsc::channel(1000);
        self.rx = Some(rx);
        self.is_running = true;

        // Spawn mock data generator for testing
        let config = self.config.clone();
        tokio::spawn(async move {
            let mut seq = 0u32;
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

                let subcarrier_count = config.bandwidth.usable_subcarriers();
                let csi_matrix: Vec<num_complex::Complex<f64>> = (0..subcarrier_count)
                    .map(|i| {
                        let phase = (i as f64 * 0.01) + (seq as f64 * 0.001);
                        num_complex::Complex::from_polar(1.0, phase)
                    })
                    .collect();

                let packet = CsiPacket {
                    timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                    antenna_id: 0,
                    subcarrier_count,
                    csi_matrix,
                    rssi: -45,
                    noise_floor: -90,
                    tx_mac_hash: 0,
                    sequence_number: seq,
                };

                if tx.send(packet).await.is_err() {
                    break;
                }

                seq = seq.wrapping_add(1);
            }
        });

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        self.is_running = false;
        self.rx = None;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.is_running
    }

    fn config(&self) -> &CsiHardwareConfig {
        &self.config
    }

    async fn recv(&mut self) -> Result<CsiPacket> {
        match &mut self.rx {
            Some(rx) => rx
                .recv()
                .await
                .ok_or_else(|| Error::CsiProcessing("Channel closed".into())),
            None => Err(Error::CsiProcessing("Acquisition not started".into())),
        }
    }

    fn try_recv(&mut self) -> Option<CsiPacket> {
        self.rx.as_mut()?.try_recv().ok()
    }
}

/// ESP32-S3 mesh node CSI acquisition
///
/// Supports distributed CSI collection from multiple ESP32-S3 nodes
/// deployed throughout the dealership.
pub struct Esp32MeshAcquisition {
    config: CsiHardwareConfig,
    is_running: bool,
    rx: Option<mpsc::Receiver<CsiPacket>>,
    /// Node addresses for the mesh network
    nodes: Vec<String>,
    /// UDP port for CSI data
    udp_port: u16,
}

impl Esp32MeshAcquisition {
    pub fn new(config: CsiHardwareConfig, nodes: Vec<String>) -> Self {
        Self {
            config,
            is_running: false,
            rx: None,
            nodes,
            udp_port: 5555,
        }
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.udp_port = port;
        self
    }

    /// Parse ESP32 CSI packet format
    fn parse_esp32_packet(data: &[u8], node_id: u8) -> Result<CsiPacket> {
        // ESP32 CSI format (simplified):
        // [0-7]:   timestamp (i64, microseconds)
        // [8-9]:   subcarrier_count (u16, typically 64 for 20MHz)
        // [10]:    rssi (i8)
        // [11..]:  CSI data (complex i16 pairs, scaled)

        if data.len() < 11 {
            return Err(Error::CsiProcessing("ESP32 packet too short".into()));
        }

        let timestamp = i64::from_le_bytes(data[0..8].try_into().unwrap()) * 1000; // Convert to nanos
        let subcarrier_count = u16::from_le_bytes(data[8..10].try_into().unwrap());
        let rssi = data[10] as i8;

        let csi_data_start = 11;
        let expected_csi_bytes = subcarrier_count as usize * 4; // 2 i16 per subcarrier

        if data.len() < csi_data_start + expected_csi_bytes {
            return Err(Error::CsiProcessing("Insufficient ESP32 CSI data".into()));
        }

        let mut csi_matrix = Vec::with_capacity(subcarrier_count as usize);
        for i in 0..subcarrier_count as usize {
            let offset = csi_data_start + i * 4;
            let real = i16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as f64
                / 32768.0;
            let imag = i16::from_le_bytes(data[offset + 2..offset + 4].try_into().unwrap()) as f64
                / 32768.0;
            csi_matrix.push(num_complex::Complex::new(real, imag));
        }

        Ok(CsiPacket {
            timestamp,
            antenna_id: node_id,
            subcarrier_count,
            csi_matrix,
            rssi,
            noise_floor: -95, // ESP32 typical
            tx_mac_hash: 0,
            sequence_number: 0,
        })
    }
}

#[async_trait]
impl CsiAcquisition for Esp32MeshAcquisition {
    async fn start(&mut self) -> Result<()> {
        if self.is_running {
            return Ok(());
        }

        let (tx, rx) = mpsc::channel(5000);
        self.rx = Some(rx);
        self.is_running = true;

        // In production, this would bind UDP socket and listen for ESP32 packets
        // For now, create mock data
        let num_nodes = self.nodes.len().max(1) as u8;
        tokio::spawn(async move {
            let mut seq = 0u32;
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

                for node_id in 0..num_nodes {
                    let subcarrier_count = 64u16; // ESP32 with 20MHz
                    let csi_matrix: Vec<num_complex::Complex<f64>> = (0..subcarrier_count)
                        .map(|i| {
                            let phase = (i as f64 * 0.02) + (seq as f64 * 0.002)
                                + (node_id as f64 * 0.1);
                            num_complex::Complex::from_polar(0.8, phase)
                        })
                        .collect();

                    let packet = CsiPacket {
                        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                        antenna_id: node_id,
                        subcarrier_count,
                        csi_matrix,
                        rssi: -55,
                        noise_floor: -95,
                        tx_mac_hash: node_id as u64,
                        sequence_number: seq,
                    };

                    if tx.send(packet).await.is_err() {
                        return;
                    }
                }

                seq = seq.wrapping_add(1);
            }
        });

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        self.is_running = false;
        self.rx = None;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.is_running
    }

    fn config(&self) -> &CsiHardwareConfig {
        &self.config
    }

    async fn recv(&mut self) -> Result<CsiPacket> {
        match &mut self.rx {
            Some(rx) => rx
                .recv()
                .await
                .ok_or_else(|| Error::CsiProcessing("Channel closed".into())),
            None => Err(Error::CsiProcessing("Acquisition not started".into())),
        }
    }

    fn try_recv(&mut self) -> Option<CsiPacket> {
        self.rx.as_mut()?.try_recv().ok()
    }
}

/// Multiplexed CSI acquisition from multiple sources
pub struct MultiplexedAcquisition {
    sources: Vec<Box<dyn CsiAcquisition>>,
    rx: Option<mpsc::Receiver<CsiPacket>>,
    is_running: bool,
}

impl MultiplexedAcquisition {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            rx: None,
            is_running: false,
        }
    }

    pub fn add_source<T: CsiAcquisition + 'static>(mut self, source: T) -> Self {
        self.sources.push(Box::new(source));
        self
    }
}

impl Default for MultiplexedAcquisition {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CsiAcquisition for MultiplexedAcquisition {
    async fn start(&mut self) -> Result<()> {
        if self.is_running {
            return Ok(());
        }

        for source in &mut self.sources {
            source.start().await?;
        }

        self.is_running = true;
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        for source in &mut self.sources {
            source.stop().await?;
        }
        self.is_running = false;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.is_running
    }

    fn config(&self) -> &CsiHardwareConfig {
        self.sources
            .first()
            .map(|s| s.config())
            .unwrap_or(&CsiHardwareConfig::default())
    }

    async fn recv(&mut self) -> Result<CsiPacket> {
        // Round-robin across sources
        for source in &mut self.sources {
            if let Some(packet) = source.try_recv() {
                return Ok(packet);
            }
        }

        // If no packets ready, wait on first source
        if let Some(source) = self.sources.first_mut() {
            source.recv().await
        } else {
            Err(Error::CsiProcessing("No sources configured".into()))
        }
    }

    fn try_recv(&mut self) -> Option<CsiPacket> {
        for source in &mut self.sources {
            if let Some(packet) = source.try_recv() {
                return Some(packet);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_core::ChannelBandwidth;

    #[tokio::test]
    async fn test_picoscenes_acquisition() {
        let config = CsiHardwareConfig::default();
        let mut acq = PicoScenesAcquisition::new(config);

        acq.start().await.unwrap();
        assert!(acq.is_running());

        // Should receive packets
        let packet = acq.recv().await.unwrap();
        assert!(packet.is_valid());
        assert_eq!(
            packet.subcarrier_count,
            ChannelBandwidth::Bw160MHz.usable_subcarriers()
        );

        acq.stop().await.unwrap();
        assert!(!acq.is_running());
    }

    #[tokio::test]
    async fn test_esp32_acquisition() {
        let config = CsiHardwareConfig::default();
        let nodes = vec!["192.168.1.100".to_string(), "192.168.1.101".to_string()];
        let mut acq = Esp32MeshAcquisition::new(config, nodes);

        acq.start().await.unwrap();

        let packet = acq.recv().await.unwrap();
        assert!(packet.is_valid());

        acq.stop().await.unwrap();
    }
}
