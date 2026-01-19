//! # Psycho-CSI
//!
//! Channel State Information (CSI) processing pipeline for WiFi-based sensing.
//!
//! This crate implements the Signal Engine that processes raw CSI data from
//! WiFi 6E (802.11ax) hardware, extracting motion signatures for DensePose estimation.
//!
//! ## CSI Background
//!
//! Channel State Information describes how a signal propagates through the environment.
//! For WiFi 6E with 160MHz bandwidth, we get 1992 usable OFDM subcarriers, each
//! providing amplitude and phase information that encodes environmental reflections.
//!
//! The raw CSI matrix H(f, t) is a complex-valued tensor:
//! - f: frequency (subcarrier index)
//! - t: time (packet timestamp)
//! - Complex value: amplitude |H| and phase ∠H
//!
//! ## Pipeline Stages
//!
//! 1. **Acquisition**: Capture raw CSI from hardware (Intel AX210 via PicoScenes)
//! 2. **Sanitization**: Remove hardware artifacts (phase unwrapping, SFO removal)
//! 3. **Filtering**: Apply Hampel filter for outlier removal
//! 4. **Doppler**: Extract motion signatures via FFT across time
//! 5. **Feature Extraction**: Compute statistical features for ML models

pub mod acquisition;
pub mod doppler;
pub mod filtering;
pub mod packet;
pub mod pipeline;
pub mod sanitizer;

pub use acquisition::*;
pub use doppler::*;
pub use filtering::*;
pub use packet::*;
pub use pipeline::*;
pub use sanitizer::*;
