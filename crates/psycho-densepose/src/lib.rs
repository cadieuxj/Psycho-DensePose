//! # Psycho-DensePose
//!
//! WiFi CSI to DensePose modality translation using deep neural networks.
//!
//! This crate implements the neural architecture that maps 1D RF signals
//! (CSI amplitude and phase) to 2D UV coordinates and body part segmentation,
//! enabling human pose estimation without cameras.
//!
//! ## Architecture Overview
//!
//! The modality translation network consists of:
//!
//! 1. **Dual-Branch Encoder**: Separate processing of amplitude |H| and phase ∠H
//! 2. **ResNet-50 Backbone**: Feature extraction with ROI Align
//! 3. **Transformer Attention**: Isolate human Doppler signatures
//! 4. **Keypoint Head**: 17 skeletal joint predictions
//! 5. **DensePose Head**: 24 body part + UV regression
//!
//! ## Cross-Modal Knowledge Distillation
//!
//! The network is trained using a Teacher-Student paradigm:
//! - Teacher: Pre-trained camera-based DensePose (provides supervision)
//! - Student: WiFi-based network (learns to match teacher output)
//!
//! Loss: L_distill = KL(P_teacher || P_student) + L_keypoint + L_uv

pub mod backbone;
pub mod distillation;
pub mod encoder;
pub mod heads;
pub mod inference;
pub mod model;
pub mod roi;

pub use backbone::*;
pub use distillation::*;
pub use encoder::*;
pub use heads::*;
pub use inference::*;
pub use model::*;
pub use roi::*;
