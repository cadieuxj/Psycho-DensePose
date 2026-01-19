//! # Psycho-LMA
//!
//! Laban Movement Analysis (LMA) implementation for extracting psychological
//! traits from human motion patterns.
//!
//! ## LMA Framework
//!
//! Rudolf Laban developed a system for describing and notating human movement
//! based on four main components:
//!
//! 1. **Body** - Which body parts are moving
//! 2. **Effort** - The quality/dynamics of movement
//! 3. **Shape** - How the body forms and changes in space
//! 4. **Space** - Where the body moves
//!
//! ## Effort Factors
//!
//! This crate focuses on the Effort component, which has four factors:
//!
//! - **Space**: Direct vs Indirect (focus of attention)
//! - **Time**: Sudden vs Sustained (urgency)
//! - **Weight**: Strong vs Light (impact)
//! - **Flow**: Bound vs Free (control)
//!
//! ## OCEAN Mapping
//!
//! LMA qualities are mapped to Big Five (OCEAN) personality traits:
//!
//! - High Path Entropy → High Openness
//! - Low Jerk, High PER → High Conscientiousness
//! - Large Kinesphere, Fast Pace → High Extraversion
//! - Smooth Flow, Mirroring → High Agreeableness
//! - Bound Flow, High Jerk, Hesitation → High Neuroticism

pub mod effort;
pub mod hesitation;
pub mod metrics;
pub mod ocean_mapper;
pub mod space;
pub mod analyzer;

pub use effort::*;
pub use hesitation::*;
pub use metrics::*;
pub use ocean_mapper::*;
pub use space::*;
pub use analyzer::*;
