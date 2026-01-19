//! # Psycho-Core
//!
//! Core types and utilities for the Psycho-DensePose WiFi-based
//! human pose estimation and psychometric profiling system.

pub mod error;
pub mod geometry;
pub mod kinematics;
pub mod ocean;
pub mod types;

pub use error::{Error, Result};
pub use geometry::*;
pub use kinematics::*;
pub use ocean::*;
pub use types::*;
