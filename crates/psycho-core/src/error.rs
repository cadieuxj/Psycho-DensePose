//! Error types for the Psycho-DensePose system.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("CSI processing error: {0}")]
    CsiProcessing(String),

    #[error("Phase unwrapping failed: discontinuity at index {index}")]
    PhaseUnwrap { index: usize },

    #[error("Invalid antenna configuration: expected {expected}, got {actual}")]
    AntennaConfig { expected: usize, actual: usize },

    #[error("Subcarrier count mismatch: expected {expected}, got {actual}")]
    SubcarrierMismatch { expected: u16, actual: u16 },

    #[error("DensePose inference error: {0}")]
    DensePoseInference(String),

    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Kinematic computation error: {0}")]
    Kinematics(String),

    #[error("Trajectory association failed: {0}")]
    TrajectoryAssociation(String),

    #[error("Privacy constraint violation: {0}")]
    PrivacyViolation(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Insufficient data: need {required} samples, have {available}")]
    InsufficientData { required: usize, available: usize },

    #[error("Timeout after {duration_ms}ms")]
    Timeout { duration_ms: u64 },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}
