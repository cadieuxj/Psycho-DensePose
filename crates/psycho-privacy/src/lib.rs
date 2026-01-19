//! # Psycho-Privacy
//!
//! Privacy-preserving tracking and association system.
//!
//! ## Privacy Principles
//!
//! 1. **Edge Processing**: Raw CSI processed locally, never stored
//! 2. **Abstract Features**: Only kinematic/LMA features stored, not raw signals
//! 3. **Anonymous IDs**: Subject IDs are ephemeral UUIDs, unlinked to identity
//! 4. **Data Minimization**: Auto-expire data after configurable retention period
//! 5. **Explicit Consent**: Kiosk interaction implies consent for session tracking
//!
//! ## Spatio-Temporal Handshake
//!
//! Links anonymous WiFi trajectories to kiosk sessions by matching:
//! - Position: Subject near kiosk (<2m distance)
//! - Orientation: Facing the kiosk screen (±45°)
//! - Timing: Deceleration coinciding with screen interaction
//! - Velocity: Nearly stopped (v < 0.2 m/s) during T_click window

pub mod handshake;
pub mod jpda;
pub mod privacy;
pub mod retention;

pub use handshake::*;
pub use jpda::*;
pub use privacy::*;
pub use retention::*;
