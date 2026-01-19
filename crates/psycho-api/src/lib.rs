//! # Psycho-API
//!
//! WebTransport API server with Axum for real-time bi-directional communication.
//!
//! ## Endpoints
//!
//! ### HTTP/3 (WebTransport)
//! - `/wt/csi` - Real-time CSI data stream
//! - `/wt/densepose` - Real-time pose estimation stream
//! - `/wt/intelligence` - Sales intelligence updates
//!
//! ### REST (HTTP/2)
//! - `POST /api/v1/sessions` - Create new session
//! - `GET /api/v1/sessions/{id}` - Get session data
//! - `POST /api/v1/questionnaire` - Submit questionnaire
//! - `GET /api/v1/intelligence/{session_id}` - Get sales intelligence
//! - `GET /api/v1/subjects/{id}/trajectory` - Get subject trajectory
//! - `GET /api/v1/health` - Health check

pub mod config;
pub mod handlers;
pub mod middleware;
pub mod routes;
pub mod server;
pub mod state;
pub mod webtransport;

pub use config::*;
pub use server::*;
pub use state::*;
