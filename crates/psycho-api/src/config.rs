//! API server configuration.

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;

/// Complete server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// HTTP server configuration
    pub http: HttpConfig,

    /// WebTransport configuration
    pub webtransport: WebTransportConfig,

    /// Database configuration
    pub database: DatabaseConfig,

    /// TLS certificate configuration
    pub tls: TlsConfig,

    /// CORS configuration
    pub cors: CorsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// HTTP/2 bind address
    pub bind_addr: SocketAddr,

    /// Request timeout (seconds)
    pub timeout_secs: u64,

    /// Maximum request body size (bytes)
    pub max_body_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebTransportConfig {
    /// HTTP/3 bind address
    pub bind_addr: SocketAddr,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Connection idle timeout (seconds)
    pub idle_timeout_secs: u64,

    /// Maximum datagram size (bytes)
    pub max_datagram_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// QuestDB HTTP endpoint
    pub questdb_host: String,

    /// QuestDB ILP (Influx Line Protocol) port
    pub questdb_ilp_port: u16,

    /// Connection pool size
    pub pool_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Path to TLS certificate
    pub cert_path: PathBuf,

    /// Path to TLS private key
    pub key_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    /// Allowed origins
    pub allowed_origins: Vec<String>,

    /// Allowed methods
    pub allowed_methods: Vec<String>,

    /// Allowed headers
    pub allowed_headers: Vec<String>,

    /// Max age (seconds)
    pub max_age_secs: u64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            http: HttpConfig {
                bind_addr: "0.0.0.0:8080".parse().unwrap(),
                timeout_secs: 30,
                max_body_size: 1024 * 1024, // 1MB
            },
            webtransport: WebTransportConfig {
                bind_addr: "0.0.0.0:4433".parse().unwrap(),
                max_connections: 1000,
                idle_timeout_secs: 60,
                max_datagram_size: 65535,
            },
            database: DatabaseConfig {
                questdb_host: "localhost".to_string(),
                questdb_ilp_port: 9009,
                pool_size: 10,
            },
            tls: TlsConfig {
                cert_path: PathBuf::from("certs/server.crt"),
                key_path: PathBuf::from("certs/server.key"),
            },
            cors: CorsConfig {
                allowed_origins: vec!["http://localhost:3000".to_string()],
                allowed_methods: vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()],
                allowed_headers: vec!["Content-Type".to_string(), "Authorization".to_string()],
                max_age_secs: 3600,
            },
        }
    }
}

impl ApiConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::with_prefix("PSYCHO"))
            .build()?;

        settings.try_deserialize()
    }

    /// Load from environment variables
    pub fn from_env() -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::Environment::with_prefix("PSYCHO"))
            .build()?;

        settings.try_deserialize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ApiConfig::default();
        assert_eq!(config.http.bind_addr.port(), 8080);
        assert_eq!(config.webtransport.bind_addr.port(), 4433);
    }
}
