//! Network client for WebTransport and WebSocket communication.

use gloo_net::websocket::{futures::WebSocket, Message};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use psycho_core::{DensePoseFrame, SubjectId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerMessage {
    DensePoseUpdate(DensePoseFrame),
    TrajectoryUpdate { subject_id: SubjectId, points: Vec<(f64, f64, f64)> },
    OceanUpdate { subject_id: SubjectId, scores: [f64; 5] },
    IntelligenceUpdate { session_id: String, data: String },
}

pub struct NetworkClient {
    ws: Option<WebSocket>,
}

impl NetworkClient {
    pub fn new() -> Self {
        Self { ws: None }
    }

    pub async fn connect(&mut self, url: &str) -> Result<(), String> {
        let ws = WebSocket::open(url).map_err(|e| e.to_string())?;
        self.ws = Some(ws);
        Ok(())
    }

    pub async fn recv(&mut self) -> Option<ServerMessage> {
        if let Some(ws) = &mut self.ws {
            if let Some(msg) = ws.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        serde_json::from_str(&text).ok()
                    }
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    pub async fn send(&mut self, msg: ServerMessage) -> Result<(), String> {
        if let Some(ws) = &mut self.ws {
            let text = serde_json::to_string(&msg).map_err(|e| e.to_string())?;
            ws.send(Message::Text(text)).await.map_err(|e| e.to_string())?;
            Ok(())
        } else {
            Err("Not connected".to_string())
        }
    }
}

impl Default for NetworkClient {
    fn default() -> Self {
        Self::new()
    }
}
