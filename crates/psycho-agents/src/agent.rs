//! Base agent trait and common types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Result type for agent operations
pub type AgentResult<T> = Result<T, AgentError>;

/// Agent error types
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("LLM inference error: {0}")]
    LlmError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Prompt template error: {0}")]
    PromptError(String),

    #[error("Timeout after {0}ms")]
    Timeout(u64),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Agent not initialized")]
    NotInitialized,
}

/// Base trait for all agents
#[async_trait]
pub trait Agent: Send + Sync {
    /// Agent name/identifier
    fn name(&self) -> &str;

    /// Process input and generate output
    async fn process(&self, input: &str) -> AgentResult<String>;

    /// Optional: validate input before processing
    fn validate_input(&self, input: &str) -> AgentResult<()> {
        if input.trim().is_empty() {
            Err(AgentError::InvalidInput("Empty input".to_string()))
        } else {
            Ok(())
        }
    }

    /// Optional: post-process output
    fn post_process(&self, output: String) -> AgentResult<String> {
        Ok(output)
    }
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Model to use (e.g., "llama3-70b", "gpt-4")
    pub model: String,
    /// Temperature for generation (0.0-1.0)
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Top-p sampling
    pub top_p: f32,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "llama3-70b".to_string(),
            temperature: 0.7,
            max_tokens: 512,
            top_p: 0.9,
            timeout_ms: 30_000,
        }
    }
}

/// Agent response with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// Agent that generated the response
    pub agent_name: String,
    /// Generated content
    pub content: String,
    /// Tokens used
    pub tokens_used: usize,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Model used
    pub model: String,
}

impl AgentResponse {
    pub fn new(agent_name: String, content: String) -> Self {
        Self {
            agent_name,
            content,
            tokens_used: 0,
            generation_time_ms: 0,
            model: "unknown".to_string(),
        }
    }
}

impl fmt::Display for AgentResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ({} tokens, {}ms)",
            self.agent_name, self.content, self.tokens_used, self.generation_time_ms
        )
    }
}

/// Chain-of-Thought reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoTStep {
    pub step_number: usize,
    pub reasoning: String,
    pub conclusion: String,
}

/// Chain-of-Thought response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoTResponse {
    pub steps: Vec<CoTStep>,
    pub final_answer: String,
}

impl CoTResponse {
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        for step in &self.steps {
            result.push_str(&format!(
                "Step {}: {}\nConclusion: {}\n\n",
                step.step_number, step.reasoning, step.conclusion
            ));
        }

        result.push_str(&format!("Final Answer: {}", self.final_answer));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.model, "llama3-70b");
        assert_eq!(config.temperature, 0.7);
    }

    #[test]
    fn test_agent_response_display() {
        let response = AgentResponse {
            agent_name: "TestAgent".to_string(),
            content: "Hello world".to_string(),
            tokens_used: 10,
            generation_time_ms: 150,
            model: "test-model".to_string(),
        };

        let display = format!("{}", response);
        assert!(display.contains("TestAgent"));
        assert!(display.contains("Hello world"));
    }
}
