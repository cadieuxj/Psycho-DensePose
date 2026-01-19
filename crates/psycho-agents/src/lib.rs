//! # Psycho-Agents
//!
//! Multi-agent system for generative sales intelligence.
//!
//! ## Architecture
//!
//! Three specialized agents work in sequence:
//!
//! 1. **Interpreter Agent**: Analyzes movement + OCEAN → behavioral summary
//! 2. **Strategist Agent**: Combines behavioral + questionnaire data → sales strategy
//! 3. **Copywriter Agent**: Generates personalized opening hooks using Chain-of-Thought
//!
//! ## Agent Flow
//!
//! ```text
//! LMA Scores + OCEAN
//!     ↓
//! [Agent 1: Interpreter]
//!     → "Customer shows hesitation and high neuroticism. Likely needs reassurance."
//!     ↓
//! [Agent 2: Strategist] ← Questionnaire Data
//!     → "Strategy: Consultative, Safety-focused. Emphasize reliability, warranty."
//!     ↓
//! [Agent 3: Copywriter]
//!     → 3 opening hooks tailored to strategy
//! ```

pub mod agent;
pub mod copywriter;
pub mod interpreter;
pub mod orchestrator;
pub mod strategist;
pub mod prompts;

pub use agent::*;
pub use copywriter::*;
pub use interpreter::*;
pub use orchestrator::*;
pub use strategist::*;
pub use prompts::*;
