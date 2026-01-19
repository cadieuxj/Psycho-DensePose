//! Agent 1: Interpreter - Behavioral analysis from movement data.

use async_trait::async_trait;
use psycho_core::{OceanPrediction, SalesPersona};
use psycho_lma::{EffortProfile, HesitationPattern};
use serde::{Deserialize, Serialize};

use crate::agent::{Agent, AgentConfig, AgentError, AgentResponse, AgentResult};
use crate::prompts::{format_interpreter_input, INTERPRETER_SYSTEM_PROMPT};

/// Interpreter Agent input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpreterInput {
    pub ocean: OceanPrediction,
    pub effort: EffortProfile,
    pub hesitation_pattern: HesitationPattern,
    pub kinesphere_volume: f64,
    pub path_entropy: f64,
}

/// Interpreter Agent output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpreterOutput {
    /// Emotional state assessment
    pub emotional_state: String,
    /// Key behavioral patterns
    pub behavioral_patterns: Vec<String>,
    /// Identified needs or concerns
    pub needs: Vec<String>,
    /// Recommended engagement approach
    pub engagement_approach: String,
    /// Full behavioral summary
    pub summary: String,
}

/// Interpreter Agent implementation
pub struct InterpreterAgent {
    config: AgentConfig,
    name: String,
}

impl InterpreterAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            name: "Interpreter".to_string(),
        }
    }

    /// Process input and generate behavioral interpretation
    pub async fn interpret(&self, input: &InterpreterInput) -> AgentResult<InterpreterOutput> {
        // Format the input prompt
        let ocean_summary = self.format_ocean_summary(&input.ocean);
        let lma_summary = self.format_lma_summary(&input.effort);
        let hesitation_summary = self.format_hesitation_summary(&input.hesitation_pattern);
        let movement_summary =
            self.format_movement_summary(input.kinesphere_volume, input.path_entropy);

        let prompt = format_interpreter_input(
            &ocean_summary,
            &lma_summary,
            &hesitation_summary,
            &movement_summary,
        );

        // In production, this would call actual LLM
        // For now, generate structured output based on input
        let response = self.generate_mock_response(input).await?;

        Ok(response)
    }

    fn format_ocean_summary(&self, ocean: &OceanPrediction) -> String {
        let mut lines = vec![format!("Confidence: {:?}", ocean.confidence)];

        for trait_name in ocean.scores.high_traits() {
            lines.push(format!("High {}: {}", trait_name.name(), trait_name.high_description()));
        }

        for trait_name in ocean.scores.low_traits() {
            lines.push(format!("Low {}: {}", trait_name.name(), trait_name.low_description()));
        }

        lines.push(format!(
            "Sales Persona: {:?}",
            SalesPersona::from_ocean(&ocean.scores)
        ));

        lines.join("\n")
    }

    fn format_lma_summary(&self, effort: &EffortProfile) -> String {
        format!(
            "Space: {} (directness: {:.2})\nTime: {} (suddenness: {:.2})\nWeight: {} (strength: {:.2})\nFlow: {} (boundness: {:.2})",
            effort.space.description(),
            effort.space.directness,
            effort.time.description(),
            effort.time.suddenness,
            effort.weight.description(),
            effort.weight.strength,
            effort.flow.description(),
            effort.flow.boundness
        )
    }

    fn format_hesitation_summary(&self, pattern: &HesitationPattern) -> String {
        format!("{}\nRecommended Response: {}", pattern.description(), pattern.recommended_response())
    }

    fn format_movement_summary(&self, volume: f64, entropy: f64) -> String {
        let kinesphere = if volume > 0.5 { "Expansive" } else { "Constrained" };
        let predictability = if entropy > 0.6 { "Exploratory" } else { "Predictable" };

        format!(
            "Kinesphere: {} ({:.2} m³)\nMovement Pattern: {} (entropy: {:.2})",
            kinesphere, volume, predictability, entropy
        )
    }

    /// Mock response generator (would be replaced with actual LLM call)
    async fn generate_mock_response(
        &self,
        input: &InterpreterInput,
    ) -> AgentResult<InterpreterOutput> {
        // Analyze emotional state based on neuroticism and hesitation
        let emotional_state = if input.ocean.scores.neuroticism > 0.6 {
            if input.hesitation_pattern == HesitationPattern::Overwhelmed {
                "Anxious and overwhelmed".to_string()
            } else {
                "Cautious and uncertain".to_string()
            }
        } else if input.ocean.scores.extraversion > 0.6 {
            "Confident and engaged".to_string()
        } else {
            "Calm and thoughtful".to_string()
        };

        // Identify behavioral patterns
        let mut behavioral_patterns = Vec::new();

        if input.effort.flow.boundness > 0.6 {
            behavioral_patterns.push("Controlled, careful movements".to_string());
        }
        if input.effort.time.suddenness > 0.6 {
            behavioral_patterns.push("Quick, responsive reactions".to_string());
        }
        if input.path_entropy > 0.6 {
            behavioral_patterns.push("Exploratory browsing behavior".to_string());
        }
        if input.kinesphere_volume < 0.3 {
            behavioral_patterns.push("Reserved body language".to_string());
        }

        // Identify needs
        let mut needs = Vec::new();

        if input.ocean.scores.neuroticism > 0.6 {
            needs.push("Reassurance and security".to_string());
        }
        if input.ocean.scores.conscientiousness > 0.6 {
            needs.push("Detailed information and data".to_string());
        }
        if input.ocean.scores.openness > 0.6 {
            needs.push("Innovation and unique features".to_string());
        }

        // Engagement approach
        let persona = SalesPersona::from_ocean(&input.ocean.scores);
        let engagement_approach = persona.recommended_approach().to_string();

        // Generate summary
        let summary = format!(
            "Customer appears {} with {} patterns. {}. Recommended approach: {}",
            emotional_state.to_lowercase(),
            if behavioral_patterns.is_empty() {
                "typical"
            } else {
                behavioral_patterns[0].as_str()
            },
            if needs.is_empty() {
                "Standard engagement appropriate"
            } else {
                &format!("Prioritize {}", needs.join(" and "))
            },
            engagement_approach
        );

        Ok(InterpreterOutput {
            emotional_state,
            behavioral_patterns,
            needs,
            engagement_approach,
            summary,
        })
    }
}

impl Default for InterpreterAgent {
    fn default() -> Self {
        Self::new(AgentConfig::default())
    }
}

#[async_trait]
impl Agent for InterpreterAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn process(&self, input: &str) -> AgentResult<String> {
        // In production, parse input JSON to InterpreterInput
        // For now, return mock response
        Ok("Behavioral interpretation: Customer shows cautious behavior with high attention to detail.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_core::{OceanScores, PredictionConfidence};
    use psycho_lma::{FlowEffort, SpaceEffort, TimeEffort, WeightEffort};

    fn create_test_input() -> InterpreterInput {
        InterpreterInput {
            ocean: OceanPrediction::new(
                OceanScores::new(0.5, 0.7, 0.4, 0.6, 0.7),
                10,
                30.0,
                [0.8, 0.8, 0.7, 0.8, 0.8],
            ),
            effort: EffortProfile {
                space: SpaceEffort {
                    per: 0.6,
                    directness: 0.6,
                    curvature: 0.3,
                    direction_changes: 5,
                },
                time: TimeEffort {
                    rms_jerk: 10.0,
                    max_jerk: 20.0,
                    jerk_variance: 5.0,
                    suddenness: 0.3,
                },
                weight: WeightEffort {
                    mean_acceleration: 2.0,
                    max_acceleration: 5.0,
                    acceleration_variance: 1.0,
                    strength: 0.4,
                },
                flow: FlowEffort {
                    smoothness: 0.7,
                    velocity_consistency: 0.8,
                    stop_frequency: 0.5,
                    boundness: 0.6,
                },
            },
            hesitation_pattern: HesitationPattern::Evaluative,
            kinesphere_volume: 0.8,
            path_entropy: 0.5,
        }
    }

    #[tokio::test]
    async fn test_interpreter_agent() {
        let agent = InterpreterAgent::default();
        let input = create_test_input();

        let output = agent.interpret(&input).await.unwrap();

        assert!(!output.emotional_state.is_empty());
        assert!(!output.summary.is_empty());
        assert!(!output.engagement_approach.is_empty());
    }
}
