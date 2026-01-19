//! Agent 3: Copywriter - Generate personalized opening hooks with Chain-of-Thought.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::{Agent, AgentConfig, AgentError, AgentResult, CoTResponse, CoTStep};
use crate::prompts::{format_copywriter_input, parse_cot_response, COPYWRITER_SYSTEM_PROMPT};
use crate::strategist::StrategistOutput;

/// Copywriter Agent input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopywriterInput {
    pub strategy: StrategistOutput,
    pub customer_profile_summary: String,
}

/// Generated opening hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpeningHook {
    /// Hook number (1-3)
    pub number: usize,
    /// Hook text (2-3 sentences)
    pub text: String,
    /// Hook type (question, statement, observation)
    pub hook_type: HookType,
    /// Confidence score [0-1]
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HookType {
    Question,
    Statement,
    Observation,
}

/// Copywriter Agent output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopywriterOutput {
    /// Three opening hooks
    pub hooks: Vec<OpeningHook>,
    /// Chain-of-Thought reasoning
    pub cot_reasoning: Vec<CoTStep>,
}

/// Copywriter Agent implementation
pub struct CopywriterAgent {
    config: AgentConfig,
    name: String,
}

impl CopywriterAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            name: "Copywriter".to_string(),
        }
    }

    /// Generate opening hooks using Chain-of-Thought
    pub async fn generate_hooks(&self, input: &CopywriterInput) -> AgentResult<CopywriterOutput> {
        // In production, call LLM with CoT prompting
        // For now, generate structured hooks based on strategy
        let output = self.generate_hooks_internal(input).await?;

        Ok(output)
    }

    async fn generate_hooks_internal(&self, input: &CopywriterInput) -> AgentResult<CopywriterOutput> {
        // Simulate CoT reasoning steps
        let cot_reasoning = vec![
            CoTStep {
                step_number: 1,
                reasoning: format!(
                    "Customer's primary need: {}. They appear {}.",
                    input.strategy.themes.get(0).unwrap_or(&"value".to_string()),
                    input.customer_profile_summary.split('.').next().unwrap_or("interested")
                ),
                conclusion: "Focus on their main concern first".to_string(),
            },
            CoTStep {
                step_number: 2,
                reasoning: format!(
                    "Communication style: {}. Approach: {:?}",
                    input.strategy.communication_style.split('.').next().unwrap_or("Professional"),
                    input.strategy.approach
                ),
                conclusion: "Match their communication preference".to_string(),
            },
            CoTStep {
                step_number: 3,
                reasoning: format!(
                    "Key themes: {}. Features: {}",
                    input.strategy.themes.join(", "),
                    input.strategy.features_to_highlight.get(0).unwrap_or(&"quality".to_string())
                ),
                conclusion: "Weave themes into natural conversation".to_string(),
            },
        ];

        // Generate three distinct hooks
        let hooks = self.craft_hooks(input);

        Ok(CopywriterOutput {
            hooks,
            cot_reasoning,
        })
    }

    fn craft_hooks(&self, input: &CopywriterInput) -> Vec<OpeningHook> {
        let primary_theme = input.strategy.themes.first()
            .map(|s| s.as_str())
            .unwrap_or("quality");

        let primary_feature = input.strategy.features_to_highlight.first()
            .map(|s| s.as_str())
            .unwrap_or("features");

        let approach = input.strategy.approach;

        vec![
            // Hook 1: Question-based
            OpeningHook {
                number: 1,
                text: self.generate_question_hook(primary_theme, &input.customer_profile_summary),
                hook_type: HookType::Question,
                confidence: 0.85,
            },
            // Hook 2: Observation-based
            OpeningHook {
                number: 2,
                text: self.generate_observation_hook(primary_feature, primary_theme),
                hook_type: HookType::Observation,
                confidence: 0.82,
            },
            // Hook 3: Statement-based
            OpeningHook {
                number: 3,
                text: self.generate_statement_hook(&input.strategy),
                hook_type: HookType::Statement,
                confidence: 0.80,
            },
        ]
    }

    fn generate_question_hook(&self, theme: &str, profile: &str) -> String {
        let theme_lower = theme.to_lowercase();

        if theme_lower.contains("safety") {
            format!(
                "I noticed you're interested in family vehicles. Have you had a chance to see how our advanced safety features work? \
                Many families tell us the collision avoidance system gives them real peace of mind."
            )
        } else if theme_lower.contains("performance") {
            format!(
                "Are you looking for something with a bit more power? \
                This model's performance specs are impressive, but what's really special is how it handles. \
                Would you like to experience that firsthand?"
            )
        } else if theme_lower.contains("efficiency") {
            format!(
                "Fuel efficiency is so important these days, isn't it? \
                This model's hybrid system can save you significant money over time. \
                Have you calculated what you're currently spending on gas?"
            )
        } else if theme_lower.contains("technology") || theme_lower.contains("innovation") {
            format!(
                "How important is technology integration for you? \
                This model's infotainment system is incredibly intuitive. \
                Do you use Apple CarPlay or Android Auto?"
            )
        } else {
            format!(
                "What's most important to you in your next vehicle? \
                I'd love to show you how this model delivers exceptional value. \
                Is this your first time considering this brand?"
            )
        }
    }

    fn generate_observation_hook(&self, feature: &str, theme: &str) -> String {
        let feature_lower = feature.to_lowercase();

        if feature_lower.contains("safety") || feature_lower.contains("assistance") {
            format!(
                "I see you're taking your time looking at the safety features—that's smart. \
                This model has one of the highest safety ratings in its class. \
                Let me show you how the adaptive cruise control works in real traffic."
            )
        } else if feature_lower.contains("infotainment") || feature_lower.contains("technology") {
            format!(
                "You seem interested in the tech features. \
                The interface is incredibly user-friendly—even my least tech-savvy customers master it quickly. \
                Want to connect your phone and try it out?"
            )
        } else if feature_lower.contains("performance") || feature_lower.contains("engine") {
            format!(
                "You're checking out the performance specs—I can tell you appreciate power. \
                The numbers are impressive, but the real story is how it delivers that power so smoothly. \
                The test drive really shows it off."
            )
        } else {
            format!(
                "I notice you've been looking at this model for a while. \
                It's one of our most popular for good reason—reliability and value. \
                What questions can I answer for you?"
            )
        }
    }

    fn generate_statement_hook(&self, strategy: &StrategistOutput) -> String {
        let communication = strategy.communication_style.to_lowercase();

        if communication.contains("calm") || communication.contains("reassuring") {
            format!(
                "Take your time looking around—there's no rush. \
                When you're ready, I'm happy to answer any questions or walk you through the features. \
                We want you to feel completely confident in your decision."
            )
        } else if communication.contains("enthusiastic") || communication.contains("energy") {
            format!(
                "This is such an exciting model—it just came in and it's already generating a lot of interest. \
                The combination of style, performance, and value is really hitting the mark. \
                What drew you to this one specifically?"
            )
        } else if communication.contains("fact-based") || communication.contains("precise") {
            format!(
                "This model consistently ranks at the top for reliability and resale value. \
                The data backs it up—Consumer Reports rates it as a best buy. \
                I can share the detailed comparison sheet if you'd like to see the numbers."
            )
        } else {
            format!(
                "Welcome! I'm here if you have any questions. \
                This model is a great choice—it offers an excellent balance of features, reliability, and value. \
                Feel free to sit in it and get a feel for the space."
            )
        }
    }
}

impl Default for CopywriterAgent {
    fn default() -> Self {
        Self::new(AgentConfig::default())
    }
}

#[async_trait]
impl Agent for CopywriterAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn process(&self, input: &str) -> AgentResult<String> {
        Ok("Generated 3 personalized opening hooks.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategist::{SalesApproach, StrategistOutput};

    fn create_test_input() -> CopywriterInput {
        CopywriterInput {
            strategy: StrategistOutput {
                approach: SalesApproach::Consultative,
                themes: vec!["Safety & Reliability".to_string()],
                features_to_highlight: vec!["Advanced Driver Assistance Systems".to_string()],
                objections: vec!["Price sensitivity".to_string()],
                communication_style: "Use calm, reassuring tone.".to_string(),
                strategy_summary: "Consultative approach".to_string(),
            },
            customer_profile_summary: "Customer appears cautious with high attention to safety.".to_string(),
        }
    }

    #[tokio::test]
    async fn test_copywriter_agent() {
        let agent = CopywriterAgent::default();
        let input = create_test_input();

        let output = agent.generate_hooks(&input).await.unwrap();

        assert_eq!(output.hooks.len(), 3);
        assert!(!output.cot_reasoning.is_empty());
        assert!(output.hooks.iter().all(|h| !h.text.is_empty()));
    }

    #[test]
    fn test_hook_types() {
        let agent = CopywriterAgent::default();
        let input = create_test_input();

        let hooks = agent.craft_hooks(&input);

        // Should have different hook types
        let types: Vec<_> = hooks.iter().map(|h| h.hook_type).collect();
        assert!(types.contains(&HookType::Question));
        assert!(types.contains(&HookType::Observation));
        assert!(types.contains(&HookType::Statement));
    }
}
