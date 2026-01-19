//! Agent 2: Strategist - Sales strategy from behavioral insights + questionnaire.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::{Agent, AgentConfig, AgentError, AgentResult};
use crate::interpreter::InterpreterOutput;
use crate::prompts::{format_strategist_input, STRATEGIST_SYSTEM_PROMPT};

/// Customer questionnaire data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionnaireData {
    /// Vehicle type interest (sedan, SUV, truck, etc.)
    pub vehicle_type: Option<String>,
    /// Budget range
    pub budget_range: Option<String>,
    /// Primary use case (commute, family, work, etc.)
    pub primary_use: Option<String>,
    /// Top priorities (safety, performance, efficiency, luxury, etc.)
    pub priorities: Vec<String>,
    /// Trade-in vehicle
    pub has_trade_in: bool,
    /// Financing preference
    pub financing_preference: Option<String>,
}

/// Sales context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalesContext {
    /// Current vehicle(s) customer is viewing
    pub viewing_vehicles: Vec<String>,
    /// Dealership inventory highlights
    pub inventory_highlights: Vec<String>,
    /// Current promotions
    pub active_promotions: Vec<String>,
    /// Time spent in dealership (minutes)
    pub time_spent_mins: u32,
}

/// Strategist Agent input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategistInput {
    pub behavioral_summary: InterpreterOutput,
    pub questionnaire: Option<QuestionnaireData>,
    pub context: SalesContext,
}

/// Sales approach type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SalesApproach {
    /// Build relationship first, understand needs
    Consultative,
    /// Lead with data, specs, comparisons
    DataDriven,
    /// Focus on emotional appeal and lifestyle
    EmotionalAppeal,
    /// Quick, efficient, transactional
    Transactional,
    /// Emphasize value and practicality
    ValueFocused,
}

impl SalesApproach {
    pub fn description(&self) -> &'static str {
        match self {
            SalesApproach::Consultative => "Consultative: Build trust through active listening and tailored recommendations",
            SalesApproach::DataDriven => "Data-Driven: Present objective information, specs, and comparisons",
            SalesApproach::EmotionalAppeal => "Emotional Appeal: Connect vehicle to lifestyle and aspirations",
            SalesApproach::Transactional => "Transactional: Efficient, straightforward, focus on close",
            SalesApproach::ValueFocused => "Value-Focused: Emphasize ROI, reliability, and long-term benefits",
        }
    }
}

/// Strategist Agent output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategistOutput {
    /// Recommended sales approach
    pub approach: SalesApproach,
    /// Key themes to emphasize (ordered by priority)
    pub themes: Vec<String>,
    /// Specific features to highlight
    pub features_to_highlight: Vec<String>,
    /// Potential objections to address
    pub objections: Vec<String>,
    /// Communication style notes
    pub communication_style: String,
    /// Complete strategy summary
    pub strategy_summary: String,
}

/// Strategist Agent implementation
pub struct StrategistAgent {
    config: AgentConfig,
    name: String,
}

impl StrategistAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            name: "Strategist".to_string(),
        }
    }

    /// Generate sales strategy
    pub async fn strategize(&self, input: &StrategistInput) -> AgentResult<StrategistOutput> {
        // In production, call LLM with formatted prompt
        // For now, generate structured strategy based on inputs
        let strategy = self.generate_strategy(input).await?;

        Ok(strategy)
    }

    async fn generate_strategy(&self, input: &StrategistInput) -> AgentResult<StrategistOutput> {
        // Determine approach based on behavioral summary
        let approach = self.select_approach(&input.behavioral_summary);

        // Identify key themes
        let mut themes = Vec::new();

        // Add themes from behavioral needs
        for need in &input.behavioral_summary.needs {
            if need.contains("security") || need.contains("Reassurance") {
                themes.push("Safety & Reliability".to_string());
                themes.push("Peace of Mind".to_string());
            }
            if need.contains("information") || need.contains("data") {
                themes.push("Performance Specs".to_string());
                themes.push("Technology Features".to_string());
            }
            if need.contains("innovation") || need.contains("unique") {
                themes.push("Cutting-Edge Design".to_string());
                themes.push("Innovation".to_string());
            }
        }

        // Add themes from questionnaire priorities
        if let Some(ref questionnaire) = input.questionnaire {
            for priority in &questionnaire.priorities {
                let priority_lower = priority.to_lowercase();
                if priority_lower.contains("safety") && !themes.contains(&"Safety & Reliability".to_string()) {
                    themes.push("Safety & Reliability".to_string());
                }
                if priority_lower.contains("performance") && !themes.contains(&"Performance Specs".to_string()) {
                    themes.push("Performance Specs".to_string());
                }
                if priority_lower.contains("efficiency") || priority_lower.contains("fuel") {
                    themes.push("Fuel Efficiency & Savings".to_string());
                }
                if priority_lower.contains("luxury") || priority_lower.contains("comfort") {
                    themes.push("Comfort & Luxury".to_string());
                }
            }
        }

        // Default themes if none identified
        if themes.is_empty() {
            themes = vec![
                "Quality & Value".to_string(),
                "Customer Satisfaction".to_string(),
            ];
        }

        // Select features to highlight
        let features_to_highlight = self.select_features(&themes, &input.questionnaire);

        // Predict objections
        let objections = self.predict_objections(&input.behavioral_summary, &input.questionnaire);

        // Communication style
        let communication_style = self.determine_communication_style(&input.behavioral_summary);

        // Generate summary
        let strategy_summary = format!(
            "{} {} Emphasize: {}. Key features: {}.",
            approach.description(),
            communication_style,
            themes.join(", "),
            features_to_highlight.get(0).unwrap_or(&"value proposition".to_string())
        );

        Ok(StrategistOutput {
            approach,
            themes,
            features_to_highlight,
            objections,
            communication_style,
            strategy_summary,
        })
    }

    fn select_approach(&self, behavioral: &InterpreterOutput) -> SalesApproach {
        let emotional_state = behavioral.emotional_state.to_lowercase();

        if emotional_state.contains("anxious") || emotional_state.contains("cautious") {
            SalesApproach::Consultative
        } else if behavioral.needs.iter().any(|n| n.contains("information") || n.contains("data")) {
            SalesApproach::DataDriven
        } else if emotional_state.contains("confident") || emotional_state.contains("engaged") {
            SalesApproach::EmotionalAppeal
        } else {
            SalesApproach::ValueFocused
        }
    }

    fn select_features(&self, themes: &[String], questionnaire: &Option<QuestionnaireData>) -> Vec<String> {
        let mut features = Vec::new();

        for theme in themes {
            let theme_lower = theme.to_lowercase();
            if theme_lower.contains("safety") {
                features.push("Advanced Driver Assistance Systems".to_string());
                features.push("5-Star Safety Rating".to_string());
            }
            if theme_lower.contains("performance") {
                features.push("Horsepower & Torque Specs".to_string());
                features.push("0-60 Time".to_string());
            }
            if theme_lower.contains("technology") || theme_lower.contains("innovation") {
                features.push("Infotainment System".to_string());
                features.push("Smartphone Integration".to_string());
            }
            if theme_lower.contains("efficiency") {
                features.push("MPG Ratings".to_string());
                features.push("Hybrid/Electric Options".to_string());
            }
        }

        if features.is_empty() {
            features = vec![
                "Warranty Coverage".to_string(),
                "Reliability Ratings".to_string(),
                "Resale Value".to_string(),
            ];
        }

        features.truncate(5); // Limit to top 5
        features
    }

    fn predict_objections(&self, behavioral: &InterpreterOutput, questionnaire: &Option<QuestionnaireData>) -> Vec<String> {
        let mut objections = Vec::new();

        // Common objections based on behavioral patterns
        if behavioral.emotional_state.to_lowercase().contains("cautious") {
            objections.push("May need more time to decide".to_string());
            objections.push("Wants to compare with competitors".to_string());
        }

        if behavioral.needs.iter().any(|n| n.contains("Reassurance")) {
            objections.push("Concerns about reliability".to_string());
            objections.push("Warranty coverage questions".to_string());
        }

        // Budget objections if questionnaire indicates
        if let Some(ref q) = questionnaire {
            if q.budget_range.is_some() {
                objections.push("Price sensitivity".to_string());
            }
            if q.has_trade_in {
                objections.push("Trade-in value expectations".to_string());
            }
        }

        if objections.is_empty() {
            objections = vec!["Standard price negotiation".to_string()];
        }

        objections
    }

    fn determine_communication_style(&self, behavioral: &InterpreterOutput) -> String {
        let emotional_state = behavioral.emotional_state.to_lowercase();

        if emotional_state.contains("anxious") || emotional_state.contains("overwhelmed") {
            "Use calm, reassuring tone. Speak slowly. Give space for questions.".to_string()
        } else if emotional_state.contains("confident") || emotional_state.contains("engaged") {
            "Match their energy. Be enthusiastic but not pushy. Keep pace brisk.".to_string()
        } else if behavioral.needs.iter().any(|n| n.contains("information")) {
            "Be precise and fact-based. Use data to support claims. Allow time for analysis.".to_string()
        } else {
            "Professional and friendly. Balanced pace. Active listening.".to_string()
        }
    }
}

impl Default for StrategistAgent {
    fn default() -> Self {
        Self::new(AgentConfig::default())
    }
}

#[async_trait]
impl Agent for StrategistAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn process(&self, input: &str) -> AgentResult<String> {
        Ok("Sales strategy: Consultative approach focusing on safety and reliability.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::InterpreterOutput;

    fn create_test_input() -> StrategistInput {
        StrategistInput {
            behavioral_summary: InterpreterOutput {
                emotional_state: "Cautious and uncertain".to_string(),
                behavioral_patterns: vec!["Controlled movements".to_string()],
                needs: vec!["Reassurance and security".to_string()],
                engagement_approach: "Consultative approach".to_string(),
                summary: "Customer appears cautious".to_string(),
            },
            questionnaire: Some(QuestionnaireData {
                vehicle_type: Some("SUV".to_string()),
                budget_range: Some("$30k-$40k".to_string()),
                primary_use: Some("Family".to_string()),
                priorities: vec!["Safety".to_string(), "Reliability".to_string()],
                has_trade_in: true,
                financing_preference: Some("Lease".to_string()),
            }),
            context: SalesContext {
                viewing_vehicles: vec!["Honda CR-V".to_string()],
                inventory_highlights: vec!["2024 models in stock".to_string()],
                active_promotions: vec!["0% APR financing".to_string()],
                time_spent_mins: 15,
            },
        }
    }

    #[tokio::test]
    async fn test_strategist_agent() {
        let agent = StrategistAgent::default();
        let input = create_test_input();

        let output = agent.strategize(&input).await.unwrap();

        assert_eq!(output.approach, SalesApproach::Consultative);
        assert!(!output.themes.is_empty());
        assert!(!output.features_to_highlight.is_empty());
    }
}
