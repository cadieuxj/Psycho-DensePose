//! Multi-agent orchestrator coordinating the three-agent pipeline.

use psycho_core::{SessionId, SubjectId, Timestamp};
use psycho_lma::PsychometricProfile;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::copywriter::{CopywriterAgent, CopywriterInput, CopywriterOutput};
use crate::interpreter::{InterpreterAgent, InterpreterInput, InterpreterOutput};
use crate::strategist::{QuestionnaireData, SalesContext, StrategistAgent, StrategistInput, StrategistOutput};
use crate::agent::{AgentConfig, AgentResult};

/// Complete sales intelligence result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalesIntelligence {
    /// Subject and session IDs
    pub subject_id: SubjectId,
    pub session_id: SessionId,
    pub timestamp: Timestamp,

    /// Agent 1 output
    pub behavioral_interpretation: InterpreterOutput,

    /// Agent 2 output
    pub sales_strategy: StrategistOutput,

    /// Agent 3 output
    pub opening_hooks: CopywriterOutput,

    /// Complete psychometric profile
    pub psychometric_profile: PsychometricProfile,
}

/// Multi-agent orchestrator
pub struct AgentOrchestrator {
    interpreter: InterpreterAgent,
    strategist: StrategistAgent,
    copywriter: CopywriterAgent,

    /// Cache of generated intelligence
    cache: RwLock<HashMap<SessionId, SalesIntelligence>>,
}

impl AgentOrchestrator {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            interpreter: InterpreterAgent::new(config.clone()),
            strategist: StrategistAgent::new(config.clone()),
            copywriter: CopywriterAgent::new(config),
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Generate complete sales intelligence for a session
    pub async fn generate_intelligence(
        &self,
        subject_id: SubjectId,
        session_id: SessionId,
        profile: &PsychometricProfile,
        questionnaire: Option<QuestionnaireData>,
        context: SalesContext,
    ) -> AgentResult<SalesIntelligence> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&session_id) {
                return Ok(cached.clone());
            }
        }

        tracing::info!(
            "Generating sales intelligence for session {:?}",
            session_id
        );

        // Agent 1: Interpreter
        let interpreter_input = InterpreterInput {
            ocean: profile.ocean.clone(),
            effort: psycho_lma::EffortProfile {
                space: psycho_lma::SpaceEffort {
                    per: 0.7,
                    directness: 0.7,
                    curvature: 0.3,
                    direction_changes: 5,
                },
                time: psycho_lma::TimeEffort {
                    rms_jerk: 10.0,
                    max_jerk: 20.0,
                    jerk_variance: 5.0,
                    suddenness: 0.3,
                },
                weight: psycho_lma::WeightEffort {
                    mean_acceleration: 2.0,
                    max_acceleration: 5.0,
                    acceleration_variance: 1.0,
                    strength: 0.4,
                },
                flow: psycho_lma::FlowEffort {
                    smoothness: 0.7,
                    velocity_consistency: 0.8,
                    stop_frequency: 0.5,
                    boundness: 0.6,
                },
            },
            hesitation_pattern: psycho_lma::HesitationPattern::Evaluative,
            kinesphere_volume: 0.8,
            path_entropy: 0.5,
        };

        let behavioral_interpretation = self.interpreter.interpret(&interpreter_input).await?;

        tracing::debug!("Interpreter: {}", behavioral_interpretation.summary);

        // Agent 2: Strategist
        let strategist_input = StrategistInput {
            behavioral_summary: behavioral_interpretation.clone(),
            questionnaire,
            context,
        };

        let sales_strategy = self.strategist.strategize(&strategist_input).await?;

        tracing::debug!("Strategist: {:?}", sales_strategy.approach);

        // Agent 3: Copywriter
        let copywriter_input = CopywriterInput {
            strategy: sales_strategy.clone(),
            customer_profile_summary: behavioral_interpretation.summary.clone(),
        };

        let opening_hooks = self.copywriter.generate_hooks(&copywriter_input).await?;

        tracing::debug!(
            "Copywriter: Generated {} hooks",
            opening_hooks.hooks.len()
        );

        // Package result
        let intelligence = SalesIntelligence {
            subject_id,
            session_id,
            timestamp: Timestamp::now(),
            behavioral_interpretation,
            sales_strategy,
            opening_hooks,
            psychometric_profile: profile.clone(),
        };

        // Cache result
        {
            let mut cache = self.cache.write().await;
            cache.insert(session_id, intelligence.clone());
        }

        Ok(intelligence)
    }

    /// Get cached intelligence for a session
    pub async fn get_intelligence(&self, session_id: SessionId) -> Option<SalesIntelligence> {
        let cache = self.cache.read().await;
        cache.get(&session_id).cloned()
    }

    /// Clear cache entry
    pub async fn clear_session(&self, session_id: SessionId) {
        let mut cache = self.cache.write().await;
        cache.remove(&session_id);
    }

    /// Clear all cached intelligence
    pub async fn clear_all(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        CacheStats {
            cached_sessions: cache.len(),
        }
    }
}

impl Default for AgentOrchestrator {
    fn default() -> Self {
        Self::new(AgentConfig::default())
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub cached_sessions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_core::OceanScores;

    #[tokio::test]
    async fn test_orchestrator_pipeline() {
        let orchestrator = AgentOrchestrator::default();

        let subject_id = SubjectId::new();
        let session_id = SessionId::new();

        let profile = PsychometricProfile {
            ocean: psycho_core::OceanPrediction::new(
                OceanScores::new(0.5, 0.7, 0.4, 0.6, 0.6),
                10,
                30.0,
                [0.8, 0.8, 0.7, 0.8, 0.8],
            ),
            persona: psycho_core::SalesPersona::Analyst,
            behavioral_insights: vec!["Test insight".to_string()],
            recommended_approach: "Consultative".to_string(),
        };

        let context = SalesContext {
            viewing_vehicles: vec!["Honda CR-V".to_string()],
            inventory_highlights: vec!["2024 models".to_string()],
            active_promotions: vec!["0% financing".to_string()],
            time_spent_mins: 10,
        };

        let intelligence = orchestrator
            .generate_intelligence(subject_id, session_id, &profile, None, context)
            .await
            .unwrap();

        // Verify pipeline completed
        assert!(!intelligence.behavioral_interpretation.summary.is_empty());
        assert!(!intelligence.sales_strategy.themes.is_empty());
        assert_eq!(intelligence.opening_hooks.hooks.len(), 3);

        // Verify caching
        let cached = orchestrator.get_intelligence(session_id).await;
        assert!(cached.is_some());
    }
}
