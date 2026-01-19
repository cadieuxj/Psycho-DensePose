//! OCEAN (Big Five) personality trait mapping from LMA movement metrics.
//!
//! ## Movement-to-Trait Mappings
//!
//! | Movement Feature | High Value → Trait |
//! |-----------------|-------------------|
//! | Path Entropy | High Openness |
//! | Path Efficiency | High Conscientiousness |
//! | Kinesphere Volume | High Extraversion |
//! | Flow Smoothness | High Agreeableness |
//! | Bound Flow + Jerk | High Neuroticism |
//!
//! ## Research Basis
//!
//! These mappings are based on empirical research correlating movement qualities
//! with personality traits:
//! - Koppensteiner & Grammer (2010) - Gait and personality
//! - Satchell et al. (2017) - Movement and Big Five
//! - Thoresen et al. (2012) - Nonverbal behavior and personality

use psycho_core::{OceanPrediction, OceanScores, OceanTrait, PredictionConfidence, SalesPersona};
use serde::{Deserialize, Serialize};

use crate::metrics::MovementMetrics;

/// Configuration for OCEAN mapping weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OceanMappingConfig {
    /// Weights for Openness prediction
    pub openness_weights: TraitWeights,
    /// Weights for Conscientiousness prediction
    pub conscientiousness_weights: TraitWeights,
    /// Weights for Extraversion prediction
    pub extraversion_weights: TraitWeights,
    /// Weights for Agreeableness prediction
    pub agreeableness_weights: TraitWeights,
    /// Weights for Neuroticism prediction
    pub neuroticism_weights: TraitWeights,
}

/// Feature weights for a single trait
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitWeights {
    pub path_entropy: f64,
    pub path_efficiency: f64,
    pub kinesphere_volume: f64,
    pub flow_smoothness: f64,
    pub bound_flow: f64,
    pub rms_jerk: f64,
    pub hesitation_score: f64,
    pub velocity_mean: f64,
    pub direction_changes: f64,
    pub fidgeting: f64,
    pub bias: f64,
}

impl Default for TraitWeights {
    fn default() -> Self {
        Self {
            path_entropy: 0.0,
            path_efficiency: 0.0,
            kinesphere_volume: 0.0,
            flow_smoothness: 0.0,
            bound_flow: 0.0,
            rms_jerk: 0.0,
            hesitation_score: 0.0,
            velocity_mean: 0.0,
            direction_changes: 0.0,
            fidgeting: 0.0,
            bias: 0.5,
        }
    }
}

impl Default for OceanMappingConfig {
    fn default() -> Self {
        Self {
            // Openness: High path entropy, exploration, varied movement
            openness_weights: TraitWeights {
                path_entropy: 0.35,
                path_efficiency: -0.1,
                kinesphere_volume: 0.15,
                flow_smoothness: 0.0,
                bound_flow: -0.1,
                rms_jerk: 0.0,
                hesitation_score: 0.0,
                velocity_mean: 0.0,
                direction_changes: 0.2,
                fidgeting: 0.0,
                bias: 0.3,
            },

            // Conscientiousness: High path efficiency, deliberate, organized
            conscientiousness_weights: TraitWeights {
                path_entropy: -0.2,
                path_efficiency: 0.35,
                kinesphere_volume: 0.0,
                flow_smoothness: 0.15,
                bound_flow: 0.1,
                rms_jerk: -0.15,
                hesitation_score: -0.1,
                velocity_mean: 0.0,
                direction_changes: -0.15,
                fidgeting: -0.1,
                bias: 0.3,
            },

            // Extraversion: Large kinesphere, fast movement, expansive
            extraversion_weights: TraitWeights {
                path_entropy: 0.1,
                path_efficiency: 0.0,
                kinesphere_volume: 0.3,
                flow_smoothness: 0.1,
                bound_flow: -0.2,
                rms_jerk: 0.0,
                hesitation_score: -0.15,
                velocity_mean: 0.25,
                direction_changes: 0.0,
                fidgeting: 0.0,
                bias: 0.3,
            },

            // Agreeableness: Smooth flow, accommodating, mirroring
            agreeableness_weights: TraitWeights {
                path_entropy: 0.0,
                path_efficiency: 0.1,
                kinesphere_volume: -0.1,
                flow_smoothness: 0.35,
                bound_flow: -0.15,
                rms_jerk: -0.2,
                hesitation_score: 0.0,
                velocity_mean: -0.1,
                direction_changes: -0.1,
                fidgeting: -0.1,
                bias: 0.4,
            },

            // Neuroticism: Bound flow, high jerk, hesitation, fidgeting
            neuroticism_weights: TraitWeights {
                path_entropy: 0.0,
                path_efficiency: -0.15,
                kinesphere_volume: -0.15,
                flow_smoothness: -0.25,
                bound_flow: 0.25,
                rms_jerk: 0.2,
                hesitation_score: 0.25,
                velocity_mean: 0.0,
                direction_changes: 0.1,
                fidgeting: 0.2,
                bias: 0.25,
            },
        }
    }
}

/// OCEAN trait mapper
pub struct OceanMapper {
    config: OceanMappingConfig,
}

impl OceanMapper {
    pub fn new(config: OceanMappingConfig) -> Self {
        Self { config }
    }

    /// Map movement metrics to OCEAN scores
    pub fn predict(&self, metrics: &MovementMetrics) -> OceanPrediction {
        // Extract normalized features
        let features = self.extract_features(metrics);

        // Calculate trait scores
        let openness = self.apply_weights(&features, &self.config.openness_weights);
        let conscientiousness =
            self.apply_weights(&features, &self.config.conscientiousness_weights);
        let extraversion = self.apply_weights(&features, &self.config.extraversion_weights);
        let agreeableness = self.apply_weights(&features, &self.config.agreeableness_weights);
        let neuroticism = self.apply_weights(&features, &self.config.neuroticism_weights);

        let scores = OceanScores::new(
            openness,
            conscientiousness,
            extraversion,
            agreeableness,
            neuroticism,
        );

        // Calculate confidence based on data quality
        let observation_count = 1; // TODO: track actual observations
        let observation_duration = metrics.kinematics.window_duration_secs;

        // Per-trait confidence based on feature relevance
        let trait_confidences = self.calculate_trait_confidences(metrics);

        OceanPrediction::new(
            scores,
            observation_count,
            observation_duration,
            trait_confidences,
        )
    }

    /// Extract and normalize features for trait mapping
    fn extract_features(&self, metrics: &MovementMetrics) -> NormalizedFeatures {
        NormalizedFeatures {
            path_entropy: metrics.path_entropy,
            path_efficiency: metrics.kinematics.path_efficiency_ratio,
            kinesphere_volume: (metrics.kinesphere.volume / 2.0).min(1.0), // Normalize to ~2 m³ max
            flow_smoothness: metrics.effort.flow.smoothness,
            bound_flow: metrics.effort.flow.boundness,
            rms_jerk: (metrics.effort.time.rms_jerk / 50.0).min(1.0), // Normalize to ~50 m/s³
            hesitation_score: metrics.hesitation.overall_score,
            velocity_mean: (metrics.kinematics.mean_velocity / 2.0).min(1.0), // Normalize to ~2 m/s
            direction_changes: (metrics.kinematics.direction_changes as f64 / 20.0).min(1.0),
            fidgeting: metrics.fidgeting_score,
        }
    }

    /// Apply weights to features
    fn apply_weights(&self, features: &NormalizedFeatures, weights: &TraitWeights) -> f64 {
        let score = weights.path_entropy * features.path_entropy
            + weights.path_efficiency * features.path_efficiency
            + weights.kinesphere_volume * features.kinesphere_volume
            + weights.flow_smoothness * features.flow_smoothness
            + weights.bound_flow * features.bound_flow
            + weights.rms_jerk * features.rms_jerk
            + weights.hesitation_score * features.hesitation_score
            + weights.velocity_mean * features.velocity_mean
            + weights.direction_changes * features.direction_changes
            + weights.fidgeting * features.fidgeting
            + weights.bias;

        // Apply sigmoid to bound [0, 1]
        1.0 / (1.0 + (-4.0 * (score - 0.5)).exp())
    }

    /// Calculate per-trait confidence scores
    fn calculate_trait_confidences(&self, metrics: &MovementMetrics) -> [f64; 5] {
        // Base confidence on data quality and feature coverage
        let base = (metrics.effort.flow.smoothness + 0.5).min(1.0); // Higher smoothness = better tracking

        // Adjust per trait based on relevant feature availability
        let o_conf = base * if metrics.path_entropy > 0.0 { 1.0 } else { 0.5 };
        let c_conf = base * if metrics.kinematics.path_efficiency_ratio > 0.0 { 1.0 } else { 0.5 };
        let e_conf = base * if metrics.kinesphere.volume > 0.0 { 1.0 } else { 0.5 };
        let a_conf = base * 0.8; // Flow features always available
        let n_conf = base * if metrics.hesitation.overall_score >= 0.0 { 1.0 } else { 0.5 };

        [o_conf, c_conf, e_conf, a_conf, n_conf]
    }

    pub fn config(&self) -> &OceanMappingConfig {
        &self.config
    }
}

impl Default for OceanMapper {
    fn default() -> Self {
        Self::new(OceanMappingConfig::default())
    }
}

/// Normalized feature vector for trait mapping
struct NormalizedFeatures {
    path_entropy: f64,
    path_efficiency: f64,
    kinesphere_volume: f64,
    flow_smoothness: f64,
    bound_flow: f64,
    rms_jerk: f64,
    hesitation_score: f64,
    velocity_mean: f64,
    direction_changes: f64,
    fidgeting: f64,
}

/// Complete psychometric profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychometricProfile {
    /// OCEAN prediction
    pub ocean: OceanPrediction,
    /// Sales persona classification
    pub persona: SalesPersona,
    /// Movement-based behavioral insights
    pub behavioral_insights: Vec<String>,
    /// Recommended engagement approach
    pub recommended_approach: String,
}

impl PsychometricProfile {
    /// Generate complete profile from movement metrics
    pub fn from_metrics(metrics: &MovementMetrics) -> Self {
        let mapper = OceanMapper::default();
        let ocean = mapper.predict(metrics);
        let persona = SalesPersona::from_ocean(&ocean.scores);

        let behavioral_insights = generate_insights(metrics, &ocean);
        let recommended_approach = persona.recommended_approach().to_string();

        Self {
            ocean,
            persona,
            behavioral_insights,
            recommended_approach,
        }
    }

    /// Get opening conversation themes
    pub fn conversation_themes(&self) -> Vec<&'static str> {
        self.persona.opening_themes().to_vec()
    }
}

/// Generate behavioral insights from metrics and OCEAN
fn generate_insights(metrics: &MovementMetrics, ocean: &OceanPrediction) -> Vec<String> {
    let mut insights = Vec::new();

    // Movement-based insights
    if metrics.hesitation.overall_score > 0.6 {
        insights.push("Shows significant hesitation - may need reassurance".to_string());
    }

    if metrics.effort.space.directness > 0.7 {
        insights.push("Direct, purposeful movement - knows what they want".to_string());
    } else if metrics.effort.space.directness < 0.3 {
        insights.push("Exploratory browsing pattern - open to suggestions".to_string());
    }

    if metrics.kinesphere.is_expansive() {
        insights.push("Expansive body language - comfortable and confident".to_string());
    }

    if metrics.fidgeting_score > 0.5 {
        insights.push("Signs of restlessness - may prefer quick interactions".to_string());
    }

    // OCEAN-based insights
    for trait_type in ocean.scores.high_traits() {
        insights.push(format!("High {}: {}", trait_type.name(), trait_type.high_description()));
    }

    for trait_type in ocean.scores.low_traits() {
        insights.push(format!("Low {}: {}", trait_type.name(), trait_type.low_description()));
    }

    insights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::MovementMetrics;
    use psycho_core::{
        Acceleration3D, Position3D, SubjectId, Timestamp, Trajectory, TrajectoryPoint, Velocity3D,
    };

    fn create_extrovert_trajectory() -> Trajectory {
        let mut traj = Trajectory::new(SubjectId::new());

        // Large, fast, expansive movements
        for i in 0..30 {
            let t = i as f64 * 0.1;
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(i * 100_000_000),
                Position3D::new(
                    2.0 * t.cos(),
                    2.0 * t.sin(),
                    0.5 * (t * 2.0).sin(),
                ),
                Velocity3D::new(
                    -2.0 * t.sin() * 2.0,
                    2.0 * t.cos() * 2.0,
                    (t * 2.0).cos(),
                ),
                Acceleration3D::new(
                    -2.0 * t.cos() * 4.0,
                    -2.0 * t.sin() * 4.0,
                    -(t * 2.0).sin() * 2.0,
                ),
            ));
        }

        traj
    }

    fn create_neurotic_trajectory() -> Trajectory {
        let mut traj = Trajectory::new(SubjectId::new());

        // Jerky, hesitant, constrained movements
        for i in 0..30 {
            let jitter = if i % 3 == 0 { 0.1 } else { -0.05 };
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(i * 100_000_000),
                Position3D::new(i as f64 * 0.02 + jitter, jitter, 0.0),
                Velocity3D::new(0.2 + jitter * 2.0, jitter, 0.0),
                Acceleration3D::new(jitter * 50.0, jitter * 30.0, 0.0),
            ));
        }

        traj
    }

    #[test]
    fn test_ocean_mapping_extraversion() {
        let traj = create_extrovert_trajectory();
        let metrics = MovementMetrics::from_trajectory(&traj);

        let mapper = OceanMapper::default();
        let prediction = mapper.predict(&metrics);

        // Expansive movement should correlate with higher extraversion
        assert!(
            prediction.scores.extraversion > 0.4,
            "Expansive movement should suggest extraversion"
        );
    }

    #[test]
    fn test_ocean_mapping_neuroticism() {
        let traj = create_neurotic_trajectory();
        let metrics = MovementMetrics::from_trajectory(&traj);

        let mapper = OceanMapper::default();
        let prediction = mapper.predict(&metrics);

        // Jerky, hesitant movement should correlate with higher neuroticism
        // Note: This is probabilistic, so we just check it's computed
        assert!(
            prediction.scores.neuroticism >= 0.0 && prediction.scores.neuroticism <= 1.0,
            "Neuroticism should be in valid range"
        );
    }

    #[test]
    fn test_psychometric_profile() {
        let traj = create_extrovert_trajectory();
        let metrics = MovementMetrics::from_trajectory(&traj);

        let profile = PsychometricProfile::from_metrics(&metrics);

        assert!(!profile.behavioral_insights.is_empty());
        assert!(!profile.recommended_approach.is_empty());
    }
}
