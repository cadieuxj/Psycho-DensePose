//! Big Five (OCEAN) personality trait modeling and scoring.

use serde::{Deserialize, Serialize};

/// OCEAN (Big Five) personality trait scores
/// Each trait is scored on a scale of 0.0 to 1.0
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OceanScores {
    /// Openness to Experience: creativity, curiosity, intellectual interests
    /// Movement: High path entropy, exploratory behavior, varied trajectories
    pub openness: f64,

    /// Conscientiousness: organization, dependability, self-discipline
    /// Movement: High path efficiency, deliberate pace, minimal direction changes
    pub conscientiousness: f64,

    /// Extraversion: sociability, assertiveness, positive emotions
    /// Movement: Larger personal space usage, faster pace, more gestures
    pub extraversion: f64,

    /// Agreeableness: cooperation, trust, compliance
    /// Movement: Smooth flow, accommodating trajectories, mirroring behavior
    pub agreeableness: f64,

    /// Neuroticism: emotional instability, anxiety, moodiness
    /// Movement: Bound flow, high jerk, hesitation patterns, fidgeting
    pub neuroticism: f64,
}

impl OceanScores {
    pub fn new(
        openness: f64,
        conscientiousness: f64,
        extraversion: f64,
        agreeableness: f64,
        neuroticism: f64,
    ) -> Self {
        Self {
            openness: openness.clamp(0.0, 1.0),
            conscientiousness: conscientiousness.clamp(0.0, 1.0),
            extraversion: extraversion.clamp(0.0, 1.0),
            agreeableness: agreeableness.clamp(0.0, 1.0),
            neuroticism: neuroticism.clamp(0.0, 1.0),
        }
    }

    pub fn neutral() -> Self {
        Self::new(0.5, 0.5, 0.5, 0.5, 0.5)
    }

    /// Get the dominant trait (highest score)
    pub fn dominant_trait(&self) -> OceanTrait {
        let traits = [
            (OceanTrait::Openness, self.openness),
            (OceanTrait::Conscientiousness, self.conscientiousness),
            (OceanTrait::Extraversion, self.extraversion),
            (OceanTrait::Agreeableness, self.agreeableness),
            (OceanTrait::Neuroticism, self.neuroticism),
        ];

        traits
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, _)| t)
            .unwrap_or(OceanTrait::Openness)
    }

    /// Get traits that are significantly high (> 0.65)
    pub fn high_traits(&self) -> Vec<OceanTrait> {
        let threshold = 0.65;
        let mut result = Vec::new();

        if self.openness > threshold {
            result.push(OceanTrait::Openness);
        }
        if self.conscientiousness > threshold {
            result.push(OceanTrait::Conscientiousness);
        }
        if self.extraversion > threshold {
            result.push(OceanTrait::Extraversion);
        }
        if self.agreeableness > threshold {
            result.push(OceanTrait::Agreeableness);
        }
        if self.neuroticism > threshold {
            result.push(OceanTrait::Neuroticism);
        }

        result
    }

    /// Get traits that are significantly low (< 0.35)
    pub fn low_traits(&self) -> Vec<OceanTrait> {
        let threshold = 0.35;
        let mut result = Vec::new();

        if self.openness < threshold {
            result.push(OceanTrait::Openness);
        }
        if self.conscientiousness < threshold {
            result.push(OceanTrait::Conscientiousness);
        }
        if self.extraversion < threshold {
            result.push(OceanTrait::Extraversion);
        }
        if self.agreeableness < threshold {
            result.push(OceanTrait::Agreeableness);
        }
        if self.neuroticism < threshold {
            result.push(OceanTrait::Neuroticism);
        }

        result
    }

    /// Convert to feature vector
    pub fn to_vector(&self) -> [f64; 5] {
        [
            self.openness,
            self.conscientiousness,
            self.extraversion,
            self.agreeableness,
            self.neuroticism,
        ]
    }

    /// Create from feature vector
    pub fn from_vector(v: &[f64]) -> Option<Self> {
        if v.len() < 5 {
            return None;
        }
        Some(Self::new(v[0], v[1], v[2], v[3], v[4]))
    }

    /// Weighted blend with another score set
    pub fn blend(&self, other: &OceanScores, alpha: f64) -> OceanScores {
        let a = alpha.clamp(0.0, 1.0);
        let b = 1.0 - a;

        OceanScores::new(
            self.openness * b + other.openness * a,
            self.conscientiousness * b + other.conscientiousness * a,
            self.extraversion * b + other.extraversion * a,
            self.agreeableness * b + other.agreeableness * a,
            self.neuroticism * b + other.neuroticism * a,
        )
    }
}

impl Default for OceanScores {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Individual OCEAN trait enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OceanTrait {
    Openness,
    Conscientiousness,
    Extraversion,
    Agreeableness,
    Neuroticism,
}

impl OceanTrait {
    pub fn name(&self) -> &'static str {
        match self {
            OceanTrait::Openness => "Openness",
            OceanTrait::Conscientiousness => "Conscientiousness",
            OceanTrait::Extraversion => "Extraversion",
            OceanTrait::Agreeableness => "Agreeableness",
            OceanTrait::Neuroticism => "Neuroticism",
        }
    }

    pub fn abbreviation(&self) -> &'static str {
        match self {
            OceanTrait::Openness => "O",
            OceanTrait::Conscientiousness => "C",
            OceanTrait::Extraversion => "E",
            OceanTrait::Agreeableness => "A",
            OceanTrait::Neuroticism => "N",
        }
    }

    /// High score behavioral description
    pub fn high_description(&self) -> &'static str {
        match self {
            OceanTrait::Openness => "Creative, curious, open to new experiences",
            OceanTrait::Conscientiousness => "Organized, dependable, disciplined",
            OceanTrait::Extraversion => "Sociable, energetic, talkative",
            OceanTrait::Agreeableness => "Cooperative, trusting, helpful",
            OceanTrait::Neuroticism => "Anxious, moody, emotionally reactive",
        }
    }

    /// Low score behavioral description
    pub fn low_description(&self) -> &'static str {
        match self {
            OceanTrait::Openness => "Practical, conventional, prefers routine",
            OceanTrait::Conscientiousness => "Flexible, spontaneous, casual",
            OceanTrait::Extraversion => "Reserved, quiet, independent",
            OceanTrait::Agreeableness => "Analytical, skeptical, competitive",
            OceanTrait::Neuroticism => "Calm, emotionally stable, resilient",
        }
    }
}

/// Confidence level for OCEAN predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionConfidence {
    /// Insufficient data for reliable prediction
    Low,
    /// Moderate confidence, some uncertainty
    Medium,
    /// High confidence prediction
    High,
}

impl PredictionConfidence {
    pub fn from_observation_count(count: usize) -> Self {
        match count {
            0..=10 => PredictionConfidence::Low,
            11..=50 => PredictionConfidence::Medium,
            _ => PredictionConfidence::High,
        }
    }
}

/// OCEAN prediction result with confidence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OceanPrediction {
    pub scores: OceanScores,
    pub confidence: PredictionConfidence,
    pub observation_count: usize,
    pub observation_duration_secs: f64,

    /// Per-trait confidence scores
    pub trait_confidences: [f64; 5],
}

impl OceanPrediction {
    pub fn new(
        scores: OceanScores,
        observation_count: usize,
        observation_duration_secs: f64,
        trait_confidences: [f64; 5],
    ) -> Self {
        Self {
            scores,
            confidence: PredictionConfidence::from_observation_count(observation_count),
            observation_count,
            observation_duration_secs,
            trait_confidences,
        }
    }

    /// Generate a textual summary of the prediction
    pub fn summary(&self) -> String {
        let dominant = self.scores.dominant_trait();
        let high = self.scores.high_traits();
        let low = self.scores.low_traits();

        let mut parts = Vec::new();

        parts.push(format!("Dominant trait: {}", dominant.name()));

        if !high.is_empty() {
            let high_names: Vec<_> = high.iter().map(|t| t.name()).collect();
            parts.push(format!("High: {}", high_names.join(", ")));
        }

        if !low.is_empty() {
            let low_names: Vec<_> = low.iter().map(|t| t.name()).collect();
            parts.push(format!("Low: {}", low_names.join(", ")));
        }

        parts.push(format!("Confidence: {:?}", self.confidence));

        parts.join(" | ")
    }
}

/// Sales persona classification based on OCEAN profile
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SalesPersona {
    /// High O, High E: Adventurous, wants unique/innovative
    Innovator,
    /// High C, Low N: Analytical, wants data and specs
    Analyst,
    /// High E, High A: Social, wants relationships
    Socializer,
    /// High N, Low E: Cautious, wants safety and reassurance
    Cautious,
    /// High C, High A: Practical, wants value and reliability
    Pragmatist,
    /// Balanced profile, adaptive approach needed
    Adaptive,
}

impl SalesPersona {
    pub fn from_ocean(scores: &OceanScores) -> Self {
        // High Openness + High Extraversion = Innovator
        if scores.openness > 0.6 && scores.extraversion > 0.6 {
            return SalesPersona::Innovator;
        }

        // High Conscientiousness + Low Neuroticism = Analyst
        if scores.conscientiousness > 0.6 && scores.neuroticism < 0.4 {
            return SalesPersona::Analyst;
        }

        // High Extraversion + High Agreeableness = Socializer
        if scores.extraversion > 0.6 && scores.agreeableness > 0.6 {
            return SalesPersona::Socializer;
        }

        // High Neuroticism + Low Extraversion = Cautious
        if scores.neuroticism > 0.6 && scores.extraversion < 0.4 {
            return SalesPersona::Cautious;
        }

        // High Conscientiousness + High Agreeableness = Pragmatist
        if scores.conscientiousness > 0.6 && scores.agreeableness > 0.6 {
            return SalesPersona::Pragmatist;
        }

        SalesPersona::Adaptive
    }

    pub fn recommended_approach(&self) -> &'static str {
        match self {
            SalesPersona::Innovator => {
                "Focus on cutting-edge features, technology, and unique selling points. \
                 Emphasize what makes this vehicle different and exciting."
            }
            SalesPersona::Analyst => {
                "Provide detailed specifications, comparison data, and objective information. \
                 Let them process information at their own pace."
            }
            SalesPersona::Socializer => {
                "Build rapport first, share stories, discuss lifestyle fit. \
                 Make the experience enjoyable and social."
            }
            SalesPersona::Cautious => {
                "Emphasize safety features, reliability, warranty, and testimonials. \
                 Provide reassurance and don't pressure."
            }
            SalesPersona::Pragmatist => {
                "Focus on value proposition, total cost of ownership, practicality. \
                 Be straightforward and efficient."
            }
            SalesPersona::Adaptive => {
                "Use a balanced approach, observe reactions, and adjust strategy \
                 based on their responses."
            }
        }
    }

    pub fn opening_themes(&self) -> &'static [&'static str] {
        match self {
            SalesPersona::Innovator => {
                &["innovation", "technology", "unique features", "design"]
            }
            SalesPersona::Analyst => {
                &["specifications", "efficiency", "comparisons", "data"]
            }
            SalesPersona::Socializer => {
                &["lifestyle", "family", "adventures", "experiences"]
            }
            SalesPersona::Cautious => {
                &["safety", "reliability", "warranty", "peace of mind"]
            }
            SalesPersona::Pragmatist => {
                &["value", "practicality", "efficiency", "durability"]
            }
            SalesPersona::Adaptive => &["needs", "preferences", "goals", "questions"],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocean_clamp() {
        let scores = OceanScores::new(1.5, -0.2, 0.5, 0.5, 0.5);
        assert_eq!(scores.openness, 1.0);
        assert_eq!(scores.conscientiousness, 0.0);
    }

    #[test]
    fn test_dominant_trait() {
        let scores = OceanScores::new(0.9, 0.3, 0.4, 0.5, 0.2);
        assert_eq!(scores.dominant_trait(), OceanTrait::Openness);
    }

    #[test]
    fn test_sales_persona() {
        let innovator_scores = OceanScores::new(0.8, 0.5, 0.8, 0.5, 0.3);
        assert_eq!(
            SalesPersona::from_ocean(&innovator_scores),
            SalesPersona::Innovator
        );

        let cautious_scores = OceanScores::new(0.4, 0.5, 0.2, 0.5, 0.8);
        assert_eq!(
            SalesPersona::from_ocean(&cautious_scores),
            SalesPersona::Cautious
        );
    }
}
