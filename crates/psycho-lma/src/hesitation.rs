//! Hesitation metric calculation.
//!
//! Hesitation is a key behavioral indicator combining:
//! - Low velocity (pausing)
//! - High angular change (looking around, uncertainty)
//!
//! ## Hesitation Formula
//!
//! H(t) = ∫_{t-Δt}^{t} (1/(v(τ) + ε)) · |dθ/dτ| dτ
//!
//! Where:
//! - v(τ) is velocity at time τ
//! - θ is direction angle
//! - ε is small constant to avoid division by zero
//! - Δt is the integration window

use psycho_core::{Trajectory, TrajectoryPoint, Timestamp};
use serde::{Deserialize, Serialize};

/// Hesitation analysis configuration
#[derive(Debug, Clone)]
pub struct HesitationConfig {
    /// Small constant to avoid division by zero (epsilon)
    pub epsilon: f64,
    /// Integration window size in seconds
    pub window_secs: f64,
    /// Minimum velocity to be considered "stopped" (m/s)
    pub stop_threshold: f64,
}

impl Default for HesitationConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            window_secs: 1.0,
            stop_threshold: 0.1,
        }
    }
}

/// Hesitation metric result
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HesitationMetric {
    /// Instantaneous hesitation value
    pub value: f64,
    /// Normalized hesitation score [0, 1]
    pub normalized: f64,
    /// Number of hesitation events in window
    pub hesitation_count: u32,
    /// Time spent hesitating in window (seconds)
    pub hesitation_duration: f64,
}

impl HesitationMetric {
    pub fn is_hesitating(&self) -> bool {
        self.normalized > 0.5
    }

    pub fn description(&self) -> &'static str {
        if self.normalized > 0.8 {
            "Very Hesitant - frequent pauses, high uncertainty"
        } else if self.normalized > 0.6 {
            "Moderately Hesitant - noticeable indecision"
        } else if self.normalized > 0.4 {
            "Slightly Hesitant - some uncertainty"
        } else if self.normalized > 0.2 {
            "Confident - minimal hesitation"
        } else {
            "Very Confident - decisive, purposeful"
        }
    }
}

/// Hesitation analyzer
pub struct HesitationAnalyzer {
    config: HesitationConfig,
}

impl HesitationAnalyzer {
    pub fn new(config: HesitationConfig) -> Self {
        Self { config }
    }

    /// Calculate hesitation metric at a specific point in the trajectory
    ///
    /// Implements: H(t) = ∫_{t-Δt}^{t} (1/(v(τ) + ε)) · |dθ/dτ| dτ
    pub fn calculate_at(&self, trajectory: &Trajectory, timestamp: Timestamp) -> HesitationMetric {
        let window_nanos = (self.config.window_secs * 1e9) as i64;
        let target_nanos = timestamp.as_nanos();
        let window_start = target_nanos - window_nanos;

        // Find points within the window
        let window_points: Vec<&TrajectoryPoint> = trajectory
            .points
            .iter()
            .filter(|p| {
                let t = p.timestamp.as_nanos();
                t >= window_start && t <= target_nanos
            })
            .collect();

        if window_points.len() < 2 {
            return HesitationMetric {
                value: 0.0,
                normalized: 0.0,
                hesitation_count: 0,
                hesitation_duration: 0.0,
            };
        }

        // Calculate the integral
        let mut integral = 0.0;
        let mut hesitation_count = 0u32;
        let mut hesitation_duration = 0.0;
        let mut in_hesitation = false;
        let mut hesitation_start = 0i64;

        for window in window_points.windows(2) {
            let p0 = window[0];
            let p1 = window[1];

            let dt = (p1.timestamp.as_nanos() - p0.timestamp.as_nanos()) as f64 / 1e9;
            if dt <= 0.0 {
                continue;
            }

            // Velocity at midpoint
            let v = (p0.velocity.magnitude() + p1.velocity.magnitude()) / 2.0;

            // Angular change rate
            let dx = p1.position.x - p0.position.x;
            let dy = p1.position.y - p0.position.y;
            let dx_prev = p0.velocity.vx;
            let dy_prev = p0.velocity.vy;

            let angle_curr = dy.atan2(dx);
            let angle_prev = dy_prev.atan2(dx_prev);
            let d_theta = angular_difference(angle_curr, angle_prev);
            let angular_rate = d_theta.abs() / dt;

            // Hesitation integrand: (1/(v + ε)) · |dθ/dτ|
            let integrand = (1.0 / (v + self.config.epsilon)) * angular_rate;
            integral += integrand * dt;

            // Track hesitation events
            let is_hesitating = v < self.config.stop_threshold && angular_rate > 0.5;

            if is_hesitating && !in_hesitation {
                in_hesitation = true;
                hesitation_start = p0.timestamp.as_nanos();
                hesitation_count += 1;
            } else if !is_hesitating && in_hesitation {
                in_hesitation = false;
                hesitation_duration +=
                    (p1.timestamp.as_nanos() - hesitation_start) as f64 / 1e9;
            }
        }

        // Handle ongoing hesitation
        if in_hesitation {
            hesitation_duration +=
                (window_points.last().unwrap().timestamp.as_nanos() - hesitation_start) as f64 / 1e9;
        }

        // Normalize the integral
        // Typical range: 0 (no hesitation) to ~10 (very hesitant)
        let normalized = (integral / 10.0).min(1.0);

        HesitationMetric {
            value: integral,
            normalized,
            hesitation_count,
            hesitation_duration,
        }
    }

    /// Calculate hesitation profile over entire trajectory
    pub fn analyze_trajectory(&self, trajectory: &Trajectory) -> HesitationProfile {
        if trajectory.points.is_empty() {
            return HesitationProfile::default();
        }

        let window_nanos = (self.config.window_secs * 1e9) as i64;
        let start_time = trajectory.start_time.as_nanos();
        let end_time = trajectory.end_time.as_nanos();

        let mut metrics = Vec::new();
        let mut current_time = start_time + window_nanos;

        while current_time <= end_time {
            let metric = self.calculate_at(trajectory, Timestamp::from_nanos(current_time));
            metrics.push((Timestamp::from_nanos(current_time), metric));
            current_time += window_nanos / 4; // 25% overlap
        }

        // Aggregate statistics
        let values: Vec<f64> = metrics.iter().map(|(_, m)| m.value).collect();
        let mean_hesitation = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        };

        let max_hesitation = values.iter().cloned().fold(0.0, f64::max);

        let total_hesitation_events: u32 = metrics.iter().map(|(_, m)| m.hesitation_count).sum();

        let total_hesitation_duration: f64 = metrics.iter().map(|(_, m)| m.hesitation_duration).sum();

        HesitationProfile {
            metrics,
            mean_hesitation,
            max_hesitation,
            total_hesitation_events,
            total_hesitation_duration,
            overall_score: (mean_hesitation / 5.0).min(1.0),
        }
    }

    pub fn config(&self) -> &HesitationConfig {
        &self.config
    }
}

impl Default for HesitationAnalyzer {
    fn default() -> Self {
        Self::new(HesitationConfig::default())
    }
}

/// Complete hesitation analysis over trajectory
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HesitationProfile {
    /// Time series of hesitation metrics
    pub metrics: Vec<(Timestamp, HesitationMetric)>,
    /// Mean hesitation value
    pub mean_hesitation: f64,
    /// Maximum hesitation value
    pub max_hesitation: f64,
    /// Total number of hesitation events
    pub total_hesitation_events: u32,
    /// Total time spent hesitating (seconds)
    pub total_hesitation_duration: f64,
    /// Overall hesitation score [0, 1]
    pub overall_score: f64,
}

impl HesitationProfile {
    pub fn is_hesitant(&self) -> bool {
        self.overall_score > 0.4
    }

    /// Get timestamps of peak hesitation
    pub fn peak_hesitation_times(&self, threshold: f64) -> Vec<Timestamp> {
        self.metrics
            .iter()
            .filter(|(_, m)| m.normalized > threshold)
            .map(|(t, _)| *t)
            .collect()
    }
}

/// Calculate angular difference handling wrap-around
fn angular_difference(angle1: f64, angle2: f64) -> f64 {
    let diff = angle1 - angle2;
    let pi = std::f64::consts::PI;

    if diff > pi {
        diff - 2.0 * pi
    } else if diff < -pi {
        diff + 2.0 * pi
    } else {
        diff
    }
}

/// Identify hesitation patterns (useful for sales context)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HesitationPattern {
    /// Brief pauses during exploration
    Exploratory,
    /// Extended stops with looking around
    Evaluative,
    /// Approach-withdrawal oscillation
    Ambivalent,
    /// Freezing in place
    Overwhelmed,
    /// No significant hesitation
    Decisive,
}

impl HesitationPattern {
    pub fn from_profile(profile: &HesitationProfile) -> Self {
        // Classify based on hesitation characteristics
        if profile.overall_score < 0.2 {
            return HesitationPattern::Decisive;
        }

        if profile.total_hesitation_events == 0 {
            return HesitationPattern::Decisive;
        }

        let avg_event_duration = if profile.total_hesitation_events > 0 {
            profile.total_hesitation_duration / profile.total_hesitation_events as f64
        } else {
            0.0
        };

        // Classify patterns
        if avg_event_duration > 5.0 && profile.max_hesitation > 0.7 {
            HesitationPattern::Overwhelmed
        } else if avg_event_duration > 2.0 {
            HesitationPattern::Evaluative
        } else if profile.total_hesitation_events > 5 && avg_event_duration < 1.0 {
            HesitationPattern::Ambivalent
        } else {
            HesitationPattern::Exploratory
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            HesitationPattern::Exploratory => "Exploratory: Brief pauses while browsing",
            HesitationPattern::Evaluative => "Evaluative: Extended consideration of options",
            HesitationPattern::Ambivalent => "Ambivalent: Uncertain, approach-withdrawal behavior",
            HesitationPattern::Overwhelmed => "Overwhelmed: Analysis paralysis, needs guidance",
            HesitationPattern::Decisive => "Decisive: Clear intent, minimal hesitation",
        }
    }

    pub fn recommended_response(&self) -> &'static str {
        match self {
            HesitationPattern::Exploratory => "Allow browsing; be available but not intrusive",
            HesitationPattern::Evaluative => "Provide detailed information when approached",
            HesitationPattern::Ambivalent => "Offer gentle guidance and reassurance",
            HesitationPattern::Overwhelmed => "Simplify options; offer structured assistance",
            HesitationPattern::Decisive => "Be ready to facilitate; don't over-explain",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_core::{Acceleration3D, Position3D, SubjectId, Velocity3D};

    fn create_hesitant_trajectory() -> Trajectory {
        let mut traj = Trajectory::new(SubjectId::new());

        // Moving, then stopping and looking around
        for i in 0..10 {
            let t = i as f64 * 0.1;
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(i * 100_000_000),
                Position3D::new(t, 0.0, 0.0),
                Velocity3D::new(1.0, 0.0, 0.0),
                Acceleration3D::zero(),
            ));
        }

        // Hesitation: stopped with direction changes
        for i in 10..20 {
            let angle = (i - 10) as f64 * 0.3; // Changing direction while stopped
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(i * 100_000_000),
                Position3D::new(1.0, 0.0, 0.0),
                Velocity3D::new(0.01 * angle.cos(), 0.01 * angle.sin(), 0.0),
                Acceleration3D::zero(),
            ));
        }

        traj
    }

    fn create_confident_trajectory() -> Trajectory {
        let mut traj = Trajectory::new(SubjectId::new());

        for i in 0..20 {
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(i * 100_000_000),
                Position3D::new(i as f64 * 0.1, 0.0, 0.0),
                Velocity3D::new(1.0, 0.0, 0.0),
                Acceleration3D::zero(),
            ));
        }

        traj
    }

    #[test]
    fn test_hesitation_detection() {
        let analyzer = HesitationAnalyzer::default();

        let hesitant = create_hesitant_trajectory();
        let confident = create_confident_trajectory();

        let hesitant_profile = analyzer.analyze_trajectory(&hesitant);
        let confident_profile = analyzer.analyze_trajectory(&confident);

        // Hesitant trajectory should have higher score
        assert!(
            hesitant_profile.overall_score >= confident_profile.overall_score,
            "Hesitant trajectory should score higher"
        );
    }

    #[test]
    fn test_hesitation_pattern_classification() {
        let profile = HesitationProfile {
            metrics: Vec::new(),
            mean_hesitation: 0.5,
            max_hesitation: 0.6,
            total_hesitation_events: 3,
            total_hesitation_duration: 6.0, // 2 sec average
            overall_score: 0.5,
        };

        let pattern = HesitationPattern::from_profile(&profile);
        assert_eq!(pattern, HesitationPattern::Evaluative);
    }

    #[test]
    fn test_angular_difference() {
        let pi = std::f64::consts::PI;

        // Simple case
        assert!((angular_difference(0.5, 0.0) - 0.5).abs() < 0.01);

        // Wrap around
        assert!((angular_difference(pi - 0.1, -pi + 0.1).abs() - 0.2).abs() < 0.01);
    }
}
