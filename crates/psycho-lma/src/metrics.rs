//! Derived movement metrics for psychometric analysis.

use psycho_core::{KinematicFeatures, Trajectory};
use serde::{Deserialize, Serialize};

use crate::effort::{EffortProfile, FlowEffort, TimeEffort, WeightEffort};
use crate::hesitation::{HesitationAnalyzer, HesitationProfile};
use crate::space::{calculate_path_entropy, Kinesphere, SpaceEffort};

/// Complete movement metrics package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovementMetrics {
    /// Kinematic features (velocity, acceleration, jerk)
    pub kinematics: KinematicFeatures,
    /// LMA effort profile
    pub effort: SerializableEffortProfile,
    /// Hesitation analysis
    pub hesitation: HesitationProfile,
    /// Kinesphere analysis
    pub kinesphere: Kinesphere,
    /// Path entropy (movement predictability)
    pub path_entropy: f64,
    /// Fidgeting score (small, repetitive movements)
    pub fidgeting_score: f64,
    /// Approach-avoidance score
    pub approach_avoidance: f64,
}

/// Serializable version of EffortProfile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableEffortProfile {
    pub space: SpaceEffort,
    pub time: TimeEffort,
    pub weight: WeightEffort,
    pub flow: FlowEffort,
}

impl From<EffortProfile> for SerializableEffortProfile {
    fn from(profile: EffortProfile) -> Self {
        Self {
            space: profile.space,
            time: profile.time,
            weight: profile.weight,
            flow: profile.flow,
        }
    }
}

impl MovementMetrics {
    /// Extract all metrics from a trajectory
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        let kinematics = KinematicFeatures::from_trajectory(trajectory, 0.1);
        let effort = EffortProfile::from_trajectory(trajectory).into();
        let hesitation = HesitationAnalyzer::default().analyze_trajectory(trajectory);
        let kinesphere = Kinesphere::from_trajectory(trajectory);
        let path_entropy = calculate_path_entropy(&trajectory.points, 8);
        let fidgeting_score = calculate_fidgeting_score(trajectory);
        let approach_avoidance = calculate_approach_avoidance(trajectory);

        Self {
            kinematics,
            effort,
            hesitation,
            kinesphere,
            path_entropy,
            fidgeting_score,
            approach_avoidance,
        }
    }

    /// Convert to feature vector for ML models
    pub fn to_feature_vector(&self) -> Vec<f64> {
        let mut features = self.kinematics.to_feature_vector();

        // Add effort features
        features.push(self.effort.space.directness);
        features.push(self.effort.time.suddenness);
        features.push(self.effort.weight.strength);
        features.push(self.effort.flow.boundness);

        // Add hesitation features
        features.push(self.hesitation.overall_score);
        features.push(self.hesitation.mean_hesitation);

        // Add spatial features
        features.push(self.kinesphere.volume);
        features.push(self.kinesphere.utilization);
        features.push(self.path_entropy);

        // Add behavioral features
        features.push(self.fidgeting_score);
        features.push(self.approach_avoidance);

        features
    }

    /// Get feature names for the feature vector
    pub fn feature_names() -> Vec<&'static str> {
        let mut names = vec![
            "path_efficiency_ratio",
            "total_displacement",
            "path_length",
            "mean_velocity",
            "max_velocity",
            "velocity_variance",
            "mean_acceleration",
            "max_acceleration",
            "acceleration_variance",
            "rms_jerk",
            "max_jerk",
            "jerk_variance",
            "direction_changes",
            "kinematic_path_entropy",
            "stop_count",
            "total_stop_duration",
            "mean_stop_duration",
        ];

        names.extend([
            "space_directness",
            "time_suddenness",
            "weight_strength",
            "flow_boundness",
            "hesitation_score",
            "mean_hesitation",
            "kinesphere_volume",
            "kinesphere_utilization",
            "path_entropy",
            "fidgeting_score",
            "approach_avoidance",
        ]);

        names
    }
}

/// Calculate fidgeting score from small, repetitive movements
fn calculate_fidgeting_score(trajectory: &Trajectory) -> f64 {
    if trajectory.points.len() < 10 {
        return 0.0;
    }

    // Detect small oscillations in position
    let mut oscillation_count = 0;
    let mut total_checks = 0;

    // Use a sliding window to detect direction reversals
    for window in trajectory.points.windows(5) {
        // Calculate displacements
        let d01 = displacement(&window[0].position, &window[1].position);
        let d12 = displacement(&window[1].position, &window[2].position);
        let d23 = displacement(&window[2].position, &window[3].position);
        let d34 = displacement(&window[3].position, &window[4].position);

        // Check for oscillation pattern (small movements, direction changes)
        let is_small = d01 < 0.1 && d12 < 0.1 && d23 < 0.1 && d34 < 0.1;

        if is_small {
            // Check for direction reversals
            let dot1 = dot_2d(&window[0].position, &window[1].position, &window[2].position);
            let dot2 = dot_2d(&window[1].position, &window[2].position, &window[3].position);
            let dot3 = dot_2d(&window[2].position, &window[3].position, &window[4].position);

            // Negative dot products indicate direction reversal
            let reversals = (dot1 < 0.0) as u32 + (dot2 < 0.0) as u32 + (dot3 < 0.0) as u32;

            if reversals >= 2 {
                oscillation_count += 1;
            }
        }

        total_checks += 1;
    }

    if total_checks > 0 {
        (oscillation_count as f64 / total_checks as f64).min(1.0)
    } else {
        0.0
    }
}

/// Calculate approach-avoidance score from trajectory towards reference point
fn calculate_approach_avoidance(trajectory: &Trajectory) -> f64 {
    if trajectory.points.len() < 2 {
        return 0.0;
    }

    // Use the end point as the implicit target
    let target = &trajectory.points.last().unwrap().position;

    let mut approach_count = 0;
    let mut avoid_count = 0;

    for window in trajectory.points.windows(2) {
        let dist_before = window[0].position.distance_to(target);
        let dist_after = window[1].position.distance_to(target);

        if dist_after < dist_before - 0.01 {
            approach_count += 1;
        } else if dist_after > dist_before + 0.01 {
            avoid_count += 1;
        }
    }

    let total = approach_count + avoid_count;
    if total == 0 {
        return 0.0;
    }

    // Score: -1 = pure avoidance, 0 = balanced, +1 = pure approach
    (approach_count as f64 - avoid_count as f64) / total as f64
}

fn displacement(p1: &psycho_core::Position3D, p2: &psycho_core::Position3D) -> f64 {
    p1.distance_to(p2)
}

fn dot_2d(
    p1: &psycho_core::Position3D,
    p2: &psycho_core::Position3D,
    p3: &psycho_core::Position3D,
) -> f64 {
    let v1x = p2.x - p1.x;
    let v1y = p2.y - p1.y;
    let v2x = p3.x - p2.x;
    let v2y = p3.y - p2.y;

    v1x * v2x + v1y * v2y
}

/// Aggregated metrics over a time window
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Mean values
    pub mean: Vec<f64>,
    /// Standard deviation
    pub std: Vec<f64>,
    /// Minimum values
    pub min: Vec<f64>,
    /// Maximum values
    pub max: Vec<f64>,
    /// Number of samples
    pub count: usize,
}

impl AggregatedMetrics {
    /// Aggregate multiple metric samples
    pub fn from_samples(samples: &[MovementMetrics]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let vectors: Vec<Vec<f64>> = samples.iter().map(|m| m.to_feature_vector()).collect();

        let n_features = vectors[0].len();
        let n_samples = vectors.len();

        let mut mean = vec![0.0; n_features];
        let mut min = vec![f64::INFINITY; n_features];
        let mut max = vec![f64::NEG_INFINITY; n_features];

        // Calculate mean, min, max
        for v in &vectors {
            for (i, &val) in v.iter().enumerate() {
                mean[i] += val;
                min[i] = min[i].min(val);
                max[i] = max[i].max(val);
            }
        }

        for m in &mut mean {
            *m /= n_samples as f64;
        }

        // Calculate standard deviation
        let mut std = vec![0.0; n_features];
        for v in &vectors {
            for (i, &val) in v.iter().enumerate() {
                std[i] += (val - mean[i]).powi(2);
            }
        }

        for s in &mut std {
            *s = (*s / n_samples as f64).sqrt();
        }

        Self {
            mean,
            std,
            min,
            max,
            count: n_samples,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_core::{Acceleration3D, Position3D, SubjectId, Timestamp, TrajectoryPoint, Velocity3D};

    fn create_test_trajectory() -> Trajectory {
        let mut traj = Trajectory::new(SubjectId::new());

        for i in 0..20 {
            let t = i as f64 * 0.1;
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(i * 100_000_000),
                Position3D::new(t, 0.5 * (t * 2.0).sin(), 0.0),
                Velocity3D::new(1.0, (t * 2.0).cos(), 0.0),
                Acceleration3D::new(0.0, -2.0 * (t * 2.0).sin(), 0.0),
            ));
        }

        traj
    }

    #[test]
    fn test_movement_metrics() {
        let traj = create_test_trajectory();
        let metrics = MovementMetrics::from_trajectory(&traj);

        let features = metrics.to_feature_vector();
        let names = MovementMetrics::feature_names();

        assert_eq!(features.len(), names.len());
        assert!(features.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn test_aggregated_metrics() {
        let traj = create_test_trajectory();
        let metrics = MovementMetrics::from_trajectory(&traj);

        let samples = vec![metrics.clone(), metrics.clone(), metrics];
        let aggregated = AggregatedMetrics::from_samples(&samples);

        assert_eq!(aggregated.count, 3);
        assert_eq!(aggregated.mean.len(), aggregated.std.len());
    }
}
