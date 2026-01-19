//! Kinematic analysis types and computations.

use serde::{Deserialize, Serialize};

use crate::types::{Acceleration3D, Position3D, SubjectId, Timestamp, Velocity3D};

/// Time-series trajectory point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub timestamp: Timestamp,
    pub position: Position3D,
    pub velocity: Velocity3D,
    pub acceleration: Acceleration3D,
}

impl TrajectoryPoint {
    pub fn new(
        timestamp: Timestamp,
        position: Position3D,
        velocity: Velocity3D,
        acceleration: Acceleration3D,
    ) -> Self {
        Self {
            timestamp,
            position,
            velocity,
            acceleration,
        }
    }

    /// Calculate jerk (derivative of acceleration) from two consecutive points
    pub fn jerk_from(&self, previous: &TrajectoryPoint) -> Jerk3D {
        let dt = (self.timestamp.as_nanos() - previous.timestamp.as_nanos()) as f64 / 1e9;
        if dt <= 0.0 {
            return Jerk3D::zero();
        }

        Jerk3D {
            jx: (self.acceleration.ax - previous.acceleration.ax) / dt,
            jy: (self.acceleration.ay - previous.acceleration.ay) / dt,
            jz: (self.acceleration.az - previous.acceleration.az) / dt,
        }
    }
}

/// Jerk vector (derivative of acceleration, m/s³)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Jerk3D {
    pub jx: f64,
    pub jy: f64,
    pub jz: f64,
}

impl Jerk3D {
    pub fn new(jx: f64, jy: f64, jz: f64) -> Self {
        Self { jx, jy, jz }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn magnitude(&self) -> f64 {
        (self.jx * self.jx + self.jy * self.jy + self.jz * self.jz).sqrt()
    }
}

/// Complete trajectory for a tracked subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub subject_id: SubjectId,
    pub points: Vec<TrajectoryPoint>,
    pub start_time: Timestamp,
    pub end_time: Timestamp,
}

impl Trajectory {
    pub fn new(subject_id: SubjectId) -> Self {
        Self {
            subject_id,
            points: Vec::new(),
            start_time: Timestamp::now(),
            end_time: Timestamp::now(),
        }
    }

    pub fn add_point(&mut self, point: TrajectoryPoint) {
        if self.points.is_empty() {
            self.start_time = point.timestamp;
        }
        self.end_time = point.timestamp;
        self.points.push(point);
    }

    pub fn duration_secs(&self) -> f64 {
        (self.end_time.as_nanos() - self.start_time.as_nanos()) as f64 / 1e9
    }

    /// Calculate total path length (actual distance traveled)
    pub fn path_length(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }

        self.points
            .windows(2)
            .map(|w| w[0].position.distance_to(&w[1].position))
            .sum()
    }

    /// Calculate Euclidean distance from start to end
    pub fn displacement(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }

        self.points
            .first()
            .unwrap()
            .position
            .distance_to(&self.points.last().unwrap().position)
    }

    /// Path Efficiency Ratio (PER) = Displacement / Path Length
    /// Higher values indicate more direct movement
    pub fn path_efficiency_ratio(&self) -> f64 {
        let path_len = self.path_length();
        if path_len < 1e-10 {
            return 1.0; // No movement = perfect efficiency
        }
        self.displacement() / path_len
    }

    /// Calculate average velocity magnitude
    pub fn average_speed(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.points.iter().map(|p| p.velocity.magnitude()).sum();
        sum / self.points.len() as f64
    }

    /// Calculate jerk profile for the trajectory
    pub fn jerk_profile(&self) -> Vec<(Timestamp, Jerk3D)> {
        if self.points.len() < 2 {
            return Vec::new();
        }

        self.points
            .windows(2)
            .map(|w| (w[1].timestamp, w[1].jerk_from(&w[0])))
            .collect()
    }

    /// Calculate root mean square jerk (movement smoothness metric)
    pub fn rms_jerk(&self) -> f64 {
        let jerks = self.jerk_profile();
        if jerks.is_empty() {
            return 0.0;
        }

        let sum_sq: f64 = jerks.iter().map(|(_, j)| j.magnitude().powi(2)).sum();
        (sum_sq / jerks.len() as f64).sqrt()
    }

    /// Get velocity at a specific time using linear interpolation
    pub fn velocity_at(&self, timestamp: Timestamp) -> Option<Velocity3D> {
        if self.points.is_empty() {
            return None;
        }

        // Find bracketing points
        let target = timestamp.as_nanos();
        let idx = self
            .points
            .binary_search_by_key(&target, |p| p.timestamp.as_nanos());

        match idx {
            Ok(i) => Some(self.points[i].velocity),
            Err(i) => {
                if i == 0 {
                    Some(self.points[0].velocity)
                } else if i >= self.points.len() {
                    Some(self.points.last().unwrap().velocity)
                } else {
                    // Linear interpolation
                    let p0 = &self.points[i - 1];
                    let p1 = &self.points[i];
                    let t0 = p0.timestamp.as_secs_f64();
                    let t1 = p1.timestamp.as_secs_f64();
                    let t = timestamp.as_secs_f64();
                    let alpha = (t - t0) / (t1 - t0);

                    Some(Velocity3D::new(
                        p0.velocity.vx + alpha * (p1.velocity.vx - p0.velocity.vx),
                        p0.velocity.vy + alpha * (p1.velocity.vy - p0.velocity.vy),
                        p0.velocity.vz + alpha * (p1.velocity.vz - p0.velocity.vz),
                    ))
                }
            }
        }
    }
}

/// Kinematic feature vector for ML analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinematicFeatures {
    pub subject_id: SubjectId,
    pub timestamp: Timestamp,
    pub window_duration_secs: f64,

    // Spatial features
    pub path_efficiency_ratio: f64,
    pub total_displacement: f64,
    pub path_length: f64,

    // Velocity features
    pub mean_velocity: f64,
    pub max_velocity: f64,
    pub velocity_variance: f64,

    // Acceleration features
    pub mean_acceleration: f64,
    pub max_acceleration: f64,
    pub acceleration_variance: f64,

    // Jerk features (movement quality)
    pub rms_jerk: f64,
    pub max_jerk: f64,
    pub jerk_variance: f64,

    // Directional features
    pub direction_changes: u32,
    pub path_entropy: f64,

    // Stopping behavior
    pub stop_count: u32,
    pub total_stop_duration_secs: f64,
    pub mean_stop_duration_secs: f64,
}

impl KinematicFeatures {
    /// Extract features from a trajectory segment
    pub fn from_trajectory(trajectory: &Trajectory, stop_velocity_threshold: f64) -> Self {
        let subject_id = trajectory.subject_id;
        let timestamp = trajectory.end_time;
        let window_duration_secs = trajectory.duration_secs();

        // Spatial features
        let path_efficiency_ratio = trajectory.path_efficiency_ratio();
        let total_displacement = trajectory.displacement();
        let path_length = trajectory.path_length();

        // Velocity features
        let velocities: Vec<f64> = trajectory
            .points
            .iter()
            .map(|p| p.velocity.magnitude())
            .collect();
        let mean_velocity = mean(&velocities);
        let max_velocity = velocities.iter().cloned().fold(0.0, f64::max);
        let velocity_variance = variance(&velocities, mean_velocity);

        // Acceleration features
        let accelerations: Vec<f64> = trajectory
            .points
            .iter()
            .map(|p| p.acceleration.magnitude())
            .collect();
        let mean_acceleration = mean(&accelerations);
        let max_acceleration = accelerations.iter().cloned().fold(0.0, f64::max);
        let acceleration_variance = variance(&accelerations, mean_acceleration);

        // Jerk features
        let jerk_profile = trajectory.jerk_profile();
        let jerks: Vec<f64> = jerk_profile.iter().map(|(_, j)| j.magnitude()).collect();
        let rms_jerk = trajectory.rms_jerk();
        let max_jerk = jerks.iter().cloned().fold(0.0, f64::max);
        let mean_jerk = mean(&jerks);
        let jerk_variance = variance(&jerks, mean_jerk);

        // Direction changes
        let direction_changes = count_direction_changes(&trajectory.points);
        let path_entropy = calculate_path_entropy(&trajectory.points);

        // Stopping behavior
        let (stop_count, total_stop_duration_secs) =
            analyze_stops(&trajectory.points, stop_velocity_threshold);
        let mean_stop_duration_secs = if stop_count > 0 {
            total_stop_duration_secs / stop_count as f64
        } else {
            0.0
        };

        Self {
            subject_id,
            timestamp,
            window_duration_secs,
            path_efficiency_ratio,
            total_displacement,
            path_length,
            mean_velocity,
            max_velocity,
            velocity_variance,
            mean_acceleration,
            max_acceleration,
            acceleration_variance,
            rms_jerk,
            max_jerk,
            jerk_variance,
            direction_changes,
            path_entropy,
            stop_count,
            total_stop_duration_secs,
            mean_stop_duration_secs,
        }
    }

    /// Convert to feature vector for ML models
    pub fn to_feature_vector(&self) -> Vec<f64> {
        vec![
            self.path_efficiency_ratio,
            self.total_displacement,
            self.path_length,
            self.mean_velocity,
            self.max_velocity,
            self.velocity_variance,
            self.mean_acceleration,
            self.max_acceleration,
            self.acceleration_variance,
            self.rms_jerk,
            self.max_jerk,
            self.jerk_variance,
            self.direction_changes as f64,
            self.path_entropy,
            self.stop_count as f64,
            self.total_stop_duration_secs,
            self.mean_stop_duration_secs,
        ]
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

fn count_direction_changes(points: &[TrajectoryPoint]) -> u32 {
    if points.len() < 3 {
        return 0;
    }

    let mut changes = 0u32;
    let mut prev_direction = nalgebra::Vector3::new(
        points[1].position.x - points[0].position.x,
        points[1].position.y - points[0].position.y,
        points[1].position.z - points[0].position.z,
    );

    for window in points.windows(2).skip(1) {
        let curr_direction = nalgebra::Vector3::new(
            window[1].position.x - window[0].position.x,
            window[1].position.y - window[0].position.y,
            window[1].position.z - window[0].position.z,
        );

        if prev_direction.norm() > 1e-6 && curr_direction.norm() > 1e-6 {
            let dot = prev_direction.normalize().dot(&curr_direction.normalize());
            // Consider it a direction change if angle > 30 degrees
            if dot < 0.866 {
                // cos(30°)
                changes += 1;
            }
        }
        prev_direction = curr_direction;
    }

    changes
}

fn calculate_path_entropy(points: &[TrajectoryPoint]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    // Discretize directions into bins
    let num_bins = 8; // 8 directional bins (N, NE, E, SE, S, SW, W, NW)
    let mut bin_counts = vec![0u32; num_bins];
    let mut total = 0u32;

    for window in points.windows(2) {
        let dx = window[1].position.x - window[0].position.x;
        let dy = window[1].position.y - window[0].position.y;

        if (dx * dx + dy * dy).sqrt() > 1e-6 {
            let angle = dy.atan2(dx);
            let bin = ((angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)
                * num_bins as f64) as usize
                % num_bins;
            bin_counts[bin] += 1;
            total += 1;
        }
    }

    if total == 0 {
        return 0.0;
    }

    // Calculate Shannon entropy
    let mut entropy = 0.0;
    for count in bin_counts {
        if count > 0 {
            let p = count as f64 / total as f64;
            entropy -= p * p.log2();
        }
    }

    // Normalize by max entropy (log2(num_bins))
    entropy / (num_bins as f64).log2()
}

fn analyze_stops(points: &[TrajectoryPoint], threshold: f64) -> (u32, f64) {
    if points.is_empty() {
        return (0, 0.0);
    }

    let mut stop_count = 0u32;
    let mut total_duration = 0.0;
    let mut in_stop = false;
    let mut stop_start = 0i64;

    for point in points {
        let speed = point.velocity.magnitude();
        if speed < threshold {
            if !in_stop {
                in_stop = true;
                stop_start = point.timestamp.as_nanos();
            }
        } else if in_stop {
            in_stop = false;
            stop_count += 1;
            total_duration += (point.timestamp.as_nanos() - stop_start) as f64 / 1e9;
        }
    }

    // Handle case where trajectory ends in a stop
    if in_stop && !points.is_empty() {
        stop_count += 1;
        total_duration +=
            (points.last().unwrap().timestamp.as_nanos() - stop_start) as f64 / 1e9;
    }

    (stop_count, total_duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trajectory() -> Trajectory {
        let mut traj = Trajectory::new(SubjectId::new());

        for i in 0..10 {
            let t = i as f64 * 0.1;
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos((i as i64) * 100_000_000),
                Position3D::new(t, t * 0.5, 0.0),
                Velocity3D::new(1.0, 0.5, 0.0),
                Acceleration3D::zero(),
            ));
        }

        traj
    }

    #[test]
    fn test_path_efficiency() {
        let traj = create_test_trajectory();
        let per = traj.path_efficiency_ratio();
        // Straight line should have high efficiency
        assert!(per > 0.9);
    }

    #[test]
    fn test_kinematic_features() {
        let traj = create_test_trajectory();
        let features = KinematicFeatures::from_trajectory(&traj, 0.1);

        assert!(features.mean_velocity > 0.0);
        assert!(features.path_efficiency_ratio > 0.0);
    }
}
