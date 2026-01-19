//! Space effort analysis - Direct vs Indirect movement.
//!
//! The Space effort factor describes how focused or diffuse attention is during movement.
//! - **Direct**: Focused, channeled attention (e.g., pointing, reaching for specific target)
//! - **Indirect**: Flexible, multi-focused attention (e.g., wandering, exploring)
//!
//! ## Metric: Path Efficiency Ratio (PER)
//!
//! PER = Euclidean Distance / Actual Path Length
//!
//! - PER ≈ 1.0: Very direct movement (straight line)
//! - PER → 0.0: Very indirect movement (meandering)

use psycho_core::{KinematicFeatures, Position3D, Trajectory, TrajectoryPoint};
use serde::{Deserialize, Serialize};

/// Space effort quality (Direct vs Indirect)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpaceEffort {
    /// Path Efficiency Ratio [0, 1]
    pub per: f64,
    /// Directness score [0=indirect, 1=direct]
    pub directness: f64,
    /// Path curvature (average over trajectory)
    pub curvature: f64,
    /// Number of direction changes
    pub direction_changes: u32,
}

impl SpaceEffort {
    /// Analyze space effort from a trajectory
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        let per = trajectory.path_efficiency_ratio();

        // Directness is a normalized version of PER
        // PER of 0.7+ is considered direct
        let directness = normalize_to_range(per, 0.3, 1.0);

        let curvature = calculate_average_curvature(&trajectory.points);
        let direction_changes = count_significant_direction_changes(&trajectory.points, 30.0);

        Self {
            per,
            directness,
            curvature,
            direction_changes,
        }
    }

    /// Is movement predominantly direct?
    pub fn is_direct(&self) -> bool {
        self.directness > 0.5
    }

    /// Qualitative description
    pub fn description(&self) -> &'static str {
        if self.directness > 0.8 {
            "Very Direct - focused, channeled attention"
        } else if self.directness > 0.6 {
            "Somewhat Direct - purposeful with some flexibility"
        } else if self.directness > 0.4 {
            "Neutral - balanced focus"
        } else if self.directness > 0.2 {
            "Somewhat Indirect - flexible, exploratory"
        } else {
            "Very Indirect - wandering, multi-focused"
        }
    }
}

/// Calculate average curvature along trajectory
fn calculate_average_curvature(points: &[TrajectoryPoint]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut total_curvature = 0.0;
    let mut count = 0;

    for window in points.windows(3) {
        let p0 = &window[0].position;
        let p1 = &window[1].position;
        let p2 = &window[2].position;

        // Vectors
        let v1 = nalgebra::Vector3::new(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        let v2 = nalgebra::Vector3::new(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);

        let len1 = v1.norm();
        let len2 = v2.norm();

        if len1 > 1e-6 && len2 > 1e-6 {
            // Angle between consecutive segments
            let cos_angle = v1.dot(&v2) / (len1 * len2);
            let angle = cos_angle.clamp(-1.0, 1.0).acos();

            // Curvature approximation: angle / arc_length
            let arc_length = (len1 + len2) / 2.0;
            if arc_length > 1e-6 {
                total_curvature += angle / arc_length;
                count += 1;
            }
        }
    }

    if count > 0 {
        total_curvature / count as f64
    } else {
        0.0
    }
}

/// Count significant direction changes (turns > threshold degrees)
fn count_significant_direction_changes(points: &[TrajectoryPoint], threshold_degrees: f64) -> u32 {
    if points.len() < 3 {
        return 0;
    }

    let threshold_rad = threshold_degrees.to_radians();
    let mut changes = 0;

    for window in points.windows(3) {
        let p0 = &window[0].position;
        let p1 = &window[1].position;
        let p2 = &window[2].position;

        let v1 = nalgebra::Vector3::new(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        let v2 = nalgebra::Vector3::new(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);

        let len1 = v1.norm();
        let len2 = v2.norm();

        if len1 > 1e-6 && len2 > 1e-6 {
            let cos_angle = v1.dot(&v2) / (len1 * len2);
            let angle = cos_angle.clamp(-1.0, 1.0).acos();

            if angle > threshold_rad {
                changes += 1;
            }
        }
    }

    changes
}

/// Kinesphere analysis - the personal space envelope of movement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kinesphere {
    /// Center position (typically torso/pelvis)
    pub center: Position3D,
    /// Maximum reach extent in each direction
    pub extent: KinesphereExtent,
    /// Volume of the movement envelope (cubic meters)
    pub volume: f64,
    /// Ratio of used space to available space
    pub utilization: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KinesphereExtent {
    pub forward: f64,
    pub backward: f64,
    pub left: f64,
    pub right: f64,
    pub up: f64,
    pub down: f64,
}

impl Kinesphere {
    /// Analyze kinesphere from trajectory
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        if trajectory.points.is_empty() {
            return Self::default();
        }

        // Calculate center (mean position)
        let mut sum = Position3D::origin();
        for point in &trajectory.points {
            sum.x += point.position.x;
            sum.y += point.position.y;
            sum.z += point.position.z;
        }
        let n = trajectory.points.len() as f64;
        let center = Position3D::new(sum.x / n, sum.y / n, sum.z / n);

        // Calculate extents relative to center
        let mut extent = KinesphereExtent {
            forward: 0.0,
            backward: 0.0,
            left: 0.0,
            right: 0.0,
            up: 0.0,
            down: 0.0,
        };

        for point in &trajectory.points {
            let dx = point.position.x - center.x;
            let dy = point.position.y - center.y;
            let dz = point.position.z - center.z;

            if dx > 0.0 {
                extent.forward = extent.forward.max(dx);
            } else {
                extent.backward = extent.backward.max(-dx);
            }

            if dy > 0.0 {
                extent.right = extent.right.max(dy);
            } else {
                extent.left = extent.left.max(-dy);
            }

            if dz > 0.0 {
                extent.up = extent.up.max(dz);
            } else {
                extent.down = extent.down.max(-dz);
            }
        }

        // Calculate volume (ellipsoid approximation)
        let a = (extent.forward + extent.backward) / 2.0;
        let b = (extent.left + extent.right) / 2.0;
        let c = (extent.up + extent.down) / 2.0;
        let volume = (4.0 / 3.0) * std::f64::consts::PI * a * b * c;

        // Utilization: actual path spread vs theoretical envelope
        // Simplified as ratio of movement variance to extent
        let utilization = calculate_utilization(&trajectory.points, &center, &extent);

        Self {
            center,
            extent,
            volume,
            utilization,
        }
    }

    /// Is this a large/expansive kinesphere?
    pub fn is_expansive(&self) -> bool {
        self.volume > 0.5 && self.utilization > 0.3
    }
}

impl Default for Kinesphere {
    fn default() -> Self {
        Self {
            center: Position3D::origin(),
            extent: KinesphereExtent {
                forward: 0.0,
                backward: 0.0,
                left: 0.0,
                right: 0.0,
                up: 0.0,
                down: 0.0,
            },
            volume: 0.0,
            utilization: 0.0,
        }
    }
}

fn calculate_utilization(
    points: &[TrajectoryPoint],
    center: &Position3D,
    extent: &KinesphereExtent,
) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    // Calculate variance in each direction
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut var_z = 0.0;

    for point in points {
        var_x += (point.position.x - center.x).powi(2);
        var_y += (point.position.y - center.y).powi(2);
        var_z += (point.position.z - center.z).powi(2);
    }

    let n = points.len() as f64;
    var_x /= n;
    var_y /= n;
    var_z /= n;

    // Compare to maximum possible variance (uniform distribution over extent)
    let max_var_x = ((extent.forward + extent.backward) / 2.0).powi(2) / 3.0;
    let max_var_y = ((extent.left + extent.right) / 2.0).powi(2) / 3.0;
    let max_var_z = ((extent.up + extent.down) / 2.0).powi(2) / 3.0;

    let util_x = if max_var_x > 1e-6 { (var_x / max_var_x).min(1.0) } else { 0.0 };
    let util_y = if max_var_y > 1e-6 { (var_y / max_var_y).min(1.0) } else { 0.0 };
    let util_z = if max_var_z > 1e-6 { (var_z / max_var_z).min(1.0) } else { 0.0 };

    (util_x + util_y + util_z) / 3.0
}

/// Path entropy calculation for movement predictability
pub fn calculate_path_entropy(points: &[TrajectoryPoint], n_bins: usize) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    // Discretize movement directions into bins
    let mut bin_counts = vec![0usize; n_bins];
    let mut total = 0usize;

    for window in points.windows(2) {
        let dx = window[1].position.x - window[0].position.x;
        let dy = window[1].position.y - window[0].position.y;

        let dist = (dx * dx + dy * dy).sqrt();
        if dist > 1e-6 {
            let angle = dy.atan2(dx); // [-π, π]
            let normalized = (angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
            let bin = ((normalized * n_bins as f64) as usize).min(n_bins - 1);
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

    // Normalize by max entropy (log2(n_bins))
    entropy / (n_bins as f64).log2()
}

/// Normalize a value to [0, 1] range
fn normalize_to_range(value: f64, min: f64, max: f64) -> f64 {
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_core::{Acceleration3D, SubjectId, Timestamp, Velocity3D};

    fn create_straight_trajectory() -> Trajectory {
        let mut traj = Trajectory::new(SubjectId::new());
        for i in 0..10 {
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(i * 100_000_000),
                Position3D::new(i as f64 * 0.1, 0.0, 0.0),
                Velocity3D::new(1.0, 0.0, 0.0),
                Acceleration3D::zero(),
            ));
        }
        traj
    }

    fn create_curved_trajectory() -> Trajectory {
        let mut traj = Trajectory::new(SubjectId::new());
        for i in 0..20 {
            let t = i as f64 * 0.1;
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(i * 100_000_000),
                Position3D::new(t.cos(), t.sin(), 0.0),
                Velocity3D::new(-t.sin(), t.cos(), 0.0),
                Acceleration3D::zero(),
            ));
        }
        traj
    }

    #[test]
    fn test_straight_path_is_direct() {
        let traj = create_straight_trajectory();
        let space = SpaceEffort::from_trajectory(&traj);

        assert!(space.per > 0.95, "Straight path should have high PER");
        assert!(space.is_direct(), "Straight path should be direct");
    }

    #[test]
    fn test_curved_path_is_indirect() {
        let traj = create_curved_trajectory();
        let space = SpaceEffort::from_trajectory(&traj);

        assert!(space.per < 0.5, "Curved path should have low PER");
        assert!(!space.is_direct(), "Curved path should be indirect");
    }

    #[test]
    fn test_path_entropy() {
        let traj = create_straight_trajectory();
        let entropy = calculate_path_entropy(&traj.points, 8);

        // Straight line should have low entropy (predictable direction)
        assert!(entropy < 0.3, "Straight path should have low entropy");
    }
}
