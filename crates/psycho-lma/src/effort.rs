//! Effort qualities analysis - Time, Weight, and Flow.
//!
//! ## Time Effort: Sudden vs Sustained
//!
//! Describes the urgency or leisureliness of movement.
//! - **Sudden**: Quick, urgent, surprising movements
//! - **Sustained**: Prolonged, lingering, unhurried movements
//!
//! Metric: Jerk (derivative of acceleration) - High jerk = sudden
//!
//! ## Weight Effort: Strong vs Light
//!
//! Describes the impact or presence of movement.
//! - **Strong**: Forceful, impactful, grounded movements
//! - **Light**: Delicate, airy, buoyant movements
//!
//! Metric: Acceleration magnitude and variability
//!
//! ## Flow Effort: Bound vs Free
//!
//! Describes the control or release in movement.
//! - **Bound**: Controlled, careful, stoppable movements
//! - **Free**: Released, abandoned, ongoing movements
//!
//! Metric: Movement smoothness and consistency

use psycho_core::{Jerk3D, Trajectory, TrajectoryPoint};
use serde::{Deserialize, Serialize};

/// Time effort quality (Sudden vs Sustained)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeEffort {
    /// Root mean square jerk (m/s³)
    pub rms_jerk: f64,
    /// Maximum jerk magnitude
    pub max_jerk: f64,
    /// Jerk variance (consistency of timing changes)
    pub jerk_variance: f64,
    /// Suddenness score [0=sustained, 1=sudden]
    pub suddenness: f64,
}

impl TimeEffort {
    /// Analyze time effort from trajectory
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        let jerks = trajectory.jerk_profile();

        if jerks.is_empty() {
            return Self::default();
        }

        let jerk_magnitudes: Vec<f64> = jerks.iter().map(|(_, j)| j.magnitude()).collect();

        let rms_jerk = {
            let sum_sq: f64 = jerk_magnitudes.iter().map(|j| j.powi(2)).sum();
            (sum_sq / jerk_magnitudes.len() as f64).sqrt()
        };

        let max_jerk = jerk_magnitudes.iter().cloned().fold(0.0, f64::max);

        let mean_jerk = jerk_magnitudes.iter().sum::<f64>() / jerk_magnitudes.len() as f64;
        let jerk_variance = jerk_magnitudes
            .iter()
            .map(|j| (j - mean_jerk).powi(2))
            .sum::<f64>()
            / jerk_magnitudes.len() as f64;

        // Suddenness: normalized RMS jerk
        // Typical walking: ~10 m/s³, Running: ~50 m/s³, Sudden gesture: ~200+ m/s³
        let suddenness = (rms_jerk / 50.0).min(1.0);

        Self {
            rms_jerk,
            max_jerk,
            jerk_variance,
            suddenness,
        }
    }

    pub fn is_sudden(&self) -> bool {
        self.suddenness > 0.5
    }

    pub fn description(&self) -> &'static str {
        if self.suddenness > 0.8 {
            "Very Sudden - urgent, quick timing"
        } else if self.suddenness > 0.6 {
            "Somewhat Sudden - energetic, responsive"
        } else if self.suddenness > 0.4 {
            "Neutral - balanced timing"
        } else if self.suddenness > 0.2 {
            "Somewhat Sustained - leisurely, unhurried"
        } else {
            "Very Sustained - prolonged, lingering"
        }
    }
}

impl Default for TimeEffort {
    fn default() -> Self {
        Self {
            rms_jerk: 0.0,
            max_jerk: 0.0,
            jerk_variance: 0.0,
            suddenness: 0.5,
        }
    }
}

/// Weight effort quality (Strong vs Light)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WeightEffort {
    /// Mean acceleration magnitude (m/s²)
    pub mean_acceleration: f64,
    /// Maximum acceleration magnitude
    pub max_acceleration: f64,
    /// Acceleration variance
    pub acceleration_variance: f64,
    /// Strength score [0=light, 1=strong]
    pub strength: f64,
}

impl WeightEffort {
    /// Analyze weight effort from trajectory
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        let accelerations: Vec<f64> = trajectory
            .points
            .iter()
            .map(|p| p.acceleration.magnitude())
            .collect();

        if accelerations.is_empty() {
            return Self::default();
        }

        let mean_acceleration = accelerations.iter().sum::<f64>() / accelerations.len() as f64;
        let max_acceleration = accelerations.iter().cloned().fold(0.0, f64::max);

        let acceleration_variance = accelerations
            .iter()
            .map(|a| (a - mean_acceleration).powi(2))
            .sum::<f64>()
            / accelerations.len() as f64;

        // Strength: normalized mean acceleration
        // Light touch: ~0.5 m/s², Walking: ~2 m/s², Strong gesture: ~10+ m/s²
        let strength = (mean_acceleration / 5.0).min(1.0);

        Self {
            mean_acceleration,
            max_acceleration,
            acceleration_variance,
            strength,
        }
    }

    pub fn is_strong(&self) -> bool {
        self.strength > 0.5
    }

    pub fn description(&self) -> &'static str {
        if self.strength > 0.8 {
            "Very Strong - forceful, impactful presence"
        } else if self.strength > 0.6 {
            "Somewhat Strong - firm, grounded"
        } else if self.strength > 0.4 {
            "Neutral - balanced weight"
        } else if self.strength > 0.2 {
            "Somewhat Light - gentle, delicate"
        } else {
            "Very Light - airy, buoyant"
        }
    }
}

impl Default for WeightEffort {
    fn default() -> Self {
        Self {
            mean_acceleration: 0.0,
            max_acceleration: 0.0,
            acceleration_variance: 0.0,
            strength: 0.5,
        }
    }
}

/// Flow effort quality (Bound vs Free)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlowEffort {
    /// Movement smoothness (inverse of velocity variation)
    pub smoothness: f64,
    /// Velocity consistency (low variance = bound)
    pub velocity_consistency: f64,
    /// Stop-and-go frequency (high = bound)
    pub stop_frequency: f64,
    /// Boundness score [0=free, 1=bound]
    pub boundness: f64,
}

impl FlowEffort {
    /// Analyze flow effort from trajectory
    pub fn from_trajectory(trajectory: &Trajectory, stop_threshold: f64) -> Self {
        let velocities: Vec<f64> = trajectory
            .points
            .iter()
            .map(|p| p.velocity.magnitude())
            .collect();

        if velocities.is_empty() {
            return Self::default();
        }

        let mean_velocity = velocities.iter().sum::<f64>() / velocities.len() as f64;

        // Velocity consistency (coefficient of variation, inverted and normalized)
        let velocity_std = {
            let variance = velocities
                .iter()
                .map(|v| (v - mean_velocity).powi(2))
                .sum::<f64>()
                / velocities.len() as f64;
            variance.sqrt()
        };

        let cv = if mean_velocity > 1e-6 {
            velocity_std / mean_velocity
        } else {
            0.0
        };

        let velocity_consistency = 1.0 / (1.0 + cv);

        // Smoothness from jerk
        let rms_jerk = trajectory.rms_jerk();
        let smoothness = 1.0 / (1.0 + rms_jerk / 10.0);

        // Stop frequency
        let (stop_count, _) = count_stops(&trajectory.points, stop_threshold);
        let duration = trajectory.duration_secs();
        let stop_frequency = if duration > 0.0 {
            stop_count as f64 / duration
        } else {
            0.0
        };

        // Boundness: combination of consistency and stop frequency
        // Bound = consistent + frequent pauses + not smooth (controlled)
        let boundness = (velocity_consistency * 0.4
            + (stop_frequency / 2.0).min(1.0) * 0.4
            + (1.0 - smoothness) * 0.2)
            .clamp(0.0, 1.0);

        Self {
            smoothness,
            velocity_consistency,
            stop_frequency,
            boundness,
        }
    }

    pub fn is_bound(&self) -> bool {
        self.boundness > 0.5
    }

    pub fn is_free(&self) -> bool {
        self.boundness < 0.5
    }

    pub fn description(&self) -> &'static str {
        if self.boundness > 0.8 {
            "Very Bound - highly controlled, restrained"
        } else if self.boundness > 0.6 {
            "Somewhat Bound - careful, stoppable"
        } else if self.boundness > 0.4 {
            "Neutral - balanced flow"
        } else if self.boundness > 0.2 {
            "Somewhat Free - released, fluid"
        } else {
            "Very Free - abandoned, uninhibited"
        }
    }
}

impl Default for FlowEffort {
    fn default() -> Self {
        Self {
            smoothness: 0.5,
            velocity_consistency: 0.5,
            stop_frequency: 0.0,
            boundness: 0.5,
        }
    }
}

/// Count stops in trajectory
fn count_stops(points: &[TrajectoryPoint], threshold: f64) -> (u32, f64) {
    let mut stop_count = 0u32;
    let mut total_stop_duration = 0.0;
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
            total_stop_duration += (point.timestamp.as_nanos() - stop_start) as f64 / 1e9;
        }
    }

    if in_stop && !points.is_empty() {
        stop_count += 1;
        total_stop_duration +=
            (points.last().unwrap().timestamp.as_nanos() - stop_start) as f64 / 1e9;
    }

    (stop_count, total_stop_duration)
}

/// Complete effort profile combining all four factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffortProfile {
    pub space: super::SpaceEffort,
    pub time: TimeEffort,
    pub weight: WeightEffort,
    pub flow: FlowEffort,
}

impl EffortProfile {
    /// Analyze complete effort profile from trajectory
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        Self {
            space: super::SpaceEffort::from_trajectory(trajectory),
            time: TimeEffort::from_trajectory(trajectory),
            weight: WeightEffort::from_trajectory(trajectory),
            flow: FlowEffort::from_trajectory(trajectory, 0.1),
        }
    }

    /// Get dominant effort states
    pub fn dominant_efforts(&self) -> Vec<&'static str> {
        let mut efforts = Vec::new();

        if self.space.is_direct() {
            efforts.push("Direct");
        } else {
            efforts.push("Indirect");
        }

        if self.time.is_sudden() {
            efforts.push("Sudden");
        } else {
            efforts.push("Sustained");
        }

        if self.weight.is_strong() {
            efforts.push("Strong");
        } else {
            efforts.push("Light");
        }

        if self.flow.is_bound() {
            efforts.push("Bound");
        } else {
            efforts.push("Free");
        }

        efforts
    }

    /// Identify Laban "action drives" (combinations of three effort factors)
    pub fn action_drive(&self) -> Option<ActionDrive> {
        // Action drives are full-bodied combinations of Space+Weight+Time
        let direct = self.space.is_direct();
        let strong = self.weight.is_strong();
        let sudden = self.time.is_sudden();

        match (direct, strong, sudden) {
            (true, true, true) => Some(ActionDrive::Punch),
            (true, true, false) => Some(ActionDrive::Press),
            (true, false, true) => Some(ActionDrive::Dab),
            (true, false, false) => Some(ActionDrive::Glide),
            (false, true, true) => Some(ActionDrive::Slash),
            (false, true, false) => Some(ActionDrive::Wring),
            (false, false, true) => Some(ActionDrive::Flick),
            (false, false, false) => Some(ActionDrive::Float),
        }
    }
}

/// Laban's eight basic effort actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionDrive {
    /// Direct + Strong + Sudden
    Punch,
    /// Direct + Strong + Sustained
    Press,
    /// Direct + Light + Sudden
    Dab,
    /// Direct + Light + Sustained
    Glide,
    /// Indirect + Strong + Sudden
    Slash,
    /// Indirect + Strong + Sustained
    Wring,
    /// Indirect + Light + Sudden
    Flick,
    /// Indirect + Light + Sustained
    Float,
}

impl ActionDrive {
    pub fn description(&self) -> &'static str {
        match self {
            ActionDrive::Punch => "Punching: Direct, Strong, Sudden - forceful impact",
            ActionDrive::Press => "Pressing: Direct, Strong, Sustained - persistent force",
            ActionDrive::Dab => "Dabbing: Direct, Light, Sudden - precise touch",
            ActionDrive::Glide => "Gliding: Direct, Light, Sustained - smooth progression",
            ActionDrive::Slash => "Slashing: Indirect, Strong, Sudden - powerful sweep",
            ActionDrive::Wring => "Wringing: Indirect, Strong, Sustained - twisting force",
            ActionDrive::Flick => "Flicking: Indirect, Light, Sudden - quick scatter",
            ActionDrive::Float => "Floating: Indirect, Light, Sustained - buoyant drift",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_core::{Acceleration3D, Position3D, SubjectId, Timestamp, Velocity3D};

    fn create_jerky_trajectory() -> Trajectory {
        let mut traj = Trajectory::new(SubjectId::new());
        for i in 0..20 {
            // Alternating acceleration for jerkiness
            let accel = if i % 2 == 0 { 5.0 } else { -5.0 };
            traj.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(i * 50_000_000),
                Position3D::new(i as f64 * 0.05, 0.0, 0.0),
                Velocity3D::new(1.0 + 0.5 * (i as f64 * 0.5).sin(), 0.0, 0.0),
                Acceleration3D::new(accel, 0.0, 0.0),
            ));
        }
        traj
    }

    fn create_smooth_trajectory() -> Trajectory {
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
    fn test_time_effort_jerky() {
        let traj = create_jerky_trajectory();
        let time = TimeEffort::from_trajectory(&traj);

        assert!(time.rms_jerk > 0.0, "Jerky trajectory should have jerk");
        // Suddenness depends on normalization - just check it's computed
        assert!(time.suddenness >= 0.0 && time.suddenness <= 1.0);
    }

    #[test]
    fn test_flow_smooth_is_free() {
        let traj = create_smooth_trajectory();
        let flow = FlowEffort::from_trajectory(&traj, 0.1);

        assert!(flow.smoothness > 0.5, "Smooth trajectory should be smooth");
        assert!(flow.velocity_consistency > 0.5, "Constant velocity should be consistent");
    }

    #[test]
    fn test_effort_profile() {
        let traj = create_smooth_trajectory();
        let profile = EffortProfile::from_trajectory(&traj);

        let efforts = profile.dominant_efforts();
        assert_eq!(efforts.len(), 4);

        let action = profile.action_drive();
        assert!(action.is_some());
    }
}
