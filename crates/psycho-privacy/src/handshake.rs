//! Spatio-temporal handshake for linking WiFi trajectories to kiosk sessions.

use psycho_core::{
    geometry::{is_facing_target, Orientation3D},
    Position3D, SessionId, SubjectId, Timestamp, Trajectory, Velocity3D,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for handshake matching
#[derive(Debug, Clone)]
pub struct HandshakeConfig {
    /// Maximum distance to kiosk for match (meters)
    pub max_distance: f64,
    /// Maximum angle deviation from facing kiosk (radians)
    pub max_angle: f64,
    /// Maximum velocity during interaction (m/s)
    pub max_velocity: f64,
    /// Time window around click event (seconds)
    pub time_window: f64,
    /// Minimum deceleration for match (m/s²)
    pub min_deceleration: f64,
}

impl Default for HandshakeConfig {
    fn default() -> Self {
        Self {
            max_distance: 2.0,
            max_angle: std::f64::consts::FRAC_PI_4, // 45 degrees
            max_velocity: 0.2,
            time_window: 2.0,
            min_deceleration: -0.5, // Negative = slowing down
        }
    }
}

/// Kiosk interaction event from UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KioskEvent {
    /// Session ID created for this interaction
    pub session_id: SessionId,
    /// Kiosk position in dealership
    pub kiosk_position: Position3D,
    /// Timestamp of first interaction (click, touch)
    pub interaction_time: Timestamp,
    /// Type of interaction
    pub event_type: KioskEventType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KioskEventType {
    InitialTouch,
    ButtonClick,
    FormSubmit,
    QuestionnaireStart,
}

/// Handshake match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeMatch {
    /// Matched subject ID
    pub subject_id: SubjectId,
    /// Linked session ID
    pub session_id: SessionId,
    /// Match confidence [0, 1]
    pub confidence: f64,
    /// Matching criteria scores
    pub scores: MatchScores,
    /// When the match was established
    pub match_time: Timestamp,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MatchScores {
    /// Distance score [0=far, 1=at kiosk]
    pub distance: f64,
    /// Orientation score [0=facing away, 1=facing kiosk]
    pub orientation: f64,
    /// Velocity score [0=moving fast, 1=stopped]
    pub velocity: f64,
    /// Timing score [0=poor timing, 1=perfect timing]
    pub timing: f64,
    /// Deceleration score [0=no decel, 1=strong decel]
    pub deceleration: f64,
}

impl MatchScores {
    /// Calculate overall confidence from individual scores
    pub fn confidence(&self) -> f64 {
        // Weighted average with emphasis on critical factors
        let weights = [0.25, 0.20, 0.25, 0.15, 0.15]; // distance, orientation, velocity, timing, decel
        let scores = [
            self.distance,
            self.orientation,
            self.velocity,
            self.timing,
            self.deceleration,
        ];

        weights
            .iter()
            .zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum()
    }
}

/// Spatio-temporal handshake matcher
pub struct HandshakeMatcher {
    config: HandshakeConfig,
    /// Pending kiosk events awaiting match
    pending_events: parking_lot::RwLock<Vec<KioskEvent>>,
    /// Confirmed matches
    matches: parking_lot::RwLock<HashMap<SessionId, HandshakeMatch>>,
}

impl HandshakeMatcher {
    pub fn new(config: HandshakeConfig) -> Self {
        Self {
            config,
            pending_events: parking_lot::RwLock::new(Vec::new()),
            matches: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Register a kiosk interaction event
    pub fn register_kiosk_event(&self, event: KioskEvent) {
        let mut pending = self.pending_events.write();
        pending.push(event);
    }

    /// Attempt to match a trajectory to pending kiosk events
    pub fn try_match(&self, subject_id: SubjectId, trajectory: &Trajectory) -> Option<HandshakeMatch> {
        let pending = self.pending_events.read();

        for event in pending.iter() {
            if let Some(match_result) = self.evaluate_match(subject_id, trajectory, event) {
                if match_result.confidence >= 0.6 {
                    return Some(match_result);
                }
            }
        }

        None
    }

    /// Confirm a match and remove from pending
    pub fn confirm_match(&self, match_result: HandshakeMatch) {
        let mut matches = self.matches.write();
        let mut pending = self.pending_events.write();

        matches.insert(match_result.session_id, match_result.clone());

        // Remove matched event from pending
        pending.retain(|e| e.session_id != match_result.session_id);
    }

    /// Evaluate if a trajectory matches a kiosk event
    fn evaluate_match(
        &self,
        subject_id: SubjectId,
        trajectory: &Trajectory,
        event: &KioskEvent,
    ) -> Option<HandshakeMatch> {
        // Find trajectory point closest in time to kiosk interaction
        let event_nanos = event.interaction_time.as_nanos();
        let window_nanos = (self.config.time_window * 1e9) as i64;

        let relevant_points: Vec<_> = trajectory
            .points
            .iter()
            .filter(|p| {
                let dt = (p.timestamp.as_nanos() - event_nanos).abs();
                dt <= window_nanos
            })
            .collect();

        if relevant_points.is_empty() {
            return None;
        }

        // Find point closest to interaction time
        let closest = relevant_points
            .iter()
            .min_by_key(|p| (p.timestamp.as_nanos() - event_nanos).abs())
            .unwrap();

        // Calculate match scores
        let distance = closest.position.distance_to(&event.kiosk_position);
        let distance_score = if distance <= self.config.max_distance {
            1.0 - (distance / self.config.max_distance)
        } else {
            0.0
        };

        // Estimate orientation from velocity direction
        let orientation = if closest.velocity.magnitude() > 0.01 {
            let vel_vec = closest.velocity.to_vector();
            let to_kiosk = nalgebra::Vector3::new(
                event.kiosk_position.x - closest.position.x,
                event.kiosk_position.y - closest.position.y,
                event.kiosk_position.z - closest.position.z,
            );

            let angle = vel_vec.normalize().dot(&to_kiosk.normalize()).acos();
            Orientation3D::from_euler(0.0, 0.0, angle)
        } else {
            Orientation3D::identity()
        };

        let is_facing = is_facing_target(
            &closest.position,
            &orientation,
            &event.kiosk_position,
            self.config.max_angle,
        );
        let orientation_score = if is_facing { 1.0 } else { 0.3 };

        // Velocity check
        let velocity = closest.velocity.magnitude();
        let velocity_score = if velocity <= self.config.max_velocity {
            1.0 - (velocity / self.config.max_velocity)
        } else {
            0.0
        };

        // Timing: how close to the click event
        let time_diff = (closest.timestamp.as_nanos() - event_nanos).abs() as f64 / 1e9;
        let timing_score = 1.0 - (time_diff / self.config.time_window).min(1.0);

        // Deceleration: check if subject was slowing down
        let decel_score = if relevant_points.len() >= 2 {
            let decel = closest.acceleration.magnitude();
            if closest.acceleration.az < self.config.min_deceleration {
                1.0 - (closest.acceleration.az / self.config.min_deceleration).abs().min(1.0)
            } else {
                0.0
            }
        } else {
            0.5 // Neutral if we can't calculate
        };

        let scores = MatchScores {
            distance: distance_score,
            orientation: orientation_score,
            velocity: velocity_score,
            timing: timing_score,
            deceleration: decel_score,
        };

        let confidence = scores.confidence();

        Some(HandshakeMatch {
            subject_id,
            session_id: event.session_id,
            confidence,
            scores,
            match_time: Timestamp::now(),
        })
    }

    /// Get confirmed match for a session
    pub fn get_match(&self, session_id: SessionId) -> Option<HandshakeMatch> {
        let matches = self.matches.read();
        matches.get(&session_id).cloned()
    }

    /// Get subject for a session
    pub fn get_subject(&self, session_id: SessionId) -> Option<SubjectId> {
        let matches = self.matches.read();
        matches.get(&session_id).map(|m| m.subject_id)
    }

    /// Clear old pending events
    pub fn cleanup_old_events(&self, max_age_secs: f64) {
        let mut pending = self.pending_events.write();
        let now = Timestamp::now().as_nanos();
        let max_age_nanos = (max_age_secs * 1e9) as i64;

        pending.retain(|e| {
            let age = now - e.interaction_time.as_nanos();
            age < max_age_nanos
        });
    }

    pub fn config(&self) -> &HandshakeConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> HandshakeStats {
        let matches = self.matches.read();
        let pending = self.pending_events.read();

        HandshakeStats {
            total_matches: matches.len(),
            pending_events: pending.len(),
            avg_confidence: if matches.is_empty() {
                0.0
            } else {
                matches.values().map(|m| m.confidence).sum::<f64>() / matches.len() as f64
            },
        }
    }
}

impl Default for HandshakeMatcher {
    fn default() -> Self {
        Self::new(HandshakeConfig::default())
    }
}

/// Handshake statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeStats {
    pub total_matches: usize,
    pub pending_events: usize,
    pub avg_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_core::{Acceleration3D, TrajectoryPoint};

    #[test]
    fn test_handshake_match() {
        let matcher = HandshakeMatcher::default();
        let subject_id = SubjectId::new();

        // Register kiosk event
        let event = KioskEvent {
            session_id: SessionId::new(),
            kiosk_position: Position3D::new(5.0, 5.0, 1.5),
            interaction_time: Timestamp::from_nanos(1_000_000_000),
            event_type: KioskEventType::InitialTouch,
        };
        matcher.register_kiosk_event(event.clone());

        // Create trajectory approaching kiosk
        let mut trajectory = Trajectory::new(subject_id);
        for i in 0..20 {
            let t = i as f64 * 0.1;
            let pos = Position3D::new(5.0 - t, 5.0, 1.5);
            let vel = Velocity3D::new(1.0 - t * 0.1, 0.0, 0.0); // Slowing down

            trajectory.add_point(TrajectoryPoint::new(
                Timestamp::from_nanos(900_000_000 + i * 50_000_000),
                pos,
                vel,
                Acceleration3D::new(-0.5, 0.0, 0.0),
            ));
        }

        // Try to match
        let match_result = matcher.try_match(subject_id, &trajectory);
        assert!(match_result.is_some());

        let match_result = match_result.unwrap();
        assert!(match_result.confidence > 0.5);
        assert_eq!(match_result.session_id, event.session_id);
    }
}
