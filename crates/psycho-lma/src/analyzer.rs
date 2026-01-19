//! Complete LMA analyzer orchestrating all movement analysis components.

use psycho_core::{DensePoseFrame, SubjectId, Timestamp, Trajectory};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::metrics::MovementMetrics;
use crate::ocean_mapper::{OceanMapper, PsychometricProfile};

/// Complete LMA analyzer
pub struct LmaAnalyzer {
    ocean_mapper: OceanMapper,
    /// Trajectory buffer per subject
    trajectories: RwLock<HashMap<SubjectId, Trajectory>>,
    /// Analysis results cache
    profiles: RwLock<HashMap<SubjectId, PsychometricProfile>>,
}

impl LmaAnalyzer {
    pub fn new() -> Self {
        Self {
            ocean_mapper: OceanMapper::default(),
            trajectories: RwLock::new(HashMap::new()),
            profiles: RwLock::new(HashMap::new()),
        }
    }

    /// Process a DensePose frame and update subject trajectory
    pub async fn process_frame(&self, frame: &DensePoseFrame) {
        let mut trajectories = self.trajectories.write().await;

        let trajectory = trajectories
            .entry(frame.subject_id)
            .or_insert_with(|| Trajectory::new(frame.subject_id));

        // Extract center of mass from skeletal pose
        if let Some(com) = frame.skeletal_pose.center_of_mass() {
            // Add trajectory point (simplified - would need velocity/acceleration from frame delta)
            let point = psycho_core::TrajectoryPoint::new(
                frame.timestamp,
                com,
                psycho_core::Velocity3D::zero(),
                psycho_core::Acceleration3D::zero(),
            );
            trajectory.add_point(point);
        }
    }

    /// Analyze a subject and generate psychometric profile
    pub async fn analyze_subject(&self, subject_id: SubjectId) -> Option<PsychometricProfile> {
        let trajectories = self.trajectories.read().await;
        let trajectory = trajectories.get(&subject_id)?;

        // Need sufficient data for reliable analysis
        if trajectory.points.len() < 10 {
            return None;
        }

        // Extract movement metrics
        let metrics = MovementMetrics::from_trajectory(trajectory);

        // Generate psychometric profile
        let profile = PsychometricProfile::from_metrics(&metrics);

        // Cache result
        let mut profiles = self.profiles.write().await;
        profiles.insert(subject_id, profile.clone());

        Some(profile)
    }

    /// Get cached profile for a subject
    pub async fn get_profile(&self, subject_id: SubjectId) -> Option<PsychometricProfile> {
        let profiles = self.profiles.read().await;
        profiles.get(&subject_id).cloned()
    }

    /// Get all active subjects
    pub async fn active_subjects(&self) -> Vec<SubjectId> {
        let trajectories = self.trajectories.read().await;
        trajectories.keys().copied().collect()
    }

    /// Clear trajectory and profile for a subject
    pub async fn clear_subject(&self, subject_id: SubjectId) {
        let mut trajectories = self.trajectories.write().await;
        let mut profiles = self.profiles.write().await;
        trajectories.remove(&subject_id);
        profiles.remove(&subject_id);
    }

    /// Clear all data
    pub async fn clear_all(&self) {
        let mut trajectories = self.trajectories.write().await;
        let mut profiles = self.profiles.write().await;
        trajectories.clear();
        profiles.clear();
    }

    /// Get trajectory for a subject
    pub async fn get_trajectory(&self, subject_id: SubjectId) -> Option<Trajectory> {
        let trajectories = self.trajectories.read().await;
        trajectories.get(&subject_id).cloned()
    }

    /// Batch analyze all subjects
    pub async fn analyze_all(&self) -> Vec<(SubjectId, PsychometricProfile)> {
        let subjects = self.active_subjects().await;
        let mut results = Vec::new();

        for subject_id in subjects {
            if let Some(profile) = self.analyze_subject(subject_id).await {
                results.push((subject_id, profile));
            }
        }

        results
    }
}

impl Default for LmaAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub subject_id: SubjectId,
    pub timestamp: Timestamp,
    pub profile: PsychometricProfile,
    pub trajectory_length: usize,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use psycho_core::{
        Acceleration3D, KeypointDetection, Position3D, SkeletalPose, Keypoint,
    };

    #[tokio::test]
    async fn test_analyzer_workflow() {
        let analyzer = LmaAnalyzer::new();
        let subject_id = SubjectId::new();

        // Create test frames
        for i in 0..20 {
            let mut keypoints = [None; Keypoint::COUNT];
            keypoints[0] = Some(KeypointDetection::new(
                Keypoint::Nose,
                Position3D::new(i as f64 * 0.1, 0.0, 0.0),
                0.9,
            ));

            let skeletal_pose = SkeletalPose {
                timestamp: Timestamp::from_nanos(i * 100_000_000),
                subject_id,
                keypoints,
                overall_confidence: 0.9,
            };

            let frame = DensePoseFrame {
                timestamp: Timestamp::from_nanos(i * 100_000_000),
                subject_id,
                skeletal_pose,
                surface_points: Vec::new(),
                body_part_mask: Vec::new(),
                mask_width: 0,
                mask_height: 0,
            };

            analyzer.process_frame(&frame).await;
        }

        // Analyze
        let profile = analyzer.analyze_subject(subject_id).await;
        assert!(profile.is_some());

        // Retrieve
        let cached = analyzer.get_profile(subject_id).await;
        assert!(cached.is_some());

        // Clear
        analyzer.clear_subject(subject_id).await;
        let after_clear = analyzer.get_profile(subject_id).await;
        assert!(after_clear.is_none());
    }
}
