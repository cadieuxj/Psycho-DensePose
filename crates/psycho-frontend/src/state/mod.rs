//! Global application state management.

use leptos::*;
use psycho_core::{DensePoseFrame, OceanScores, SubjectId};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct AppState {
    pub active_subjects: HashMap<SubjectId, SubjectState>,
    pub current_session: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SubjectState {
    pub subject_id: SubjectId,
    pub latest_pose: Option<DensePoseFrame>,
    pub ocean_scores: Option<OceanScores>,
    pub trajectory_points: Vec<(f64, f64, f64)>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            active_subjects: HashMap::new(),
            current_session: None,
        }
    }

    pub fn update_pose(&mut self, frame: DensePoseFrame) {
        let subject_state = self.active_subjects
            .entry(frame.subject_id)
            .or_insert_with(|| SubjectState {
                subject_id: frame.subject_id,
                latest_pose: None,
                ocean_scores: None,
                trajectory_points: Vec::new(),
            });

        subject_state.latest_pose = Some(frame);
    }

    pub fn update_ocean(&mut self, subject_id: SubjectId, scores: OceanScores) {
        if let Some(subject_state) = self.active_subjects.get_mut(&subject_id) {
            subject_state.ocean_scores = Some(scores);
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
