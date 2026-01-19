//! Fundamental types for the Psycho-DensePose system.

use chrono::{DateTime, Utc};
use nalgebra::{Point2, Point3, Vector3};
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for tracked subjects (anonymized)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubjectId(pub Uuid);

impl SubjectId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SubjectId {
    fn default() -> Self {
        Self::new()
    }
}

/// Session identifier linking kiosk interactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Timestamp wrapper with nanosecond precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp(pub i64);

impl Timestamp {
    pub fn now() -> Self {
        Self(Utc::now().timestamp_nanos_opt().unwrap_or(0))
    }

    pub fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    pub fn as_nanos(&self) -> i64 {
        self.0
    }

    pub fn as_secs_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }

    pub fn to_datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_nanos(self.0)
    }
}

/// WiFi 6E frequency band specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrequencyBand {
    /// 2.4 GHz band (20/40 MHz channels)
    Band2_4GHz,
    /// 5 GHz band (20/40/80/160 MHz channels)
    Band5GHz,
    /// 6 GHz band (WiFi 6E, 20/40/80/160/320 MHz channels)
    Band6GHz,
}

/// Channel bandwidth configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelBandwidth {
    Bw20MHz,
    Bw40MHz,
    Bw80MHz,
    Bw160MHz,
    Bw320MHz,
}

impl ChannelBandwidth {
    /// Returns the number of OFDM subcarriers for this bandwidth
    /// WiFi 6E uses 4x subcarrier density compared to WiFi 5
    pub fn subcarrier_count(&self) -> u16 {
        match self {
            ChannelBandwidth::Bw20MHz => 256,   // 242 usable
            ChannelBandwidth::Bw40MHz => 512,   // 484 usable
            ChannelBandwidth::Bw80MHz => 1024,  // 996 usable
            ChannelBandwidth::Bw160MHz => 2048, // 1992 usable
            ChannelBandwidth::Bw320MHz => 4096, // 3984 usable (WiFi 7)
        }
    }

    /// Returns usable subcarriers (excluding guard and DC)
    pub fn usable_subcarriers(&self) -> u16 {
        match self {
            ChannelBandwidth::Bw20MHz => 242,
            ChannelBandwidth::Bw40MHz => 484,
            ChannelBandwidth::Bw80MHz => 996,
            ChannelBandwidth::Bw160MHz => 1992,
            ChannelBandwidth::Bw320MHz => 3984,
        }
    }
}

/// Antenna configuration for MIMO systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AntennaConfig {
    /// Number of transmit antennas
    pub n_tx: u8,
    /// Number of receive antennas
    pub n_rx: u8,
}

impl AntennaConfig {
    pub fn new(n_tx: u8, n_rx: u8) -> Self {
        Self { n_tx, n_rx }
    }

    /// Total number of spatial streams (Tx * Rx)
    pub fn total_streams(&self) -> usize {
        self.n_tx as usize * self.n_rx as usize
    }

    /// Intel AX210 typical configuration
    pub fn ax210_default() -> Self {
        Self { n_tx: 2, n_rx: 2 }
    }
}

/// 3D position in dealership coordinate system (meters)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Position3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn to_nalgebra(&self) -> Point3<f64> {
        Point3::new(self.x, self.y, self.z)
    }

    pub fn from_nalgebra(p: Point3<f64>) -> Self {
        Self::new(p.x, p.y, p.z)
    }

    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// 3D velocity vector (meters/second)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Velocity3D {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

impl Velocity3D {
    pub fn new(vx: f64, vy: f64, vz: f64) -> Self {
        Self { vx, vy, vz }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn magnitude(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz).sqrt()
    }

    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(self.vx, self.vy, self.vz)
    }
}

/// 3D acceleration vector (meters/second^2)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Acceleration3D {
    pub ax: f64,
    pub ay: f64,
    pub az: f64,
}

impl Acceleration3D {
    pub fn new(ax: f64, ay: f64, az: f64) -> Self {
        Self { ax, ay, az }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn magnitude(&self) -> f64 {
        (self.ax * self.ax + self.ay * self.ay + self.az * self.az).sqrt()
    }

    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(self.ax, self.ay, self.az)
    }
}

/// UV coordinate for DensePose body surface mapping
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UvCoordinate {
    pub u: f32,
    pub v: f32,
}

impl UvCoordinate {
    pub fn new(u: f32, v: f32) -> Self {
        Self { u, v }
    }

    pub fn to_point2(&self) -> Point2<f32> {
        Point2::new(self.u, self.v)
    }
}

/// Body part classification for DensePose (24 parts)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum BodyPart {
    Background = 0,
    Torso = 1,
    RightHand = 2,
    LeftHand = 3,
    LeftFoot = 4,
    RightFoot = 5,
    UpperLegRight = 6,
    UpperLegLeft = 7,
    LowerLegRight = 8,
    LowerLegLeft = 9,
    UpperArmLeft = 10,
    UpperArmRight = 11,
    LowerArmLeft = 12,
    LowerArmRight = 13,
    Head = 14,
    // Extended parts for fine-grained analysis
    RightShoulder = 15,
    LeftShoulder = 16,
    RightElbow = 17,
    LeftElbow = 18,
    RightWrist = 19,
    LeftWrist = 20,
    RightKnee = 21,
    LeftKnee = 22,
    RightAnkle = 23,
    LeftAnkle = 24,
}

impl BodyPart {
    pub fn from_index(idx: u8) -> Option<Self> {
        match idx {
            0 => Some(Self::Background),
            1 => Some(Self::Torso),
            2 => Some(Self::RightHand),
            3 => Some(Self::LeftHand),
            4 => Some(Self::LeftFoot),
            5 => Some(Self::RightFoot),
            6 => Some(Self::UpperLegRight),
            7 => Some(Self::UpperLegLeft),
            8 => Some(Self::LowerLegRight),
            9 => Some(Self::LowerLegLeft),
            10 => Some(Self::UpperArmLeft),
            11 => Some(Self::UpperArmRight),
            12 => Some(Self::LowerArmLeft),
            13 => Some(Self::LowerArmRight),
            14 => Some(Self::Head),
            15 => Some(Self::RightShoulder),
            16 => Some(Self::LeftShoulder),
            17 => Some(Self::RightElbow),
            18 => Some(Self::LeftElbow),
            19 => Some(Self::RightWrist),
            20 => Some(Self::LeftWrist),
            21 => Some(Self::RightKnee),
            22 => Some(Self::LeftKnee),
            23 => Some(Self::RightAnkle),
            24 => Some(Self::LeftAnkle),
            _ => None,
        }
    }
}

/// 17-joint skeletal keypoint definition (COCO format)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Keypoint {
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
}

impl Keypoint {
    pub const COUNT: usize = 17;

    pub fn from_index(idx: u8) -> Option<Self> {
        match idx {
            0 => Some(Self::Nose),
            1 => Some(Self::LeftEye),
            2 => Some(Self::RightEye),
            3 => Some(Self::LeftEar),
            4 => Some(Self::RightEar),
            5 => Some(Self::LeftShoulder),
            6 => Some(Self::RightShoulder),
            7 => Some(Self::LeftElbow),
            8 => Some(Self::RightElbow),
            9 => Some(Self::LeftWrist),
            10 => Some(Self::RightWrist),
            11 => Some(Self::LeftHip),
            12 => Some(Self::RightHip),
            13 => Some(Self::LeftKnee),
            14 => Some(Self::RightKnee),
            15 => Some(Self::LeftAnkle),
            16 => Some(Self::RightAnkle),
            _ => None,
        }
    }

    /// Returns skeleton connectivity pairs for visualization
    pub fn skeleton_pairs() -> &'static [(Keypoint, Keypoint)] {
        &[
            (Keypoint::LeftAnkle, Keypoint::LeftKnee),
            (Keypoint::LeftKnee, Keypoint::LeftHip),
            (Keypoint::RightAnkle, Keypoint::RightKnee),
            (Keypoint::RightKnee, Keypoint::RightHip),
            (Keypoint::LeftHip, Keypoint::RightHip),
            (Keypoint::LeftShoulder, Keypoint::LeftHip),
            (Keypoint::RightShoulder, Keypoint::RightHip),
            (Keypoint::LeftShoulder, Keypoint::RightShoulder),
            (Keypoint::LeftShoulder, Keypoint::LeftElbow),
            (Keypoint::RightShoulder, Keypoint::RightElbow),
            (Keypoint::LeftElbow, Keypoint::LeftWrist),
            (Keypoint::RightElbow, Keypoint::RightWrist),
            (Keypoint::LeftEye, Keypoint::RightEye),
            (Keypoint::Nose, Keypoint::LeftEye),
            (Keypoint::Nose, Keypoint::RightEye),
            (Keypoint::LeftEye, Keypoint::LeftEar),
            (Keypoint::RightEye, Keypoint::RightEar),
            (Keypoint::LeftEar, Keypoint::LeftShoulder),
            (Keypoint::RightEar, Keypoint::RightShoulder),
        ]
    }
}

/// Keypoint detection with confidence score
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KeypointDetection {
    pub keypoint: Keypoint,
    pub position: Position3D,
    pub confidence: f32,
}

impl KeypointDetection {
    pub fn new(keypoint: Keypoint, position: Position3D, confidence: f32) -> Self {
        Self {
            keypoint,
            position,
            confidence,
        }
    }
}

/// Complete skeletal pose with all 17 keypoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletalPose {
    pub timestamp: Timestamp,
    pub subject_id: SubjectId,
    pub keypoints: [Option<KeypointDetection>; Keypoint::COUNT],
    pub overall_confidence: f32,
}

impl SkeletalPose {
    pub fn center_of_mass(&self) -> Option<Position3D> {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;
        let mut count = 0.0;

        for kp in self.keypoints.iter().flatten() {
            sum_x += kp.position.x * kp.confidence as f64;
            sum_y += kp.position.y * kp.confidence as f64;
            sum_z += kp.position.z * kp.confidence as f64;
            count += kp.confidence as f64;
        }

        if count > 0.0 {
            Some(Position3D::new(sum_x / count, sum_y / count, sum_z / count))
        } else {
            None
        }
    }
}

/// DensePose surface point with UV coordinates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DensePosePoint {
    pub body_part: BodyPart,
    pub uv: UvCoordinate,
    pub position_3d: Position3D,
    pub confidence: f32,
}

/// Complete DensePose estimation for a single frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensePoseFrame {
    pub timestamp: Timestamp,
    pub subject_id: SubjectId,
    pub skeletal_pose: SkeletalPose,
    pub surface_points: Vec<DensePosePoint>,
    pub body_part_mask: Vec<u8>, // Flattened 2D mask
    pub mask_width: u32,
    pub mask_height: u32,
}

/// Complex number type alias for CSI data
pub type CsiComplex = Complex<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_bandwidth_subcarriers() {
        assert_eq!(ChannelBandwidth::Bw160MHz.usable_subcarriers(), 1992);
        assert_eq!(ChannelBandwidth::Bw80MHz.usable_subcarriers(), 996);
    }

    #[test]
    fn test_position_distance() {
        let p1 = Position3D::new(0.0, 0.0, 0.0);
        let p2 = Position3D::new(3.0, 4.0, 0.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_body_part_roundtrip() {
        for i in 0..=24 {
            if let Some(part) = BodyPart::from_index(i) {
                assert_eq!(part as u8, i);
            }
        }
    }
}
