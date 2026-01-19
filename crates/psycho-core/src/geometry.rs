//! Geometric utilities for spatial computations.

use nalgebra::{Matrix3, Point3, Rotation3, Unit, Vector3};
use serde::{Deserialize, Serialize};

use crate::types::Position3D;

/// Orientation in 3D space using quaternion representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Orientation3D {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Orientation3D {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        let norm = (w * w + x * x + y * y + z * z).sqrt();
        Self {
            w: w / norm,
            x: x / norm,
            y: y / norm,
            z: z / norm,
        }
    }

    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create from Euler angles (roll, pitch, yaw) in radians
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        let rotation = Rotation3::from_euler_angles(roll, pitch, yaw);
        let quat = nalgebra::UnitQuaternion::from_rotation_matrix(&rotation);
        Self {
            w: quat.w,
            x: quat.i,
            y: quat.j,
            z: quat.k,
        }
    }

    /// Get yaw angle (heading direction) in radians
    pub fn yaw(&self) -> f64 {
        let siny_cosp = 2.0 * (self.w * self.z + self.x * self.y);
        let cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        siny_cosp.atan2(cosy_cosp)
    }

    /// Get pitch angle in radians
    pub fn pitch(&self) -> f64 {
        let sinp = 2.0 * (self.w * self.y - self.z * self.x);
        if sinp.abs() >= 1.0 {
            std::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        }
    }

    /// Get roll angle in radians
    pub fn roll(&self) -> f64 {
        let sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z);
        let cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        sinr_cosp.atan2(cosr_cosp)
    }

    /// Get the forward direction vector
    pub fn forward(&self) -> Vector3<f64> {
        let rotation = self.to_rotation_matrix();
        rotation * Vector3::new(1.0, 0.0, 0.0)
    }

    pub fn to_rotation_matrix(&self) -> Matrix3<f64> {
        let quat = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            self.w, self.x, self.y, self.z,
        ));
        *quat.to_rotation_matrix().matrix()
    }
}

impl Default for Orientation3D {
    fn default() -> Self {
        Self::identity()
    }
}

/// Bounding box in 3D space
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox3D {
    pub min: Position3D,
    pub max: Position3D,
}

impl BoundingBox3D {
    pub fn new(min: Position3D, max: Position3D) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: &[Position3D]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for p in points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            min_z = min_z.min(p.z);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
            max_z = max_z.max(p.z);
        }

        Some(Self {
            min: Position3D::new(min_x, min_y, min_z),
            max: Position3D::new(max_x, max_y, max_z),
        })
    }

    pub fn center(&self) -> Position3D {
        Position3D::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )
    }

    pub fn volume(&self) -> f64 {
        (self.max.x - self.min.x) * (self.max.y - self.min.y) * (self.max.z - self.min.z)
    }

    pub fn contains(&self, point: &Position3D) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    pub fn intersects(&self, other: &BoundingBox3D) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

/// Ray for intersection testing
#[derive(Debug, Clone, Copy)]
pub struct Ray3D {
    pub origin: Point3<f64>,
    pub direction: Unit<Vector3<f64>>,
}

impl Ray3D {
    pub fn new(origin: Position3D, direction: Vector3<f64>) -> Self {
        Self {
            origin: origin.to_nalgebra(),
            direction: Unit::new_normalize(direction),
        }
    }

    pub fn point_at(&self, t: f64) -> Point3<f64> {
        self.origin + self.direction.as_ref() * t
    }

    /// Test intersection with axis-aligned bounding box
    pub fn intersects_aabb(&self, bbox: &BoundingBox3D) -> Option<f64> {
        let inv_dir = Vector3::new(
            1.0 / self.direction.x,
            1.0 / self.direction.y,
            1.0 / self.direction.z,
        );

        let t1 = (bbox.min.x - self.origin.x) * inv_dir.x;
        let t2 = (bbox.max.x - self.origin.x) * inv_dir.x;
        let t3 = (bbox.min.y - self.origin.y) * inv_dir.y;
        let t4 = (bbox.max.y - self.origin.y) * inv_dir.y;
        let t5 = (bbox.min.z - self.origin.z) * inv_dir.z;
        let t6 = (bbox.max.z - self.origin.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if tmax < 0.0 || tmin > tmax {
            None
        } else {
            Some(if tmin < 0.0 { tmax } else { tmin })
        }
    }
}

/// Zone definition for dealership areas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DealershipZone {
    pub id: String,
    pub name: String,
    pub zone_type: ZoneType,
    pub bounds: BoundingBox3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoneType {
    Showroom,
    VehicleDisplay,
    Kiosk,
    SalesDesk,
    ServiceArea,
    Entrance,
    Exit,
    Walkway,
}

impl DealershipZone {
    pub fn contains_position(&self, pos: &Position3D) -> bool {
        self.bounds.contains(pos)
    }
}

/// Calculate angle between two vectors
pub fn angle_between(v1: &Vector3<f64>, v2: &Vector3<f64>) -> f64 {
    let dot = v1.dot(v2);
    let norms = v1.norm() * v2.norm();
    if norms < 1e-10 {
        0.0
    } else {
        (dot / norms).clamp(-1.0, 1.0).acos()
    }
}

/// Check if a subject is facing a target position
pub fn is_facing_target(
    subject_pos: &Position3D,
    subject_orientation: &Orientation3D,
    target_pos: &Position3D,
    threshold_radians: f64,
) -> bool {
    let to_target = Vector3::new(
        target_pos.x - subject_pos.x,
        target_pos.y - subject_pos.y,
        target_pos.z - subject_pos.z,
    );

    let forward = subject_orientation.forward();
    let angle = angle_between(&forward, &to_target);

    angle < threshold_radians
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_orientation_from_euler() {
        let ori = Orientation3D::from_euler(0.0, 0.0, PI / 2.0);
        assert!((ori.yaw() - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox3D::new(Position3D::new(0.0, 0.0, 0.0), Position3D::new(1.0, 1.0, 1.0));

        assert!(bbox.contains(&Position3D::new(0.5, 0.5, 0.5)));
        assert!(!bbox.contains(&Position3D::new(1.5, 0.5, 0.5)));
    }

    #[test]
    fn test_is_facing_target() {
        let subject_pos = Position3D::origin();
        let ori = Orientation3D::identity(); // Facing +X
        let target = Position3D::new(1.0, 0.0, 0.0);

        assert!(is_facing_target(
            &subject_pos,
            &ori,
            &target,
            PI / 4.0
        ));
    }
}
