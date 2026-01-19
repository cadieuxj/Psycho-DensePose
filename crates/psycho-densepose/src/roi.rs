//! ROI Align implementation for WiFi-based human localization.
//!
//! ROI Align extracts fixed-size features from regions of interest,
//! enabling the model to focus on human bodies detected in the CSI signal.

use candle_core::{DType, Device, Result, Tensor};

/// Region of Interest specification
#[derive(Debug, Clone, Copy)]
pub struct Roi {
    /// Batch index
    pub batch_idx: usize,
    /// Start position (normalized 0-1)
    pub start: f32,
    /// End position (normalized 0-1)
    pub end: f32,
    /// Confidence score
    pub score: f32,
}

impl Roi {
    pub fn new(batch_idx: usize, start: f32, end: f32, score: f32) -> Self {
        Self {
            batch_idx,
            start: start.clamp(0.0, 1.0),
            end: end.clamp(0.0, 1.0),
            score,
        }
    }

    pub fn width(&self) -> f32 {
        (self.end - self.start).abs()
    }

    pub fn center(&self) -> f32 {
        (self.start + self.end) / 2.0
    }
}

/// Configuration for ROI Align
#[derive(Debug, Clone)]
pub struct RoiAlignConfig {
    /// Output size after alignment
    pub output_size: usize,
    /// Spatial scale (feature_size / input_size)
    pub spatial_scale: f32,
    /// Sampling ratio for bilinear interpolation
    pub sampling_ratio: usize,
}

impl Default for RoiAlignConfig {
    fn default() -> Self {
        Self {
            output_size: 14,
            spatial_scale: 1.0 / 16.0,
            sampling_ratio: 2,
        }
    }
}

/// ROI Align module for 1D feature extraction
pub struct RoiAlign1d {
    config: RoiAlignConfig,
}

impl RoiAlign1d {
    pub fn new(config: RoiAlignConfig) -> Self {
        Self { config }
    }

    /// Extract aligned features for given ROIs
    ///
    /// # Arguments
    /// * `features` - Feature tensor [batch, channels, length]
    /// * `rois` - List of regions of interest
    ///
    /// # Returns
    /// Aligned features [n_rois, channels, output_size]
    pub fn forward(&self, features: &Tensor, rois: &[Roi]) -> Result<Tensor> {
        if rois.is_empty() {
            let (_, c, _) = features.dims3()?;
            return Tensor::zeros(
                (0, c, self.config.output_size),
                features.dtype(),
                features.device(),
            );
        }

        let (batch, channels, length) = features.dims3()?;
        let n_rois = rois.len();
        let output_size = self.config.output_size;

        // Pre-allocate output
        let mut aligned_features = Vec::with_capacity(n_rois);

        for roi in rois {
            if roi.batch_idx >= batch {
                continue;
            }

            // Get batch features
            let batch_features = features.i(roi.batch_idx)?;

            // Calculate ROI bounds in feature space
            let roi_start = (roi.start * length as f32 * self.config.spatial_scale) as f32;
            let roi_end = (roi.end * length as f32 * self.config.spatial_scale) as f32;
            let roi_length = (roi_end - roi_start).max(1.0);

            // Sample points for each output bin
            let bin_size = roi_length / output_size as f32;
            let mut roi_output = Vec::with_capacity(channels * output_size);

            for c in 0..channels {
                let channel_data: Vec<f32> = batch_features.i(c)?.to_vec1()?;

                for out_idx in 0..output_size {
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    // Sample within the bin
                    for sy in 0..self.config.sampling_ratio {
                        let y = roi_start
                            + (out_idx as f32 + (sy as f32 + 0.5) / self.config.sampling_ratio as f32)
                                * bin_size;

                        // Bilinear interpolation
                        let y_low = (y.floor() as usize).min(length - 1);
                        let y_high = (y.ceil() as usize).min(length - 1);
                        let y_frac = y - y.floor();

                        if y_low < length && y_high < length {
                            let interpolated = channel_data[y_low] * (1.0 - y_frac)
                                + channel_data[y_high] * y_frac;
                            sum += interpolated;
                            count += 1;
                        }
                    }

                    roi_output.push(if count > 0 { sum / count as f32 } else { 0.0 });
                }
            }

            // Reshape to [channels, output_size]
            let roi_tensor = Tensor::from_vec(
                roi_output,
                (channels, output_size),
                features.device(),
            )?;
            aligned_features.push(roi_tensor);
        }

        // Stack all ROI features
        if aligned_features.is_empty() {
            Tensor::zeros(
                (0, channels, output_size),
                features.dtype(),
                features.device(),
            )
        } else {
            Tensor::stack(&aligned_features, 0)
        }
    }

    pub fn config(&self) -> &RoiAlignConfig {
        &self.config
    }
}

/// Simple ROI proposal generator based on CSI activity
pub struct RoiProposer {
    /// Minimum proposal width
    pub min_width: f32,
    /// Maximum number of proposals
    pub max_proposals: usize,
    /// Activity threshold for proposal generation
    pub threshold: f32,
}

impl Default for RoiProposer {
    fn default() -> Self {
        Self {
            min_width: 0.1,
            max_proposals: 10,
            threshold: 0.5,
        }
    }
}

impl RoiProposer {
    pub fn new(min_width: f32, max_proposals: usize, threshold: f32) -> Self {
        Self {
            min_width,
            max_proposals,
            threshold,
        }
    }

    /// Generate ROI proposals from activity map
    ///
    /// # Arguments
    /// * `activity_map` - Activity scores [batch, length]
    ///
    /// # Returns
    /// List of ROI proposals
    pub fn propose(&self, activity_map: &Tensor) -> Result<Vec<Roi>> {
        let (batch, length) = activity_map.dims2()?;
        let mut proposals = Vec::new();

        for b in 0..batch {
            let activity: Vec<f32> = activity_map.i(b)?.to_vec1()?;

            // Find contiguous regions above threshold
            let mut in_region = false;
            let mut region_start = 0;
            let mut region_max = 0.0f32;

            for (i, &score) in activity.iter().enumerate() {
                if score > self.threshold {
                    if !in_region {
                        in_region = true;
                        region_start = i;
                        region_max = score;
                    } else {
                        region_max = region_max.max(score);
                    }
                } else if in_region {
                    // End of region
                    let start = region_start as f32 / length as f32;
                    let end = i as f32 / length as f32;

                    if end - start >= self.min_width {
                        proposals.push(Roi::new(b, start, end, region_max));
                    }

                    in_region = false;
                }
            }

            // Handle region that extends to end
            if in_region {
                let start = region_start as f32 / length as f32;
                let end = 1.0;

                if end - start >= self.min_width {
                    proposals.push(Roi::new(b, start, end, region_max));
                }
            }
        }

        // Sort by score and limit
        proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        proposals.truncate(self.max_proposals);

        Ok(proposals)
    }
}

/// Non-Maximum Suppression for ROI filtering
pub fn nms_1d(rois: &[Roi], iou_threshold: f32) -> Vec<Roi> {
    if rois.is_empty() {
        return Vec::new();
    }

    let mut sorted_rois = rois.to_vec();
    sorted_rois.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; sorted_rois.len()];

    for i in 0..sorted_rois.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(sorted_rois[i]);

        for j in (i + 1)..sorted_rois.len() {
            if suppressed[j] {
                continue;
            }

            let iou = compute_iou_1d(&sorted_rois[i], &sorted_rois[j]);
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// Compute 1D IoU between two ROIs
fn compute_iou_1d(roi1: &Roi, roi2: &Roi) -> f32 {
    let intersection_start = roi1.start.max(roi2.start);
    let intersection_end = roi1.end.min(roi2.end);
    let intersection = (intersection_end - intersection_start).max(0.0);

    let union = roi1.width() + roi2.width() - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roi_align_shapes() -> Result<()> {
        let device = Device::Cpu;
        let features = Tensor::randn(0f32, 1.0, (2, 64, 128), &device)?;

        let config = RoiAlignConfig {
            output_size: 7,
            spatial_scale: 1.0 / 4.0,
            sampling_ratio: 2,
        };

        let roi_align = RoiAlign1d::new(config);

        let rois = vec![
            Roi::new(0, 0.1, 0.4, 0.9),
            Roi::new(0, 0.5, 0.8, 0.8),
            Roi::new(1, 0.2, 0.6, 0.7),
        ];

        let aligned = roi_align.forward(&features, &rois)?;
        assert_eq!(aligned.dims(), &[3, 64, 7]);

        Ok(())
    }

    #[test]
    fn test_nms() {
        let rois = vec![
            Roi::new(0, 0.1, 0.4, 0.9),
            Roi::new(0, 0.15, 0.45, 0.8), // Overlaps with first
            Roi::new(0, 0.6, 0.9, 0.7),   // No overlap
        ];

        let kept = nms_1d(&rois, 0.5);

        // First and third should be kept
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 0.01);
        assert!((kept[1].score - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_iou_calculation() {
        let roi1 = Roi::new(0, 0.0, 0.5, 1.0);
        let roi2 = Roi::new(0, 0.25, 0.75, 1.0);

        let iou = compute_iou_1d(&roi1, &roi2);

        // Intersection: 0.25-0.5 = 0.25
        // Union: 0.5 + 0.5 - 0.25 = 0.75
        // IoU = 0.25 / 0.75 = 0.333...
        assert!((iou - 0.333).abs() < 0.01);
    }
}
