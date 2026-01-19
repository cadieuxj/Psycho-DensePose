// Trajectory heatmap compute shader

struct TrajectoryPoint {
    position: vec2<f32>,
    timestamp: f32,
    weight: f32,
}

struct Params {
    width: u32,
    height: u32,
    sigma: f32,
    max_intensity: f32,
}

@group(0) @binding(0)
var<storage, read> points: array<TrajectoryPoint>;

@group(0) @binding(1)
var heatmap: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var<uniform> params: Params;

// Gaussian kernel for smooth heatmap
fn gaussian_2d(distance: f32, sigma: f32) -> f32 {
    let sigma_sq = sigma * sigma;
    return exp(-(distance * distance) / (2.0 * sigma_sq));
}

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coords = vec2<u32>(global_id.xy);

    if (pixel_coords.x >= params.width || pixel_coords.y >= params.height) {
        return;
    }

    // Normalize pixel coordinates to [0, 1]
    let normalized_pos = vec2<f32>(
        f32(pixel_coords.x) / f32(params.width),
        f32(pixel_coords.y) / f32(params.height)
    );

    // Accumulate contribution from all trajectory points
    var intensity: f32 = 0.0;
    let num_points = arrayLength(&points);

    for (var i: u32 = 0u; i < num_points; i++) {
        let point = points[i];
        let diff = normalized_pos - point.position;
        let distance = length(diff);

        // Add Gaussian contribution
        intensity += gaussian_2d(distance, params.sigma) * point.weight;
    }

    // Normalize intensity
    intensity = min(intensity / params.max_intensity, 1.0);

    // Write to heatmap texture
    textureStore(heatmap, pixel_coords, vec4<f32>(intensity, 0.0, 0.0, 0.0));
}
