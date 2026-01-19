// Heatmap display shader with color mapping

@group(0) @binding(0)
var heatmap_texture: texture_2d<f32>;

@group(0) @binding(1)
var heatmap_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen quad vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    // Generate fullscreen quad
    let x = f32((vertex_index & 1u) << 1u) - 1.0;
    let y = f32((vertex_index & 2u)) - 1.0;

    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return output;
}

// Heat colormap (blue -> cyan -> green -> yellow -> red)
fn heat_color(value: f32) -> vec3<f32> {
    let v = clamp(value, 0.0, 1.0);

    if (v < 0.25) {
        // Blue to cyan
        let t = v / 0.25;
        return mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0), t);
    } else if (v < 0.5) {
        // Cyan to green
        let t = (v - 0.25) / 0.25;
        return mix(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), t);
    } else if (v < 0.75) {
        // Green to yellow
        let t = (v - 0.5) / 0.25;
        return mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), t);
    } else {
        // Yellow to red
        let t = (v - 0.75) / 0.25;
        return mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), t);
    }
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let intensity = textureSample(heatmap_texture, heatmap_sampler, input.uv).r;

    if (intensity < 0.01) {
        discard;
    }

    let color = heat_color(intensity);
    return vec4<f32>(color, intensity * 0.8);
}
