// src/composite.wgsl

// Full-res radiance storage texture (read-only)
@group(0) @binding(0)
var radiance: texture_storage_2d<rgba32float, read>;

struct VSOut {
    @builtin(position) Position: vec4<f32>,
    @location(0) TexCoord: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0)
    );
    var out: VSOut;
    out.Position = vec4<f32>(pos[vi], 0.0, 1.0);
    out.TexCoord = uv[vi];
    return out;
}

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>, @location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(i32(fragCoord.x), i32(fragCoord.y));
    let light = textureLoad(radiance, coord);
    var color = light.rgb;
    
    // Apply stronger exposure to make shadows more visible
    let exposure = 2.0;
    color *= exposure;
    
    // Apply ACES tonemapping
    color = tonemap_aces(color);
    
    // Very minimal ambient light
    let ambient = 0.0005;  // Reduced ambient for darker shadows
    color += vec3<f32>(ambient);
    
    // Enhance shadow contrast
    color = pow(color, vec3<f32>(0.7));  // Gamma less than 1 to enhance shadows
    
    // Final gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(color, 1.0);
}
