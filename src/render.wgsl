// src/render.wgsl

struct RenderParams {
    resolution: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> renderParams: RenderParams;

@group(0) @binding(1)
var radianceIn: texture_storage_2d<r32float, read>;

@group(0) @binding(2)
var renderTarget: texture_storage_2d<rgba8unorm, write>;

const EXPOSURE: f32 = 2.0;
const GAMMA: f32 = 2.2;
const COLOR_TEMP: vec3<f32> = vec3<f32>(1.0, 0.7, 0.4); // Warm light color

fn tonemap(color: vec3<f32>) -> vec3<f32> {
    // ACES-like tonemapping
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = i32(gid.x);
    let py = i32(gid.y);
    if (px >= i32(renderParams.resolution.x) ||
        py >= i32(renderParams.resolution.y)) {
        return;
    }

    // Load radiance value
    let radiance = textureLoad(radianceIn, vec2<i32>(px, py));
    
    // Apply exposure and color temperature
    var color = vec3<f32>(radiance.r) * COLOR_TEMP * EXPOSURE;
    
    // Apply tonemapping
    color = tonemap(color);
    
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / GAMMA));
    
    // Store final color
    textureStore(renderTarget, vec2<i32>(px, py), vec4<f32>(color, 1.0));
}
