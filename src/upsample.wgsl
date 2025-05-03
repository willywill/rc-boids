// src/upsample.wgsl

struct UpsampleParams {
    srcSize: vec2<f32>,  // coarse resolution
};

@group(0) @binding(0)
var<uniform> upParams: UpsampleParams;

@group(0) @binding(1)
var coarse: texture_storage_2d<rgba32float, read>;

@group(0) @binding(2)
var fineIn: texture_storage_2d<rgba32float, read>;

@group(0) @binding(3)
var fineOut: texture_storage_2d<rgba32float, write>;

const BLEND_FACTOR: f32 = 0.6; // How much of coarse level to blend in

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fx = i32(gid.x);
    let fy = i32(gid.y);

    let cw = i32(upParams.srcSize.x);
    let ch = i32(upParams.srcSize.y);
    let fw = cw * 2;
    let fh = ch * 2;

    if (fx < 0 || fy < 0 || fx >= fw || fy >= fh) {
        return;
    }

    // Get base coarse coordinates
    let cx_base = fx / 2;
    let cy_base = fy / 2;
    
    // Get fractional coordinates for interpolation
    let fx_fract = f32(fx % 2) / 2.0;
    let fy_fract = f32(fy % 2) / 2.0;
    
    // Sample 4 nearest coarse pixels with wider sampling
    let c00 = textureLoad(coarse, vec2<i32>(cx_base, cy_base));
    let c10 = textureLoad(coarse, vec2<i32>(cx_base + 1, cy_base));
    let c01 = textureLoad(coarse, vec2<i32>(cx_base, cy_base + 1));
    let c11 = textureLoad(coarse, vec2<i32>(cx_base + 1, cy_base + 1));
    
    // Bilinear interpolation
    let mix_x0 = mix(c00, c10, fx_fract);
    let mix_x1 = mix(c01, c11, fx_fract);
    let coarse_sample = mix(mix_x0, mix_x1, fy_fract);
    
    // Blend with existing fine level
    let fine_sample = textureLoad(fineIn, vec2<i32>(fx, fy));
    let result = fine_sample + coarse_sample * BLEND_FACTOR;
    
    textureStore(fineOut, vec2<i32>(fx, fy), result);
}
