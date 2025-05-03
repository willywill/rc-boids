// src/diffuse.wgsl

struct DiffuseParams {
    resolution: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> diffuseParams: DiffuseParams;

@group(0) @binding(1)
var radianceIn: texture_storage_2d<rgba32float, read>;

@group(0) @binding(2)
var barrierMask: texture_2d<f32>;

@group(0) @binding(3)
var radianceOut: texture_storage_2d<rgba32float, write>;

const DECAY_RATE: f32 = 0.97;       // Faster decay for sharper shadows
const DIFFUSE_STRENGTH: f32 = 0.9;  // Reduced diffusion to prevent light bleeding
const CENTER_WEIGHT: f32 = 0.5;     // Higher center weight
const ADJACENT_WEIGHT: f32 = 0.1;   // Lower adjacent for less bleeding
const DIAGONAL_WEIGHT: f32 = 0.025; // Very small diagonal contribution

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = i32(gid.x);
    let py = i32(gid.y);
    if (px >= i32(diffuseParams.resolution.x) ||
        py >= i32(diffuseParams.resolution.y)) {
        return;
    }

    // Get barrier mask value for current pixel
    let barrier = textureLoad(barrierMask, vec2<i32>(px, py), 0).r;
    
    // If we're inside a barrier, block all light
    if (barrier < 0.5) {
        textureStore(radianceOut, vec2<i32>(px, py), vec4<f32>(0.0));
        return;
    }

    // Start with center light
    var accumLight = textureLoad(radianceIn, vec2<i32>(px, py)) * CENTER_WEIGHT;
    
    // Adjacent samples (up, down, left, right)
    let adjacentDirs = array<vec2<i32>, 4>(
        vec2<i32>( 1,  0),
        vec2<i32>(-1,  0),
        vec2<i32>( 0,  1),
        vec2<i32>( 0, -1)
    );
    
    // Sample adjacent pixels
    for (var i = 0u; i < 4u; i = i + 1u) {
        let samplePos = vec2<i32>(px + adjacentDirs[i].x, py + adjacentDirs[i].y);
        
        // Skip if out of bounds
        if (samplePos.x < 0 || samplePos.y < 0 || 
            samplePos.x >= i32(diffuseParams.resolution.x) ||
            samplePos.y >= i32(diffuseParams.resolution.y)) {
            continue;
        }

        // Check for barriers
        let barrierCheck = textureLoad(barrierMask, samplePos, 0).r;
        if (barrierCheck > 0.5) {
            accumLight += textureLoad(radianceIn, samplePos) * ADJACENT_WEIGHT;
        }
    }

    // Diagonal samples
    let diagonalDirs = array<vec2<i32>, 4>(
        vec2<i32>( 1,  1),
        vec2<i32>(-1, -1),
        vec2<i32>( 1, -1),
        vec2<i32>(-1,  1)
    );
    
    // Sample diagonal pixels
    for (var i = 0u; i < 4u; i = i + 1u) {
        let samplePos = vec2<i32>(px + diagonalDirs[i].x, py + diagonalDirs[i].y);
        
        // Skip if out of bounds
        if (samplePos.x < 0 || samplePos.y < 0 || 
            samplePos.x >= i32(diffuseParams.resolution.x) ||
            samplePos.y >= i32(diffuseParams.resolution.y)) {
            continue;
        }

        // Check both cardinal directions for barriers to block diagonal light
        let barrierH = textureLoad(barrierMask, vec2<i32>(px + diagonalDirs[i].x, py), 0).r;
        let barrierV = textureLoad(barrierMask, vec2<i32>(px, py + diagonalDirs[i].y), 0).r;
        let barrierD = textureLoad(barrierMask, samplePos, 0).r;
        
        // Only let light through if there's a clear diagonal path
        if (barrierH > 0.5 && barrierV > 0.5 && barrierD > 0.5) {
            accumLight += textureLoad(radianceIn, samplePos) * DIAGONAL_WEIGHT;
        }
    }

    // Apply diffusion and decay
    let finalLight = accumLight * DIFFUSE_STRENGTH * DECAY_RATE;
    
    // Store result
    textureStore(radianceOut, vec2<i32>(px, py), finalLight);
}
