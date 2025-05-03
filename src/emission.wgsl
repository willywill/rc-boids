// src/emission.wgsl

struct EmissionParams {
    resolution: vec2<f32>,
    time: f32,
    numBoids: u32,
};

@group(0) @binding(0)
var<uniform> params: EmissionParams;

@group(0) @binding(1)
var<storage, read> boids: array<vec4<f32>>;

@group(0) @binding(2)
var radianceOut: texture_storage_2d<rgba32float, write>;  // Changed to rgba32float

const EMISSION_STRENGTH: f32 = 1.0;  // Increased for stronger shadows
const EMISSION_RADIUS: f32 = 0.8;   // Smaller for more point-like sources
const SOFT_FACTOR: f32 = 1.5;

// Hash function for pseudo-random numbers
fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

// Get color for boid index - more vibrant colors
fn getBoidColor(index: u32) -> vec3<f32> {
    let h1 = hash(f32(index) * 1.618);
    let h2 = hash(f32(index) * 2.618);
    let h3 = hash(f32(index) * 3.618);
    
    // Generate more saturated colors
    var colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.2, 0.2),  // Red
        vec3<f32>(0.2, 1.0, 0.2),  // Green
        vec3<f32>(0.2, 0.2, 1.0)   // Blue
    );
    return colors[index];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = i32(gid.x);
    let py = i32(gid.y);
    if (px >= i32(params.resolution.x) ||
        py >= i32(params.resolution.y)) {
        return;
    }

    // Convert pixel coordinates to NDC space [-1, 1]
    let ndc = vec2<f32>(
        (f32(px) / params.resolution.x) * 2.0 - 1.0,
        (f32(py) / params.resolution.y) * 2.0 - 1.0
    );
    
    // Accumulate light from all boids with their unique colors
    var totalLight = vec3<f32>(0.0);
    for (var i = 0u; i < params.numBoids; i++) {
        let boidPos = boids[i].xy;
        let dist = length(boidPos - ndc);
        
        if (dist < EMISSION_RADIUS * SOFT_FACTOR) {
            // Sharper falloff for better defined shadows
            let falloff = pow(1.0 - smoothstep(0.0, EMISSION_RADIUS, dist), 3.0);
            let color = getBoidColor(i);
            totalLight += color * falloff;
        }
    }
    
    // Add slight pulsing effect
    let pulse = sin(params.time * 1.5) * 0.15 + 0.85;
    
    // Store colored result directly
    textureStore(radianceOut, vec2<i32>(px, py), vec4<f32>(totalLight * EMISSION_STRENGTH * pulse, 1.0));
}
