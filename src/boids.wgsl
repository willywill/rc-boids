// src/boids.wgsl

const MIN_DIST: f32 = 0.1;           // ← tune this to set how close they get

struct SimParams {
    mouse:         vec2<f32>,
    dt:            f32,
    count:         u32,
    accelStrength: f32,
    maxSpeed:      f32,
};

@group(0) @binding(1)
var<uniform> params: SimParams;

@group(0) @binding(0)
var<storage, read_write> boids: array<vec4<f32>>;

@group(0) @binding(2)
var barrierMask: texture_2d<f32>;

fn clampSpeed(v: vec2<f32>, maxSpd: f32) -> vec2<f32> {
    let len = length(v);
    if (len > maxSpd) {
        return normalize(v) * maxSpd;
    }
    return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }

    var b   = boids[i];
    let pos = b.xy;
    var vel = b.zw;

    // steer toward mouse, but only if farther than MIN_DIST
    let toMouse = params.mouse - pos;
    let dist    = length(toMouse);
    if (dist > MIN_DIST) {
        vel += normalize(toMouse) * params.dt * params.accelStrength;
    }

    // clamp & integrate
    vel = clampSpeed(vel, params.maxSpeed);
    var newPos = pos + vel * params.dt;

    // wrap NDC box
    if (newPos.x >  1.0) { newPos.x = -1.0; }
    if (newPos.x < -1.0) { newPos.x =  1.0; }
    if (newPos.y >  1.0) { newPos.y = -1.0; }
    if (newPos.y < -1.0) { newPos.y =  1.0; }

    // Check for collisions with obstacles
    let uv = (newPos * 0.5 + vec2<f32>(0.5));
    let maskValue = textureLoad(barrierMask, vec2<i32>(uv * vec2<f32>(f32(params.count), f32(params.count))), 0).r;
    if (maskValue < 0.5) {
        vel = -vel; // Reverse velocity on collision
        newPos = pos + vel * params.dt; // Recalculate position
    }

    boids[i] = vec4<f32>(newPos, vel);
}
