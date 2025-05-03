// src/main.ts
import { type GPUContext, initializeWebGPU } from "./gpu";
import { createBoidBuffer, makeBoidArray } from "./boids";
import { addMouseListener } from "./input";
// raw‐import WGSL
import boidsCode from "./boids.wgsl?raw";
import emissionCode from "./emission.wgsl?raw";
import diffuseCode from "./diffuse.wgsl?raw";
import upsampleCode from "./upsample.wgsl?raw";
import compositeCode from "./composite.wgsl?raw";

const NUM_BOIDS = 10;  // Reduced to just 3 boids
const WORKGROUP_SIZE = 64;
const ACCEL_STRENGTH = 1.5;  // Slightly reduced for better control
const MAX_SPEED = 1.8;  // Reduced for smoother movement
const numWorkgroups = Math.ceil(NUM_BOIDS / WORKGROUP_SIZE);

const main = async (): Promise<void> => {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const info = document.getElementById("info") as HTMLDivElement;
  const ctx = await initializeWebGPU(canvas);
  if (!ctx) {
    console.error("Failed to init WebGPU"); return;
  }
  const { device, context, format } = ctx;
  const width = canvas.width;
  const height = canvas.height;

  // 1) Boid simulation
  const boidData = makeBoidArray(NUM_BOIDS);
  const boidBuffer = createBoidBuffer(device, boidData);
  const SIM_UNIFORM_SIZE = 24;
  const align = device.limits.minUniformBufferOffsetAlignment;
  const simSize = Math.ceil(SIM_UNIFORM_SIZE / align) * align;
  const simParamBuffer = device.createBuffer({
    size: simSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const boidModule = device.createShaderModule({ code: boidsCode });
  const boidPipeline = device.createComputePipeline({
    layout: "auto", compute: { module: boidModule, entryPoint: "main" }
  });

  // 2) Radiance cascades (3 levels), all rgba32float for color
  const cascadeSizes: [number, number][] = [
    [width, height],
    [width >> 1, height >> 1],
    [width >> 2, height >> 2],
  ];
  const radianceCascades = cascadeSizes.map(([w, h]) =>
    device.createTexture({
      size: [w, h],
      format: "rgba32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
    })
  );
  const radianceTemp = cascadeSizes.map(([w, h]) =>
    device.createTexture({
      size: [w, h],
      format: "rgba32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
    })
  );
  const radianceViews = radianceCascades.map(t => t.createView());
  const radianceTempViews = radianceTemp.map(t => t.createView());

  // 3) Emission texture (rgba32float)
  const emissionTex = device.createTexture({
    size: [width, height],
    format: "rgba32float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST |
      GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
  });
  const emissionView = emissionTex.createView();
  const EMIT_UNIFORM_SIZE = 16;
  const emitSize = Math.ceil(EMIT_UNIFORM_SIZE / align) * align;
  const emitParamBuffer = device.createBuffer({
    size: emitSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  // fill EmitParams once
  {
    const buf = new ArrayBuffer(EMIT_UNIFORM_SIZE);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);
    f32[0] = width; f32[1] = height; f32[2] = 0.0; // resolution.xy, time
    u32[3] = NUM_BOIDS; // number of boids
    device.queue.writeBuffer(emitParamBuffer, 0, buf);
  }
  const emitModule = device.createShaderModule({ code: emissionCode });
  const emitPipeline = device.createComputePipeline({
    layout: "auto", compute: { module: emitModule, entryPoint: "main" }
  });
  const emitBindGroup = device.createBindGroup({
    layout: emitPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: emitParamBuffer } },
      { binding: 1, resource: { buffer: boidBuffer } },
      { binding: 2, resource: emissionView },
    ],
  });

  // 4) Barrier mask (r8unorm) + writeTexture with aligned rows
  const barrierMaskTex = device.createTexture({
    size: [width, height], format: "r8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const bytesPerTexel = 1;
  const bytesPerRow = Math.ceil((width * bytesPerTexel) / 256) * 256;
  const maskBuffer = new Uint8Array(bytesPerRow * height);
  for (let y = 0; y < height; y++) {
    maskBuffer.fill(255, y * bytesPerRow, y * bytesPerRow + width);
  }

  // Add rectangular obstacles
  const addRectangle = (x: number, y: number, w: number, h: number) => {
    const startX = Math.floor(x * width);
    const startY = Math.floor(y * height);
    const rectWidth = Math.floor(w * width);
    const rectHeight = Math.floor(h * height);

    for (let y = startY; y < startY + rectHeight && y < height; y++) {
      for (let x = startX; x < startX + rectWidth && x < width; x++) {
        maskBuffer[y * bytesPerRow + x] = 0; // Mark as obstacle
      }
    }
  };

  // Add three thinner rectangles
  addRectangle(0.45, 0.3, 0.05, 0.4);   // Vertical rectangle
  addRectangle(0.2, 0.5, 0.4, 0.05);    // Horizontal rectangle
  addRectangle(0.7, 0.3, 0.05, 0.3);    // Another vertical rectangle

  device.queue.writeTexture(
    { texture: barrierMaskTex },
    maskBuffer,
    { offset: 0, bytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 }
  );
  const barrierMaskView = barrierMaskTex.createView();

  // Now create the boid bind group with all required bindings
  const boidBindGroup = device.createBindGroup({
    layout: boidPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: boidBuffer } },
      { binding: 1, resource: { buffer: simParamBuffer } },
      { binding: 2, resource: barrierMaskView },
    ],
  });

  // 5) Diffuse pipelines & bind-groups
  const DIFFUSE_UNIFORM_SIZE = 8;
  const diffuseParamBuffers = cascadeSizes.map(([w, h]) => {
    const sz = Math.ceil(DIFFUSE_UNIFORM_SIZE / align) * align;
    const buf = device.createBuffer({
      size: sz, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([w, h]).buffer);
    return buf;
  });
  const diffuseModule = device.createShaderModule({ code: diffuseCode });
  const diffusePipeline = device.createComputePipeline({
    layout: "auto", compute: { module: diffuseModule, entryPoint: "main" }
  });
  const diffuseBindGroups = cascadeSizes.map((_, i) => {
    const inView = i === 0 ? emissionView : radianceViews[i - 1];
    const outView = radianceViews[i];
    return device.createBindGroup({
      layout: diffusePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: diffuseParamBuffers[i] } },
        { binding: 1, resource: inView },
        { binding: 2, resource: barrierMaskView },
        { binding: 3, resource: outView },
      ],
    });
  });

  // 6) Upsample pipelines & bind-groups
  const UPSAMPLE_UNIFORM_SIZE = 8;
  const upsampleParamBuffers = cascadeSizes.slice(1).map(([w, h]) => {
    const sz = Math.ceil(UPSAMPLE_UNIFORM_SIZE / align) * align;
    const buf = device.createBuffer({
      size: sz, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([w, h]).buffer);
    return buf;
  });
  const upModule = device.createShaderModule({ code: upsampleCode });
  const upsamplePipeline = device.createComputePipeline({
    layout: "auto", compute: { module: upModule, entryPoint: "main" }
  });
  const upsampleBindGroups = cascadeSizes.slice(1).map((_, i) => {
    return device.createBindGroup({
      layout: upsamplePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: upsampleParamBuffers[i] } },
        { binding: 1, resource: radianceViews[i + 1] },
        { binding: 2, resource: radianceViews[i] },
        { binding: 3, resource: radianceTempViews[i] },
      ],
    });
  });

  // 7) Composite pipeline
  const compositeModule = device.createShaderModule({ code: compositeCode });
  const compositePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: compositeModule, entryPoint: "vs_main" },
    fragment: { module: compositeModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const compositeBindGroup = device.createBindGroup({
    layout: compositePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: radianceViews[0] },
    ],
  });

  // Create clear pipeline for emission texture
  const clearCode = `
    @group(0) @binding(0)
    var emissionTex: texture_storage_2d<rgba32float, write>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        textureStore(emissionTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
  `;
  const clearModule = device.createShaderModule({ code: clearCode });
  const clearPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: clearModule, entryPoint: "main" }
  });
  const clearBindGroup = device.createBindGroup({
    layout: clearPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: emissionView }
    ]
  });

  // Mouse & timing
  const mouse = { x: 0, y: 0 };
  addMouseListener(canvas, (nx, ny) => {
    mouse.x = nx; mouse.y = ny;
    info.textContent = `x=${nx.toFixed(2)}, y=${ny.toFixed(2)}`;
  });
  let lastTime = performance.now();

  // Frame loop
  const frame = (now: number) => {
    const dt = (now - lastTime) / 1000; lastTime = now;

    // Update emission time
    device.queue.writeBuffer(emitParamBuffer, 8, new Float32Array([now / 1000]));

    // Boid compute
    {
      const buf = new ArrayBuffer(SIM_UNIFORM_SIZE);
      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);
      f32[0] = mouse.x * 2 - 1; f32[1] = 1 - mouse.y * 2; f32[2] = dt;
      u32[3] = NUM_BOIDS; f32[4] = ACCEL_STRENGTH; f32[5] = MAX_SPEED;
      device.queue.writeBuffer(simParamBuffer, 0, buf);
      const ce = device.createCommandEncoder();
      const pass = ce.beginComputePass();
      pass.setPipeline(boidPipeline);
      pass.setBindGroup(0, boidBindGroup);
      pass.dispatchWorkgroups(numWorkgroups);
      pass.end();
      device.queue.submit([ce.finish()]);
    }

    // Radiance cascade
    const encoder = device.createCommandEncoder();

    // Clear emission texture
    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(clearPipeline);
    clearPass.setBindGroup(0, clearBindGroup);
    clearPass.dispatchWorkgroups(
      Math.ceil(width / 8),
      Math.ceil(height / 8)
    );
    clearPass.end();

    // emission
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(emitPipeline);
      pass.setBindGroup(0, emitBindGroup);
      // Use proper workgroup dispatch based on texture dimensions
      pass.dispatchWorkgroups(
        Math.ceil(width / 8),
        Math.ceil(height / 8)
      );
      pass.end();
    }
    // diffuse
    for (let i = 0; i < cascadeSizes.length; i++) {
      const [w, h] = cascadeSizes[i];
      const pass = encoder.beginComputePass();
      pass.setPipeline(diffusePipeline);
      pass.setBindGroup(0, diffuseBindGroups[i]);
      pass.dispatchWorkgroups(
        Math.ceil(w / 8), Math.ceil(h / 8)
      );
      pass.end();
    }
    // upsample
    for (let i = 0; i < upsampleBindGroups.length; i++) {
      const [w, h] = cascadeSizes[i];
      const pass = encoder.beginComputePass();
      pass.setPipeline(upsamplePipeline);
      pass.setBindGroup(0, upsampleBindGroups[i]);
      pass.dispatchWorkgroups(
        Math.ceil(w / 8), Math.ceil(h / 8)
      );
      pass.end();

      // Copy the temp result back to the main cascade texture
      encoder.copyTextureToTexture(
        { texture: radianceTemp[i] },
        { texture: radianceCascades[i] },
        { width: w, height: h }
      );
    }
    // composite
    {
      const view = context.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view,
          loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store"
        }]
      });
      pass.setPipeline(compositePipeline);
      pass.setBindGroup(0, compositeBindGroup);
      pass.draw(6);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
};

window.addEventListener("DOMContentLoaded", main);
