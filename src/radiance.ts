// src/radiance.ts
import type { GPUContext } from "./gpu";
import emissionCode from "./emission.wgsl?raw";
import diffuseCode from "./diffuse.wgsl?raw";
import upsampleCode from "./upsample.wgsl?raw";
import compositeCode from "./composite.wgsl?raw";

export interface RadianceContext {
  barrierMask: GPUTexture;
  emissionTex: GPUTexture;
  radianceCascades: GPUTexture[];
  pipelines: {
    emit: GPUComputePipeline;
    diffuse: GPUComputePipeline;
    upsample: GPUComputePipeline;
    composite: GPURenderPipeline;
  };
  bindGroups: {
    emit: GPUBindGroup;
    diffuse: GPUBindGroup[];
    upsample: GPUBindGroup[];
    composite: GPUBindGroup;
  };
}

export const initRadiance = (ctx: GPUContext): RadianceContext => {
  const { device, context, format } = ctx;
  const w = context.canvas.width;
  const h = context.canvas.height;
  const align = device.limits.minUniformBufferOffsetAlignment;

  // 1) Barrier‐mask texture (r8unorm) for your diffuse pass
  const barrierMask = device.createTexture({
    size: [w, h],
    format: "r8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  // Upload an all-1.0 mask with proper row alignment:
  {
    const bytesPerRow = Math.ceil(w / 256) * 256;
    const data = new Uint8Array(bytesPerRow * h);
    for (let y = 0; y < h; y++) {
      data.fill(255, y * bytesPerRow, y * bytesPerRow + w);
    }
    device.queue.writeTexture(
      { texture: barrierMask },
      data,
      { offset: 0, bytesPerRow, rowsPerImage: h },
      { width: w, height: h, depthOrArrayLayers: 1 }
    );
  }

  // 2) Emission texture (r32float, write‐only storage)
  const emissionTex = device.createTexture({
    size: [w, h],
    format: "r32float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  // 3) Radiance cascades (r32float) at full, half, quarter resolution
  const cascadeSizes: [number, number][] = [
    [w, h],
    [w >> 1, h >> 1],
    [w >> 2, h >> 2],
  ];
  const radianceCascades = cascadeSizes.map(([cw, ch]) =>
    device.createTexture({
      size: [cw, ch],
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
  );

  // Create uniform buffers for all passes
  const EMIT_UNIFORM_SIZE = 12; // vec2<f32> resolution + f32 time
  const emitParamBuffer = device.createBuffer({
    size: Math.ceil(EMIT_UNIFORM_SIZE / align) * align,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Initialize emission parameters
  const emitParams = new Float32Array([w, h, 0.0]); // resolution.xy, time
  device.queue.writeBuffer(emitParamBuffer, 0, emitParams);

  // Create diffuse parameter buffers (one per cascade)
  const DIFFUSE_UNIFORM_SIZE = 8; // vec2<f32> resolution
  const diffuseParamBuffers = cascadeSizes.map(([cw, ch]) => {
    const buf = device.createBuffer({
      size: Math.ceil(DIFFUSE_UNIFORM_SIZE / align) * align,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([cw, ch]));
    return buf;
  });

  // Create upsample parameter buffers
  const UPSAMPLE_UNIFORM_SIZE = 8; // vec2<f32> srcSize
  const upsampleParamBuffers = cascadeSizes.slice(1).map(([cw, ch]) => {
    const buf = device.createBuffer({
      size: Math.ceil(UPSAMPLE_UNIFORM_SIZE / align) * align,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([cw, ch]));
    return buf;
  });

  // 4) Compute pipelines for emission, diffuse, upsample
  const mkCompute = (code: string) =>
    device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code }),
        entryPoint: "main",
      },
    });
  const emitP = mkCompute(emissionCode);
  const diffuseP = mkCompute(diffuseCode);
  const upsampleP = mkCompute(upsampleCode);

  // 5) Composite render pipeline (reads storage‐texture)
  const compModule = device.createShaderModule({ code: compositeCode });
  const compositeP = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: compModule, entryPoint: "vs_main" },
    fragment: { module: compModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // 6) Bind‐groups
  // — emission —
  const emitBG = device.createBindGroup({
    layout: emitP.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: emitParamBuffer } },
      { binding: 1, resource: radianceCascades[0].createView() },
      { binding: 2, resource: emissionTex.createView() },
    ],
  });

  // — diffuse (one per cascade level) —
  const diffuseBG = cascadeSizes.map((_, i) =>
    device.createBindGroup({
      layout: diffuseP.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: diffuseParamBuffers[i] } },
        { binding: 1, resource: (i === 0 ? emissionTex : radianceCascades[i - 1]).createView() },
        { binding: 2, resource: barrierMask.createView() },
        { binding: 3, resource: radianceCascades[i].createView() },
      ],
    })
  );

  // — upsample (one per cascade except last) —
  const upsampleBG = cascadeSizes.slice(1).map((_, i) =>
    device.createBindGroup({
      layout: upsampleP.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: upsampleParamBuffers[i] } },
        { binding: 1, resource: radianceCascades[i + 1].createView() },
        { binding: 2, resource: radianceCascades[i].createView() },
      ],
    })
  );

  // — composite —
  const compositeBG = device.createBindGroup({
    layout: compositeP.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: radianceCascades[0].createView() },
    ],
  });

  return {
    barrierMask,
    emissionTex,
    radianceCascades,
    pipelines: {
      emit: emitP,
      diffuse: diffuseP,
      upsample: upsampleP,
      composite: compositeP,
    },
    bindGroups: {
      emit: emitBG,
      diffuse: diffuseBG,
      upsample: upsampleBG,
      composite: compositeBG,
    },
  };
};
