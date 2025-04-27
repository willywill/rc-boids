import type { GPUContext } from "./gpu";

export const render = async (ctx: GPUContext, clearColor: GPUColor) => {
  const { device, context } = ctx;

  /**
   * Establishes an encoder which gives us a system to record commands
   * that will be executed on the GPU. This is where we can
   * define what we want to render and how we want to render it
   */
  const encoder = device.createCommandEncoder();
  const view = context.getCurrentTexture().createView();

  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view,
        loadOp: "clear",
        clearValue: clearColor,
        storeOp: "store",
      },
    ],
  });

  pass.end();

  device.queue.submit([encoder.finish()]);
}