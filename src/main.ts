const initializeWebGPU = async () => {
  /**
   * We need to first check if the browser supports WebGPU
   * as this is a newer standard and not all browsers support it yet
   */
  if (!navigator.gpu.requestAdapter()) {
    alert("WebGPU not supported on this browser. Please update your browser.");
    return;
  }

  /**
   * Retrieve the GPU adapter which allows us to access and 
   * interface with the users GPU through the browser
   */
  const adapter = await navigator.gpu.requestAdapter();

  if (!adapter) {
    console.error("Failed to get GPU adapter.");
    return;
  }

  /**
   * Request a device from the adapter. This is the main interface
   * we will use to interact with the GPU
   */
  const device = await adapter.requestDevice();

  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const context = canvas.getContext("webgpu")!;
  const format = navigator.gpu.getPreferredCanvasFormat();

  // Properly scales the canvas viewport
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * pixelRatio;
  canvas.height = canvas.clientHeight * pixelRatio;

  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  /**
   * Establishes a render pass which is a series of rendering commands
   * that will be executed on the GPU. This is where we will
   * define what we want to render and how we want to render it
   */
  const encoder = device.createCommandEncoder();
  const view = context.getCurrentTexture().createView();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view,
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 1, a: 1 },
        storeOp: "store",
      },
    ],
  });

  pass.end();

  device.queue.submit([encoder.finish()]);
};

window.addEventListener("DOMContentLoaded", initializeWebGPU);
