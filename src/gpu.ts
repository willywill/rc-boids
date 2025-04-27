export interface GPUContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  pixelRatio: number;
  canvas: HTMLCanvasElement;
}

export const initializeWebGPU = async (canvas: HTMLCanvasElement): Promise<GPUContext | undefined> => {
  /**
   * We need to first check if the browser supports WebGPU
   * as this is a newer standard and not all browsers support it yet
   */
  const adapter = await navigator.gpu.requestAdapter();

  if (!adapter) {
    alert("WebGPU not supported on this browser. Please update your browser.");
    return;
  }

  /**
   * Request a device from the adapter. This is the main interface
   * we will use to interact with the GPU
   */
  const device = await adapter.requestDevice();

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

  return {
    device,
    context,
    format,
    pixelRatio,
    canvas,
  }
};