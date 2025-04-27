import { type GPUContext, initializeWebGPU } from "./gpu";
import { addMouseListener } from "./input";
import { render } from "./renderer";

const renderLoop = (ctx: GPUContext, mouse: { x: number; y: number }) => {
  const clearColor = { r: mouse.x, g: mouse.y, b: 1 - mouse.x, a: 1 };
  render(ctx, clearColor);
  requestAnimationFrame(() => renderLoop(ctx, mouse));
};

const main = async () => {
  const info = document.getElementById("info") as HTMLDivElement;
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const ctx = await initializeWebGPU(canvas);

  if (!ctx) {
    console.error("Failed to initialize WebGPU.");
    return;
  }

  const mouse = { x: 0, y: 0 };

  addMouseListener(canvas, (nx, ny) => {
    mouse.x = nx;
    mouse.y = ny;
    info.textContent = `x=${nx.toFixed(2)}, y=${ny.toFixed(2)}`;
  });

  // Start the frame loop
  requestAnimationFrame(renderLoop.bind(null, ctx, mouse));
};

window.addEventListener("DOMContentLoaded", main);
