export type MouseCallback = (x: number, y: number) => void;

export const addMouseListener = (
  canvas: HTMLCanvasElement,
  callback: MouseCallback
) => {
  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    // normalize to [0,1]
    const normalizedX = (event.clientX - rect.left) / rect.width;
    const normalizedY = (event.clientY - rect.top) / rect.height;
    callback(normalizedX, normalizedY);
  });
}