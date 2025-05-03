export const makeBoidArray = (numBoids: number): Float32Array => {
  const arr = new Float32Array(numBoids * 4);
  for (let i = 0; i < numBoids; i++) {
    // Random position in normalized device coords (NDC)
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    // Small random velocity
    const vx = (Math.random() * 2 - 1) * 0.01;
    const vy = (Math.random() * 2 - 1) * 0.01;
    const base = i * 4;
    arr[base + 0] = x;
    arr[base + 1] = y;
    arr[base + 2] = vx;
    arr[base + 3] = vy;
  }
  return arr;
}

/**
 * Creates a GPUBuffer to hold boid data, and uploads the initial state.
 */
export const createBoidBuffer = (
  device: GPUDevice,
  initialData: Float32Array
): GPUBuffer => {
  const byteLength = initialData.byteLength;

  // STORAGE so compute shader can read/write... COPY_DST so we can upload
  const buffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  // Write initial data into the mapped buffer, then unmap
  const mapping = new Float32Array(buffer.getMappedRange());
  mapping.set(initialData);
  buffer.unmap();
  return buffer;
}