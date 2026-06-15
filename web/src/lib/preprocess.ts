const SIZE = 32;

let sharedCanvas: HTMLCanvasElement | null = null;
function getCanvas(): HTMLCanvasElement {
  if (!sharedCanvas) {
    sharedCanvas = document.createElement("canvas");
    sharedCanvas.width = SIZE;
    sharedCanvas.height = SIZE;
  }
  return sharedCanvas;
}

export function imageToInput(
  source: HTMLImageElement | HTMLCanvasElement | ImageBitmap
): Float32Array {
  const canvas = getCanvas();
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2D canvas context unavailable");

  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.drawImage(source, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);

  const out = new Float32Array(SIZE * SIZE * 3);
  for (let p = 0; p < SIZE * SIZE; p++) {
    out[p * 3 + 0] = data[p * 4 + 0] / 255;
    out[p * 3 + 1] = data[p * 4 + 1] / 255;
    out[p * 3 + 2] = data[p * 4 + 2] / 255;
  }
  return out;
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

export function fileToObjectURL(file: File): string {
  return URL.createObjectURL(file);
}
