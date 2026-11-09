import { useEffect, useRef } from "react";
import type { Activations, Tensor3 } from "../model/forward";

const LOW = [244, 240, 230];
const HIGH = [30, 26, 21];
const CHANNELS = 12;

export type LayerKind = "input" | "conv1" | "pool1" | "conv2" | "pool2" | "dense" | "output";

export interface LayerSpec {
  key: LayerKind;
  layer: string;
  shape: string;
}

function shade(px: Uint8ClampedArray, i: number, t: number) {
  px[i] = LOW[0] + (HIGH[0] - LOW[0]) * t;
  px[i + 1] = LOW[1] + (HIGH[1] - LOW[1]) * t;
  px[i + 2] = LOW[2] + (HIGH[2] - LOW[2]) * t;
  px[i + 3] = 255;
}

function paint(canvas: HTMLCanvasElement | null, w: number, h: number, fill: (px: Uint8ClampedArray) => void) {
  if (!canvas) return;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(w, h);
  fill(img.data);
  ctx.putImageData(img, 0, 0);
}

function drawChannel(canvas: HTMLCanvasElement | null, t: Tensor3, c: number) {
  const n = t.h * t.w;
  let min = Infinity;
  let max = -Infinity;
  for (let p = 0; p < n; p++) {
    const v = t.data[p * t.c + c];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min;
  paint(canvas, t.w, t.h, (px) => {
    for (let p = 0; p < n; p++) {
      shade(px, p * 4, range > 1e-9 ? (t.data[p * t.c + c] - min) / range : 0);
    }
  });
}

function drawColor(canvas: HTMLCanvasElement | null, t: Tensor3) {
  paint(canvas, t.w, t.h, (px) => {
    for (let p = 0; p < t.h * t.w; p++) {
      px[p * 4] = t.data[p * 3] * 255;
      px[p * 4 + 1] = t.data[p * 3 + 1] * 255;
      px[p * 4 + 2] = t.data[p * 3 + 2] * 255;
      px[p * 4 + 3] = 255;
    }
  });
}

function drawStrip(canvas: HTMLCanvasElement | null, values: Float32Array) {
  let max = 0;
  for (const v of values) if (v > max) max = v;
  paint(canvas, values.length, 1, (px) => {
    values.forEach((v, i) => shade(px, i * 4, max > 0 ? v / max : 0));
  });
}

function MapRow({ tensor, color }: { tensor: Tensor3 | null; color?: boolean }) {
  const refs = useRef<(HTMLCanvasElement | null)[]>([]);
  const count = color ? tensor?.c ?? 3 : CHANNELS;

  useEffect(() => {
    if (!tensor) return;
    if (color) drawColor(refs.current[0], tensor);
    for (let c = 0; c < count; c++) {
      drawChannel(refs.current[color ? c + 1 : c], tensor, c);
    }
  }, [tensor, color, count]);

  return (
    <div className="maps" aria-hidden="true">
      {Array.from({ length: color ? count + 1 : count }, (_, i) => (
        <canvas key={i} ref={(el) => (refs.current[i] = el)} width={1} height={1} />
      ))}
    </div>
  );
}

function Strip({ values, mark }: { values: Float32Array | null; mark?: number | null }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (values) drawStrip(ref.current, values);
  }, [values]);

  return (
    <div className="strip" aria-hidden="true">
      <canvas ref={ref} width={1} height={1} />
      {mark != null && values && (
        <span
          className="strip-mark"
          style={{
            left: `${(mark / values.length) * 100}%`,
            width: `${100 / values.length}%`,
          }}
        />
      )}
    </div>
  );
}

export function Network({
  layers,
  activations,
  probs,
  pred,
}: {
  layers: LayerSpec[];
  activations: Activations | null;
  probs: Float32Array | null;
  pred: number | null;
}) {
  const a = activations;
  const visual = (key: LayerKind) => {
    switch (key) {
      case "input":
        return <MapRow tensor={a?.input ?? null} color />;
      case "dense":
        return <Strip values={a?.dense ?? null} />;
      case "output":
        return <Strip values={probs} mark={pred} />;
      default:
        return <MapRow tensor={a?.[key] ?? null} />;
    }
  };

  return (
    <div className="layers">
      {layers.map((l) => (
        <div className="layer" key={l.key}>
          <div className="layer-name">
            <span>{l.layer}</span>
            <span className="mono">{l.shape}</span>
          </div>
          {visual(l.key)}
        </div>
      ))}
    </div>
  );
}
