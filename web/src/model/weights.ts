export interface ParamMeta {
  dtype: "float32";
  shape: number[];
  offset: number;
  count: number;
}

export interface WeightManifest {
  order: string[];
  totalFloats: number;
  params: Record<string, ParamMeta>;
}

export interface ModelWeights {

  w0: Float32Array;
  b0: Float32Array;

  w1: Float32Array;
  b1: Float32Array;

  w2: Float32Array;
  b2: Float32Array;

  w3: Float32Array;
  b3: Float32Array;
}

function sliceParam(blob: Float32Array, meta: ParamMeta): Float32Array {
  return blob.subarray(meta.offset, meta.offset + meta.count);
}

export async function loadWeights(baseUrl = ""): Promise<ModelWeights> {
  const [manifestRes, binRes] = await Promise.all([
    fetch(`${baseUrl}/weights.json`),
    fetch(`${baseUrl}/weights.bin`),
  ]);
  if (!manifestRes.ok) throw new Error(`Failed to load weights.json (${manifestRes.status})`);
  if (!binRes.ok) throw new Error(`Failed to load weights.bin (${binRes.status})`);

  const manifest = (await manifestRes.json()) as WeightManifest;
  const buffer = await binRes.arrayBuffer();
  const blob = new Float32Array(buffer);

  if (blob.length !== manifest.totalFloats) {
    throw new Error(
      `weights.bin size mismatch: got ${blob.length} floats, manifest expects ${manifest.totalFloats}`
    );
  }

  const get = (key: keyof ModelWeights) => sliceParam(blob, manifest.params[key]);
  return {
    w0: get("w0"), b0: get("b0"),
    w1: get("w1"), b1: get("b1"),
    w2: get("w2"), b2: get("b2"),
    w3: get("w3"), b3: get("b3"),
  };
}
