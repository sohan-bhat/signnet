import { useCallback, useEffect, useRef, useState } from "react";
import { Logo } from "./components/Logo";
import { Classifier } from "./components/Classifier";
import { Gallery, type SampleItem } from "./components/Gallery";
import { Network, type LayerSpec } from "./components/Network";
import { loadWeights, type ModelWeights } from "./model/weights";
import { BASE } from "./lib/base";
import { predict, type Prediction } from "./model/forward";
import { imageToInput, loadImage } from "./lib/preprocess";

const REPO_URL = "https://github.com/sohan-bhat/signnet";
const PORTFOLIO_URL = "https://sohanbhat.me";
const FIRST_SAMPLE = 14;

const ARCHITECTURE: LayerSpec[] = [
  { key: "input", layer: "Input", shape: "32×32×3" },
  { key: "conv1", layer: "Conv 3×3, 32 filters + ReLU", shape: "30×30×32" },
  { key: "pool1", layer: "MaxPool 2×2", shape: "15×15×32" },
  { key: "conv2", layer: "Conv 3×3, 64 filters + ReLU", shape: "13×13×64" },
  { key: "pool2", layer: "MaxPool 2×2", shape: "6×6×64" },
  { key: "dense", layer: "Dense 128 + ReLU", shape: "128" },
  { key: "output", layer: "Dense 43 + Softmax", shape: "43" },
];

interface Selection {
  imageUrl: string;
  truthClassId: number | null;
  file: string | null;
  result: Prediction;
}

export default function App() {
  const [weights, setWeights] = useState<ModelWeights | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [selection, setSelection] = useState<Selection | null>(null);

  const objectUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef(0);
  const autoRef = useRef(false);

  useEffect(() => {
    loadWeights(BASE).then(setWeights).catch((e) => setLoadError(String(e)));
    fetch(`${BASE}/samples/manifest.json`)
      .then((r) => r.json())
      .then((list: SampleItem[]) => setSamples(list.sort((a, b) => a.classId - b.classId)))
      .catch(() => setSamples([]));
  }, []);

  const handleUrl = useCallback(
    async (url: string, truth: number | null, file: string | null) => {
      if (!weights) return;
      const request = ++requestRef.current;
      try {
        const img = await loadImage(url);
        const result = predict(imageToInput(img), weights);
        if (request === requestRef.current) {
          setSelection({ imageUrl: url, truthClassId: truth, file, result });
        }
      } catch (e) {
        setLoadError(String(e));
      }
    },
    [weights]
  );

  const handleFile = useCallback(
    (file: File) => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      void handleUrl(url, null, null);
    },
    [handleUrl]
  );

  const handleSample = useCallback(
    (s: SampleItem) => void handleUrl(`${BASE}/samples/${s.file}`, s.classId, s.file),
    [handleUrl]
  );

  useEffect(() => {
    if (autoRef.current || !weights || samples.length === 0) return;
    const first = samples.find((s) => s.classId === FIRST_SAMPLE) ?? samples[0];
    autoRef.current = true;
    handleSample(first);
  }, [weights, samples, handleSample]);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            handleFile(file);
            break;
          }
        }
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [handleFile]);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  const result = selection?.result ?? null;

  return (
    <div className="app">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f && f.type.startsWith("image/")) handleFile(f);
          e.target.value = "";
        }}
      />
      <header className="topbar">
        <div className="container topbar-inner">
          <Logo />
          <a className="text-link" href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </div>
      </header>

      <main>
        <section className="hero container">
          <div className="intro">
            <h1>Traffic sign classification, built from scratch.</h1>
            <p className="lede">
              A convolutional network in pure math with NumPy, running in your browser.
            </p>
          </div>

          <div className="demo">
            <Classifier
              imageUrl={selection?.imageUrl ?? null}
              truthClassId={selection?.truthClassId ?? null}
              result={result}
              onFile={handleFile}
              onUploadClick={() => fileInputRef.current?.click()}
            />
            {loadError && <p className="load-error">{loadError}</p>}
            <Gallery samples={samples} activeFile={selection?.file ?? null} onPick={handleSample} />
          </div>

          <dl className="figures">
            <div>
              <dt>Test accuracy</dt>
              <dd className="mono">90.06%</dd>
            </div>
            <div>
              <dt>Sign classes</dt>
              <dd className="mono">43</dd>
            </div>
            <div>
              <dt>Parameters</dt>
              <dd className="mono">320K</dd>
            </div>
          </dl>
        </section>

        <section className="network" aria-labelledby="network-title">
          <div className="container">
            <h2 id="network-title">Inside the network</h2>
            <Network
              layers={ARCHITECTURE}
              activations={result?.activations ?? null}
              probs={result?.probs ?? null}
              pred={result?.pred ?? null}
            />
          </div>
        </section>
      </main>

      <footer className="footer">
        <div className="container">
          <a className="text-link" href={PORTFOLIO_URL} target="_blank" rel="noreferrer">
            Sohan Bhat
          </a>
        </div>
      </footer>
    </div>
  );
}
