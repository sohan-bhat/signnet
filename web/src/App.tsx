import { useCallback, useEffect, useRef, useState } from "react";
import { Logo } from "./components/Logo";
import { Classifier } from "./components/Classifier";
import { Disclaimer } from "./components/Disclaimer";
import { Gallery, type SampleItem } from "./components/Gallery";
import { loadWeights, type ModelWeights } from "./model/weights";
import { predict, type Prediction } from "./model/forward";
import { imageToInput, loadImage } from "./lib/preprocess";

const GITHUB_URL = "https://github.com/sohan-bhat";
const REPO_URL = "https://github.com/sohan-bhat/signnet";
const PORTFOLIO_URL = "https://sohan-bhat.github.io";
const ACK_KEY = "signnet-custom-ack";

const ARCHITECTURE = [
  { layer: "Input", shape: "32 × 32 × 3", io: true },
  { layer: "Conv 3×3, 32 filters + ReLU", shape: "30 × 30 × 32" },
  { layer: "MaxPool 2×2", shape: "15 × 15 × 32" },
  { layer: "Conv 3×3, 64 filters + ReLU", shape: "13 × 13 × 64" },
  { layer: "MaxPool 2×2", shape: "6 × 6 × 64" },
  { layer: "Flatten", shape: "2304" },
  { layer: "Dense 128 + ReLU", shape: "128" },
  { layer: "Dense 43 + Softmax", shape: "43", io: true },
];

export default function App() {
  const [weights, setWeights] = useState<ModelWeights | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [truthClassId, setTruthClassId] = useState<number | null>(null);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);

  const objectUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pendingUploadRef = useRef(false);

  const [showDisclaimer, setShowDisclaimer] = useState(false);
  const [acknowledged, setAcknowledged] = useState(
    () => localStorage.getItem(ACK_KEY) === "1"
  );

  useEffect(() => {
    loadWeights().then(setWeights).catch((e) => setLoadError(String(e)));
    fetch("/samples/manifest.json")
      .then((r) => r.json())
      .then((list: SampleItem[]) => setSamples(list.sort((a, b) => a.classId - b.classId)))
      .catch(() => setSamples([]));
  }, []);

  const handleUrl = useCallback(
    async (url: string, truth: number | null, file: string | null) => {
      if (!weights) return;
      setImageUrl(url);
      setTruthClassId(truth);
      setActiveFile(file);
      setResult(null);
      try {
        const img = await loadImage(url);
        setResult(predict(imageToInput(img), weights));
      } catch (e) {
        setLoadError(String(e));
      }
    },
    [weights]
  );

  const handleFile = useCallback(
    (file: File) => {
      if (!acknowledged) setShowDisclaimer(true);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      void handleUrl(url, null, null);
    },
    [handleUrl, acknowledged]
  );

  const handleUploadClick = useCallback(() => {
    if (!acknowledged) {
      pendingUploadRef.current = true;
      setShowDisclaimer(true);
    } else {
      fileInputRef.current?.click();
    }
  }, [acknowledged]);

  const acknowledgeDisclaimer = useCallback(() => {
    setShowDisclaimer(false);
    setAcknowledged(true);
    localStorage.setItem(ACK_KEY, "1");
    if (pendingUploadRef.current) {
      pendingUploadRef.current = false;
      fileInputRef.current?.click();
    }
  }, []);

  const handleSample = useCallback(
    (s: SampleItem) => void handleUrl(`/samples/${s.file}`, s.classId, s.file),
    [handleUrl]
  );

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

  return (
    <div className="app" id="top">
      {showDisclaimer && <Disclaimer onAcknowledge={acknowledgeDisclaimer} />}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f && f.type.startsWith("image/")) handleFile(f);
          e.target.value = "";
        }}
      />
      <header className="topbar">
        <div className="container topbar-inner">
          <Logo />
          <nav className="topbar-links">
            <a className="topbar-link hide-mobile" href="#classifier">
              Classifier
            </a>
            <a className="topbar-link hide-mobile" href="#about">
              About
            </a>
            <a className="topbar-link" href={REPO_URL} target="_blank" rel="noreferrer">
              GitHub
            </a>
          </nav>
        </div>
      </header>

      <section className="hero container">
        <div>
          <div className="hero-eyebrow">Convolutional neural network · NumPy only</div>
          <h1>Traffic sign classification, built from scratch.</h1>
          <p className="hero-tagline">
            SignNet recognizes 43 German traffic sign types. Every layer, including
            backpropagation, is written by hand in NumPy. This page runs the trained model
            directly in your browser.
          </p>
          <div className="hero-actions">
            <a className="btn btn-primary" href="#classifier">
              Try the classifier
            </a>
            <a className="btn btn-secondary" href={REPO_URL} target="_blank" rel="noreferrer">
              View source
            </a>
          </div>
        </div>
        <dl className="spec-list">
          <div>
            <dt>Test accuracy</dt>
            <dd>90.06%</dd>
          </div>
          <div>
            <dt>Test images</dt>
            <dd>12,630</dd>
          </div>
          <div>
            <dt>Sign classes</dt>
            <dd>43</dd>
          </div>
          <div>
            <dt>Parameters</dt>
            <dd>320K</dd>
          </div>
          <div>
            <dt>Input size</dt>
            <dd>32 × 32 RGB</dd>
          </div>
        </dl>
      </section>

      <section className="section container" id="classifier" style={{ scrollMarginTop: 24 }}>
        <div className="section-head">
          <h2>Classifier</h2>
          <p>Upload a sign or pick a sample. Inference runs locally, nothing is sent anywhere.</p>
        </div>
        <Classifier
          imageUrl={imageUrl}
          truthClassId={truthClassId}
          result={result}
          onFile={handleFile}
          onUploadClick={handleUploadClick}
        />
        {loadError && <p className="load-error">{loadError}</p>}
      </section>

      <section className="section container">
        <div className="section-head">
          <h2>Sample test images</h2>
          <p>One held-out GTSRB test image per class. Click any sign to classify it.</p>
        </div>
        <Gallery samples={samples} activeFile={activeFile} onPick={handleSample} />
      </section>

      <section className="about" id="about">
        <div className="container about-grid">
          <div>
            <h2>About the model</h2>
            <p>
              SignNet is a convolutional neural network written in NumPy: convolution, max
              pooling, dense layers, ReLU, softmax, and the backpropagation training loop. No
              PyTorch, no TensorFlow, no autograd. It reaches 90.06% accuracy on the GTSRB test
              set across all 43 classes.
            </p>
            <p>
              Convolution is implemented with im2col, unfolding image patches into a matrix so
              the whole operation becomes a single matrix multiply. That is what makes training
              on a CPU practical.
            </p>
            <p>
              This page reimplements the same forward pass in TypeScript. The weights are
              exported directly from the NumPy model, and the browser predictions match the
              Python ones to within about 1e-6 on identical inputs.
            </p>
          </div>
          <div className="arch">
            <div className="arch-title">Architecture</div>
            {ARCHITECTURE.map((row) => (
              <div className={`arch-row${row.io ? " io" : ""}`} key={row.layer}>
                <span className="layer">{row.layer}</span>
                <span className="shape">{row.shape}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="container footer-inner">
          <span>SignNet · Sohan Bhat</span>
          <div className="footer-links">
            <a href={GITHUB_URL} target="_blank" rel="noreferrer">
              GitHub
            </a>
            <a href={PORTFOLIO_URL} target="_blank" rel="noreferrer">
              Portfolio
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
