import { useCallback, useEffect, useRef, useState } from "react";
import { Logo } from "./components/Logo";
import { ThemeToggle } from "./components/ThemeToggle";
import { Classifier, type Status } from "./components/Classifier";
import { Disclaimer } from "./components/Disclaimer";
import { Gallery, type SampleItem } from "./components/Gallery";
import { loadWeights, type ModelWeights } from "./model/weights";
import { predict, type Prediction } from "./model/forward";
import { imageToInput, loadImage } from "./lib/preprocess";

const GITHUB_URL = "https://github.com/sohan-bhat";
const REPO_URL = "https://github.com/sohan-bhat/signnet";
const PORTFOLIO_URL = "https://sohan-bhat.github.io";
const THINK_MS = 280;

export default function App() {
  const [weights, setWeights] = useState<ModelWeights | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [truthClassId, setTruthClassId] = useState<number | null>(null);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<Prediction | null>(null);

  const objectUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pendingUploadRef = useRef(false);

  const [showDisclaimer, setShowDisclaimer] = useState(false);
  const [acknowledged, setAcknowledged] = useState(
    () => localStorage.getItem("signnet-custom-ack") === "1"
  );

  useEffect(() => {
    loadWeights().then(setWeights).catch((e) => setLoadError(String(e)));
    fetch("/samples/manifest.json")
      .then((r) => r.json())
      .then((list: SampleItem[]) => setSamples(list.sort((a, b) => a.classId - b.classId)))
      .catch(() => setSamples([]));
  }, []);

  const runPrediction = useCallback(
    async (src: HTMLImageElement, w: ModelWeights) => {
      setStatus("thinking");

      await new Promise((res) => setTimeout(res, THINK_MS));
      const input = imageToInput(src);
      const out = predict(input, w);
      setResult(out);
      setStatus("ready");
    },
    []
  );

  const handleUrl = useCallback(
    async (url: string, truth: number | null, file: string | null) => {
      if (!weights) return;
      setImageUrl(url);
      setTruthClassId(truth);
      setActiveFile(file);
      setResult(null);
      try {
        const img = await loadImage(url);
        await runPrediction(img, weights);
      } catch (e) {
        setLoadError(String(e));
        setStatus("idle");
      }
    },
    [weights, runPrediction]
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

  // Clicking the image to upload: show the disclaimer first (before the OS file
  // picker) if it hasn't been acknowledged yet; otherwise open the picker.
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
    localStorage.setItem("signnet-custom-ack", "1");
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
          <div className="topbar-links">
            <a className="topbar-link hide-mobile" href="#classifier">
              Classifier
            </a>
            <a className="topbar-link hide-mobile" href="#about">
              About
            </a>
            <a className="topbar-link" href={GITHUB_URL} target="_blank" rel="noreferrer">
              GitHub
            </a>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <section className="hero container">
        <h1>
          Traffic signs, classified by a network <br />
          built <span className="grad">from scratch</span>.
        </h1>
        <p className="hero-tagline">
          A convolutional neural network built from scratch, classifying German traffic signs.
        </p>
        <div className="hero-actions">
          <a className="btn btn-primary" href={REPO_URL} target="_blank" rel="noreferrer">
            View source
          </a>
        </div>
      </section>

      <section className="container" id="classifier" style={{ scrollMarginTop: 80 }}>
        <Classifier
          imageUrl={imageUrl}
          truthClassId={truthClassId}
          status={status}
          result={result}
          onFile={handleFile}
          onUploadClick={handleUploadClick}
        />
        {loadError && (
          <p style={{ color: "var(--text-faint)", fontSize: "0.85rem", marginTop: 12 }}>
            {loadError}
          </p>
        )}
      </section>

      <section className="section container">
        <div className="section-head">
          <h2>Sample test images</h2>
          <p>
            One held-out GTSRB test image per class. Click any sign to classify it instantly.
          </p>
        </div>
        <Gallery samples={samples} activeFile={activeFile} onPick={handleSample} />
      </section>

      <section className="about" id="about" style={{ scrollMarginTop: 64 }}>
        <div className="container about-grid">
          <div>
            <h3>Built from scratch</h3>
            <p>
              SignNet is a convolutional neural network written in NumPy: convolution, max
              pooling, dense layers, ReLU, softmax, and the backpropagation training loop. It
              reaches about 90% accuracy on the GTSRB test set across all 43 classes.
            </p>
            <p>
              This page reimplements the same forward pass in TypeScript. The trained weights
              come directly from the NumPy model, and the JavaScript predictions match the
              Python ones to within about 1e-6 on identical inputs.
            </p>
          </div>
          <div className="stat-grid">
            <div className="stat">
              <div className="num mono">~90%</div>
              <div className="lbl">Test accuracy</div>
            </div>
            <div className="stat">
              <div className="num mono">43</div>
              <div className="lbl">Sign classes</div>
            </div>
            <div className="stat">
              <div className="num mono">320K</div>
              <div className="lbl">Parameters</div>
            </div>
            <div className="stat">
              <div className="num mono">32×32</div>
              <div className="lbl">Input size</div>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="container footer-inner">
          <span>signnet · Built by Sohan Bhat</span>
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
