import { useRef, useState } from "react";
import type { DragEvent } from "react";
import type { Prediction } from "../model/forward";
import { topK } from "../model/forward";
import { signName } from "../lib/labels";
import { ConfidenceBars } from "./ConfidenceBars";

export type Status = "idle" | "thinking" | "ready";

export function Classifier({
  imageUrl,
  truthClassId,
  status,
  result,
  onFile,
}: {
  imageUrl: string | null;
  truthClassId: number | null;
  status: Status;
  result: Prediction | null;
  onFile: (file: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const pickFile = (files: FileList | null) => {
    const f = files?.[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    pickFile(e.dataTransfer.files);
  };

  const top5 = result
    ? topK(result.probs, 5).map((classId) => ({ classId, prob: result.probs[classId] }))
    : [];
  const correct = result && truthClassId !== null && result.pred === truthClassId;

  return (
    <div className="classifier">
      <div className="panel">
        <div className="panel-title">Input image</div>
        {truthClassId !== null && (
          <div className="truth-tag">
            True label:&nbsp;<b>{signName(truthClassId)}</b>
            {status === "ready" && result && (
              <span style={{ color: correct ? "var(--accent)" : "var(--text-faint)" }}>
                &nbsp;{correct ? "✓ correct" : "✗ mismatch"}
              </span>
            )}
          </div>
        )}
        <div
          className={`image-stage interactive${dragging ? " dragging" : ""}`}
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          title="Click to upload, paste, or drag & drop an image"
        >
          {imageUrl ? (
            <>
              <img src={imageUrl} alt="Selected input" />
              <div className="image-badge">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <path d="M17 8l-5-5-5 5" />
                  <path d="M12 3v12" />
                </svg>
                Click or paste to change
              </div>
            </>
          ) : (
            <div className="image-empty">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="9" cy="9" r="2" />
                <path d="m21 15-3.5-3.5a2 2 0 0 0-3 0L5 21" />
              </svg>
              <span>
                <strong>Click to upload</strong>, paste, or drag &amp; drop an image
              </span>
              <span className="image-empty-hint">or pick a sample below</span>
            </div>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => pickFile(e.target.files)}
        />
      </div>

      <div className="panel">
        <div className="panel-title">Prediction</div>

        {status === "idle" && (
          <div className="pred-empty">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a10 10 0 1 0 10 10" />
              <path d="M12 6v6l4 2" />
            </svg>
            <span>Choose an image to see the network's prediction.</span>
          </div>
        )}

        {status === "thinking" && (
          <div className="pred-empty">
            <div className="thinking">
              <span className="pulse">
                <span />
                <span />
                <span />
              </span>
              Running forward pass…
            </div>
          </div>
        )}

        {status === "ready" && result && (
          <>
            <div className="prediction-head">
              <div>
                <div className="pred-label">{signName(result.pred)}</div>
                <div className="pred-sub mono">class index {result.pred}</div>
              </div>
              <div className="pred-confidence">
                <div className="value mono">{(result.probs[result.pred] * 100).toFixed(1)}%</div>
                <div className="caption">confidence</div>
              </div>
            </div>
            <ConfidenceBars entries={top5} truthClassId={truthClassId} />
          </>
        )}
      </div>
    </div>
  );
}
