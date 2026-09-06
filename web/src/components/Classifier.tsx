import { useState } from "react";
import type { DragEvent } from "react";
import type { Prediction } from "../model/forward";
import { topK } from "../model/forward";
import { signName } from "../lib/labels";
import { ConfidenceBars } from "./ConfidenceBars";

export function Classifier({
  imageUrl,
  truthClassId,
  result,
  onFile,
  onUploadClick,
}: {
  imageUrl: string | null;
  truthClassId: number | null;
  result: Prediction | null;
  onFile: (file: File) => void;
  onUploadClick: () => void;
}) {
  const [dragging, setDragging] = useState(false);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  };

  const top5 = result
    ? topK(result.probs, 5).map((classId) => ({ classId, prob: result.probs[classId] }))
    : [];
  const correct = result !== null && truthClassId !== null && result.pred === truthClassId;

  return (
    <div className="classifier">
      <div className="panel">
        <div className="panel-title">Input</div>
        {truthClassId !== null && (
          <div className="truth-tag">
            <span>True label:</span>
            <b>{signName(truthClassId)}</b>
            {result && (
              <span className={correct ? "ok" : "bad"}>{correct ? "Correct" : "Mismatch"}</span>
            )}
          </div>
        )}
        <div
          className={`image-stage${dragging ? " dragging" : ""}`}
          role="button"
          tabIndex={0}
          onClick={onUploadClick}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onUploadClick();
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          title="Click to upload, paste, or drag and drop an image"
        >
          {imageUrl ? (
            <>
              <img src={imageUrl} alt="Selected input" />
              <div className="image-badge">Click to replace</div>
            </>
          ) : (
            <div className="image-empty">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="9" cy="9" r="2" />
                <path d="m21 15-3.5-3.5a2 2 0 0 0-3 0L5 21" />
              </svg>
              <span>
                <strong>Upload an image</strong>, paste, or drag and drop
              </span>
              <span className="image-empty-hint">or choose a sample below</span>
            </div>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">Prediction</div>

        {!result && (
          <div className="pred-empty">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a10 10 0 1 0 10 10" />
              <path d="M12 6v6l4 2" />
            </svg>
            <span>Choose an image to see the network's prediction.</span>
          </div>
        )}

        {result && (
          <>
            <div className="prediction-head">
              <div>
                <div className="pred-label">{signName(result.pred)}</div>
                <div className="pred-sub mono">class {result.pred}</div>
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
