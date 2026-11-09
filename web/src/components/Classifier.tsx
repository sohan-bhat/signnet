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
    <div className="result">
      <button
        type="button"
        className={`stage${dragging ? " dragging" : ""}`}
        onClick={onUploadClick}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <span className="stage-image">
          {imageUrl && <img src={imageUrl} alt="" />}
        </span>
        <span className="stage-label">Upload a sign</span>
      </button>

      <p className="hint">SignNet classifies only these 43 sign types.</p>

      <p className="truth" aria-live="polite">
        {result && truthClassId !== null && (
          <>
            <span className="truth-key">True label</span>
            <span className="truth-name">{signName(truthClassId)}</span>
            <span className={correct ? "ok" : "bad"}>{correct ? "Correct" : "Mismatch"}</span>
          </>
        )}
      </p>

      <ConfidenceBars entries={top5} truthClassId={truthClassId} />
    </div>
  );
}
