import { signName } from "../lib/labels";

export interface BarEntry {
  classId: number;
  prob: number;
}

const ROWS = 5;

export function ConfidenceBars({
  entries,
  truthClassId,
}: {
  entries: BarEntry[];
  truthClassId: number | null;
}) {
  const tone = (entry: BarEntry, index: number) => {
    if (truthClassId === null) return index === 0 ? "ink" : "";
    if (entry.classId === truthClassId) return index === 0 ? "ink" : "ok";
    return index === 0 ? "bad" : "";
  };

  return (
    <ol className="bars">
      {Array.from({ length: ROWS }, (_, i) => {
        const e = entries[i];
        return (
          <li className={`bar-row${i === 0 ? " top" : ""}`} key={i}>
            <span className="bar-label">
              <span>{e ? signName(e.classId) : ""}</span>
            </span>
            <span className="bar-value mono">{e ? `${(e.prob * 100).toFixed(1)}%` : ""}</span>
            <span className="bar-track">
              <span
                className={`bar-fill ${e ? tone(e, i) : "empty"}`}
                style={{ width: e ? `${e.prob * 100}%` : "100%" }}
              />
            </span>
          </li>
        );
      })}
    </ol>
  );
}
