import { signName } from "../lib/labels";

export interface BarEntry {
  classId: number;
  prob: number;
}

export function ConfidenceBars({
  entries,
  truthClassId,
}: {
  entries: BarEntry[];
  truthClassId: number | null;
}) {
  const fillColor = (entry: BarEntry, index: number) => {
    if (truthClassId === null) {
      return index === 0 ? "var(--accent)" : "var(--border-strong)";
    }
    return entry.classId === truthClassId ? "var(--ok)" : "var(--bad)";
  };

  return (
    <div className="bars">
      {entries.map((e, i) => (
        <div className={`bar-row${i === 0 ? " top" : ""}`} key={e.classId}>
          <span className="bar-label">{signName(e.classId)}</span>
          <span className="bar-value mono">{(e.prob * 100).toFixed(1)}%</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{
                width: `${Math.max(e.prob * 100, 0.6)}%`,
                background: fillColor(e, i),
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
