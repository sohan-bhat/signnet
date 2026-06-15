import { signName } from "../lib/labels";

export interface SampleItem {
  classId: number;
  file: string;
}

export function Gallery({
  samples,
  activeFile,
  onPick,
}: {
  samples: SampleItem[];
  activeFile: string | null;
  onPick: (sample: SampleItem) => void;
}) {
  return (
    <div className="gallery">
      {samples.map((s) => {
        const url = `/samples/${s.file}`;
        return (
          <button
            key={s.file}
            className={`sample${activeFile === s.file ? " active" : ""}`}
            onClick={() => onPick(s)}
            title={`${signName(s.classId)}, click to classify`}
          >
            <img src={url} alt={signName(s.classId)} loading="lazy" />
          </button>
        );
      })}
    </div>
  );
}
