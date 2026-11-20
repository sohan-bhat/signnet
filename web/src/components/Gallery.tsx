import { BASE } from "../lib/base";
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
    <div className="picker">
      {samples.map((s) => (
        <button
          type="button"
          key={s.file}
          className="sample"
          aria-pressed={activeFile === s.file}
          onClick={() => onPick(s)}
        >
          <img src={`${BASE}/samples/${s.file}`} alt={signName(s.classId)} width={48} height={48} />
        </button>
      ))}
    </div>
  );
}
