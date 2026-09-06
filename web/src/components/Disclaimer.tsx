export function Disclaimer({ onAcknowledge }: { onAcknowledge: () => void }) {
  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="disclaimer-title">
      <div className="modal">
        <h3 id="disclaimer-title">Before you upload</h3>
        <p>
          The model only knows the 43 GTSRB traffic sign types. If you give it an image that
          is not a sign, it will still report the closest sign it can find.
        </p>
        <button className="btn btn-primary" onClick={onAcknowledge} autoFocus>
          Continue
        </button>
      </div>
    </div>
  );
}
