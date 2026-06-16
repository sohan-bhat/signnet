export function Disclaimer({ onAcknowledge }: { onAcknowledge: () => void }) {
  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="disclaimer-title">
      <div className="modal">
        <h3 id="disclaimer-title">Heads up</h3>
        <p>
          This model only recognizes the 43 GTSRB traffic sign types. Given a non-sign image, it
          will still pick the closest sign, that's expected for a classifier trained only on signs.
        </p>
        <button className="btn btn-primary" onClick={onAcknowledge} autoFocus>
          I understand
        </button>
      </div>
    </div>
  );
}
