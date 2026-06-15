export function Cone({ size = 28 }: { size?: number }) {

  return (
    <svg
      width={(size * 24) / 28}
      height={size}
      viewBox="0 0 24 28"
      fill="none"
      aria-hidden="true"
      role="img"
    >
      <rect x="2.5" y="22" width="19" height="3.4" rx="1.7" fill="var(--accent)" />
      <path
        d="M12 2.2c.62 0 1.16.4 1.36.99l5.9 17.3c.27.8-.32 1.62-1.16 1.62H5.9c-.84 0-1.43-.82-1.16-1.62l5.9-17.3c.2-.59.74-.99 1.36-.99Z"
        fill="var(--accent)"
      />
      <path d="M9.95 9.1h4.1l1.0 2.95H8.95l1.0-2.95Z" fill="#fff" />
      <path d="M8.35 15.5h7.3l1.05 3.05H7.3l1.05-3.05Z" fill="#fff" />
      <circle cx="12" cy="3.1" r="1.15" fill="var(--accent-strong)" />
    </svg>
  );
}

export function Logo() {
  return (
    <a href="#top" className="wordmark" aria-label="SignNet home">
      <span className="wordmark-text">
        <span className="s-hat">
          <span className="cone-hat">
            <Cone size={25} />
          </span>
          s
        </span>
        ign<span className="accent">net</span>
      </span>
    </a>
  );
}
