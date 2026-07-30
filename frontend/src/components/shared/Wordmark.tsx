/**
 * The VentAssist wordmark. `.wordmark` (styles/index.css) carries the gradient and
 * display face; size and any extra spacing come from the caller.
 */
export function Wordmark({ className = "text-3xl" }: { className?: string }) {
  return <span className={`wordmark ${className}`}>VentAssist</span>;
}
