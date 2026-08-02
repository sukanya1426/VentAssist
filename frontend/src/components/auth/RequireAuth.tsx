import { useEffect, type ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useAuth } from "../../store/useAuth";

/**
 * Wraps every route that shows patient data. Anonymous visitors are sent to the
 * sign-in page, and where they were going is carried along so the sign-in lands
 * them there rather than dumping everyone on the roster.
 *
 * The `checking` state matters: a returning clinician holds a token that has not
 * been validated yet, and rendering the redirect during that window would bounce
 * them to sign-in and straight back, losing the page they asked for.
 */
export function RequireAuth({ children }: { children: ReactNode }) {
  const status = useAuth((s) => s.status);
  const restore = useAuth((s) => s.restore);
  const location = useLocation();

  useEffect(() => {
    if (status === "checking") restore();
  }, [status, restore]);

  if (status === "checking") {
    return (
      <div className="grid min-h-screen place-items-center">
        <Loader2 className="animate-spin text-slate-300" size={26} />
      </div>
    );
  }

  if (status === "anonymous") {
    return <Navigate to="/signin" replace state={{ from: location.pathname + location.search }} />;
  }

  return <>{children}</>;
}
