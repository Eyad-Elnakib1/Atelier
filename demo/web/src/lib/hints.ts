// Remembers which controls a visitor has already used, so their hints stop showing.
// Kept in localStorage; if storage is blocked, hints simply show for the session.

import { useCallback, useState } from "react";

export type HintId = "compare" | "tags";

const KEY = "atelier-hints-seen";

function load(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "{}");
  } catch {
    return {};
  }
}

export function useHints(): [Record<string, boolean>, (id: HintId) => void] {
  const [seen, setSeen] = useState<Record<string, boolean>>(load);
  const markSeen = useCallback((id: HintId) => {
    setSeen((cur) => {
      if (cur[id]) return cur;
      const next = { ...cur, [id]: true };
      try {
        localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
        /* storage blocked: remember for this visit only */
      }
      return next;
    });
  }, []);
  return [seen, markSeen];
}
