// Client for the model server (demo/server/app.py), usually a Hugging Face Space.
// Its address comes from VITE_API_URL at build time.

export const API_URL: string =
  ((import.meta.env.VITE_API_URL as string | undefined) || "http://localhost:7860").replace(/\/+$/, "");

export interface SegClass {
  id: number;
  name: string;
  percent: number;
  garment: boolean;
}

export interface SegResult {
  width: number;
  height: number;
  original_width: number;
  original_height: number;
  mask_png: string; // base64 PNG, pixel value = class id
  classes: SegClass[];
  ms: number;
  flip: boolean;
  source?: string;
}

async function withTimeout<T>(ms: number, run: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    return await run(ctrl.signal);
  } finally {
    clearTimeout(timer);
  }
}

/** True when the server answers and has the model loaded. */
export async function isReady(timeoutMs = 8000): Promise<boolean> {
  try {
    return await withTimeout(timeoutMs, async (signal) => {
      const r = await fetch(`${API_URL}/api/health`, { signal, cache: "no-store" });
      if (!r.ok) return false;
      const j = await r.json();
      return j.status === "ok";
    });
  } catch {
    return false;
  }
}

/** Send one photo to the model. A sleeping free server can take a minute or two to answer. */
export async function segment(file: Blob, flip: boolean): Promise<SegResult> {
  const form = new FormData();
  form.append("image", file, "photo");
  const r = await withTimeout(180_000, (signal) =>
    fetch(`${API_URL}/api/segment?flip=${flip}`, { method: "POST", body: form, signal }),
  ).catch((e: unknown) => {
    const aborted = e instanceof DOMException && e.name === "AbortError";
    throw new Error(aborted ? "The service did not answer in time. It may still be starting up; try again in a minute."
                            : "The service cannot be reached right now. The sample looks still work.");
  });
  if (!r.ok) {
    let detail = `Something went wrong (error ${r.status}). Please try again.`;
    try {
      const j = await r.json();
      if (typeof j.detail === "string") detail = j.detail;
    } catch {
      /* not JSON: keep the generic message */
    }
    throw new Error(detail);
  }
  return r.json();
}
