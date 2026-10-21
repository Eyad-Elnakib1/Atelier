import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { LIVE, segment, type SegResult } from "../lib/api";
import { CLASSES, css } from "../lib/classes";
import { prepare, render, type Prepared, type View } from "../lib/compose";
import { useHints } from "../lib/hints";
import SectionHead from "./SectionHead";

export interface Sample {
  id: string;
  title: string;
  image: string;
  result: string;
}

export interface Look {
  title: string;
  photo: string;
  result: SegResult;
  prepared: Prepared;
  live: boolean;
}

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "map", label: "Colour map", hint: "each piece in its own colour" },
  { id: "clothes", label: "Clothes only", hint: "everything else in black" },
  { id: "overlay", label: "Overlay", hint: "colours over the photo" },
  { id: "person", label: "Person", hint: "background removed" },
  { id: "photo", label: "Original", hint: "the photo as uploaded" },
];

const SCAN_LINES = ["Reading the silhouette", "Finding the seams", "Sorting fabric from skin", "Labelling every pixel"];

interface Props {
  samples: Sample[];
  modelState: "checking" | "ready" | "waking" | "offline";
  look: Look | null;
  onLook: (look: Look) => void;
  pickerRef: React.MutableRefObject<(() => void) | null>;
}

export default function Studio({ samples, modelState, look, onLook, pickerRef }: Props) {
  const [view, setView] = useState<View>("map");
  const [focus, setFocus] = useState<number | null>(null);
  const [split, setSplit] = useState(50);
  const [seen, markSeen] = useHints();
  const dragging = useRef(false);
  const flip = false; // the faster setting; flip averaging (the report's setting) is not offered on the site
  const [busy, setBusy] = useState<{ photo: string; title: string; started: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const topRef = useRef<HTMLCanvasElement>(null);
  const baseRef = useRef<HTMLCanvasElement>(null);

  pickerRef.current = () => fileRef.current?.click();
  const liveTest = useRef(false);

  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => setElapsed((Date.now() - busy.started) / 1000), 100);
    return () => clearInterval(t);
  }, [busy]);

  useEffect(() => {
    if (!look) return;
    const draw = (c: HTMLCanvasElement | null, v: View, f: number | null) => {
      if (!c) return;
      c.width = look.prepared.w;
      c.height = look.prepared.h;
      c.getContext("2d")?.putImageData(render(look.prepared, v, f), 0, 0);
    };
    draw(baseRef.current, "photo", null);
    draw(topRef.current, view, focus);
  }, [look, view, focus, busy]);

  const runSample = useCallback(async (s: Sample) => {
    setError(null);
    setBusy({ photo: s.image, title: s.title, started: Date.now() });
    try {
      const [result] = await Promise.all([
        fetch(s.result).then((r) => r.json() as Promise<SegResult>),
        new Promise((r) => setTimeout(r, 1100)), // let the scan play once; the result itself is pre-computed
      ]);
      const prepared = await prepare(s.image, result.mask_png, result.width, result.height);
      onLook({ title: s.title, photo: s.image, result, prepared, live: false });
      setFocus(null);
      setSplit(50);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, [onLook]);

  const runFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please choose a photo (JPEG, PNG or WebP).");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("This photo is larger than 10 MB. Please choose a smaller one.");
      return;
    }
    setError(null);
    const url = URL.createObjectURL(file);
    setBusy({ photo: url, title: file.name, started: Date.now() });
    try {
      const result = await segment(file, flip);
      const prepared = await prepare(url, result.mask_png, result.width, result.height);
      onLook({ title: file.name, photo: url, result, prepared, live: true });
      setFocus(null);
      setSplit(50);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, [flip, onLook]);

  // Test switch: ?live=<sample id> sends that sample photo to the live model, as an upload would.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("live");
    const s = samples.find((x) => x.id === id);
    if (!s || liveTest.current || modelState !== "ready") return;
    liveTest.current = true;
    fetch(s.image).then((r) => r.blob()).then((b) => runFile(new File([b], `${s.id}.jpg`, { type: "image/jpeg" })));
  }, [samples, modelState, runFile]);

  const pickTag = (id: number | null) => {
    setFocus(id);
    if (id !== null) markSeen("tags");
  };

  const found = look ? look.result.classes.filter((c) => c.percent >= 0.05) : [];
  const garments = found.filter((c) => c.garment);
  const body = found.filter((c) => !c.garment);
  const scanText = SCAN_LINES[Math.min(SCAN_LINES.length - 1, Math.floor(elapsed / 1.4))];

  return (
    <section className="studio" id="studio" aria-labelledby="studio-title">
      <SectionHead eyebrow="No. 02 — The fitting room" title="Step in front of the mirror." id="studio-title">
        {LIVE ? "Upload a photo of a person, or try one of the looks." : "Pick one of the five looks."} Every garment is found and outlined; switch between the views below.
      </SectionHead>

      <div className="studio-grid">
        <div
          className={`mirror ${drag ? "is-drag" : ""}`}
          onDragOver={(e) => { if (!LIVE) return; e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            if (!LIVE) return;
            e.preventDefault();
            setDrag(false);
            const f = e.dataTransfer.files?.[0];
            if (f) runFile(f);
          }}
        >
          <div className="mirror-glass">
            <AnimatePresence initial={false}>
              {busy ? (
                <motion.div key="scan" className="mirror-scan" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <img src={busy.photo} alt="" />
                  <div className="scan-beam" aria-hidden="true" />
                  <div className="scan-caption" role="status" aria-live="polite">
                    <span>{scanText}…</span>
                    <span className="mono">{elapsed.toFixed(1)} s</span>
                  </div>
                </motion.div>
              ) : look ? (
                <motion.div
                  key={look.title + look.result.ms}
                  className="mirror-result"
                  initial={{ opacity: 0, scale: 0.985 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5 }}
                  onPointerDown={(e) => {
                    dragging.current = true;
                    e.currentTarget.setPointerCapture(e.pointerId);
                    const r = e.currentTarget.getBoundingClientRect();
                    setSplit(Math.round(Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100))));
                    markSeen("compare");
                  }}
                  onPointerMove={(e) => {
                    if (!dragging.current) return;
                    const r = e.currentTarget.getBoundingClientRect();
                    setSplit(Math.round(Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100))));
                  }}
                  onPointerUp={() => (dragging.current = false)}
                  onPointerCancel={() => (dragging.current = false)}
                >
                  <canvas ref={baseRef} className="layer-base" aria-hidden="true" />
                  <canvas
                    ref={topRef}
                    className="layer-top"
                    style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
                    role="img"
                    aria-label={`Segmentation of ${look.title}: ${garments.map((g) => CLASSES[g.id].label).join(", ") || "no garments found"}`}
                  />
                  <div className="split-line" style={{ left: `${split}%` }} aria-hidden="true">
                    <span className="split-knob">↔</span>
                  </div>
                  <span className="split-label split-label-left" aria-hidden="true">{VIEWS.find((v) => v.id === view)?.label}</span>
                  <span className="split-label split-label-right" aria-hidden="true">Photo</span>
                  <AnimatePresence>
                    {!seen.compare && (
                      <motion.span className="hint hint-mirror" style={{ x: "-50%" }} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ delay: 0.6 }}>
                        <i aria-hidden="true" />
                        Drag the line ↔ to compare with the photo
                      </motion.span>
                    )}
                  </AnimatePresence>
                </motion.div>
              ) : !LIVE ? (
                <motion.div key="loading" className="mirror-empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <span className="serif">Loading the looks…</span>
                </motion.div>
              ) : (
                <motion.button key="empty" className="mirror-empty" onClick={() => fileRef.current?.click()} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <span className="mirror-empty-icon" aria-hidden="true">＋</span>
                  <span className="serif">Drop a photo here</span>
                  <span className="muted">or click to choose one · JPEG, PNG, WebP · up to 10 MB</span>
                </motion.button>
              )}
            </AnimatePresence>
          </div>
          {look && !busy && (
            <label className="split-control">
              <span className="mono">Compare ↔</span>
              <input type="range" min={0} max={100} value={split} onChange={(e) => { setSplit(Number(e.target.value)); markSeen("compare"); }} aria-label="Compare the view with the original photo" />
            </label>
          )}
          {LIVE && <p className="mirror-tip">Tip: you can also drop a photo onto the mirror.</p>}
          <input
            ref={fileRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) runFile(f);
              e.target.value = "";
            }}
          />
        </div>

        <aside className="studio-panel">
          <div className="panel-block">
            <span className="panel-title mono">{LIVE ? "Try a look" : "Pick a look"}</span>
            <div className="sample-row">
              {samples.map((s) => (
                <button key={s.id} className={`sample ${look?.title === s.title ? "is-on" : ""}`} onClick={() => runSample(s)} disabled={!!busy} aria-label={`Try the sample look: ${s.title}`}>
                  <img src={s.image} alt="" loading="lazy" />
                  <span>{s.title}</span>
                </button>
              ))}
            </div>
            {LIVE && (
              <>
                <div className="upload-row">
                  <button className="btn btn-ink" onClick={() => fileRef.current?.click()} disabled={!!busy}>Upload your photo</button>
                </div>
                <ModelNote state={modelState} />
              </>
            )}
          </div>

          <AnimatePresence>
            {error && (
              <motion.p className="error" role="alert" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                {error}
              </motion.p>
            )}
          </AnimatePresence>

          {look && !busy && (
            <motion.div className="panel-block" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
              <span className="panel-title mono">Choose a view</span>
              <div className="swatches" role="radiogroup" aria-label="View">
                {VIEWS.map((v) => (
                  <button key={v.id} role="radio" aria-checked={view === v.id} className={`swatch swatch-${v.id} ${view === v.id ? "is-on" : ""}`} onClick={() => setView(v.id)} title={v.hint}>
                    <span className="swatch-cloth" aria-hidden="true" />
                    <span>{v.label}</span>
                  </button>
                ))}
              </div>

              <span className="panel-title mono">Found in this photo</span>
              <AnimatePresence initial={false}>
                {!seen.tags && (
                  <motion.span className="hint" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}>
                    <i aria-hidden="true" />
                    Hover or tap a tag to isolate that piece in the mirror
                  </motion.span>
                )}
              </AnimatePresence>
              <TagList items={garments} focus={focus} setFocus={pickTag} nudge={!seen.tags} />
              {body.length > 0 && <TagList items={body} focus={focus} setFocus={pickTag} quiet />}

              <p className="meta mono">
                {look.live ? `Ready in ${(look.result.ms / 1000).toFixed(1)} s.` : "Sample look."}
              </p>
            </motion.div>
          )}
        </aside>
      </div>
    </section>
  );
}

function TagList({ items, focus, setFocus, quiet = false, nudge = false }: { items: { id: number; percent: number }[]; focus: number | null; setFocus: (id: number | null) => void; quiet?: boolean; nudge?: boolean }) {
  return (
    <ul className={`tags ${quiet ? "tags-quiet" : ""}`}>
      {items.map((c, i) => {
        const info = CLASSES[c.id];
        return (
          <li key={c.id}>
            <button
              className={`tag ${focus === c.id ? "is-on" : ""} ${nudge && i === 0 ? "is-nudge" : ""}`}
              onMouseEnter={() => setFocus(c.id)}
              onMouseLeave={() => setFocus(null)}
              onFocus={() => setFocus(c.id)}
              onBlur={() => setFocus(null)}
              onClick={() => setFocus(focus === c.id ? null : c.id)}
              style={{ ["--tag" as string]: css(info.color) }}
            >
              <span className="tag-hole" aria-hidden="true" />
              <span className="tag-name">{info.label}</span>
              <span className="tag-pct mono">{c.percent < 1 ? c.percent.toFixed(2) : c.percent.toFixed(1)}%</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function ModelNote({ state }: { state: Props["modelState"] }) {
  const text = {
    checking: "Connecting…",
    ready: "Ready. Your photo is processed in memory and never stored.",
    waking: "Starting up — this can take a minute. The sample looks work right away.",
    offline: "Photo upload is unavailable right now. The sample looks still work.",
  }[state];
  return <p className={`model-note state-${state}`}>{text}</p>;
}
