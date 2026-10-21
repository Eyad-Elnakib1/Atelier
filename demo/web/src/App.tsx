import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { isReady, LIVE, type SegResult } from "./lib/api";
import { prepare, type Prepared } from "./lib/compose";
import { piecesOf } from "./lib/pieces";
import type { ToolId } from "./lib/tools";
import Hero from "./components/Hero";
import Studio, { type Look, type Sample } from "./components/Studio";
import Rail from "./components/Rail";
import Layers from "./components/Layers";
import Workroom from "./components/Workroom";
import { Footer, HowItWorks } from "./components/Info";

type ModelState = "checking" | "ready" | "waking" | "offline";

/** The sample look selected when the page opens. */
const DEFAULT_SAMPLE = "ccp-0777"; // Bomber and dress

export default function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [previews, setPreviews] = useState<{ title: string; prepared: Prepared }[]>([]);
  const [look, setLook] = useState<Look | null>(null);
  const [model, setModel] = useState<ModelState>(LIVE ? "checking" : "offline");
  const picker = useRef<(() => void) | null>(null);
  // Garments of the current look with their colours, used by the style studio.
  const pieces = useMemo(() => (look ? piecesOf(look.prepared, look.result) : []), [look]);

  // Sample looks (pre-computed) feed the hero's background, one after another; "Bomber and dress" opens in the fitting room.
  useEffect(() => {
    fetch("/samples/index.json")
      .then((r) => r.json() as Promise<Sample[]>)
      .then(async (list) => {
        setSamples(list);
        const defaultId = list.some((x) => x.id === DEFAULT_SAMPLE) ? DEFAULT_SAMPLE : list[0]?.id;
        for (const s of list) {
          try {
            const result = (await fetch(s.result).then((r) => r.json())) as SegResult;
            const prepared = await prepare(s.image, result.mask_png, result.width, result.height);
            setPreviews((cur) => (cur.some((x) => x.title === s.title) ? cur : [...cur, { title: s.title, prepared }]));
            if (s.id === defaultId) setLook((cur) => cur ?? { title: s.title, photo: s.image, result, prepared, live: false });
          } catch {
            /* a sample that fails to load is skipped */
          }
        }
      })
      .catch(() => undefined);
  }, []);

  // A free model server sleeps when unused; keep asking for a while so it can wake up. No server in showcase mode.
  useEffect(() => {
    if (!LIVE) return;
    let stop = false;
    (async () => {
      for (let attempt = 0; attempt < 30 && !stop; attempt++) {
        if (await isReady(attempt === 0 ? 6000 : 10000)) {
          if (!stop) setModel("ready");
          return;
        }
        if (!stop) setModel(attempt < 24 ? "waking" : "offline");
        await new Promise((r) => setTimeout(r, 5000));
      }
      if (!stop) setModel("offline");
    })();
    return () => {
      stop = true;
    };
  }, []);

  const toStudio = useCallback((open: boolean) => {
    document.getElementById("studio")?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (open) setTimeout(() => picker.current?.(), 450);
  }, []);

  // The shortcut at the top scrolls to the style studio.
  const openTool = useCallback((_id: ToolId) => {
    document.getElementById("workroom")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  return (
    <>
      <nav className="nav" aria-label="Site">
        <span className="nav-mark serif">Atelier</span>
        {LIVE && (
          <span className={`status status-${model}`} role="status" aria-live="polite">
            <i aria-hidden="true" />
            {{ checking: "Connecting", ready: "Ready", waking: "Starting up", offline: "Offline · samples available" }[model]}
          </span>
        )}
      </nav>
      <main>
        <Hero previews={previews} onTry={LIVE ? () => toStudio(true) : undefined} onSample={() => toStudio(false)} onTool={openTool} />
        <Studio samples={samples} modelState={model} look={look} onLook={setLook} pickerRef={picker} />
        <Rail look={look} />
        <Layers look={look} />
        <Workroom look={look} pieces={pieces} />
        <HowItWorks />
      </main>
      <Footer />
    </>
  );
}
