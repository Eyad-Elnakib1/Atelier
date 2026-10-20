import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { render, type Prepared } from "../lib/compose";
import { TOOLS, type ToolId } from "../lib/tools";

interface Preview {
  title: string;
  prepared: Prepared;
}

interface Props {
  previews: Preview[];
  onTry: () => void;
  onSample: () => void;
  onTool: (id: ToolId) => void;
}

const SLIDE_MS = 4200;

/**
 * The first screen: the sample looks fill the background one after another, each shown with
 * its garments coloured over the person; the headline, buttons and tool shortcuts sit centred on top.
 */
export default function Hero({ previews, onTry, onSample, onTool }: Props) {
  const [index, setIndex] = useState(0);
  const reduce = useReducedMotion();
  const count = previews.length;

  // A fresh timer for every look, so a look picked with the dots also gets its full time.
  useEffect(() => {
    if (reduce || count < 2) return;
    const t = setTimeout(() => setIndex((i) => (i + 1) % count), SLIDE_MS);
    return () => clearTimeout(t);
  }, [index, count, reduce]);

  const current = count ? previews[index % count] : null;

  return (
    <header className="hero">
      <div className="tape" aria-hidden="true">
        {Array.from({ length: 40 }, (_, i) => (
          <span key={i}>{i % 5 === 0 ? i : ""}</span>
        ))}
      </div>
      <div className="hero-stage">
        <div className="hero-bg" aria-hidden="true">
          {current ? (
            <AnimatePresence initial={false}>
              <motion.div
                key={current.title}
                className="hero-slide"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 2 }}
              >
                <Slide prepared={current.prepared} />
              </motion.div>
            </AnimatePresence>
          ) : (
            <div className="arch-placeholder" />
          )}
          <div className="hero-scrim" />
        </div>

        <motion.div className="hero-copy" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: [0.2, 0.7, 0.2, 1] }}>
          <span className="eyebrow">Atelier · a digital fitting room</span>
          <h1>
            Your outfit,
            <br />
            <em>reimagined.</em>
          </h1>
          <p className="lede">
            Upload a photo of what you are wearing and try every piece in a new colour or fabric.
            Find your next look — before you buy, dye or sew a thing.
          </p>
          <div className="hero-actions">
            <button className="btn btn-cream" onClick={onTry}>Try your photo</button>
            <button className="btn btn-glass" onClick={onSample}>See a sample look</button>
          </div>
          <nav className="hero-tools" aria-label="Jump to a tool">
            {TOOLS.map((t, i) => (
              <motion.button
                key={t.id}
                className="hero-tool"
                onClick={() => onTool(t.id)}
                title={t.blurb}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + i * 0.06, duration: 0.4 }}
              >
                <span className="hero-tool-glyph" aria-hidden="true">{t.glyph}</span>
                <span>{t.label}</span>
              </motion.button>
            ))}
          </nav>
        </motion.div>

        {current && (
          <div className="hero-caption">
            <AnimatePresence mode="wait">
              <motion.span key={current.title} className="serif" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.3 }}>
                {current.title}
              </motion.span>
            </AnimatePresence>
            {count > 1 && (
              <span className="hero-dots">
                {previews.map((p, i) => (
                  <button key={p.title} className={i === index % count ? "is-on" : ""} onClick={() => setIndex(i)} aria-label={`Show the sample look: ${p.title}`} />
                ))}
              </span>
            )}
          </div>
        )}
      </div>
    </header>
  );
}

/** One background look: a blurred copy fills the sides, the sharp photo sits in the middle. */
function Slide({ prepared }: { prepared: Prepared }) {
  const front = useRef<HTMLCanvasElement>(null);
  const back = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const img = render(prepared, "photo");
    for (const c of [front.current, back.current]) {
      if (!c) continue;
      c.width = prepared.w;
      c.height = prepared.h;
      c.getContext("2d")?.putImageData(img, 0, 0);
    }
  }, [prepared]);
  return (
    <>
      <canvas ref={back} className="hero-bg-fill" />
      <canvas ref={front} className="hero-bg-photo" />
    </>
  );
}
