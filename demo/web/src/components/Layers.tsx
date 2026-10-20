import { motion, useReducedMotion } from "framer-motion";
import { useMemo, useState } from "react";
import { BODY_IDS, CLASSES, css } from "../lib/classes";
import { layer, present } from "../lib/compose";
import SectionHead from "./SectionHead";
import type { Look } from "./Studio";

// Bottom to top: the body, then the garments roughly in the order they are worn.
const ORDER = [5, 6, 7, 4, 8, 14, 9, 13, 1, 3];

/** The outfit taken apart into stacked layers, one per class, in 3D. */
export default function Layers({ look }: { look: Look | null }) {
  const [open, setOpen] = useState(true);
  const reduce = useReducedMotion();

  const layers = useMemo(() => {
    if (!look) return [];
    const have = present(look.prepared);
    const list: { key: string; label: string; color: string; url: string }[] = [];
    if (BODY_IDS.some((id) => have.has(id))) {
      list.push({ key: "body", label: "Body · hair, face, arms, legs", color: css([210, 174, 136]), url: layer(look.prepared, BODY_IDS) });
    }
    for (const id of ORDER) {
      const pct = look.result.classes.find((c) => c.id === id)?.percent ?? 0;
      if (have.has(id) && pct >= 0.15) {
        list.push({ key: CLASSES[id].key, label: CLASSES[id].label, color: css(CLASSES[id].color), url: layer(look.prepared, [id]) });
      }
    }
    return list;
  }, [look]);

  const gap = layers.length > 6 ? 46 : 58;

  return (
    <section className="layers-section" aria-labelledby="layers-title">
      <SectionHead eyebrow="No. 04 — Anatomy of an outfit" title="Taken apart, layer by layer." id="layers-title">
        The look, stacked in depth: one sheet for the body and one for each garment. Not one picture, but separate pieces.
      </SectionHead>
      {layers.length === 0 ? (
        <p className="empty-note">Try a look in the fitting room, then come back here.</p>
      ) : (
        <div className="layers-wrap">
          <button className="btn btn-line layers-toggle" onClick={() => setOpen((o) => !o)} aria-pressed={open}>
            {open ? "Assemble the outfit" : "Take it apart"}
          </button>
          <div className="layers-body">
          <div className="layers-stage" style={{ height: 380 + (open ? layers.length * gap * 0.55 : 0) }}>
            {layers.map((l, i) => (
              <motion.div
                key={l.key}
                className="layer-sheet"
                initial={false}
                animate={
                  reduce
                    ? { opacity: 1 }
                    : { rotateX: open ? 58 : 0, rotateZ: open ? -32 : 0, y: open ? -i * gap : 0, opacity: 1 }
                }
                transition={{ type: "spring", stiffness: 70, damping: 16, delay: open ? i * 0.05 : (layers.length - i) * 0.03 }}
                style={{ zIndex: i + 1, ["--tag" as string]: l.color }}
              >
                <img src={l.url} alt="" />
              </motion.div>
            ))}
          </div>
          <ol className="layers-legend" aria-label="Layers, from top to bottom">
            {[...layers].reverse().map((l, i) => (
              <motion.li key={l.key} style={{ ["--tag" as string]: l.color }} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.05 }}>
                <i aria-hidden="true" />
                <span>{l.label}</span>
              </motion.li>
            ))}
          </ol>
          </div>
        </div>
      )}
    </section>
  );
}
