import { motion } from "framer-motion";
import { useMemo } from "react";
import { CLASSES, css } from "../lib/classes";
import { cutout } from "../lib/compose";
import SectionHead from "./SectionHead";
import type { Look } from "./Studio";

/** Each garment found, cut out with a transparent background and hung on a rail. */
export default function Rail({ look }: { look: Look | null }) {
  const items = useMemo(() => {
    if (!look) return [];
    return look.result.classes
      .filter((c) => c.garment && c.percent >= 0.15)
      .map((c) => ({ ...c, info: CLASSES[c.id], cut: cutout(look.prepared, [c.id]) }))
      .filter((x) => x.cut);
  }, [look]);

  const stem = look ? look.title.replace(/\.[^.]+$/, "").replace(/[^a-z0-9]+/gi, "-").toLowerCase() : "look";

  return (
    <section className="rail-section" aria-labelledby="rail-title">
      <SectionHead eyebrow="No. 03 — The rail" title="Each piece, cut out and hung up." id="rail-title">
        Every garment in the photo, separated with a transparent background and ready to use anywhere. Click a piece to download it as a PNG.
      </SectionHead>
      {items.length === 0 ? (
        <p className="empty-note">Try a look in the fitting room above, and its garments will hang here.</p>
      ) : (
        <div className="rail" role="list">
          <div className="rail-bar" aria-hidden="true" />
          {items.map((x, i) => (
            <motion.a
              role="listitem"
              key={`${stem}-${x.id}`}
              className="hanger"
              href={x.cut!.url}
              download={`${stem}-${x.info.key}.png`}
              initial={{ opacity: 0, y: -30, rotate: -6 }}
              animate={{ opacity: 1, y: 0, rotate: 0 }}
              transition={{ type: "spring", stiffness: 140, damping: 11, delay: i * 0.08 }}
              whileHover={{ rotate: [0, -3, 2.5, -1.5, 0], transition: { duration: 0.9 } }}
              aria-label={`Download the ${x.info.label.toLowerCase()} as a PNG`}
            >
              <svg className="hanger-hook" viewBox="0 0 120 54" aria-hidden="true">
                <path d="M60 2c6 0 9 4 9 8s-4 6-9 9v6L8 48h104L60 25" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <div className="hanger-piece">
                <img src={x.cut!.url} alt={x.info.label} />
                <span className="download-chip" aria-hidden="true">↓ Download PNG</span>
              </div>
              <span className="hanger-tag" style={{ ["--tag" as string]: css(x.info.color) }}>
                <span className="tag-hole" aria-hidden="true" />
                <b>{x.info.label}</b>
                <span className="mono">{x.percent.toFixed(1)}% of photo · PNG ↓</span>
              </span>
            </motion.a>
          ))}
        </div>
      )}
    </section>
  );
}
