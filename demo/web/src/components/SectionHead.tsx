import { motion, useMotionValue, useMotionValueEvent, useReducedMotion, useScroll, useTransform, type MotionValue } from "framer-motion";
import { Fragment, useMemo, useRef, type ReactNode } from "react";

interface Props {
  eyebrow: string;
  title: string;
  id: string;
  children?: ReactNode;
}

/** Small deterministic random numbers, so every visit scatters the letters the same way. */
function seeded(n: number): number {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

/**
 * Section heading whose letters are driven by scrolling: they start scattered and move
 * into place as the heading scrolls up the screen, each one stopping in its own spot.
 * Once assembled, the heading stays assembled (scrolling back up does not scatter it).
 * The eyebrow line, the eyebrow and the paragraph follow the same scroll. Static with reduced motion.
 */
export default function SectionHead({ eyebrow, title, id, children }: Props) {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLHeadingElement>(null);
  // 0 when the heading enters at the bottom of the screen, 1 when it reaches 35% from the top.
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "start 0.35"] });
  const settled = useMotionValue(0);
  useMotionValueEvent(scrollYProgress, "change", (v) => {
    if (v > settled.get()) settled.set(Math.min(1, v)); // only ever moves towards "assembled"
  });

  // The eyebrow and the paragraph follow the same scroll: line first, then label, paragraph last.
  const lineScale = useTransform(settled, [0, 0.25], [0, 1], { clamp: true });
  const eyebrowOpacity = useTransform(settled, [0.1, 0.35], [0, 1], { clamp: true });
  const eyebrowX = useTransform(settled, [0.1, 0.35], [-12, 0], { clamp: true });
  const paraOpacity = useTransform(settled, [0.65, 1], [0, 1], { clamp: true });
  const paraY = useTransform(settled, [0.65, 1], [16, 0], { clamp: true });

  const words = useMemo(() => title.split(" "), [title]);
  const total = title.replace(/ /g, "").length;

  if (reduce) {
    return (
      <div className="section-head">
        <span className="eyebrow">{eyebrow}</span>
        <h2 id={id} ref={ref}>{title}</h2>
        {children && <p>{children}</p>}
      </div>
    );
  }

  let index = 0;
  return (
    <div className="section-head">
      <span className="eyebrow eyebrow-animated">
        <motion.i className="eyebrow-line" aria-hidden="true" style={{ scaleX: lineScale }} />
        <motion.span style={{ opacity: eyebrowOpacity, x: eyebrowX }}>{eyebrow}</motion.span>
      </span>
      <h2 id={id} ref={ref} aria-label={title}>
        {words.map((w, wi) => (
          <Fragment key={wi}>
            <span className="word-keep" aria-hidden="true">
              {[...w].map((ch) => {
                const i = index++;
                return <Letter key={i} ch={ch} i={i} total={total} progress={settled} />;
              })}
            </span>
            {wi < words.length - 1 && " "}
          </Fragment>
        ))}
      </h2>
      {children && <motion.p style={{ opacity: paraOpacity, y: paraY }}>{children}</motion.p>}
    </div>
  );
}

function Letter({ ch, i, total, progress }: { ch: string; i: number; total: number; progress: MotionValue<number> }) {
  // Each letter has its own scatter and its own window of the scroll, so they land one after another.
  const start = (i / Math.max(1, total)) * 0.45;
  const end = Math.min(1, start + 0.55);
  const dx = (seeded(i) - 0.5) * 2 * 260;
  const dy = 60 + seeded(i + 99) * 160;
  const rot = (seeded(i + 7) - 0.5) * 2 * 70;
  const x = useTransform(progress, [start, end], [dx, 0], { clamp: true });
  const y = useTransform(progress, [start, end], [dy, 0], { clamp: true });
  const rotate = useTransform(progress, [start, end], [rot, 0], { clamp: true });
  const opacity = useTransform(progress, [start, start + (end - start) * 0.6], [0, 1], { clamp: true });
  const blur = useTransform(progress, [start, end], [6, 0], { clamp: true });
  const filter = useTransform(blur, (b) => (b > 0.05 ? `blur(${b}px)` : "none"));
  return (
    <motion.span className="letter" style={{ x, y, rotate, opacity, filter }}>
      {ch}
    </motion.span>
  );
}
