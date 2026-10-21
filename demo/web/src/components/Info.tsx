import { motion } from "framer-motion";
import { LIVE } from "../lib/api";
import SectionHead from "./SectionHead";

const reveal = {
  initial: { opacity: 0, y: 24 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.25 },
  transition: { duration: 0.7, ease: [0.2, 0.7, 0.2, 1] },
} as const;

export function HowItWorks() {
  const steps = [
    LIVE
      ? { n: "01", t: "Upload a photo", d: "One person, head to toe, any size or orientation. Or start with one of the sample looks." }
      : { n: "01", t: "Pick a look", d: "Five street-style photos, each segmented by the model ahead of time." },
    { n: "02", t: "Every piece found", d: "Each garment and accessory is outlined and separated from the body and the background." },
    { n: "03", t: "Restyle it", d: "Try any piece in a new colour or fabric and compare before and after — right in your browser." },
    LIVE
      ? { n: "04", t: "Private by design", d: "Your photo is processed in memory and never stored. Nothing you try is saved anywhere." }
      : { n: "04", t: "Runs in your browser", d: "Every view, cut-out and restyle is drawn on your device. Nothing is sent anywhere." },
  ];
  return (
    <section className="how-section" aria-labelledby="how-title">
      <SectionHead eyebrow="No. 06 — How it works" title="From photo to new look in seconds." id="how-title" />
      <ol className="how-steps">
        {steps.map((s, i) => (
          <motion.li key={s.n} {...reveal} transition={{ ...reveal.transition, delay: i * 0.08 }}>
            <span className="how-n mono">{s.n}</span>
            <h3>{s.t}</h3>
            <p>{s.d}</p>
          </motion.li>
        ))}
      </ol>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="footer">
      <div className="footer-mark serif">Atelier</div>
      <p>Your outfit, piece by piece.</p>
      <p className="muted small">
        {LIVE ? "Photos you upload are processed in memory and never stored. " : ""}Sample photos: Clothing Co-Parsing (Apache-2.0). Model: for non-commercial research and evaluation use.
      </p>
    </footer>
  );
}
