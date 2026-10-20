import { useEffect, useRef, useState } from "react";
import { css } from "../lib/classes";
import { colourWords, type Piece } from "../lib/pieces";
import SectionHead from "./SectionHead";
import type { Look } from "./Studio";
import Fabric from "./tools/Fabric";

/** The style studio: restyle the pieces of the look above. */
export default function Workroom({ look, pieces }: { look: Look | null; pieces: Piece[] }) {
  return (
    <section className="workroom-section" id="workroom" aria-labelledby="workroom-title">
      <SectionHead eyebrow="No. 05 — The style studio" title="Restyle any piece." id="workroom-title">
        Pick one or more pieces and dye them any colour or dress them in a new fabric. Drag across the photo to compare before and after.
      </SectionHead>
      {!look ? (
        <p className="empty-note">Try a look in the fitting room first.</p>
      ) : pieces.length === 0 ? (
        <p className="empty-note">No garment was found in this look.</p>
      ) : (
        <div className="tool-panel">
          <Fabric key={look.title} look={look} pieces={pieces} />
        </div>
      )}
    </section>
  );
}

/** Garment chips; one or several can be on. */
export function PiecePicker({ pieces, selected, onPick, multi = false, label }: { pieces: Piece[]; selected: number[]; onPick: (id: number) => void; multi?: boolean; label: string }) {
  return (
    <div className="piece-picker" role={multi ? "group" : "radiogroup"} aria-label={label}>
      {pieces.map((p) => {
        const on = selected.includes(p.id);
        return (
          <button key={p.id} role={multi ? "checkbox" : "radio"} aria-checked={on} className={`piece-chip ${on ? "is-on" : ""}`} onClick={() => onPick(p.id)} style={{ ["--tag" as string]: css(p.info.color) }}>
            <Dot colour={p.colour.main.rgb} second={p.colour.second?.rgb} />
            <span>{p.info.label}</span>
            <span className="muted small">{colourWords(p.colour)}</span>
          </button>
        );
      })}
    </div>
  );
}

export function Dot({ colour, second }: { colour: [number, number, number]; second?: [number, number, number] }) {
  const bg = second ? `linear-gradient(135deg, ${css(colour)} 0 55%, ${css(second)} 55%)` : css(colour);
  return <i className="colour-dot" style={{ background: bg }} aria-hidden="true" />;
}

function Paint({ source, className }: { source: HTMLCanvasElement; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.width = source.width;
    c.height = source.height;
    c.getContext("2d")?.drawImage(source, 0, 0);
  }, [source]);
  return <canvas ref={ref} className={className} />;
}

/** Before and after, one over the other, with a handle to drag between them. */
export function Compare({ before, after, label }: { before: HTMLCanvasElement; after: HTMLCanvasElement; label: string }) {
  const [pos, setPos] = useState(50);
  return (
    <div className="compare" role="group" aria-label={label} style={{ aspectRatio: `${after.width} / ${after.height}` }}>
      <Paint source={after} className="compare-layer" />
      <div className="compare-before" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <Paint source={before} className="compare-layer" />
      </div>
      <div className="compare-line" style={{ left: `${pos}%` }} aria-hidden="true">
        <span className="compare-knob">⟷</span>
      </div>
      <span className="compare-tag compare-tag-l mono" aria-hidden="true">Before</span>
      <span className="compare-tag compare-tag-r mono" aria-hidden="true">After</span>
      <input className="compare-range" type="range" min={0} max={100} value={pos} onChange={(e) => setPos(Number(e.target.value))} aria-label="Drag to compare before and after" />
    </div>
  );
}
