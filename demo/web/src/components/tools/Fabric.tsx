import { useMemo, useState } from "react";
import type { RGB } from "../../lib/classes";
import { hex, partners } from "../../lib/colour";
import { applyFabric, download, PATTERNS, patternTile, photoCanvas, textureFrom, type Finish, type Pattern } from "../../lib/effects";
import { colourWords, WORDS, type Piece } from "../../lib/pieces";
import { Compare, Dot, PiecePicker } from "../Workroom";
import type { Look } from "../Studio";

// Colour stories to start from. The groupings and shades are the site's own (judgement).
const PALETTES: { name: string; colours: string[] }[] = [
  { name: "Neutrals", colours: ["#f3efe6", "#d7c3a1", "#b8894f", "#808080", "#1f2a46", "#1c1c1c"] },
  { name: "Earth", colours: ["#a3482a", "#6b6a2e", "#c99a1f", "#7b4b2a", "#d9b48f", "#4a5a3a"] },
  { name: "Pastel", colours: ["#f2c4cf", "#b9a6d7", "#a3c4e6", "#a9dcc2", "#f6e3a1", "#f3d2b8"] },
  { name: "Brights", colours: ["#c1272d", "#e8772e", "#f2d43b", "#3b8a3e", "#2f5fb3", "#e0457b"] },
  { name: "Jewel", colours: ["#0f6b4f", "#1d3f8f", "#8e1b3a", "#5b2a86", "#b8860b", "#0e5e6f"] },
];

const FINISHES: { id: Finish; label: string }[] = [
  { id: "matte", label: "Matte" },
  { id: "satin", label: "Satin" },
  { id: "gloss", label: "Gloss" },
];

/** The piece covering the most of the photo: the natural one to start with. */
export const biggest = (pieces: Piece[]) => [...pieces].sort((p, q) => q.percent - p.percent)[0].id;

const toRgb = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];

/** Fabric studio: dye or re-pattern one or several garments; the folds come from the photo's own shading. */
export default function Fabric({ look, pieces }: { look: Look; pieces: Piece[] }) {
  const [ids, setIds] = useState<number[]>(() => [biggest(pieces)]);
  const [kind, setKind] = useState<Pattern>("solid");
  const [a, setA] = useState("#8e1b3a");
  const [b, setB] = useState("#f3efe6");
  const [slot, setSlot] = useState<"a" | "b">("a");
  const [scale, setScale] = useState(1);
  const [angle, setAngle] = useState(0);
  const [shading, setShading] = useState(1);
  const [finish, setFinish] = useState<Finish>("satin");
  const [own, setOwn] = useState<HTMLImageElement | null>(null);
  const p = look.prepared;
  const base = (Math.max(p.w, p.h) / 18) * scale;
  const twoColours = !own && kind !== "solid";

  const toggle = (id: number) => setIds((cur) => (cur.includes(id) ? (cur.length > 1 ? cur.filter((x) => x !== id) : cur) : [...cur, id]));
  const pick = (c: string) => (slot === "b" && twoColours ? setB(c) : setA(c));

  const original = useMemo(() => photoCanvas(p), [p]);
  const result = useMemo(() => {
    const tex = own
      ? textureFrom(own, base * 4, (base * 4 * own.naturalHeight) / own.naturalWidth, p.w, p.h, angle)
      : textureFrom(patternTile(kind, toRgb(a), toRgb(b), base), base, base, p.w, p.h, angle);
    return applyFabric(p, ids, tex, shading, finish);
  }, [p, ids, kind, a, b, base, angle, shading, finish, own]);

  // Colours that go with the first chosen piece's own colour (neutrals, neighbours, opposite).
  const lead = pieces.find((x) => x.id === ids[0])!;
  const ideas = useMemo(() => partners(lead.colour.main), [lead]);

  const thumbs = useMemo(() => PATTERNS.map((x) => ({ ...x, url: patternTile(x.id, toRgb(a), toRgb(b), 44).toDataURL() })), [a, b]);
  const names = pieces.filter((x) => ids.includes(x.id)).map((x) => x.info.label.toLowerCase());

  const upload = (file: File | undefined) => {
    if (!file || !file.type.startsWith("image/")) return;
    const img = new Image();
    img.onload = () => setOwn(img);
    img.src = URL.createObjectURL(file);
  };

  const reset = () => {
    setKind("solid"); setA("#8e1b3a"); setB("#f3efe6"); setScale(1); setAngle(0); setShading(1); setFinish("satin"); setOwn(null);
  };

  return (
    <div className="studio">
      <div className="studio-head">
        <span className="panel-title mono">Apply to · choose one or more pieces</span>
        <PiecePicker pieces={pieces} selected={ids} onPick={toggle} multi label="Pieces to restyle" />
      </div>
      <div className="tool-grid">
        <div className="stage">
          <Compare before={original} after={result} label={`The ${names.join(" and ")} before and after`} />
          <div className="stage-actions">
            <button className="btn btn-ink" onClick={() => download(result, `atelier-${look.title.toLowerCase().replace(/\W+/g, "-")}.png`)}>Download image</button>
            <button className="btn btn-line" onClick={reset}>Reset</button>
          </div>
        </div>

        <div className="tool-controls">
          <div className="control-block">
            <span className="panel-title mono">Fabric</span>
            <div className="fabric-grid" role="radiogroup" aria-label="Fabric">
              {thumbs.map((x) => (
                <button key={x.id} role="radio" aria-checked={!own && kind === x.id} className={`fabric ${!own && kind === x.id ? "is-on" : ""}`} onClick={() => { setOwn(null); setKind(x.id); }}>
                  <img src={x.url} alt="" />
                  <span>{x.label}</span>
                </button>
              ))}
              <label className={`fabric fabric-own ${own ? "is-on" : ""}`}>
                {own ? <img src={own.src} alt="" /> : <span className="fabric-plus" aria-hidden="true">＋</span>}
                <span>Your fabric</span>
                <input type="file" accept="image/*" className="sr-only" onChange={(e) => { upload(e.target.files?.[0]); e.target.value = ""; }} />
              </label>
            </div>
          </div>

          {!own && (
            <div className="control-block">
              <div className="row-between">
                <span className="panel-title mono">Colour</span>
                {twoColours && (
                  <div className="seg seg-small" role="radiogroup" aria-label="Which colour the swatches set">
                    <button role="radio" aria-checked={slot === "a"} className={slot === "a" ? "is-on" : ""} onClick={() => setSlot("a")}><i style={{ background: a }} />Main</button>
                    <button role="radio" aria-checked={slot === "b"} className={slot === "b" ? "is-on" : ""} onClick={() => setSlot("b")}><i style={{ background: b }} />Second</button>
                  </div>
                )}
              </div>
              <div className="suggest">
                <span className="small">Suggested for the <b>{colourWords(lead.colour)} {WORDS[lead.id].noun}</b></span>
                {ideas.map((g) => (
                  <div key={g.label} className="idea-row">
                    <span className="small muted">{g.label}</span>
                    <div className="idea-swatches">
                      {g.colours.map((c, i) => {
                        const h = hex(c.rgb);
                        const on = (slot === "b" && twoColours ? b : a).toLowerCase() === h;
                        return (
                          <button key={c.name + i} className={`idea-btn ${on ? "is-on" : ""}`} onClick={() => pick(h)} aria-pressed={on}>
                            <Dot colour={c.rgb} />{c.name}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
              {PALETTES.map((pal) => (
                <div key={pal.name} className="palette-row">
                  <span className="small muted">{pal.name}</span>
                  <div className="palette-swatches">
                    {pal.colours.map((c) => {
                      const on = (slot === "b" && twoColours ? b : a).toLowerCase() === c;
                      return <button key={c} className={`swatch-btn ${on ? "is-on" : ""}`} style={{ background: c }} onClick={() => pick(c)} aria-label={`Use ${c}`} title={c} />;
                    })}
                  </div>
                </div>
              ))}
              <div className="field-row">
                <label className="field field-colour">
                  <span>Main · {a}</span>
                  <input type="color" value={a} onChange={(e) => setA(e.target.value)} />
                </label>
                {twoColours && (
                  <label className="field field-colour">
                    <span>Second · {b}</span>
                    <input type="color" value={b} onChange={(e) => setB(e.target.value)} />
                  </label>
                )}
                <button className="btn btn-line btn-small" onClick={() => setA(hex(pieces.find((x) => x.id === ids[0])!.colour.main.rgb))} title="Start from the piece's own colour">Its own colour</button>
              </div>
            </div>
          )}

          <div className="control-block">
            <span className="panel-title mono">Finish</span>
            <div className="seg" role="radiogroup" aria-label="Finish">
              {FINISHES.map((f) => (
                <button key={f.id} role="radio" aria-checked={finish === f.id} className={finish === f.id ? "is-on" : ""} onClick={() => setFinish(f.id)}>{f.label}</button>
              ))}
            </div>
            {(kind !== "solid" || own) && (
              <div className="field-row">
                <label className="field">
                  <span>Pattern size · {scale.toFixed(1)}×</span>
                  <input type="range" min={0.4} max={3} step={0.1} value={scale} onChange={(e) => setScale(Number(e.target.value))} />
                </label>
                <label className="field">
                  <span>Angle · {angle}°</span>
                  <input type="range" min={-90} max={90} step={5} value={angle} onChange={(e) => setAngle(Number(e.target.value))} />
                </label>
              </div>
            )}
            <label className="field">
              <span>Folds and shadows · {Math.round(shading * 100)}%</span>
              <input type="range" min={0} max={1.5} step={0.05} value={shading} onChange={(e) => setShading(Number(e.target.value))} />
            </label>
          </div>
          <p className="tool-note">The fabric is laid flat over the piece, so a pattern does not bend around the body the way real cloth would. Its edges follow the detected outline, so small errors at the edges can show.</p>
        </div>
      </div>
    </div>
  );
}
