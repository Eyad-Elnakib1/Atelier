// The garments of one look, with their colours and everyday names, shared by every tool.

import { CLASSES, type ClassInfo } from "./classes";
import { judge, pieceColour, type PieceColour, type Verdict } from "./colour";
import type { Prepared } from "./compose";
import type { SegResult } from "./api";

export interface Piece {
  id: number;
  info: ClassInfo;
  percent: number;
  colour: PieceColour;
}

/** Everyday words for each garment class. "Pants" also covers shorts in ATR's labels. */
export const WORDS: Record<number, { noun: string; plural: boolean; category: string; slot: Slot }> = {
  1: { noun: "hat", plural: false, category: "Accessories › Hats", slot: "accessory" },
  3: { noun: "sunglasses", plural: true, category: "Accessories › Sunglasses", slot: "accessory" },
  4: { noun: "top", plural: false, category: "Tops", slot: "top" },
  5: { noun: "skirt", plural: false, category: "Bottoms › Skirts", slot: "bottom" },
  6: { noun: "trousers", plural: true, category: "Bottoms › Trousers & shorts", slot: "bottom" },
  7: { noun: "dress", plural: false, category: "Dresses", slot: "dress" },
  8: { noun: "belt", plural: false, category: "Accessories › Belts", slot: "accessory" },
  9: { noun: "shoes", plural: true, category: "Shoes", slot: "shoes" },
  13: { noun: "bag", plural: false, category: "Bags", slot: "accessory" },
  14: { noun: "scarf", plural: false, category: "Accessories › Scarves", slot: "accessory" },
};

export type Slot = "top" | "bottom" | "dress" | "shoes" | "accessory";

/** Head to toe, then what is carried. */
const ORDER = [1, 3, 14, 4, 7, 8, 5, 6, 9, 13];

export function piecesOf(prepared: Prepared, result: SegResult, minPercent = 0.15): Piece[] {
  return result.classes
    .filter((c) => c.garment && c.percent >= minPercent && WORDS[c.id])
    .map((c) => ({ id: c.id, info: CLASSES[c.id], percent: c.percent, colour: pieceColour(prepared, c.id) }))
    .filter((x): x is Piece => !!x.colour)
    .sort((a, b) => ORDER.indexOf(a.id) - ORDER.indexOf(b.id));
}

/** "navy", "brown and black patterned". */
export function colourWords(c: PieceColour): string {
  if (c.second) return `${c.main.name} and ${c.second.name}`;
  return c.patterned ? `patterned ${c.main.name}` : c.main.name;
}

export function outfitVerdict(pieces: Piece[]): Verdict {
  // Small accessories are left out: a black belt does not change how an outfit's colours work.
  const main = pieces.filter((p) => [4, 5, 6, 7, 9, 14, 1].includes(p.id));
  return judge(main.map((p) => p.colour.main));
}
