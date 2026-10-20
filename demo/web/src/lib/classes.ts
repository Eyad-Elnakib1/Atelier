// The 15 classes of the model, in the model's own order (src/clothseg/labels.py).
// Display colours are the website's own: body parts in muted sand tones, garments in
// strong fabric colours, so the clothes stand out. Only the colours differ from the report.

export type RGB = [number, number, number];

export interface ClassInfo {
  id: number;
  key: string;
  label: string;
  garment: boolean;
  color: RGB;
}

const hex = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];

export const CLASSES: ClassInfo[] = [
  { id: 0, key: "background", label: "Background", garment: false, color: hex("#000000") },
  { id: 1, key: "hat", label: "Hat", garment: true, color: hex("#8C2F39") },
  { id: 2, key: "hair", label: "Hair", garment: false, color: hex("#5B4636") },
  { id: 3, key: "sunglasses", label: "Sunglasses", garment: true, color: hex("#26221F") },
  { id: 4, key: "upper-clothes", label: "Upper clothes", garment: true, color: hex("#D9572B") },
  { id: 5, key: "skirt", label: "Skirt", garment: true, color: hex("#6E8F57") },
  { id: 6, key: "pants", label: "Pants", garment: true, color: hex("#2E5C78") },
  { id: 7, key: "dress", label: "Dress", garment: true, color: hex("#7A4A95") },
  { id: 8, key: "belt", label: "Belt", garment: true, color: hex("#B7832A") },
  { id: 9, key: "shoes", label: "Shoes", garment: true, color: hex("#EBB92E") },
  { id: 10, key: "face", label: "Face", garment: false, color: hex("#E3C6A6") },
  { id: 11, key: "legs", label: "Legs", garment: false, color: hex("#D2AE88") },
  { id: 12, key: "arms", label: "Arms", garment: false, color: hex("#C59E78") },
  { id: 13, key: "bag", label: "Bag", garment: true, color: hex("#1F8A8A") },
  { id: 14, key: "scarf", label: "Scarf", garment: true, color: hex("#C46A92") },
];

export const GARMENT_IDS = CLASSES.filter((c) => c.garment).map((c) => c.id);
export const BODY_IDS = [2, 10, 11, 12];

export const css = (c: RGB, a = 1) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
