# Atelier — your outfit, reimagined

Atelier is a web app built on the clothes-segmentation model in this repository. Upload a photo of what you are wearing: every garment is found and separated from the body and the background, and you can try any piece in a new colour or fabric before you buy, dye or sew a thing. Everything after the segmentation runs in your browser, and photos are never stored.

Two parts:

| Part | Folder | Runs on |
|---|---|---|
| Model server (FastAPI around the evaluated `Segmenter`) | `demo/server/` | A Hugging Face Space (Docker, free CPU) |
| Website (Vite, React, TypeScript, Framer Motion) | `demo/web/` | Vercel, as a static site |

The model cannot run on Vercel itself: a Vercel function may be at most 250 MB uncompressed, and PyTorch plus the 189 MB weights are larger ([Vercel limits](https://vercel.com/docs/functions/limitations)).

## What the website does

- **First screen:** four sample looks fade one into the next behind the headline; one click goes to your own photo or a sample.
- **Fitting room:** upload a photo or pick a sample; an arched mirror shows the result with a comparison slider. Views: colour map, clothes only, overlay, person without background, original. Hover a garment tag to isolate it.
- **The rail:** every garment found, cut out with a transparent background, downloadable as PNG.
- **Anatomy of an outfit:** the outfit as stacked 3D layers, one per garment.
- **Style studio:** restyle one or several pieces at once.
  - 12 fabrics (plain dye, Breton, pinstripe, gingham, windowpane, tartan, houndstooth, herringbone, chevron, argyle, polka dot, denim twill) or a photo of your own fabric;
  - colours suggested for the piece (neutrals, neighbours and opposite on the colour wheel) and five ready colour sets;
  - matte, satin or gloss finish, pattern size and angle, and how strongly the folds and shadows show;
  - a before/after slider, and a PNG download of the restyled photo.
  The new fabric keeps the garment's own folds: it is applied in OKLab colour space, adding each pixel's lightness difference from its garment's average to the fabric, so it works on dark and light clothes alike.

The server returns only the label mask (one class id per pixel); every view and tool is computed in the browser. Photos are processed in memory and not stored. The four sample looks are photos from the Clothing Co-Parsing dataset (Apache-2.0) with results pre-computed by the same model, so they work even when the free server is asleep.

## Run locally

```bash
# 1. model server (needs best.pt in the project root)
py -3 -m uvicorn demo.server.app:app --port 7860
# 2. website
cd demo/web
npm install
npm run dev          # http://localhost:5173, talks to http://localhost:7860
```

Tests of the server: `py -3 demo/server/test_app.py <folder with .jpg photos>`.

## Deploy

1. **Model server:** `py -3 tools/build_space.py` assembles `demo/space_build/` (app, Dockerfile, the `clothseg` package and `best.pt`). Create a Hugging Face Space with the Docker SDK and upload that folder:
   `hf upload <user>/clothes-segmentation-api demo/space_build . --repo-type=space`
2. **Website:** import the repository in Vercel with **Root Directory** `demo/web`, and set the environment variable `VITE_API_URL=https://<user>-clothes-segmentation-api.hf.space`. Or, without GitHub, from `demo/web`: `npx vercel --prod`.

A free Space goes to sleep when it is not used; the first photo after that takes longer while it wakes up. The website shows the server's state and keeps the samples available meanwhile.
