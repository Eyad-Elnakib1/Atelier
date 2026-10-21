# Atelier — your outfit, reimagined

Atelier is a web app built on the clothes-segmentation model in this repository. Upload a photo of what you are wearing: every garment is found and separated from the body and the background, and you can try any piece in a new colour or fabric before you buy, dye or sew a thing. Everything after the segmentation runs in your browser, and photos are never stored.

Two modes:

| Mode | When | What visitors can do |
|---|---|---|
| **Showcase** (the public site) | `VITE_API_URL` not set | Explore five sample looks segmented by the model ahead of time: every view, cut-out, layer and restyle. No server; hosted free as a static site on Vercel. |
| **Live** | `VITE_API_URL` points to the model server (`demo/server/`) | Everything above, plus uploading their own photo. |

The model cannot run on Vercel itself: a Vercel function may be at most 250 MB uncompressed, and PyTorch plus the 189 MB weights are larger ([Vercel limits](https://vercel.com/docs/functions/limitations)). Hosting the server on a Hugging Face Docker Space now needs a PRO subscription ([pricing](https://huggingface.co/pricing)), so the public site runs as a showcase.

## What the website does

- **First screen:** the five sample looks fade one into the next behind the headline.
- **Fitting room:** upload a photo or pick a sample; an arched mirror shows the result with a comparison slider. Views: colour map, clothes only, overlay, person without background, original. Hover a garment tag to isolate it.
- **The rail:** every garment found, cut out with a transparent background, downloadable as PNG.
- **Anatomy of an outfit:** the outfit as stacked 3D layers, one per garment.
- **Style studio:** restyle one or several pieces at once.
  - 12 fabrics (plain dye, Breton, pinstripe, gingham, windowpane, tartan, houndstooth, herringbone, chevron, argyle, polka dot, denim twill) or a photo of your own fabric;
  - colours suggested for the piece (neutrals, neighbours and opposite on the colour wheel) and five ready colour sets;
  - matte, satin or gloss finish, pattern size and angle, and how strongly the folds and shadows show;
  - a before/after slider, and a PNG download of the restyled photo.
  The new fabric keeps the garment's own folds: it is applied in OKLab colour space, adding each pixel's lightness difference from its garment's average to the fabric, so it works on dark and light clothes alike.

The server returns only the label mask (one class id per pixel); every view and tool is computed in the browser. Photos are processed in memory and not stored. The five sample looks are photos from the Clothing Co-Parsing dataset (Apache-2.0) with results pre-computed by the same model and the same server code (`flip=true`).

## Run locally

```bash
# Showcase only (sample looks, no upload)
cd demo/web
npm install
npm run dev          # http://localhost:5173

# Live mode: also start the model server (needs best.pt in the project root, from the GitHub Release)
py -3 -m uvicorn demo.server.app:app --port 7860
# and in demo/web create .env.local with:  VITE_API_URL=http://localhost:7860
```

Tests of the server: `py -3 demo/server/test_app.py <folder with .jpg photos>`.

## Deploy the showcase (free)

Import the GitHub repository in Vercel (Hobby plan) with **Root Directory** `demo/web` and no environment variables, then deploy. Every push to `main` redeploys the site.

## Deploy the live mode (optional)

`py -3 tools/build_space.py` assembles `demo/space_build/` (app, Dockerfile, the `clothseg` package and `best.pt`) for a Docker host such as a Hugging Face Docker Space (PRO) or Google Cloud Run. Then set `VITE_API_URL` in Vercel to the server's address and `ALLOWED_ORIGINS` on the server to the site's address.
