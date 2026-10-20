---
title: Clothes Segmentation API
emoji: 👗
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: SegFormer-B3 clothes segmentation API for the demo website
---

# Clothes segmentation API

The model behind the demo website: SegFormer-B3 fine-tuned on the ATR human-parsing dataset (15 classes). Test split: mIoU 0.789, garment mIoU 0.744.

- `GET /api/health`: status
- `GET /api/info`: classes, colours and the model card
- `POST /api/segment` (multipart field `image`, optional `?flip=true`): label mask as a base64 PNG (pixel value = class id) and the classes found

Photos are processed in memory and not stored. Research and evaluation use only: the SegFormer weights are under the NVIDIA research licence, and ATR states no licence.
