# PyTorch baseline (seg-in / seg-out)

Real PyTorch model inside the sandbox. This baseline is identity at the ID
level (passes input IDs through unchanged) but runs a small CNN on every mask
to prove the full torch / CUDA stack works end-to-end. Replace `predict_one`
with your trained codebook logic.

## Build and submit

```bash
docker build --platform linux/amd64 -t lemanichack/pytorch-baseline:v1 .
docker save lemanichack/pytorch-baseline:v1 | gzip > pytorch_baseline.tar.gz

curl -H "Authorization: Bearer $TOKEN" \
     -T pytorch_baseline.tar.gz \
     https://<orchestrator>/api/upload
```
