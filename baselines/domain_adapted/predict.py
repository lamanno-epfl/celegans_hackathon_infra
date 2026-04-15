"""Predict for domain-adapted baseline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.common.rotation6d import numpy_rotation_6d_to_matrix
from baselines.domain_adapted.model import DomainAdaptedRegressor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=os.environ.get("INPUT_DIR", "/input"))
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    p.add_argument("--weights", default=os.environ.get("WEIGHTS", "/app/model.pt"))
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((input_dir / "manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomainAdaptedRegressor().to(device)
    weights = Path(args.weights)
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location=device))
    else:
        print(f"[warn] no weights at {weights}", file=sys.stderr)
    model.eval()

    poses, embs = {}, []
    with torch.no_grad():
        for name in manifest:
            img = np.load(input_dir / "images" / name).astype(np.float32)
            x = torch.from_numpy(img).unsqueeze(0).to(device)
            emb, rot6, trans, _ = model(x, alpha=0.0)
            R = numpy_rotation_6d_to_matrix(rot6.cpu().numpy()[0])
            poses[name] = {"rotation": R.tolist(), "translation": trans.cpu().numpy()[0].tolist()}
            embs.append(emb.cpu().numpy()[0])
    embeddings = np.stack(embs, axis=0).astype(np.float32)
    (output_dir / "poses.json").write_text(json.dumps(poses))
    np.save(output_dir / "embeddings.npy", embeddings)
    (output_dir / "metadata.json").write_text(
        json.dumps({"embedding_dim": int(embeddings.shape[1]), "model_name": "domain_adapted_resnet18"})
    )


if __name__ == "__main__":
    main()
