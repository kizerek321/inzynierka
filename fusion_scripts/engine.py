import os
import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "SAMPolyBuild"))
sys.path.append(str(BASE_DIR / "Pix2Poly"))

from shapely.geometry import Polygon, mapping


class SatelliteFusionEngine:
    def __init__(self, sam_checkpoint, pix2poly_checkpoint, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing fusion engine on: {self.device}")

        # SAMPolyBuild  = segmentation
        from segment_anything import sam_model_registry
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint) #check for the model in .\SAMPolyBuild\segment_anything
        self.sam.to(device=self.device)
        self.sam.eval()

        from segment_anything.predictor import SamPredictor
        self.sam_predictor = SamPredictor(self.sam)

        # Pix2Poly = vectorization 
        self.pix2poly, self.tokenizer, self.pix2poly_cfg = self._load_pix2poly(pix2poly_checkpoint)
        self.pix2poly.to(device=self.device)
        self.pix2poly.eval()

    def _load_pix2poly(self, checkpoint_path):
        """
        Initialize the Pix2Poly architecture (Encoder-Decoder transformer
        with ScoreNet for the Optimal Matching Network) and load weights.

        Architecture per README:
          (i)   Discrete Sequence Tokenizer  – tokenizer.py
          (ii)  Vertex Sequence Detector     – Encoder + Decoder
          (iii) Optimal Matching Network     – ScoreNet (x2) inside EncoderDecoder
        """
        from config import CFG
        from tokenizer import Tokenizer
        from models.model import Encoder, Decoder, EncoderDecoder

        # Build the tokenizer (Discrete Sequence Tokenizer)
        tokenizer = Tokenizer(
            num_classes=1,
            num_bins=CFG.NUM_BINS,
            width=CFG.INPUT_WIDTH,
            height=CFG.INPUT_HEIGHT,
            max_len=CFG.MAX_LEN,
        )
        CFG.PAD_IDX = tokenizer.PAD_code

        # Build encoder (ViT backbone)
        encoder = Encoder(
            model_name=CFG.MODEL_NAME,
            pretrained=False,   # weights come from the checkpoint
            out_dim=256,
        )

        # Build decoder (Transformer decoder)
        decoder = Decoder(
            cfg=CFG,
            vocab_size=tokenizer.vocab_size,
            encoder_len=CFG.NUM_PATCHES,
            dim=256,
            num_heads=8,
            num_layers=6,
        )

        # Build full model (Vertex Sequence Detector + Optimal Matching Network)
        model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)

        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Pix2Poly loaded from epoch {checkpoint.get('epochs_run', '?')}")

        return model, tokenizer, CFG

    # ------------------------------------------------------------------
    #  Pix2Poly inference helpers (adapted from Pix2Poly/utils.py)
    # ------------------------------------------------------------------
    def _pix2poly_generate(self, image_tensor):
        """
        Autoregressive sequence generation for a batch of images.
        Returns predicted token sequences, confidences, and permutation matrices.
        """
        from utils import scores_to_permutations

        cfg = self.pix2poly_cfg
        tok = self.tokenizer

        x = image_tensor.to(self.device)
        batch_preds = torch.ones((x.size(0), 1), device=self.device).fill_(tok.BOS_code).long()
        confs = []

        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)

        with torch.no_grad():
            for i in range(cfg.generation_steps):
                preds, feats = self.pix2poly.predict(x, batch_preds)
                if i % 2 == 0:
                    confs_ = torch.softmax(preds, dim=-1).sort(dim=-1, descending=True)[0][:, 0].cpu()
                    confs.append(confs_)
                preds = sample(preds)
                batch_preds = torch.cat([batch_preds, preds], dim=1)

            # Optimal Matching Network – produce permutation matrix
            perm_preds = self.pix2poly.scorenet1(feats) + torch.transpose(
                self.pix2poly.scorenet2(feats), 1, 2
            )
            perm_preds = scores_to_permutations(perm_preds)

        return batch_preds.cpu(), confs, perm_preds

    def _pix2poly_postprocess(self, batch_preds, batch_confs):
        """Decode token sequences into (x, y) coordinates."""
        tok = self.tokenizer

        EOS_idxs = (batch_preds == tok.EOS_code).float().argmax(dim=-1)
        # Sanity check – EOS must come after an even number of coordinate tokens
        invalid_idxs = ((EOS_idxs - 1) % 2 != 0).nonzero().view(-1)
        EOS_idxs[invalid_idxs] = 0

        all_coords, all_confs = [], []
        for i, eos in enumerate(EOS_idxs.tolist()):
            if eos == 0:
                all_coords.append(None)
                all_confs.append(None)
                continue
            coords = tok.decode(batch_preds[i, :eos + 1])
            c = [round(batch_confs[j][i].item(), 3) for j in range(len(coords))]
            all_coords.append(coords)
            all_confs.append(c)

        return all_coords, all_confs

    @torch.no_grad()
    def process_image(self, image_path):
        """
        Main fusion pipeline: Image -> SAM Mask -> Pix2Poly Polygon

        Returns:
            masks:    list of binary masks (np.ndarray, H×W each)
            polygons: list of polygon coordinate arrays (N×2 each)
        """
        from utils import permutations_to_polygons
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        cfg = self.pix2poly_cfg

        # ---- 1. Read & preprocess ----
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]

        # ---- 2. SAM inference: generating binary masks ----
        self.sam_predictor.set_image(image_rgb)
        # Use a grid of foreground points to prompt SAM (whole-image segmentation)
        grid_size = 8
        point_coords = np.array([
            [x, y]
            for y in np.linspace(0, original_h - 1, grid_size, dtype=int)
            for x in np.linspace(0, original_w - 1, grid_size, dtype=int)
        ])
        point_labels = np.ones(len(point_coords), dtype=int)  # all foreground

        masks_np, iou_preds, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        # Pick the mask with the highest predicted IoU
        best_idx = int(np.argmax(iou_preds))
        sam_mask = masks_np[best_idx].astype(np.uint8)  # H×W binary

        self.sam_predictor.reset_image()

        # ---- 3. Pix2Poly inference: generating vertex coordinates ----
        # Prepare the masked image crop for Pix2Poly (resize to 224×224)
        masked_image = image_rgb.copy()
        masked_image[sam_mask == 0] = 0  # zero-out background

        pix2poly_transform = A.Compose([
            A.Resize(height=cfg.INPUT_HEIGHT, width=cfg.INPUT_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ])
        transformed = pix2poly_transform(image=masked_image)
        img_tensor = transformed["image"].unsqueeze(0)  # 1×3×224×224

        batch_preds, batch_confs, perm_preds = self._pix2poly_generate(img_tensor)
        vertex_coords, confs = self._pix2poly_postprocess(batch_preds, batch_confs)

        # Build graph tensor for permutation→polygon conversion
        coords_for_perm = []
        for vc in vertex_coords:
            if vc is not None:
                coord = torch.from_numpy(vc)
            else:
                coord = torch.tensor([])
            pad = torch.ones((cfg.N_VERTICES - len(coord), 2)).fill_(cfg.PAD_IDX)
            coord = torch.cat([coord, pad], dim=0)
            coords_for_perm.append(coord)

        batch_polygons = permutations_to_polygons(perm_preds, coords_for_perm, out="torch")

        # Scale polygons from Pix2Poly space (224×224) back to original image size
        scale_x = original_w / cfg.INPUT_WIDTH
        scale_y = original_h / cfg.INPUT_HEIGHT
        output_polygons = []
        for sample_polys in batch_polygons:
            for poly_tensor in sample_polys:
                poly = poly_tensor.cpu().numpy()
                poly = poly[poly[:, 0] != cfg.PAD_IDX]  # remove padding
                if len(poly) < 3:
                    continue
                # Pix2Poly outputs (x, y); scale back
                poly[:, 0] *= scale_x
                poly[:, 1] *= scale_y
                output_polygons.append(poly)

        return [sam_mask], output_polygons

    def save_to_geojson(self, vertices_list, output_path):
        """Save the obtained polygons to GeoJSON format.

        Args:
            vertices_list: list of Nx2 numpy arrays (polygon vertices)
            output_path:   path for the output .geojson file
        """
        features = []
        for idx, verts in enumerate(vertices_list):
            if len(verts) < 3:
                continue
            # Close the polygon ring if necessary
            if not np.array_equal(verts[0], verts[-1]):
                verts = np.vstack([verts, verts[0]])

            poly = Polygon(verts.tolist())
            if not poly.is_valid:
                poly = poly.buffer(0)  # attempt to fix self-intersections

            feature = {
                "type": "Feature",
                "id": idx,
                "properties": {"building_id": idx},
                "geometry": mapping(poly),
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"Saved {len(features)} polygon(s) to {output_path}")