# starBoard Model Upgrade Plan: MegaStarID v2 (mAP 0.7001)

**Date**: 2026-04-07
**Status**: PLAN — not yet executed

## Summary

Replace the current MegaStarID embedding checkpoint (Phase 1, mAP=66.4%) with the
new best checkpoint (Phase 2, mAP=70.01%, Rank-1=67.13%). This is a minimal
drop-in change because both checkpoints use the identical architecture.

## Architecture Compatibility Verified

Both old and new checkpoints are:
- ConvNeXt-Tiny backbone
- 512-dim embeddings
- Multiscale fusion (stages 2,3,4)
- BNNeck
- 384×384 image size
- ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

Key verification:
- Old checkpoint: 211 state_dict keys, NO config dict
- New checkpoint: 391 state_dict keys (superset), HAS config dict
- All 211 old keys exist in new checkpoint with IDENTICAL shapes
- The 180 extra keys are raw `backbone.stages.N` duplicates + `id_classifier` (training-only)
- `load_state_dict(strict=False)` in reid_adapter.py will match all 211 expected keys perfectly

## What Changes

ONE file swap:
```
OLD: star_identification/checkpoints/default/best.pth  (current Phase 1 checkpoint)
NEW: /home/weertman/Documents/star_identification/pretrain_depth_sweep/results/ablated_standard_ep003_bs96_ls_finetune_1gpu_bs96/best.pth
```

## What Must Be Regenerated After the Swap

All precomputed artifacts under `archive/_dl_precompute/default_megastarid_v1/` are
keyed to the old checkpoint hash. After swapping the checkpoint, these are STALE
and must be regenerated from the DL tab in starBoard:

  embeddings/
    gallery_embeddings.npz          <- centroid embeddings per gallery ID
    query_embeddings.npz            <- centroid embeddings per query ID
    gallery_image_embeddings.npz    <- per-image embeddings
    query_image_embeddings.npz      <- per-image embeddings
    gallery_image_paths.json
    query_image_paths.json
  
  similarity/
    query_gallery_scores.npz        <- cosine similarity matrix
    image_similarity_matrix.npz
    id_mapping.json
    gallery_image_index.json
    query_image_index.json
    metadata.json

  Also stale: verification artifacts if verification model used embedding backbone

The `_dl_registry.json` checkpoint_hash will be auto-updated when starBoard
detects the hash mismatch and re-precomputes.

## Places in Code That Touch the Model

### Desktop App (src/)
1. `src/dl/reid_adapter.py` — ReIDAdapter.load_model() / _create_model_from_checkpoint()
   - Auto-detects ConvNeXt from `layer_scale` in keys → NO CODE CHANGE NEEDED
   - Config inference: backbone, embedding_dim, multiscale, bnneck → all correct
   - Image size defaults to 384 when no config.model.image_size → correct
   - New checkpoint HAS config dict (plain dict, not dataclass) — the code checks
     `hasattr(config, 'model')` which returns False for dicts, so it falls through
     to state_dict inference → WORKS CORRECTLY

2. `src/dl/registry.py` — DLRegistry tracks checkpoint_path + checkpoint_hash
   - Default path: `star_identification/checkpoints/default/best.pth` → same path
   - Hash auto-updates on re-precompute

3. `src/dl/precompute.py` — Runs embedding extraction + similarity computation
   - No architecture assumptions → NO CODE CHANGE NEEDED

4. `src/ui/tab_dl.py` — UI for DL tab
   - Just orchestrates registry + precompute → NO CODE CHANGE NEEDED

### Mobile Portal (mobile_portal/)
5. `mobile_portal/app/adapters/megastar_model_adapter.py` — Loads model for mobile inference
   - Uses ReIDAdapter internally → NO CODE CHANGE NEEDED
   - Falls through same config inference path

6. `mobile_portal/app/adapters/megastar_artifact_loader.py` — Validates precomputed artifacts
   - Checks artifact files exist + metadata valid → NO CODE CHANGE NEEDED
   - Will need re-generated artifacts on the deployment machine

7. `mobile_portal/megastar_worker/` — Standalone inference worker
   - Delegates to same model loading code → NO CODE CHANGE NEEDED

### Training Code (star_identification/)
8. `star_identification/megastar_identity_verification/` — Verification model
   - Separate model, not affected by embedding checkpoint swap
   - BUT: verification model uses ConvNeXt-Small backbone (not Tiny), loaded independently

## Execution Steps

### On this machine (workstation):
1. Back up current checkpoint:
   ```
   cp star_identification/checkpoints/default/best.pth \
      star_identification/checkpoints/default/best.pth.bak.phase1
   ```

2. Copy new checkpoint:
   ```
   cp /home/weertman/Documents/star_identification/pretrain_depth_sweep/results/ablated_standard_ep003_bs96_ls_finetune_1gpu_bs96/best.pth \
      star_identification/checkpoints/default/best.pth
   ```

3. Launch starBoard desktop app

4. Go to DL tab → click "Precompute All" (this re-extracts all embeddings + similarity)
   - ~6063 images, expect ~10-20 min on GPU, ~1-2 hours on CPU

5. Verify: check that mAP/Rank metrics in DL tab evaluation match expected ~70%

### On Mac mini (security@10.246.1.219) for mobile portal:
6. SCP the new checkpoint to Mac mini:
   ```
   scp star_identification/checkpoints/default/best.pth \
       security@10.246.1.219:/path/to/starBoard/star_identification/checkpoints/default/best.pth
   ```

7. SCP the regenerated artifacts:
   ```
   scp -r archive/_dl_precompute/default_megastarid_v1/ \
       security@10.246.1.219:/path/to/starBoard/archive/_dl_precompute/default_megastarid_v1/
   ```

8. Restart the megastar worker service on Mac mini

## Risk Assessment

- **Architecture mismatch**: NONE — verified identical key shapes
- **Code changes**: ZERO — pure checkpoint swap
- **Rollback**: Restore `best.pth.bak.phase1` and old precomputed artifacts
- **TTA**: Not recommended (only +0.6% gain, adds inference cost)
