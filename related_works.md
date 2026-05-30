# Related Works: Production Soccer Video Analytics (2024–2025)

Research sources: 110-agent deep research sweep (May 2025) — 27 sources, 130 claims extracted, 25 adversarially verified (12 confirmed, 13 killed). Cross-referenced against independent research report "Building a Production-Quality, Scalable AI Soccer Video Analysis System" (May 2026).

**Camera setup note:** Rosco AI uses a **pan camera that follows the ball**, not a fixed wide camera. This changes the difficulty table significantly:

```
Problem              Fixed cam        Pan cam (ours)
────────────────     ─────────────    ──────────────
Ball detection       Hard             Easy (always near center)
Ball tracking        Hard             Easy (camera follows it)
Homography           Easy (once)      Hard (every frame)
Player re-entry      Rare             Constant (pan cuts players)
Player re-ID         Manageable       The hardest problem
```

This makes the SoccerNet broadcast pipeline research more applicable than typical "fixed camera" sports analytics literature.

---

## 1. Multi-Object Tracking for Sports Video

### The Core Problem
Standard general-purpose trackers (Kalman filter + IoU matching) fail on sports footage for three compounding reasons: **similar player appearances**, **irregular non-linear motion**, and **significant camera movement**. The SportsMOT benchmark (arxiv 2304.05170) quantified this — HOTA in the 55–70 range for soccer, with IoU-based methods consistently underperforming. Kalman filter methods specifically fail to capture the abrupt, non-linear trajectories common in sports.

### YOLO + ByteTrack: The Production Baseline
As of early 2026, YOLO + ByteTrack is the production-viable baseline for multi-class soccer detection and tracking (arxiv 2602.18504). It is not state-of-the-art. ByteTrack's critical flaw for sports: it prunes lost tracklets after `max_frames_lost` frames, making re-identification of players who leave and re-enter the frame impossible by design. Strong through partial occlusion but falls apart when a player exits and re-enters.

### Tracker Comparison for Fixed-Camera Soccer

| Tracker | Year | Strengths | Notes |
|---------|------|-----------|-------|
| **ByteTrack** | 2021 (ECCV'22) | Associates high and low-confidence boxes; fast | Default soccer baseline; prunes lost tracklets [arXiv:2110.06864] |
| **BoT-SORT** | 2022 | Adds appearance (Re-ID) + camera motion compensation | Appearance helps ID switches; camera motion comp less needed on fixed cam [arXiv:2206.14651] |
| **OC-SORT** | 2022 (CVPR'23) | Observation-centric; handles non-linear motion/occlusion | Good for fast direction changes, no appearance model [arXiv:2203.14360] |
| **StrongSORT** | 2022 | Appearance-based, accurate, heavier | Best ID stability when compute allows [arXiv:2202.13514] |
| **Deep-EIoU** | 2025 | Extended IoU + deep appearance features | SoccerNet 2025 winner; 28% higher HOTA, 48% fewer ID switches vs ByteTrack |
| **Deep HM-SORT** | 2024 | Retains all tracklets indefinitely | Designed for sports re-entry scenarios [arXiv:2406.12081] |

**Practical implementation:** `boxmot` (github.com/mikel-brostrom/boxmot) plugs ByteTrack/BoT-SORT/OC-SORT/StrongSORT into any YOLO detector. The recommended upgrade path for this project.

### Deep-EIoU + OSNet ReID (SoccerNet 2025 winner)
The winning pipeline for SoccerNet 2025 Game State Reconstruction used YOLO-X for detection, **Deep-EIoU** for tracking, and **OSNet** (from torchreid) for appearance-based re-identification. Deep-EIoU achieved **28% higher HOTA (0.54 vs 0.42)** and **48% fewer identity switches** than ByteTrack on sports footage. Current state-of-the-art for challenge benchmarks.

**Directly applicable to a pan camera:** SoccerNet broadcast footage also uses panning cameras, so these results transfer directly. Player re-ID after leaving/re-entering frame due to camera pan is exactly the problem Deep-EIoU + OSNet solves.

### GTATrack (arxiv 2602.00484, SoccerTrack 2025 winner)
Independently confirms the three sports tracking challenges. Uses Deep-EIoU at its core. Validates that the SoccerNet challenge results generalize to other sports tracking benchmarks.

### SoccerNet Challenge Results
| Year | Winner | System | GS-HOTA | Baseline |
|------|--------|---------|---------|----------|
| 2024 | Constructor.tech | YOLOv5m + DeepSORT (pitch coords, TeamID constraints) | 63.81 | 23.36 |
| 2025 | KIST-GSR | YOLO-X + Deep-EIoU + OSNet + LLaMA-3.2-Vision | 63.90 | 29.01 |

Sources: arxiv 2409.10587v1 (CVPR 2024 CVSports), arxiv 2508.19182v1 (August 2025), arxiv 2504.06357.

### LLaMA-3.2-Vision for Jersey Number ReID
The SoccerNet 2025 winner incorporated LLaMA-3.2-Vision to read jersey numbers as a hard identity anchor. **This solves a broadcast TV problem** — camera cuts and zoom transitions cause complete player disappearance with no spatial continuity. Not applicable to fixed single-camera setups. Per-frame OCR is unreliable; the correct approach (if attempted) is tracklet-level aggregation via confidence-weighted voting across all legible frames (PARSeq/TrOCR transformer OCR > CRNN). Reported accuracy is ~85–92% on legible tracklets, much lower once illegible cases (far side, facing away, blurred) are included — a large fraction of tracklets must be flagged "unknown."

**Re-identification fundamental limit:** Standard appearance Re-ID fails in soccer because teammates wear identical kits — embeddings cannot separate them. Robust identity requires combining tracking (short-term continuity) + jersey number (tracklet-aggregated) + team assignment + spatial constraints. For a fixed single camera, team assignment + tracking IDs are reliable; per-player jersey ID is best-effort.

---

## 2. Ball Tracking — The Hardest Problem

Ball tracking is the single biggest failure point in soccer analytics. The ball is small (often under 10px on a wide camera), fast, frequently occluded by players, subject to motion blur, and easily confused with white field lines. Generic YOLO detection achieves only ~0.6–0.8 mAP@0.5 on the ball, far below the ~0.90–0.97 achievable for players. Ball tracking accuracy has no clean published benchmark for amateur footage — it varies enormously with resolution and must be measured on your own data.

### TrackNet Family — Relevant for Fixed Cameras, Less So for Pan Cameras
TrackNet (Huang et al., 2019, IEEE AVSS; arXiv:1907.03698) introduced **temporal heatmap models** that consume several consecutive frames to exploit ball motion, predicting a heatmap of ball position rather than a bounding box. Designed for cases where the ball is tiny, fast, and frequently exits the detection window.

- **TrackNetV2** (Sun et al., IEEE ICPAI 2020) — improved training efficiency
- **TrackNetV3** (2023) — adds rectification and motion attention

**For a pan camera: likely skip this.** The camera follows the ball, so it stays near the center of frame and is always a large-ish detection target. The hard ball-tracking problems (ball leaving frame, <10px size at far end) don't apply. YOLO detection + Kalman interpolation (as in v1) is sufficient. Revisit TrackNet only if ball detection is observed to drop significantly in practice.

### SAHI — Small Object Fix for Wide Cameras
**SAHI (Slicing Aided Hyper Inference, arXiv:2202.06934)** tiles each frame into overlapping sub-images, runs detection on each tile independently, then merges the results. Materially improves tiny-object recall (ball, far-side players) on wide-angle footage at the cost of extra compute. This is the standard practical fix for the small-object problem on a single fixed wide camera.

---

## 3. Camera Calibration and Field Homography

### The Paradigm Shift: Homography vs. Full 3D Calibration
Most prior work (including the Rosco AI v1 pipeline) treats field registration as **homography estimation** — mapping pixel coordinates to a flat 2D field plane. The field has moved toward **full 3D camera calibration**: estimating camera position, rotation, and focal length. This enables offside detection using depth, virtual stadium overlays, and correct perspective for player height.

### Pan-Camera Specifics
A pan camera that follows the ball is the **hardest calibration case** — the homography changes every frame and must be recomputed continuously.

```
Fixed camera          Pan camera (ours)
────────────          ─────────────────
Calibrate once        Recalibrate every frame
Static background     Background shifts constantly
Lines always visible  Lines drift in/out of frame
```

**The right pattern:** run the keypoint model every N frames, compute a new homography, and smooth it with exponential averaging (α=0.8 as in v1). The smoothing prevents jitter from noisy keypoint detections. This must be fast — the keypoint model needs TensorRT export to keep up.

**Kalman smoothing before computing speed/distance is non-negotiable** — a few pixels of reprojection error translates to 0.5–2m of positional error, and that error gets amplified when you take the derivative to compute speed.

### Calibration Methods (Ranked by Robustness for Soccer)

1. **Line/keypoint-based registration** — detect pitch line intersections (known geometry), solve homography from correspondences. Most robust because pitch markings are standardized.

2. **Learned keypoint detection + fit** — model segments lines/keypoints, then fit camera parameters. The SoccerNet calibration baseline and Roboflow Sports minimap approach.

3. **Optimization-based (TVCalib, PnLCalib)** — gradient descent or PnL refinement over pitch segments.

### TVCalib (arxiv 2207.11709, WACV 2023)
Reframes field registration as true camera calibration (position, rotation, focal length) rather than homography. Performs 3D registration via gradient descent on reprojection error — no annotated homography matrices as training targets. The calibration module is entirely optimization-based; only the field line segmentation sub-module requires training data. Note: TVCalib's absolute accuracy is not state-of-the-art; the paradigm framing is the contribution.

Open-source: `MM4SPA/tvcalib`

### 57-Keypoint HRNetV2-w48 Pipeline (SoccerNet Camera Calibration 2023 winner)
Defines 57 landmarks: 30 line-line intersections, 6 line-conic intersections, 8 conic tangent points, 13 axis/circle/quarter-turn markers. Uses HRNetV2-w48 backbone for high-resolution feature preservation. Achieved **Acc@5 = 73.22%** vs baseline 13.54% and 2nd-place 52.95%. Large improvement over the 12-keypoint v1 approach.

Sources: winner blog (nikolasent.github.io), arxiv 2410.07401, arxiv 2309.06006.

### PnLCalib (arxiv 2404.08401v4) — "No Bells, Just Whistles"
Adds a non-linear PnL (Points and Lines) refinement module on top of keypoint-based initialization. Uses detected field lines as additional constraints. Ablation shows **+4.3% final score on SoccerNet-Calibration** and **+7.6% on WorldCup 2014** from the PnL module. Earlier foundational calibration work: Jiang et al. "Optimizing Through Learned Errors" (WACV'20, arXiv:1909.08034); Nie et al. "A Robust and Efficient Framework for Sports-Field Registration" (WACV'21, CVF Open Access).

Note: PnLCalib's headline absolute accuracy numbers were refuted 1-2 in adversarial verification — rely on the ablation deltas, not the headline numbers.

### Open-Source Starting Points
`SoccerNet/sn-calibration`, `MM4SPA/tvcalib`, Roboflow Sports keypoint pitch detection.

---

## 4. Object Detection

### YOLO Family Remains Dominant
Despite transformer-based challengers, YOLO-family detectors dominate production sports analytics. Both SoccerNet challenge winners (2024, 2025) used YOLO variants. YOLOv5m, YOLOv8, YOLO-X, and YOLOv11 are all in active production use. Fine-tuned YOLOv8 reaches ~0.90–0.97 mAP@0.5 for players; YOLOv9/v10/v11 give incremental small-object gains via improved feature pyramids. Ball detection lags at ~0.6–0.8 mAP@0.5 regardless of YOLO variant — this is a fundamental small-object problem, not a model version problem.

### RT-DETR (arxiv 2304.08069, CVPR 2024)
Transformer-based, NMS-free. RT-DETR-R50 achieves 53.1 mAP on COCO at 108 FPS on T4 TensorRT FP16. Competitive and handles occlusion slightly better, but heavier. The claim that RT-DETR is a clear replacement for YOLO was **refuted 0-3** in adversarial verification. RT-DETR is suitable for deployment but YOLO remains dominant in production sports pipelines.

---

## 5. Team Assignment and Player Identification

### Team Assignment (Reliable)
Crop each detected player, extract a color or deep embedding, cluster into two teams plus goalkeeper(s) and referee. The Roboflow Sports approach uses **SigLIP embeddings → UMAP → K-Means**, which is robust when kits are visually distinct and degrades when kit colors or lighting are similar. This step generally works well and is the most reliable form of player identity in a single-camera setup.

### Re-identification Fundamental Limit
Standard person Re-ID fails in soccer because teammates wear identical kits — appearance embeddings cannot distinguish between players on the same team. SoccerNet Re-ID baselines sit well below street-person Re-ID benchmarks. **Appearance Re-ID alone cannot identify individuals on a team.** Robust individual identity requires combining: tracking (short-term continuity) + jersey number (tracklet-level OCR aggregation) + team assignment + spatial constraints.

### What Is Realistic on a Single Fixed Wide Camera
- Reliable team assignment and persistent tracking IDs across the match
- Partial jersey-number ID: good for near-side players with large crops, frequently "unknown" for the far touchline
- Full per-player season-long attribution requires either higher-resolution/zoomed capture, or a player-worn tag (as Trace does commercially)

---

## 6. Event Detection and Tactical Metrics

### Two Paradigms for Event Detection

**(a) Action spotting on raw video** — The SoccerNet Action Spotting challenge spots 17 match event classes (goal, card, substitution, corner, foul, etc.) with precise temporal location. SoccerNet Ball Action Spotting (2023+) targets finer ball-centric actions (pass, drive, header, shot). Methods evolved from NetVLAD++ (arXiv:2104.06779) → temporal transformers (T-DEED, E2E-Spot), with Average-mAP climbing from ~50s to ~70s over 2021–2024. Best for discrete events: goals, cards, set pieces.

**(b) Events derived from tracking geometry** — Assign the ball to the nearest player meeting proximity + velocity criteria; possession changes define passes; same-team transfer = completed pass; trajectory toward goal = shot. This is interpretable and the better route for continuous possession/passing stats. No additional model required beyond the tracking pipeline.

**Production systems fuse both:** action spotting for discrete match events, tracking-derived logic for continuous possession and passing.

### Tactical Metrics Available from Calibrated Tracking

| Metric | Source | Requirements |
|--------|--------|-------------|
| Possession % / territory | Tracking-derived | Ball tracking + player positions |
| Heatmaps | Tracking-derived | Calibrated player positions |
| Distance covered / sprint counts | Tracking-derived | Kalman-smoothed positions + homography |
| Pitch control | Spearman 2017/2018 | Player positions + velocities |
| Space value / EPV | Fernández & Bornn 2018/2019 | Positions + possession value model |
| xT (Expected Threat) | Karun Singh 2019 | Grid + ball location over possessions |
| VAEP | Decroos et al., KDD 2019 (arXiv:1802.07127) | Event stream |
| xG | StatsBomb open model | Accurate shot location + context |
| Formation detection | Bialkowski et al. 2014 | Average positions + role assignment |
| Pass networks | Tracking-derived | Player ID success required |
| Pressing (PPDA) | Industry standard | Events / tracking |

**Realistically computable from a single amateur wide camera:** possession %, territory/heatmaps, distance covered, sprint counts, team shape/formation, pitch-control surfaces, pass networks (where player ID succeeds). **Degrades or fails:** precise xG (needs accurate shot location + context), packing/line-breaking (needs precise positions across the whole pitch), anything off-ball that leaves frame. Every metric inherits noise from upstream tracking + homography.

---

## 7. Open-Source Frameworks and Ecosystem

| Tool | License | What it provides |
|------|---------|-----------------|
| **Roboflow `sports`** (github.com/roboflow/sports) | Apache-2.0 | Soccer reference scaffold: detection, SigLIP+UMAP+KMeans team clustering, pitch keypoint detection, homography/radar view, ball tracking. **Best place to fork and start.** |
| **Roboflow `supervision`** (github.com/roboflow/supervision) | MIT | ByteTrack integration, annotators, zone detection, polygon zones for possession tracking |
| **`boxmot`** (github.com/mikel-brostrom/boxmot) | AGPL-3.0 | Pluggable MOT trackers: ByteTrack, BoT-SORT, OC-SORT, StrongSORT, into any YOLO detector |
| **`SoccerNet/sn-gamestate`** | MIT | Full GSR reference pipeline matching challenge winner architecture |
| **`SoccerNet/sn-calibration`** | MIT | 57-keypoint camera calibration baseline, pretrained HRNetV2-w48 weights |
| **`MM4SPA/tvcalib`** | MIT | TVCalib 3D camera calibration via optimization |
| **StatsBomb Open Data** (github.com/statsbomb/open-data) | Custom (free) | Event + 360° freeze-frame data for validating analytics logic |

---

## 8. Datasets and Benchmarks

| Dataset | Content | License | Relevance |
|---------|---------|---------|-----------|
| **SoccerNet v1–v3** (arxiv 1804.04527, 2011.13367, Nature Sci Data 2022) | 500 broadcast matches, action spotting, tracking, calibration, re-ID | **Research/non-commercial only** | Primary benchmark for all components |
| **SoccerTrack** (CVSports 2022; code: AtomScott/SportsLabKit) | Fish-eye wide-view + drone footage, GNSS ground truth | v2: **CC BY 4.0** (commercial-friendly) | Closest match to your fixed wide camera setup |
| **SportsMOT** (arxiv 2304.05170) | Basketball, volleyball, football tracking | Public | Tracker evaluation |
| **Roboflow Universe** soccer/football sets | YOLO-ready, varied | Often permissive | Fine-tuning starting point |
| **WorldCup 2014** | Broadcast soccer | Standard | Calibration benchmark |
| **DFL Bundesliga Data Shootout** (Kaggle 2022) | Event detection | Competition | Event detection |
| **StatsBomb Open Data** | Event + 360° freeze-frames | Free | Analytics validation |

**Licensing note:** SoccerNet video is gated behind a research/non-commercial access agreement. For any commercial product, train on your own footage plus permissively licensed Roboflow/SoccerTrack sets, and use SoccerNet only for benchmarking prototypes.

---

## 9. Production Pipeline Architecture

### Recommended Architecture for 90-Minute Match Processing (Batch on Rosie)

```
Video file
    → GPU decode (NVDEC / PyAV)
    → SAHI tiling for tiny-object detection (ball, far-side players)
    → Detection worker: YOLOv11 TensorRT FP16 (~2–5x speedup vs PyTorch)
      [Run at 5–15 fps for player positions]
    → Ball tracking worker: TrackNet temporal heatmap + Kalman interpolation
      [Run at full frame rate]
    → Tracker: ByteTrack → BoT-SORT/Deep-EIoU (stateful per match)
    → Team assignment: SigLIP → UMAP → K-Means (once per match)
    → Calibration: sn-calibration / TVCalib (once per camera setup)
    → Analytics aggregator: Kalman-smoothed speed/distance, possession, zones, formations
    → Output: persisted intermediate artifacts (detections, tracks, homography)
             + JSON event stream + annotated video
```

**Artifact persistence is critical:** The CV stage (detection + tracking) is the expensive part. Store all intermediate outputs (per-frame detections, tracklets, homography matrices) so analytics can be recomputed for new metrics or bug fixes without re-running GPU inference.

**Adaptive frame sampling:** Don't run every model at full frame rate. Detection/tracking at 5–15 fps is sufficient for player position analytics. The ball model needs higher cadence. Action spotting runs on its own schedule. This alone can cut compute cost 3-5x.

### Cost Estimates (Batch Processing, Single Match)
Rough order of magnitude for a TensorRT detection+tracking stack:
- ~1–3 GPU-hours per 90-minute match on T4/A10G at ~real-time throughput
- ~**$0.50–$3 of GPU compute** (plus storage/egress) at spot pricing
- Adding heavier models (TrackNet ball, action spotting, re-ID) raises this

GPU options: AWS g4dn.xlarge (T4, ~$0.526/hr), g5.xlarge (A10G, ~$1.006/hr); spot cuts ~60–70%. Serverless GPU (Modal, RunPod, Replicate) bills per-second and scales to zero — ideal for "match uploaded" workloads that aren't continuous.

On Rosie (SLURM): one SLURM array job per match, fanned out by time segment for parallelism. Make jobs idempotent and resumable (checkpoint per segment).

### NVIDIA DeepStream + Triton
Production-grade GPU inference serving with native Triton integration (confirmed 3-0). The enterprise path for many concurrent live streams. Not necessary for batch match processing on Rosie — TensorRT export of YOLO achieves comparable throughput without the serving overhead. Note: DeepStream does not support YOLO out-of-the-box (refuted 0-3) — requires custom plugin work.

### MLOps and Data Flywheel
- **Job orchestration:** Celery+Redis / SQS, or Modal/RunPod native queues. One match = one job, fanned out by time segment.
- **Model versioning:** MLflow or W&B for models; DVC or Roboflow for datasets.
- **Evaluation pipeline:** Hold-out labeled set; track HOTA/IDF1/mAP per model version; regression-test before deploy.
- **Automated QA gates:** Per-job: % frames with ball detected, number of ID switches, calibration reprojection error → auto-flag low-quality outputs for human review instead of shipping bad stats silently.
- **Active learning loop:** Route low-confidence detections and broken tracklets to annotation (CVAT / Label Studio / Roboflow) → retrain. This is how you climb past the prototype ceiling on your own footage distribution.

---

## 10. Competitive Landscape

Single-camera fixed CV → stats is commercially proven. Every serious player mitigates the resolution ceiling somehow.

| Company | Approach | Resolution Mitigation |
|---------|----------|----------------------|
| **Veo** | Fixed dual-lens 4K; AI auto-follows ball; analytics/heatmaps | Wide multi-lens capture |
| **Pixellot** | AI panoramic stitched unit; auto-production + stats; powers NFHS Network | Wide multi-lens stitching |
| **Trace** | Autonomous camera + player-worn GPS tag; per-player clips + stats | Worn tag for guaranteed individual ID |
| **Spiideo** | Cloud multi-cam + Perform AI analytics | Multi-cam |
| **SkillCorner** | Broadcast-video CV tracking (no in-stadium cameras) → physical + tactical data | High-quality broadcast feeds |
| **Hudl** | Dominant; acquired Wyscout + StatsBomb (2024) | Full data platform |
| **Stats Perform (Opta)** | Leading event data; "Opta Vision" fuses event + optical tracking | In-stadium multi-camera |
| **Genius Sports / Second Spectrum** | Acquired 2021; multi-camera optical tracking for major leagues | Multi-camera in-stadium |

**Takeaway:** Plan a resolution-ceiling mitigation. Options: higher-res camera, multi-lens capture (Veo/Pixellot approach), or optional player-worn tag (Trace approach) for guaranteed per-player stats when individual ID matters.

---

## 11. What Was Killed by Adversarial Verification

Claims that sounded plausible but did not survive 3-agent adversarial review:

- **"RT-DETR outperforms YOLO for sports production"** — 0-3 refuted. YOLO still dominates.
- **"YOLO+ByteTrack achieves high precision/recall for soccer"** — 0-3 refuted. Unqualified; occlusion and re-entry are known failure modes.
- **"Deep HM-SORT achieves 80.1 HOTA on SportsMOT"** — 0-3 refuted. Published numbers didn't hold under cross-verification.
- **"PnLCalib achieves 80.6% JaC5 accuracy"** — 1-2 refuted. Headline numbers suspect; ablation deltas are reliable.
- **"TVCalib achieves superior accuracy vs state-of-the-art"** — 0-3 refuted. Paradigm contribution is real; accuracy claim is not.
- **"DeepStream supports YOLO out-of-the-box"** — 0-3 refuted. Requires custom plugin work.
- **"Ball detection is the primary remaining challenge"** (as a singular claim) — 0-3 refuted due to insufficient evidence for primacy. The substance is correct but the wording overstated.

---

## 12. Open Questions

1. **Ball-tracking accuracy on amateur footage** has no clean published benchmark — must be measured on your own data. The dominant project risk.
2. **Calibration generalization:** Whether pretrained sn-calibration models (broadcast TV angles) transfer to a fixed D3 sideline camera is empirical — test zero-shot first.
3. **Deep-EIoU vs BoT-SORT:** Deep-EIoU wins the SoccerNet 2025 benchmark; BoT-SORT is more broadly established in production. Both warrant evaluation on local footage.
4. **Per-player ID coverage** on a single wide camera will be partial — quantify the legible fraction early to set expectations for the coaching staff.
5. **Resolution ceiling:** Current camera setup may be the binding constraint for far-side players and ball tracking regardless of model improvements.

---

## Key Takeaways for Rosco AI v2

1. **Replace SAM2 with ByteTrack → BoT-SORT or Deep-EIoU** (via `boxmot`) — removes the largest scalability bottleneck; appearance re-ID is essential because the pan camera constantly cuts players out of frame.
2. **Player re-ID is the hardest problem, not ball tracking** — camera follows the ball so detection is easy; players leaving/re-entering due to pan is the core challenge.
3. **Keep YOLO + Kalman for ball** — TrackNet is overkill for a pan camera; skip it unless ball detection is observed to fail in practice.
4. **Add SAHI tiling** for far-side players who are small pixels when the camera is zoomed out.
5. **Upgrade team clustering to SigLIP + UMAP + K-Means** (Roboflow Sports pattern) — more robust than raw color.
6. **Expand 12 → 57 keypoints** using pretrained `sn-calibration` HRNetV2-w48 — run every N frames, smooth with α=0.8 (already doing this in v1).
7. **Export keypoint model + YOLO to TensorRT FP16** — per-frame homography recomputation requires fast inference; 2–5x speedup is critical.
8. **Kalman-smooth all positions before computing speed/distance** — non-negotiable on a pan camera.
9. **Persist intermediate artifacts** (detections, tracks, homography) — decouple expensive CV from cheap analytics recompute.
10. **Restructure as SLURM job array** with per-stage workers, adaptive frame sampling, and automated QA gates.
11. **Skip jersey number re-ID** — same conclusion as fixed camera; OSNet appearance embeddings + team assignment are sufficient.
