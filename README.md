<div align="center">

<h2>Memorizon: Training World Models Beyond Their Context Window</h2>

<p>
  <a href="https://tingtingliao.github.io/"><b>Tingting Liao</b></a> &nbsp;·&nbsp;
  <b>Xuezhi Liang</b> &nbsp;·&nbsp;
  <a href="https://www.hao-li.com/Hao_Li/Hao_Li_-_about_me.html"><b>Hao Li</b></a> &nbsp;·&nbsp;
  <a href="https://guangyiliu.com/"><b>Guangyi Liu</b></a>
  <br>
  <sub>Institute of Foundation Models (IFM), MBZUAI</sub>
</p>

<div align="center">
  <a href=''><img src='https://img.shields.io/badge/arXiv-coming%20soon-red?logo=arxiv&logoColor=darkred&labelColor=white'></a>  &ensp;
  <a href='https://tingtingliao.github.io/memorizon/'><img src='https://img.shields.io/badge/project-page-green?logo=googlechrome&logoColor=green&labelColor=white'></a>  &ensp;
  <a href='https://huggingface.co/Luffuly/memorizon'><img src='https://img.shields.io/badge/HuggingFace-model-yellow?logo=huggingface&logoColor=yellow&labelColor=white'></a>  &ensp;
  <a href='LICENSE'><img src='https://img.shields.io/badge/license-Apache--2.0-blue?logo=C&logoColor=blue&labelColor=white'></a>  &ensp;
</div>

<br>

<img src="assets/media/teaser.webp" width="100%">

</div>

---

## 🔧 Installation

```bash
git clone https://github.com/TingtingLiao/memorizon.git && cd memorizon
conda create -n memorizon python=3.11 -y && conda activate memorizon
pip install -e .
```

> Inference needs a GPU with ≥ 40 GB memory and `ffmpeg` on `PATH`. Weights download on first use.

## 🚀 Quick Start

One image + a camera path → a long video. Drive the camera with a **pose trajectory** or **keyboard actions**:

```bash
# recorded trajectory: the example of the project page's comparison (60 s)
python scripts/generate.py --checkpoint Luffuly/memorizon --image assets/examples/residential_road.jpg \
    --trajectory assets/examples/residential_road_60s.npy --output out.mp4

# keyboard actions (one per 0.25 s)
python scripts/generate.py --checkpoint Luffuly/memorizon --image photo.jpg \
    --actions "w*16 l*24 w*16 l*24" --output out.mp4
```

```python
from memorizon import MemorizonPipeline, actions_to_c2w, save_video

pipe = MemorizonPipeline.from_pretrained("Luffuly/memorizon")
video = pipe("photo.jpg", actions_to_c2w("w*16 l*24 w*16 l*24"))
save_video(video, "out.mp4")                     # 864x480, 16 fps
```

### 🎮 Keyboard actions

| Key | Action | Key | Action |
|:---:|:---|:---:|:---|
| `w` `s` | forward / backward · 0.25 m | `j` `l` | turn left / right · 7.5° |
| `a` `d` | left / right · 0.25 m | `i` `k` | look up / down · 7.5° |
| `.` | stay | `*N` | repeat N times |

Keys combine in one step, e.g. `wl` = forward while turning right.

<details>
<summary><b>Prompts</b> — captioned automatically; write your own in the same format</summary>

<br>

Without `--prompt`, the first frame is captioned by Qwen3-VL-8B with the training
instruction ([`memorizon/caption.py`](memorizon/caption.py)). Custom prompts should
follow the same format — a perspective prefix, then one paragraph:

> First-person perspective — character not visible. A winding paved path curves
> beneath a canopy of vibrant pink cherry blossoms, their petals scattered across the
> ground. Lush green grass borders the path, flanked by a low stone wall and metal
> railing. …

</details>

<details>
<summary><b>Trajectory format</b></summary>

<br>

`[T, 4, 4]` camera-to-world poses, `T = 1 + 4 × seconds`, in the first frame's
coordinates (OpenCV: x right, y down, z forward), translations in metres / 4.
`assets/trajectories/` holds walks recorded in Unreal Engine scenes; `assets/examples/` pairs a photo with a 60 s walk.

</details>

## 🏋️ Training

```bash
# 1. train (from Wan2.2-TI2V-5B, or a Memorizon checkpoint via model.from_pretrained=...)
torchrun --nnodes 4 --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint $MASTER:29500 \
    -m memorizon.train configs/train_100s.yaml data.dataset_path=data/shards

# 2. distill to 4 steps (Self-Forcing + DMD)
torchrun ... -m memorizon.self_forcing configs/self_forcing_4step.yaml \
    model.generator_path=path/to/memorizon data.dataset_path=data/shards
```

<details>
<summary><b>Details</b></summary>

<br>

- **Data** — pre-encoded shards: Wan2.2 VAE latents, one camera pose per latent frame,
  first-frame encodes and umT5 caption embeddings. Layout: [`memorizon/data.py`](memorizon/data.py).
- **Released model** — 10k steps on 32 H200 (batch 32), 95 h of walks in 50 Unreal Engine scenes.
- **Longer spans** — set `data.m_max` (199 → 200 s, 399 → 400 s); same cost per step.

</details>

## 📖 Citation

```bibtex
@article{memorizon2026,
  title   = {Memorizon: Training World Models Beyond Their Context Window},
  author  = {Tingting Liao, Xuezhi Liang, Hao Li, Guangyi Liu},
  year    = {2026},
  note    = {arXiv preprint}
}
```
