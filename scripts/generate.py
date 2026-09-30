"""Generate a video from one image and a camera trajectory.

    python scripts/generate.py --checkpoint CKPT --image photo.jpg \
        --trajectory assets/trajectories/walk_00.npy --output out.mp4 --seconds 60
    python scripts/generate.py --checkpoint CKPT --image photo.jpg \
        --actions "w*16 wl*8 w*16 j*12" --output out.mp4

Without --prompt, the image is captioned automatically in the format the model
was trained on. The trajectory is a ``[T, 4, 4]`` numpy array of camera-to-world poses, one per
latent frame (4 per second of video), expressed in the first frame's coordinates
(OpenCV axes: x right, y down, z forward) with translations in metres / 4.
Instead of a trajectory, --actions gives one keyboard action per latent frame
(see memorizon/actions.py) and the trajectory is built from it.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memorizon import DEFAULT_INTRINSICS, NEGATIVE_PROMPT, MemorizonPipeline, actions_to_c2w, save_video


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Memorizon checkpoint (local dir or HF repo)")
    ap.add_argument("--image", required=True, help="first frame")
    camera = ap.add_mutually_exclusive_group(required=True)
    camera.add_argument("--trajectory", help=".npy of [T, 4, 4] camera-to-world poses")
    camera.add_argument("--actions", help='keyboard actions, one per latent frame, e.g. "w*16 wl*8 j*12"')
    ap.add_argument("--prompt", default=None,
                    help="scene description in the training caption format (default: caption the image)")
    ap.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    ap.add_argument("--output", default="output.mp4")
    ap.add_argument("--seconds", type=int, default=None, help="length (default: the whole trajectory)")
    ap.add_argument("--steps", type=int, default=None, help="denoising steps per chunk (default: the checkpoint's)")
    ap.add_argument("--guidance", type=float, default=None, help="CFG scale (default: the checkpoint's)")
    ap.add_argument("--memory", type=int, default=6, help="memory frames retrieved per chunk")
    ap.add_argument("--intrinsics", type=float, nargs=4, default=DEFAULT_INTRINSICS,
                    metavar=("FX", "FY", "CX", "CY"), help="pixels of the 864x480 frame")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    c2w = np.load(a.trajectory) if a.trajectory else actions_to_c2w(a.actions)
    chunks = (len(c2w) - 1) // 4 if a.seconds is None else a.seconds
    pipe = MemorizonPipeline.from_pretrained(a.checkpoint)
    video = pipe(a.image, c2w, a.prompt, intrinsics=a.intrinsics, num_chunks=chunks,
                 negative_prompt=a.negative_prompt, guidance_scale=a.guidance, num_steps=a.steps,
                 memory_size=a.memory, seed=a.seed)
    save_video(video, a.output)
    print(f"wrote {a.output} ({len(video) / 16:.0f} s)")


if __name__ == "__main__":
    main()
