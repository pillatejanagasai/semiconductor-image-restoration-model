# KLA Submission Package

Maps exactly to the four required components on slide 17 of
`Problem_Statement_01_KLA.pdf`.

**Data format note:** confirmed directly from real uploaded sample files
(not assumed from the deck) — all data is `.npy` (float32 NumPy arrays),
in `NoisyLR/`+`GT/` folder pairs. See the main `README.md`, "Data format"
section, for the full confirmed details (value ranges, exact 2x shape
ratio, etc.).

| # | KLA requirement | Where it is in this repo |
|---|---|---|
| 1 | Evaluation Script — standalone Python script (non-notebook), accepts test-images dir + output dir, loads model, runs inference, writes outputs, runs with zero manual edits | `submission/infer_test_set.py` — reads/writes `.npy`, see usage below |
| 2 | Training Script — reproduces training of the submitted model | `scripts/train.py` (+ `configs/`) — see usage below |
| 3 | Denoised Test Outputs — model output on KLA's test set | Generate by running against `Test_NoisyLR/NoisyLR/` once you have it (see below) |
| 4 | Environment Specification — complete `pip freeze` output | `submission/environment_freeze.txt` — regenerate before final submission (see below) |

**None of these four can be fully completed by pushing code alone — #3 is
literally the output of running inference, and #1 only works during
KLA's benchmarking if a trained checkpoint is actually included and
reachable.** See "Before you submit" below for the exact order.

## Before you submit — required order of operations

**Do not just push this repo and submit the link.** Do this in order:

1. **Get training data.** You now have KLA's real released training set
   (`NoisyLR/`+`GT/` `.npy` pairs — point
   `configs/data/kla_data.yaml -> data_roots` at it directly, no
   conversion needed) — optionally supplement it with your own synthetic
   pairs via `src/data/degradation.py` for wider coverage.
2. **Actually train a model** (section 2 below) until you have a
   checkpoint at `outputs/checkpoints/best.pt` you're satisfied with.
   Run the ablations in `docs/experiment_log_template.md` — this is
   where your real score comes from, not from the code structure alone.
3. **Run the evaluation script against KLA's real test set** once
   released (section 3 below) to produce the actual denoised outputs.
4. **Test that the evaluation script runs cleanly, unmodified, from a
   fresh checkout** — clone your own repo to a clean folder and run the
   exact command KLA will run. If it needs any manual tweak to work,
   fix that before submitting; KLA states it will be used "as-is."
5. **Freeze the environment** (section 4 below) from the environment you
   actually used for the run in step 3.
6. **Include the trained checkpoint in what you submit.** `.gitignore`
   excludes `outputs/` by default (correct during development — you
   don't want dozens of experimental checkpoints bloating the repo), but
   your FINAL `best.pt` must be reachable when KLA's harness runs your
   eval script, since step 1 of the requirements ("loads the trained
   model") depends on it existing. Two ways to do this:
   - **Force-add it to git despite `.gitignore`:**
     `git add -f outputs/checkpoints/best.pt`, then commit and push as
     normal. **Check the file size first** (`ls -lh outputs/checkpoints/best.pt`)
     — GitHub blocks any single file over 100MB unless you use
     [Git LFS](https://git-lfs.com/). If your checkpoint is large, install
     Git LFS, run `git lfs track "*.pt"`, then add/commit/push as usual.
   - **Host it externally** (e.g. a release asset on your GitHub repo,
     Google Drive, HuggingFace) and have your submission notes/README
     point to it — but only do this if KLA's process allows a separate
     artifact; if their harness expects to `git clone` your repo and run
     the eval script immediately with zero manual steps, an external
     download the script doesn't automate itself would violate the "runs
     without manual edits" requirement. Committing the checkpoint
     directly (via Git LFS if needed) is the safer default.
7. **Confirm your actual submission mechanism with KLA/the hackathon
   portal.** The deck lists the four required components but doesn't
   specify in the pages I have whether they want a GitHub link, a zip
   upload, or something else — that detail isn't in
   `Problem_Statement_01_KLA.pdf` as shared, so check the hackathon's
   submission portal/instructions directly rather than assuming.

## 1. Evaluation script

```bash
python submission/infer_test_set.py \
    --test_dir /path/to/kla/test/images \
    --output_dir /path/to/write/restored/images
```

- Requires nothing beyond `--test_dir` and `--output_dir` — every other flag
  has a working default (`--checkpoint` defaults to
  `outputs/checkpoints/best.pt`).
- Writes one restored `.npy` (float32, matching KLA's real GT format) per
  input array, plus a `timing_report.json`
  in `--output_dir` recording script startup, model init, and
  inference+I/O time — matching KLA's own stated timing methodology
  (slide 15: startup + model init + reading inputs + inference + writing
  outputs, all counted).
- Reads the model architecture from the checkpoint's saved config, so it
  works unmodified regardless of which `model=` variant was trained.
- Optional `--compile` / `--tta` / `--batch_size` flags exist for
  speed/quality tuning — see the script's own `--help` and
  `docs/experiment_log_template.md` EXP-010/EXP-011 before enabling them
  for your final submission command (they're off by default for a
  reason — measure before you flip them on).

## 2. Training script

```bash
python scripts/train.py data=kla_data model=hybrid_restorer train=default
```

Reproduces training end-to-end: loads KLA's real `NoisyLR/`+`GT/` `.npy`
pairs directly (no conversion needed — `src/data/dataset.py` reads KLA's
native format as-is), trains `HybridRestorer` with the composite
PSNR/SSIM/LPIPS-targeted loss, and saves checkpoints to
`outputs/checkpoints/`.

Point `configs/data/kla_data.yaml -> data_roots` at the folder that
directly contains `NoisyLR/` and `GT/` (from KLA's release — watch for a
nested `train/train/` folder after unzipping, that's just an archive
artifact, point at the innermost folder). If you also generated
supplementary synthetic pairs via `src/data/degradation.py` (which writes
into the identical `NoisyLR/`+`GT/` `.npy` layout), add that folder as a
second entry in `data_roots` — both sources are read through the same
code path with zero special-casing.

## 3. Denoised test outputs

Once KLA releases the official test set (already confirmed to be
`Test_NoisyLR/NoisyLR/`, containing `.npy` files, no GT), run:

```bash
python submission/infer_test_set.py --test_dir Test_NoisyLR/NoisyLR --output_dir submission/test_outputs
```

and include the resulting `submission/test_outputs/` folder (restored
images + `timing_report.json`) in your final submission.

## 4. Environment specification

Regenerate this immediately before packaging your final submission, from
the exact environment you trained/evaluated in:

```bash
pip freeze > submission/environment_freeze.txt
```

A `submission/environment_freeze.txt` placeholder is not pre-generated
here since it must reflect the *actual* installed environment used for
your final run, not this repo's abstract `requirements.txt`.
