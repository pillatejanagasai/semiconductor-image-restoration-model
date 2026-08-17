# Pushing this project to GitHub

Target repo: https://github.com/pillatejanagasai/semiconductor-image-restoration-model

## Prerequisites (one-time, if not already done)

1. Install git if you don't have it: https://git-scm.com/downloads
2. Make sure you can access the repo in your browser and are logged in
   to the `pillatejanagasai` GitHub account.

## Option A — run the script (recommended)

```bash
cd semiconductor-restoration
chmod +x push_to_github.sh
./push_to_github.sh
```

The script initializes git (if needed), stages everything (respecting
`.gitignore`, so `outputs/`, `data/` (KLA's real + any synthetic training data), checkpoints, and venvs are
never committed), commits, points the remote at your repo, and pushes.
It will ask for your name/email the first time only, if git doesn't
already have them configured on your machine.

## Option B — run the commands yourself

```bash
cd semiconductor-restoration
git init
git add .
git commit -m "AI-based restoration pipeline for KLA AI Hackathon"
git branch -M main
git remote add origin https://github.com/pillatejanagasai/semiconductor-image-restoration.git
git push -u origin main
```

## Authentication — the most common snag

GitHub no longer accepts your account password over HTTPS `git push`. If
the push prompts for a password and then fails with something like
`Support for password authentication was removed`, you need a **Personal
Access Token (PAT)** instead:

1. Go to https://github.com/settings/tokens → "Generate new token"
   (classic is simplest) → tick the `repo` scope → generate.
2. Copy the token (you only see it once).
3. When `git push` prompts for a password, paste the token instead of
   your account password (the username field is still your GitHub
   username).
4. To avoid re-entering it every time, either use a credential manager
   (`git config --global credential.helper manager` on Windows/Mac) or
   switch the remote to SSH instead (see below).

### Alternative: SSH instead of HTTPS (no token prompts at all)

If you already have an SSH key set up with GitHub:

```bash
git remote set-url origin git@github.com:pillatejanagasai/semiconductor-image-restoration.git
git push -u origin main
```

If you don't have an SSH key yet, GitHub's guide is here:
https://docs.github.com/en/authentication/connecting-to-github-with-ssh

## If the repo on GitHub already has files (e.g. an auto-created README)

`git push` will be rejected because the remote has commits your local
repo doesn't. Pull and merge first, then push:

```bash
git pull origin main --allow-unrelated-histories
# resolve any conflicts if git reports them, then:
git push -u origin main
```

## What does NOT get pushed (by design)

`.gitignore` excludes `outputs/` (checkpoints, TensorBoard logs, eval
reports), `data/` (KLA's real released dataset plus any regenerable synthetic training pairs -- large, re-downloadable/regenerable, not meant to be versioned), any
`.pt`/`.onnx` files, and `submission/environment_freeze.txt` /
`submission/test_outputs/` — these are all either large/regenerable or
meant to be freshly produced right before you submit, not committed as
stale copies. Everything else (code, configs, docs) is pushed.
