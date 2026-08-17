#!/usr/bin/env bash
# Pushes this entire project to:
#   https://github.com/pillatejanagasai/semiconductor-image-restoration
#
# Run this from INSIDE the semiconductor-restoration folder, on YOUR
# computer (not in any sandboxed/AI environment) — it needs your own git
# credentials and network access.
#
# Usage:
#   chmod +x push_to_github.sh
#   ./push_to_github.sh
#
# If you'd rather run the commands yourself one at a time instead of via
# this script, see the numbered steps in the echo statements below — each
# echo shows exactly the command that runs next.

set -e  # stop immediately if any command fails, instead of pushing a partial/broken state

REPO_URL="https://github.com/pillatejanagasai/semiconductor-image-restoration.git"
BRANCH="main"

echo "Step 1/6 — Checking this is a git repo (initializing if not)..."
if [ ! -d ".git" ]; then
  git init
fi

echo "Step 2/6 — Checking git identity is configured..."
if [ -z "$(git config user.email)" ]; then
  echo "  git user.email is not set."
  read -p "  Enter the email to commit as: " GIT_EMAIL
  git config user.email "$GIT_EMAIL"
fi
if [ -z "$(git config user.name)" ]; then
  echo "  git user.name is not set."
  read -p "  Enter the name to commit as: " GIT_NAME
  git config user.name "$GIT_NAME"
fi

echo "Step 3/6 — Staging all files (respecting .gitignore)..."
git add .

if [ ! -f "outputs/checkpoints/best.pt" ]; then
  echo ""
  echo "  NOTE: outputs/checkpoints/best.pt does not exist yet — this push"
  echo "  will contain code only, no trained model. That's fine for early"
  echo "  development pushes, but is NOT a valid final submission (KLA's"
  echo "  evaluation script needs an actual checkpoint to load — see"
  echo "  submission/README.md, 'Before you submit')."
  echo ""
else
  echo "  Found outputs/checkpoints/best.pt — .gitignore excludes it by"
  echo "  default. If this push IS your final submission, stop here and"
  echo "  run 'git add -f outputs/checkpoints/best.pt' first (see"
  echo "  submission/README.md, step 6, for file-size / Git LFS notes),"
  echo "  then re-run this script."
  echo ""
fi

echo "Step 4/6 — Committing..."
if git diff --cached --quiet; then
  echo "  Nothing new to commit (working tree already matches last commit) — skipping."
else
  git commit -m "AI-based restoration pipeline for KLA AI Hackathon (speckle+Gaussian denoise, fixed 2x SR)"
fi

echo "Step 5/6 — Setting the remote..."
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

echo "Step 6/6 — Renaming local branch to '$BRANCH' and pushing..."
git branch -M "$BRANCH"
git push -u origin "$BRANCH"

echo ""
echo "Done. Your code is now at: https://github.com/pillatejanagasai/semiconductor-image-restoration"
echo ""
echo "NOTE: if the push above asked for a password and rejected it, GitHub no"
echo "longer accepts account passwords over HTTPS — you need a Personal"
echo "Access Token instead. See the troubleshooting note in GITHUB_PUSH_INSTRUCTIONS.md."
