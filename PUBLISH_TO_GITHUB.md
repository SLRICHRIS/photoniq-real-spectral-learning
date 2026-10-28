# Publish PhotonIQ on GitHub

**GitHub account:** `SLRICHRIS`  
**Recommended repository:** `photoniq-real-spectral-learning`

The linked ChatGPT GitHub connection can write files to existing repositories but cannot create a new repository. Do not put PhotonIQ in an unrelated existing repository.

## Option A: finish from ChatGPT

1. Create a **new empty repository** at <https://github.com/new> under account `SLRICHRIS`.
2. Name it `photoniq-real-spectral-learning`. Select public or private as you prefer. **Do not** pre-create a README, license, or `.gitignore` (these already exist).
3. Return to ChatGPT and say: `I created SLRICHRIS/photoniq-real-spectral-learning; push PhotonIQ.`

The connected GitHub tool can then publish the verified source files to that new repository.

## Option B: use Git from your own computer

Unzip the project, open a terminal in the directory containing `README.md`, and run:

```bash
git init -b main
git add .
git commit -m "Initial PhotonIQ research release"
git remote add origin https://github.com/SLRICHRIS/photoniq-real-spectral-learning.git
git push -u origin main
```

Create the empty GitHub repository before pushing. Configure your own Git identity and GitHub sign-in if prompted. Do not put authentication tokens into project files.

## Option C: preserve the prepared Git commit

If you downloaded the companion `PhotonIQ_Git_Repository.bundle`, after creating the empty GitHub repository run:

```bash
git clone PhotonIQ_Git_Repository.bundle photoniq-real-spectral-learning
cd photoniq-real-spectral-learning
git remote remove origin
git remote add origin https://github.com/SLRICHRIS/photoniq-real-spectral-learning.git
git push -u origin main
```

The bundle contains the source history, not third-party raw data, credentials, or model artifacts.

## Verification before publishing

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_notebook.py
```

The original Mendeley experimental dataset is **not included** and must be downloaded separately under its own license. No verified real-world model accuracy is claimed.
