# 🚀 Deployment Guide — GitHub + Hugging Face Spaces

Follow these steps exactly to go from zero to a live shareable URL.
**Total time: ~20 minutes.**

---

## Prerequisites

- [ ] GitHub account (free): [github.com/signup](https://github.com/signup)
- [ ] Hugging Face account (free): [huggingface.co/join](https://huggingface.co/join)
- [ ] Git installed locally: `git --version`
- [ ] Python 3.10+ (for local testing only)

---

## Step 1 — Get a Hugging Face Token

1. Log in to [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens**
3. Click **New token** → name it `policyforge` → Role: **Write**
4. Copy the token (starts with `hf_…`) — **save it, you'll need it twice**

---

## Step 2 — Push to GitHub

```bash
# Clone this repo or create a new one
git init policyforge-ai
cd policyforge-ai

# Copy all project files into this directory
# (app.py, rag_pipeline.py, requirements.txt, data/, docs/, README.md)

git add .
git commit -m "feat: initial PolicyForge AI RAG system"

# Create repo on GitHub (via browser or gh CLI)
# Then:
git remote add origin https://github.com/YOUR_USERNAME/policyforge-ai.git
git push -u origin main
```

Your GitHub repo is now live at:
`https://github.com/YOUR_USERNAME/policyforge-ai`

---

## Step 3 — Create the Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Owner**: your HF username
   - **Space name**: `policyforge-ai`
   - **License**: MIT
   - **SDK**: **Gradio** ← important
   - **SDK version**: 4.44.0
   - **Hardware**: CPU Basic (free) ← free tier
   - **Visibility**: Public (so you can share the link)
3. Click **Create Space**

---

## Step 4 — Add Your HF Token as a Secret

This allows the Space to call the HF Inference API for free generation.

1. In your Space, go to **Settings → Variables and Secrets**
2. Click **New secret**
3. Name: `HF_TOKEN`
4. Value: paste your `hf_…` token
5. Click **Save**

> ⚠️ Use "Secret" not "Variable" — secrets are not exposed in logs.

---

## Step 5 — Push Code to the Space

HF Spaces uses a Git repo. Push your code directly:

```bash
# From inside your policyforge-ai directory:

# Add the Space as a remote
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/policyforge-ai

# Push to the Space (use your HF token as the password when prompted)
git push space main
```

When prompted for credentials:
- Username: your HF username
- Password: your `hf_…` token (not your HF account password)

> 💡 To avoid typing the token every time:
> ```bash
> git remote set-url space https://YOUR_HF_USERNAME:hf_YOUR_TOKEN@huggingface.co/spaces/YOUR_HF_USERNAME/policyforge-ai
> ```

---

## Step 6 — Wait for Build (~3–5 minutes)

1. Go to your Space: `https://huggingface.co/spaces/YOUR_HF_USERNAME/policyforge-ai`
2. Click the **Logs** tab to watch the build
3. You'll see: pip installing → embedding model downloading → knowledge base loading → `Running on public URL`
4. The **App** tab will show your live UI

**Your shareable link:**
```
https://YOUR_HF_USERNAME-policyforge-ai.hf.space
```

Share this with anyone — no login required to use it.

---

## Step 7 — Update README for GitHub Portfolio

Edit your GitHub `README.md` to replace `YOUR_USERNAME` with your actual username:

```bash
# In README.md, update these lines:
# Live Demo: https://huggingface.co/spaces/YOUR_USERNAME/policyforge-ai
# git clone https://github.com/YOUR_USERNAME/policyforge-ai

git add README.md
git commit -m "docs: add live demo link"
git push origin main
```

---

## Future Updates

To update the live demo after code changes:

```bash
git add .
git commit -m "your change message"

# Push to GitHub (portfolio)
git push origin main

# Push to HF Spaces (live demo)
git push space main
```

Both remotes update independently — push to both to keep them in sync.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Build fails with `ModuleNotFoundError` | Check `requirements.txt` has all packages |
| `HF_TOKEN` not found | Add it as a Space **Secret** (not Variable) |
| Generation returns error | Free Inference API has rate limits — wait 1 min and retry |
| Space shows "Building" forever | Check Logs tab for Python errors |
| 429 Too Many Requests | HF free tier has ~1000 requests/day; sufficient for demos |
| Embedding model slow first time | Normal — MiniLM downloads once (~90 MB), then cached |

---

## Free Tier Limits

| Resource | Free Tier | Notes |
|----------|-----------|-------|
| CPU | 2 vCPU | Enough for retrieval + API calls |
| RAM | 16 GB | Plenty (embeddings use ~500 MB) |
| Storage | In-memory only | Resets on restart (acceptable for demo) |
| Inference API calls | ~1000/day | More than enough for portfolio demos |
| Space uptime | Always-on (public spaces) | Free public spaces stay up |
| Bandwidth | Unlimited | No cost for traffic |
