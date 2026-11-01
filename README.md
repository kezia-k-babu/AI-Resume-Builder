# AI Resume Builder (Streamlit + CrewAI + Gemini)

Three AI agents (Extractor → Writer → Formatter) turn rough notes into a polished
Markdown resume. Visitors need no API key; only you do.

## 1. Get a free Gemini API key
https://aistudio.google.com/app/apikey → **Create API Key**.

## 2. Run locally
Use Python 3.10–3.13.

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your real key
streamlit run app.py
```

Never commit `.streamlit/secrets.toml` (it is in `.gitignore`).

## 3. Deploy free on Streamlit Community Cloud
1. Push to GitHub (`git status` must NOT list `.streamlit/secrets.toml`).
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, file `app.py`.
3. **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-real-key-here"
   ```
4. Deploy. Every `git push` to `main` redeploys automatically.

## Other hosts (Render, Railway, Hugging Face Spaces)
`app.py` already reads `GEMINI_API_KEY` (and optional `GEMINI_MODEL`) from environment
variables when no Streamlit secret exists, so no code change is needed. Start command:
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Changing the model
Set `GEMINI_MODEL` in secrets or env (e.g. `gemini-2.5-flash`). Check current names at
https://ai.google.dev/gemini-api/docs/models

## Rate limiting
30-second cooldown and 5 generations per browser session. This is session-based and
resets in a new incognito window; add IP-based limiting if you get real public traffic.

## Limitations
- Each generation takes ~20-40 seconds (three sequential LLM calls).
- Gemini's free tier has request caps; failures show an error, never a fake result.
- Nothing is saved server-side; download the resume before leaving the page.
