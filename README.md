# AI Mood Mirror

Simple Streamlit app that detects facial emotions using DeepFace, fer, or a Mediapipe-based fallback.

Deployment (Streamlit Cloud)
1. Create a GitHub repository and push this project to it.
2. Ensure `requirements.txt` is in the repository root (this project includes one). Pin exact versions with
   `python -m pip freeze > requirements.txt` if you want reproducible installs.
3. Go to https://share.streamlit.io/ and log in with your GitHub account.
4. Click **New app**, choose the repository and branch, and set the main file to `app.py`.
5. (Optional but recommended) In the app settings -> Advanced -> Environment variables, set:
   - `DISABLE_DEEPFACE=1` to prevent DeepFace/TensorFlow model downloads and force the app to use lighter backends.
6. Deploy and open the app URL provided by Streamlit Cloud.

Notes and tips
- DeepFace uses TensorFlow and will download model weights on first run; this can be slow and may exceed free cloud resource limits.
- If you see performance issues on Streamlit Cloud, set `DISABLE_DEEPFACE=1` and use the `fer` or `mediapipe` fallback.
- If you need camera access on Streamlit Cloud, note that the cloud-hosted app cannot access your local webcam — the app's camera features work when running locally. For remote demos, use a video upload or stream from a URL instead.

Local testing
- Run locally:
```
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Contact
If you want me to set up the repo structure (README, .gitignore, CI) and prepare a production-ready requirements file, tell me and I'll do it.
