# Render Deployment Guide

Follow these steps to host your Gemini Object Detection backend on Render.com.

## 1. Push Code to GitHub
Render needs to connect to your source code repository. Make sure you've pushed your current files (including `requirements.txt` and the updated `main.py`) to your repository:
[https://github.com/krishSibin/GeminiDetecterBackend.git](https://github.com/krishSibin/GeminiDetecterBackend.git)

## 2. Create a Web Service on Render
1. Log in to [Render.com](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.

## 3. Configuration Settings
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python main.py`

## 4. Environment Variables (CRITICAL)
1. In the Render Dashboard for your service, go to the **Environment** tab.
2. Click **Add Environment Variable**.
3. Set **Key** to: `GEMINI_API_KEY`
4. Set **Value** to: `[Your Actual API Key from your .env file]`

## 5. Get Your URL
Once Render finishes building, it will provide you with a URL (e.g., `https://my-gemini-app.onrender.com`).

**Update your Flutter app** to use this new URL!
