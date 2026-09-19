# KrishiKavach

An explainable crop disease and pest prediction prototype for farmer-uploaded leaf images. It provides a model prediction, Grad-CAM visual evidence, separately labelled curated advisory, weather-risk alerts, and farmer-friendly action summaries with recent scan tracking.

Click **Use my current location** in the app to permit browser geolocation. The app sends only the resulting coordinates to Open-Meteo to retrieve current temperature and relative humidity; coordinates are not saved by the app. Manual weather inputs remain available if permission is denied.

## Project files

- `app.py` — Streamlit localhost interface
- `train.py` — training script
- `model.py` — shared compact CNN and Grad-CAM code
- `knowledge_base.py` — curated advisory, separate from model output
- `translations.py` — Hindi/English UI and scheduled Indian-language selector

## Dataset layout

Put labelled images in class-named folders. For example:

```text
dataset/
  Apple___Healthy/
  Apple___Black_rot/
  Tomato___Late_blight/
  Corn___Common_rust/
```

Use several images per class. `train.py` creates a reproducible 80/20 training-validation split and, by default, uses a balanced 150-image-per-class subset for quick CPU training. Set `MAX_IMAGES_PER_CLASS = None` in `train.py` to train on every image.

## Run on Windows PowerShell

```powershell
cd path\to\project
py -m pip install -r requirements.txt
py train.py
py -m streamlit run app.py --server.port 8502
```

Open `http://localhost:8502` in a browser. Do not use the VS Code **Run Python File** button for `app.py`.

## Important

This is a decision-support prototype. A high model confidence does not confirm disease. Get advice from a qualified local agricultural expert before applying any crop-protection product.
