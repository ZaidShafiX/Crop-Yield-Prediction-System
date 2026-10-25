# Crop Yield Prediction System: Web App & Deployment

Files in this folder:

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web app (5 sliders, a **Predict yield** button) |
| `requirements.txt` | Libraries the app needs on the server |
| `crop_yield_model_full_dataset.pkl` | **You create this** by running the notebook (Step 10.3) and put it here |

## 1. Get the model file

Run the notebook on Google Colab (**Runtime → Run all**). Step 10.3 saves
`crop_yield_model_full_dataset.pkl` and downloads it to your computer. Copy it into this folder.

`requirements.txt` already pins the versions printed in **Step 1** of the Colab run
(`scikit-learn==1.6.1`, `numpy==2.1.3`, `pandas==2.2.3`, Python 3.13). A pickled model only loads
reliably with the same scikit-learn version that created it, so if you re-run the notebook on Colab
and Step 1 prints different versions, update these lines to match.

## 2. Test the app on your computer first

```
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Check that the prediction works before deploying.

## 3. Deploy to the cloud (Streamlit Community Cloud, free)

**a) Put the three files on GitHub.** Create an **empty public repository** on github.com, then in a
terminal inside this folder:

```
git init
git add app.py requirements.txt crop_yield_model_full_dataset.pkl
git commit -m "Crop yield prediction app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> The model file is about 52 MB. GitHub's **browser upload is limited to 25 MB per file**, so use the
> terminal commands above (or GitHub Desktop) instead of drag-and-drop in the browser.

**b) Deploy.**

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Create a new app and choose your repository, the `main` branch and the main file `app.py`.
3. Open **Advanced settings** and select **Python 3.13**, the version Colab used to train the model
   (a `runtime.txt` file is ignored by Community Cloud). If 3.13 is not offered, 3.12 also works
   with the pinned libraries.
4. Click **Deploy** and wait a few minutes while it installs the libraries.

**c) Finish.** Copy the public app URL, open it and make a test prediction, then paste the URL into
the first cell of the notebook (`🔗 Live Deployment Link`).

## Troubleshooting

| Message | Fix |
|---|---|
| *The model file could not be loaded* | `scikit-learn` in `requirements.txt` does not match the version that trained the model. Use the version from Step 1 of your notebook. |
| *Model file ... was not found* | `crop_yield_model_full_dataset.pkl` is missing from the GitHub repository (check you committed it). |
| *Error installing requirements* | Check the spelling of the package names, and select a Python version that supports your pinned `scikit-learn` in **Advanced settings**. |
