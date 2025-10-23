# 224410083_ZeynepBorekci_Proje

Small project for data processing and GUI. Contains Python scripts and UI files.

How to push to GitHub

1. Create a new repository on GitHub (do NOT initialize with a README).
2. On your machine, set the remote and push:

   git remote add origin <YOUR_GITHUB_REPO_URL>
   git branch -M main
   git push -u origin main

Or use the GitHub CLI:

   gh repo create <OWNER/REPO> --source=. --public --push

Files of interest

- `main.py` — entry point
- `arayuz.ui`, `tahmin_et.ui` — UI files
- `data.csv`, `imbalanced_data.csv`, `noisy_data_with_nan.csv` — data files
