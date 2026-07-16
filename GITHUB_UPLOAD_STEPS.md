# How to Upload This Project to GitHub

## Option A: Upload using the GitHub website

1. Create a new GitHub repository.
2. Name it, for example: `bank-account-fraud-detection`.
3. Do not add another README if you will upload this folder's README.md.
4. Upload the contents of this folder.
5. Commit the uploaded files.

## Option B: Upload using Git command line

```bash
cd bank_account_fraud_detection_github_ready
git init
git add .
git commit -m "Initial capstone project upload"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/bank-account-fraud-detection.git
git push -u origin main
```

## Important data note

Do not commit the raw Kaggle CSV files if they are too large or if redistribution is not required. The `.gitignore` excludes raw CSV files by default. Keep the data download instructions in the README instead.
