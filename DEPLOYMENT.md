# Deploy Brokkr's Eye to Render.com (FREE)

## Step 1: Prepare Your Code

Your code is ready! These files have been created:
- `render.yaml` - Render configuration
- `runtime.txt` - Python version
- `requirements.txt` - Updated with pinned versions

## Step 2: Push to GitHub

1. **Create a new GitHub repository:**
   - Go to https://github.com/new
   - Name it: `brokkrs-eye` (or whatever you want)
   - Make it Public or Private (both work)
   - Don't initialize with README

2. **Push your webapp folder to GitHub:**
   ```bash
   cd c:\Personal\MjolnirNN\webapp
   git init
   git add .
   git commit -m "Initial commit - Brokkr's Eye webapp"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/brokkrs-eye.git
   git push -u origin main
   ```

## Step 3: Deploy on Render.com

1. **Sign up for Render:**
   - Go to https://render.com
   - Click "Get Started for Free"
   - Sign up with your GitHub account

2. **Create a New Web Service:**
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub repository (`brokkrs-eye`)
   - Render will auto-detect the settings from `render.yaml`

3. **Configure (if needed):**
   - **Name:** brokkrs-eye
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free

4. **Deploy:**
   - Click "Create Web Service"
   - Wait 5-10 minutes for build and deployment
   - You'll get a URL like: `https://brokkrs-eye.onrender.com`

## Step 4: Access Your App

Once deployed, your app will be live at the URL Render provides!

## Important Notes

### Free Tier Limitations:
- ✓ Apps spin down after 15 minutes of inactivity
- ✓ First request after sleep takes 30-60 seconds to wake up
- ✓ 750 hours/month free (more than enough)
- ✓ Automatic HTTPS included

### Large Model File Warning:
Your `Model_A_B500_E300_V2.hdf5` is 17MB. This should deploy fine on Render's free tier.

### If Build Fails:
- TensorFlow is large (~1GB). Build might take 10-15 minutes
- Free tier has 512MB RAM - should be enough for your model
- If RAM issues occur, you may need to upgrade to paid tier ($7/month)

## Alternative: Railway.app

If Render doesn't work, try Railway:
1. Sign up at https://railway.app
2. "New Project" → "Deploy from GitHub repo"
3. Select your repo
4. Railway auto-detects and deploys
5. Free $5/month credit (usually enough)

## Updating Your App

After initial deployment, just push changes to GitHub:
```bash
git add .
git commit -m "Update webapp"
git push
```

Render will automatically redeploy!

## Troubleshooting

**Build fails with memory error:**
- TensorFlow is heavy. Consider upgrading to Render's paid tier ($7/month)

**App is slow on first load:**
- Normal on free tier. App "wakes up" from sleep.

**Need faster performance:**
- Upgrade to paid tier for always-on service

## Your Live URL

After deployment, share your URL with anyone to use Brokkr's Eye!
Example: `https://brokkrs-eye.onrender.com`
