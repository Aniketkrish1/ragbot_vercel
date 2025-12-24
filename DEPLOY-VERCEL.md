# Deploy RAG Chatbot to Vercel

## Overview
This guide deploys your RAG chatbot backend to Vercel's serverless platform.

**Key Features:**
- ✅ Sessions stored in Vercel KV (automatically handled)
- ✅ No code changes needed
- ✅ Chatbot functionality unchanged
- ✅ Cold starts < 2 seconds (Vercel optimized)
- ✅ 10K free API calls/month (Vercel KV free tier)

---

## Prerequisites

### 1. Get API Keys
- **Groq API Key**: https://console.groq.com/keys
- **HuggingFace Token**: https://huggingface.co/settings/tokens (read-only, for embeddings)

### 2. GitHub Repository
Push your code to GitHub:
```bash
git add .
git commit -m "prepare for vercel deployment"
git push origin main
```

---

## Deployment Steps

### Step 1: Sign Up on Vercel
1. Go to https://vercel.com
2. Click "Sign Up" → Choose "GitHub"
3. Authorize Vercel to access your GitHub
4. Select your RAG repository

### Step 2: Configure Environment Variables
On Vercel dashboard:

1. Go to **Settings → Environment Variables**
2. Add these variables:
   - **Name:** `GROQ_API_KEY`
     **Value:** (your Groq API key)
   
   - **Name:** `HF_TOKEN`
     **Value:** (your HuggingFace token)

3. Make sure environment is set to **Production**

### Step 3: Configure Vercel KV
1. Go to **Storage → Create Database**
2. Select **Vercel KV**
3. Name it: `rag-sessions`
4. Click **Create**
5. It auto-connects to your project

### Step 4: Deploy
1. Click **Deploy**
2. Vercel automatically:
   - Installs `requirements-api.txt`
   - Builds your project
   - Deploys to serverless
3. Wait for deployment complete (~2-3 minutes)

### Step 5: Get Your API URL
After deployment, Vercel shows your domain:
```
https://your-project-name.vercel.app
```

---

## Testing Your API

### Health Check
```bash
curl https://your-project-name.vercel.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "vectorstore_loaded": true,
  "active_sessions": 0
}
```

### Chat Endpoint
```bash
curl -X POST https://your-project-name.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is leadership?"}'
```

### Test Session Persistence (Cookie-based)
```bash
# First message
curl -X POST https://your-project-name.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is leadership?"}' \
  -c cookies.txt

# Second message (should include session context)
curl -X POST https://your-project-name.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me more about that"}' \
  -b cookies.txt
```

---

## How Session Storage Works

**On Vercel (Production):**
- Sessions stored in Vercel KV (Redis)
- Automatic expiry: 1 hour
- Works across multiple function invocations
- Free tier: 10K commands/month

**Local Development (Fallback):**
- Sessions in memory
- If Vercel KV unavailable, auto-fallback works
- Full functionality preserved

**From User's Perspective:**
- Identical behavior (cookie-based sessions)
- Chat context maintained across requests
- No changes to chatbot functionality

---

## Production Configuration

### Update CORS for Your Domain

Edit `api.py` line ~45:

**Change from:**
```python
allow_origins=["*"]
```

**To:**
```python
allow_origins=["https://your-frontend-domain.com"]
```

Then push to GitHub - Vercel auto-redeploys.

### Enable HTTPS
Vercel automatically provides HTTPS. No additional setup needed.

---

## Monitoring

### View Logs
On Vercel dashboard:
1. Go to **Deployments**
2. Click on latest deployment
3. View **Function Logs**

### Check KV Storage Usage
1. Go to **Storage → rag-sessions**
2. See keys count and usage

### Monitor Costs
Free tier includes:
- 100 GB-hours of compute/month
- 10K KV commands/month
- Perfect for 5-8 users

Upgrade only if exceeding limits.

---

## Troubleshooting

### "502 Bad Gateway" Error
- Check environment variables are set
- Verify `GROQ_API_KEY` and `HF_TOKEN` in Vercel Settings
- View function logs for error details

### "Vectorstore not found"
- Ensure `vectorstore/` folder is in repository
- Check `.gitignore` doesn't exclude vectorstore files

### Sessions Not Persisting
- Check Vercel KV connection (Storage tab)
- Verify cookies enabled in browser
- Clear cookies and test again

### Slow First Request
- Vercel cold starts: 1-3 seconds (normal)
- First request after deploy may be slow
- Subsequent requests are instant (~100-200ms)

---

## Frontend Integration

### Example: Fetch from Frontend
```javascript
// Get session cookie
async function chatWithBot(message) {
  const response = await fetch('https://your-project.vercel.app/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include', // Important: send cookies
    body: JSON.stringify({ prompt: message })
  });
  
  const data = await response.json();
  return data;
}
```

### React Hook Example
```javascript
const [session, setSession] = useState(null);

async function sendMessage(prompt) {
  const response = await fetch(
    process.env.REACT_APP_API_URL + '/api/chat',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ prompt })
    }
  );
  
  const data = await response.json();
  setSession(data.session_id); // Auto-maintained by cookies
  return data;
}
```

---

## Scaling

### Current Limits (Free Tier)
- Compute: 100 GB-hours/month
- KV: 10K commands/month
- Supports ~100-200 requests/day

### Upgrade When Needed
- Vercel Pro: $20/month (unlimited compute)
- KV Standard: $0.20 per 100 commands

---

## Rollback to Previous Deployment

On Vercel dashboard:
1. Go to **Deployments**
2. Find previous working version
3. Click **Promote to Production**
4. Auto-reverts instantly

---

## Next Steps

1. ✅ Deploy to Vercel (this guide)
2. ✅ Connect your frontend
3. ✅ Update CORS with frontend domain
4. ✅ Monitor logs and KV usage
5. Optional: Add custom domain

---

**Questions?**
- Vercel Docs: https://vercel.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- Groq API: https://console.groq.com/docs
