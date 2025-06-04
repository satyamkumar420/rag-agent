# 🔑 API Keys Setup Guide

## How to Get Pinecone API Key

### Step 1: Create Pinecone Account

1. Go to [https://www.pinecone.io/](https://www.pinecone.io/)
2. Click **"Sign Up"** or **"Get Started Free"**
3. Create account with your email or sign up with Google/GitHub
4. Verify your email address if required

### Step 2: Access Dashboard

1. Log into your Pinecone account
2. You'll be taken to the Pinecone Console/Dashboard
3. Look for **"API Keys"** in the left sidebar or navigation menu

### Step 3: Get Your API Key

1. Click on **"API Keys"** in the dashboard
2. You'll see your default API key listed
3. Click **"Copy"** or the copy icon next to the API key
4. Save this key securely - you'll need it for the application

## How to Get Gemini API Key

### Step 1: Go to Google AI Studio

1. Visit [https://aistudio.google.com/](https://aistudio.google.com/)
2. Sign in with your Google account

### Step 2: Get API Key

1. Click **"Get API Key"** in the top navigation
2. Click **"Create API Key"**
3. Select your Google Cloud project (or create a new one)
4. Copy the generated API key

## How to Get Tavily API Key

### Step 1: Create Tavily Account

1. Go to [https://app.tavily.com/](https://app.tavily.com/)
2. Click **"Sign Up"** and register with your email or use a social login
3. Verify your email address if prompted

### Step 2: Access API Keys

1. Log into your Tavily account
2. Navigate to the **"API Keys"** section in your dashboard
3. Click **"Create API Key"** if you don't have one yet
4. Copy the generated API key and store it securely

## 🚀 Quick Start Guide

### Option 1: Set Environment Variables Temporarily

**Windows Command Prompt:**

```cmd
set PINECONE_API_KEY=pc-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
set GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
set TAVILY_API_KEY=tvly_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python app.py
```

**Windows PowerShell:**

```powershell
$env:PINECONE_API_KEY="pc-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:GEMINI_API_KEY="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:TAVILY_API_KEY="tvly_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python app.py
```

### Option 2: Create .env File (Recommended)

1. Create a file named `.env` in your project root:

```
PINECONE_API_KEY=pc-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # Optional
```

2. Run the application:

```cmd
python app.py
```

## 📋 Free Tier Information

### Pinecone Free Tier:

- ✅ 1 project
- ✅ 1 index
- ✅ 100K vectors
- ✅ Perfect for hackathons and testing

### Gemini Free Tier:

- ✅ 15 requests per minute
- ✅ 1 million tokens per day
- ✅ Sufficient for development and demos

### Tavily Free Tier:

- ✅ Generous free tier for testing and development
- ✅ Check [Tavily Pricing](https://www.tavily.com/pricing) for current limits

## 🔧 Troubleshooting

### If you get "Invalid API Key" errors:

1. Double-check the API key is copied correctly
2. Make sure there are no extra spaces
3. Verify the environment variable is set: `echo %PINECONE_API_KEY%`

### If Pinecone connection fails:

1. Check your internet connection
2. Verify your Pinecone account is active
3. Make sure you're using the correct region (default is usually fine)

## 🎯 Ready to Launch

Once you have all API keys:

1. **Set the environment variables**
2. **Run the application:**
   ```cmd
   python app.py
   ```
3. **Open your browser to:** `http://localhost:7860`
4. **Start uploading documents and asking questions!**

The application will now have full functionality with:

- ✅ Document processing and embedding
- ✅ Vector storage in Pinecone
- ✅ AI-powered question answering
- ✅ Beautiful Gradio interface

**🎉 Your AI Embedded Knowledge Agent will be fully operational!**
