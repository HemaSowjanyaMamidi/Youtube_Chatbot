# Youtube_Chatbot

This repository contains a conversational chatbot designed to interact with a given YouTube URL. The chatbot leverages the Gemma model from Groq API and the 'all-MiniLM-L6-v2' model from Hugging Face for embedding-based responses.

## Getting Started

### Prerequisites

1. **Groq API:**
   - Go to the [Groq API Keys](https://console.groq.com/keys) page.
   - Create an API key and store it securely.

2. **Hugging Face API:**
   - Sign up at [Hugging Face](https://huggingface.co/).
   - Go to Profile Settings and create an access token.
   - Save the token in a file named `.env` with the following content:
   - HUGGING_FACE_API_KEY=<your_hugging_face_api_key>
   - Ensure that the .env file is present in the root directory of the project to allow the application to access the necessary API keys.


## Create and Activate a Virtual Environment
This application is tested with python 3.11.9 on Windows. Create a virtual environment using the below commands
   ```bash
   python -m venv <envname>
   <envname>\Scripts\activate
```

### Clone the Repository

```bash
git clone <repository_url>
```

### Install Dependencies
```bash
cd <package>
pip install -r requirements.txt
```

## Start the Chatbot Application
`streamlit run app.py`

### Using the Application
Provide your Groq API key and the YouTube video URL you want to interact with.
Once the initial prompt is displayed, you can start chatting with the bot.
Happy chatting! 😊

