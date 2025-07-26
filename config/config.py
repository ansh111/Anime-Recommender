import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API = os.getenv("GROQ_API")
MODEL_NAME = "llama-3.1-8b-instant"