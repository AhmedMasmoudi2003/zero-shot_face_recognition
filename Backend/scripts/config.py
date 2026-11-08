# config.py
import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------- DEVICE ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- PINECONE ----------
# Set these in your environment before running:
# export PINECONE_API_KEY="..."
# export PINECONE_INDEX_NAME="faces-index"
# export PINECONE_ENVIRONMENT="us-east-1-aws"  # or your region

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "faces-index")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", None)

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set in environment variables")

if not PINECONE_ENVIRONMENT:
    # For new Pinecone client versions, you may not need environment;
    # if you do, set it explicitly.
    print("Warning: PINECONE_ENVIRONMENT is not set. "
          "If your Pinecone setup requires it, please set it.")
