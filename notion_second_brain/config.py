import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

if NOTION_TOKEN is None:
    raise ValueError("Error: NOTION_TOKEN environment variable not set. Please create a .env file or set the environment variable.")

if DATABASE_ID is None:
    raise ValueError("Error: NOTION_DATABASE_ID environment variable not set. Please create a .env file or set the environment variable.") 