# --- Configuration Constants ---
# INDEX_PATH = "index.faiss"
MAPPING_PATH = "index_mapping.json"
METADATA_CACHE_PATH = "metadata_cache.json"
DATABASE_SCHEMA_PATH = "schema.json"
OPENAI_EMBEDDING_MODEL_ID = "text-embedding-ada-002"  # Used for initial embedding
DEFAULT_QUERY_ANALYSIS_MODEL_KEY = "gpt-4o"  # User-facing name for query analysis
DEFAULT_FINAL_ANSWER_MODEL_KEY = "gpt-4o-mini"  # Default to gpt-4o-mini for cost-effectiveness

TOP_K = 15
MAX_EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 5

# --- Model Configuration ---
MODEL_CONFIG = {
    "gpt-4o": {
        "api_id": "gpt-4o",
        "provider": "openai",
        "cost_per_input_token": 5.00 / 1_000_000,  # $5.00 / 1M tokens
        "cost_per_output_token": 15.00 / 1_000_000, # $15.00 / 1M tokens
        "max_output_tokens": 4096,
    },
    "gpt-4o-mini": {
        "api_id": "gpt-4o-mini",
        "provider": "openai",
        "cost_per_input_token": 0.15 / 1_000_000, # $0.15 / 1M tokens
        "cost_per_output_token": 0.60 / 1_000_000,  # $0.60 / 1M tokens
        "max_output_tokens": 16384,
    },
    "text-embedding-ada-002": {
        "api_id": "text-embedding-ada-002",
        "provider": "openai",
        "cost_per_input_token": 0.10 / 1_000_000, # $0.10 / 1M tokens (OpenAI pricing page)
        "cost_per_output_token": 0,
        "max_output_tokens": 0,
    },
    "claude-3-opus-20240229": {
        "api_id": "claude-3-opus-20240229",
        "provider": "anthropic",
        "cost_per_input_token": 15.00 / 1_000_000,
        "cost_per_output_token": 75.00 / 1_000_000,
        "max_output_tokens": 4096,
    },
    "claude-3-sonnet-20240229": {
        "api_id": "claude-3-sonnet-20240229",
        "provider": "anthropic",
        "cost_per_input_token": 3.00 / 1_000_000,
        "cost_per_output_token": 15.00 / 1_000_000,
        "max_output_tokens": 4096,
    },
     "claude-3-haiku-20240307": {
        "api_id": "claude-3-haiku-20240307",
        "provider": "anthropic",
        "cost_per_input_token": 0.25 / 1_000_000,
        "cost_per_output_token": 1.25 / 1_000_000,
        "max_output_tokens": 4096,
    },
    "claude-3-5-haiku-20241022": { # Added based on user provided link and screenshot
        "api_id": "claude-3-5-haiku-20241022",
        "provider": "anthropic",
        "cost_per_input_token": 0.80 / 1_000_000, # $0.80 / 1M input tokens
        "cost_per_output_token": 4.00 / 1_000_000,  # $4.00 / 1M output tokens
        "max_output_tokens": 8192, # Max output for Claude 3.5 Haiku
    },
    # Add other models here as needed, e.g., Claude 3.5 Sonnet when UI supports it
    # "claude-3-5-sonnet-20240620": {
    #     "api_id": "claude-3-5-sonnet-20240620",
    #     "provider": "anthropic",
    #     "cost_per_input_token": 3.00 / 1_000_000,
    #     "cost_per_output_token": 15.00 / 1_000_000,
    #     "max_output_tokens": 8192,
    # },
} 