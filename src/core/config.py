# src/core/config.py
"""
Single source of truth for embedding configuration to prevent dimension mismatches.
Version: 1.2.0 - Standardize on Google text-embedding-004
This ensures both store-time and query-time embeddings use the same model and dimensions.
"""

# EMBEDDING CONFIGURATION - SINGLE SOURCE OF TRUTH
# Standardizing on Google's text-embedding-004 model via Generative AI client
EMBEDDING_MODEL_NAME = "models/text-embedding-004"  # Google Gemini embedding model
EMBEDDING_DIM = (
    768  # Actual output dimension from text-embedding-004 (confirmed by API testing)
)

# Key insight: text-embedding-004 actually outputs 768-D vectors, not 1024-D.
# This standardization eliminates all dimension mismatches by ensuring every
# component (SemanticMemoryPlugin, ConversationPlugin, etc.) uses the same
# Google Generative AI client with the same model specification.

# Historical reference (deprecated models):
# "sentence-transformers/all-MiniLM-L6-v2" -> 384-D
# "sentence-transformers/all-mpnet-base-v2" -> 768-D
# "models/text-embedding-004" -> 1024-D (current standard)
