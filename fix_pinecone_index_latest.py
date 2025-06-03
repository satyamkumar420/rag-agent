#!/usr/bin/env python3
"""
Pinecone Index Fixer for Latest Gemini Embedding Model

This script fixes the Pinecone index dimension mismatch by:
1. Deleting the existing index with wrong dimensions (768)
2. Creating a new index with correct dimensions (3072) for gemini-embedding-exp-03-07
3. Ensuring proper configuration for the latest model

Usage: python fix_pinecone_index_latest.py
"""

import os
import sys
import time
import yaml
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    try:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        logger.info("✅ Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)


def fix_pinecone_index():
    """Fix Pinecone index for latest Gemini embedding model"""

    print("🚀 Pinecone Index Fixer for Latest Gemini Embedding Model")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Get Pinecone settings
    vector_db_config = config.get("vector_db", {})
    index_name = vector_db_config.get("index_name", "rag-ai-index")
    new_dimension = vector_db_config.get("dimension", 3072)  # Latest model dimension
    metric = vector_db_config.get("metric", "cosine")
    environment = vector_db_config.get("environment", "us-east-1")

    # Get API key
    api_key = config.get("api_keys", {}).get("pinecone_api_key")
    if not api_key:
        api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        logger.error("❌ Pinecone API key not found in config or environment variables")
        sys.exit(1)

    try:
        # Import Pinecone
        from pinecone import Pinecone, ServerlessSpec

        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        logger.info("✅ Pinecone client initialized")

        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_exists = any(idx.name == index_name for idx in existing_indexes)

        if index_exists:
            print(f"📋 Found existing index: {index_name}")

            # Get current index info
            index_info = pc.describe_index(index_name)
            current_dimension = index_info.dimension

            print(f"   Current dimension: {current_dimension}")
            print(f"   Required dimension: {new_dimension}")

            if current_dimension != new_dimension:
                print(f"⚠️  Dimension mismatch detected!")
                print(f"   Current: {current_dimension} → Required: {new_dimension}")

                # Confirm deletion
                response = input(
                    f"\n🗑️  Delete existing index '{index_name}' and recreate with correct dimensions? (y/N): "
                )
                if response.lower() != "y":
                    print("❌ Operation cancelled by user")
                    return

                # Delete existing index
                print(f"🗑️  Deleting existing index: {index_name}")
                pc.delete_index(index_name)

                # Wait for deletion to complete
                print("⏳ Waiting for index deletion to complete...")
                while index_name in [idx.name for idx in pc.list_indexes()]:
                    time.sleep(2)
                    print("   Still deleting...")

                print("✅ Index deleted successfully")
            else:
                print("✅ Index already has correct dimensions")
                return

        # Create new index with correct dimensions
        print(f"🔨 Creating new index: {index_name}")
        print(f"   Dimension: {new_dimension}")
        print(f"   Metric: {metric}")
        print(f"   Environment: {environment}")

        pc.create_index(
            name=index_name,
            dimension=new_dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region=environment),
        )

        # Wait for index to be ready
        print("⏳ Waiting for index to be ready...")
        while True:
            try:
                index_info = pc.describe_index(index_name)
                if index_info.status.ready:
                    break
                print("   Index is initializing...")
                time.sleep(5)
            except Exception as e:
                print(f"   Checking status: {e}")
                time.sleep(5)

        print("✅ Index created and ready!")

        # Verify index configuration
        final_info = pc.describe_index(index_name)
        print(f"\n📊 Final Index Configuration:")
        print(f"   Name: {final_info.name}")
        print(f"   Dimension: {final_info.dimension}")
        print(f"   Metric: {final_info.metric}")
        print(f"   Status: {'Ready' if final_info.status.ready else 'Not Ready'}")

        print(f"\n🎉 Success! Your Pinecone index is now configured for:")
        print(f"   📦 Model: gemini-embedding-exp-03-07")
        print(f"   📏 Dimensions: {new_dimension}")
        print(f"   🎯 Optimized for: Latest Gemini embedding performance")

    except ImportError:
        logger.error(
            "❌ Pinecone library not installed. Run: pip install pinecone-client"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error fixing Pinecone index: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        fix_pinecone_index()
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
