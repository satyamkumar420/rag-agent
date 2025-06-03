#!/usr/bin/env python3
"""
Fix Pinecone Index Dimension Mismatch

This script deletes the existing Pinecone index and recreates it with the correct
dimension (3072) to match Gemini embeddings.

⚠️  WARNING: This will delete all existing data in the index!
"""

import os
import sys
import time
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "python-dotenv not installed. Please ensure PINECONE_API_KEY is set as environment variable."
    )

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    print("❌ Pinecone library not installed. Please run: pip install pinecone-client")
    sys.exit(1)

from utils.config_manager import ConfigManager


def fix_pinecone_index():
    """Fix the Pinecone index dimension mismatch."""

    print("🔧 Pinecone Index Dimension Fix")
    print("=" * 50)

    # Load configuration
    config_manager = ConfigManager()
    vector_db_config = config_manager.get_section("vector_db")

    # Get API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set!")
        print("Please set your Pinecone API key and try again.")
        return False

    # Get configuration
    index_name = vector_db_config.get("index_name", "rag-ai-index")
    dimension = vector_db_config.get("dimension", 3072)
    metric = vector_db_config.get("metric", "cosine")
    environment = vector_db_config.get("environment", "us-east-1")

    print(f"📋 Configuration:")
    print(f"   Index Name: {index_name}")
    print(f"   Dimension: {dimension}")
    print(f"   Metric: {metric}")
    print(f"   Environment: {environment}")
    print()

    try:
        # Initialize Pinecone client
        print("🔌 Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)

        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        print(f"📋 Existing indexes: {existing_indexes}")

        if index_name in existing_indexes:
            print(f"⚠️  Index '{index_name}' exists. Checking dimensions...")

            # Get current index info
            index_info = pc.describe_index(index_name)
            current_dimension = index_info.dimension

            print(f"   Current dimension: {current_dimension}")
            print(f"   Required dimension: {dimension}")

            if current_dimension == dimension:
                print("✅ Index already has correct dimensions!")
                return True

            print(f"❌ Dimension mismatch: {current_dimension} != {dimension}")
            print()

            # Confirm deletion
            response = input(
                f"⚠️  Delete and recreate index '{index_name}'? This will remove ALL data! (yes/no): "
            )
            if response.lower() != "yes":
                print("❌ Operation cancelled.")
                return False

            print(f"🗑️  Deleting index '{index_name}'...")
            pc.delete_index(index_name)

            # Wait for deletion to complete
            print("⏳ Waiting for deletion to complete...")
            while index_name in [idx.name for idx in pc.list_indexes()]:
                time.sleep(2)
                print("   Still deleting...")

            print("✅ Index deleted successfully!")

        # Create new index
        print(f"🔨 Creating new index '{index_name}' with dimension {dimension}...")

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region=environment),
        )

        # Wait for index to be ready
        print("⏳ Waiting for index to be ready...")
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                index_info = pc.describe_index(index_name)
                if index_info.status.ready:
                    print("✅ Index is ready!")
                    break
                print("   Index still initializing...")
                time.sleep(10)
            except Exception as e:
                print(f"   Checking status: {e}")
                time.sleep(5)
        else:
            print("❌ Index creation timed out!")
            return False

        # Verify index
        print("🔍 Verifying index...")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        print(f"✅ Index created successfully!")
        print(f"   Dimension: {stats.dimension}")
        print(f"   Metric: {stats.metric}")
        print(f"   Vector count: {stats.total_vector_count}")
        print()

        print("🎉 Pinecone index fix completed successfully!")
        print("You can now restart your application.")

        return True

    except Exception as e:
        print(f"❌ Error fixing Pinecone index: {str(e)}")
        return False


if __name__ == "__main__":
    success = fix_pinecone_index()
    sys.exit(0 if success else 1)
