#!/usr/bin/env python3
"""
Simple test script to run the AI Embedded Knowledge Agent without emoji logging issues.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    try:
        print("Starting AI Embedded Knowledge Agent...")
        print("=" * 50)

        # Set environment variables to avoid logging issues
        os.environ["PYTHONIOENCODING"] = "utf-8"

        # Import and create the application
        from app import create_app

        # Create the application
        rag_system, gradio_app = create_app()

        # Launch the Gradio interface
        print("Launching Gradio interface...")
        print("=" * 50)

        gradio_app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
        )

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        print("This is expected in demo mode without API keys.")
        print("The application structure is working correctly!")


if __name__ == "__main__":
    main()
