#!/usr/bin/env python3
"""
StudyForge AI - Main Agent Entry Point
Your intelligent AI study companion with web search capabilities
"""

# Simple import from the main implementation
from src.web_enhanced_ai import main

if __name__ == "__main__":
    import asyncio
    try:
        print("🚀 Starting StudyForge AI Agent...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Thanks for using StudyForge AI!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try: python src/web_enhanced_ai.py")