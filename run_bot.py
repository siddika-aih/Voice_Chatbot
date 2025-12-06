"""Main entry point for DCB Voice Bot"""
import asyncio
from voice_bot.dcb_bot import DCBVoiceBot

async def main():
    """Start the voice bot"""
    bot = DCBVoiceBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
