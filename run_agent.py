"""Run the agentic voice bot"""
import asyncio
from voice_bot.agentic_bot import AgenticVoiceBot

async def main():
    """Start the agentic assistant"""
    bot = AgenticVoiceBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
