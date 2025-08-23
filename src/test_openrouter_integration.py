import asyncio
import os

from dotenv import load_dotenv

from open_deep_research.deep_researcher import deep_researcher

async def main():
    load_dotenv()
    os.environ["YOUR_SITE_URL"] = "https://example.com"
    os.environ["YOUR_SITE_NAME"] = "Example"

    config = {
        "configurable": {
            "research_model": "openrouter:deepseek/deepseek-chat-v3-0324:free",
            "allow_clarification": False,
        }
    }

    input_state = {
        "messages": [("user", "What is the meaning of life?")]
    }

    async for event in deep_researcher.astream(input_state, config=config):
        for key, value in event.items():
            print(f"Event: {key}")
            print(value)
            print("---")

if __name__ == "__main__":
    asyncio.run(main())
