"""Test file for Google AI Studio integration with Gemini 2.0 Flash-Lite model."""

import asyncio
import json
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from code_consultant.code_consultant import code_consultant
from code_consultant.utils import get_all_tools, get_api_key_for_model

load_dotenv()


async def test_google_ai_studio_integration():
    """Test Google AI Studio integration with Gemini 2.0 Flash-Lite model."""

    # Configure Google AI Studio with Gemini 2.0 Flash-Lite model
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            # Google AI Studio configuration
            "analyst_model": "google:gemini-2.0-flash-lite",
            "compression_model": "google:gemini-2.0-flash-lite",
            "final_report_model": "google:gemini-2.0-flash-lite",
            "summarization_model": "google:gemini-2.0-flash-lite",
            # Research parameters - reduced for rate limiting
            "max_structured_output_retries": 2,  # Reduced from 3
            "allow_clarification": False,
            "max_concurrent_tasks": 1,  # Reduced from 2 for rate limiting
            "max_iterations": 1,  # Reduced from 2 for rate limiting
            "max_react_tool_calls": 3,  # Reduced from 5 for rate limiting
            # Token limits - conservative for rate limiting
            "analyst_model_max_tokens": 1000000,  # Reduced from 4000
            "compression_model_max_tokens": 1000000,  # Reduced from 4000
            "final_report_model_max_tokens": 1000000,  # Reduced from 4000
            "summarization_model_max_tokens": 1000000,  # Reduced from 2000
            # mcp server config
            "mcp_config": {
                "url": "http://localhost:8080",  # HTTP URL instead of stdio
                "tools": ["extract_relevant_file_content"],
                "auth_required": False,
            },
            "mcp_prompt": """

**CRITICAL FILE READING INSTRUCTIONS:**
**FOR SUPERVISORS:** When asked about local files (like README, config files, etc.), you MUST delegate these tasks to analysts via ExecuteTask. The analysts have access to the `extract_relevant_file_content` tool. Do NOT refuse local file requests - delegate them!

**FOR ANALYSTS:** When conducting file-related analysis, you have access to:
1. `extract_relevant_file_content(query: str)`: To extract content from relevant files based on a query. The query should be a descriptive sentence about what you are looking for.
2. NEVER provide generic information about files when you can read the actual content.
3. Local file questions should ALWAYS use your file reading tools."""
        }
    }

    tools = await get_all_tools(config)  
    print(f"🔧 Available tools: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]}")  
    print(f"🔧 Total tools loaded: {len(tools)}")

    # Simple test query
    test_query = "What query is used in test_openrouter_integration file?"

    try:
        # Initialize the graph
        graph = code_consultant.compile(checkpointer=MemorySaver())

        print("🧪 Testing Google AI Studio integration with Gemini 2.0 Flash-Lite...")
        print(f"📝 Query: {test_query}")
        print("⚙️  Model: gemini-2.0-flash-lite")
        print("-" * 50)

        # Run the research
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": test_query}]}, config
        )
        print(f"Supervisor messages: {result.get('supervisor_messages', [])}")  
        print(f"Task brief: {result.get('task_brief', '')}")

        # Save raw notes to file
        if "raw_notes" in result and result["raw_notes"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"google_ai_studio_raw_notes_{timestamp}.txt"

            print(f"📝 Raw notes found: {len(result['raw_notes'])} entries")

            # Create output directory if it doesn't exist
            os.makedirs("test_outputs", exist_ok=True)

            with open(f"test_outputs/{filename}", "w", encoding="utf-8") as f:
                f.write(f"Google AI Studio Integration Test - Raw Notes\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Query: {test_query}\n")
                f.write(f"Model: gemini-2.0-flash-lite\n")
                f.write("=" * 80 + "\n\n")

                for i, note in enumerate(result["raw_notes"]):
                    f.write(f"RAW NOTE {i+1}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(str(note))
                    f.write("\n\n" + "=" * 80 + "\n\n")

            print(f"💾 Raw notes saved to: test_outputs/{filename}")

            # Also print a preview
            for i, note in enumerate(result["raw_notes"]):
                print(f"📄 Raw Note {i+1} Preview: {str(note)[:200]}...")
        else:
            print("📝 No raw notes found in result")
            print(f"Available result keys: {list(result.keys())}")

        # Check results
        if "final_report" in result and result["final_report"]:
            print("✅ SUCCESS: Google AI Studio integration working!")
            print(f"📄 Report length: {len(result['final_report'])} characters")

            # Also save the full report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"google_ai_studio_full_report_{timestamp}.txt"

            with open(f"test_outputs/{report_filename}", "w", encoding="utf-8") as f:
                f.write(f"Google AI Studio Integration Test - Full Report\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Query: {test_query}\n")
                f.write(f"Model: gemini-2.0-flash-lite\n")
                f.write("=" * 80 + "\n\n")
                f.write(result["final_report"])

            print(f"💾 Full report saved to: test_outputs/{report_filename}")

            print("\n📋 Generated Report Preview:")
            print("-" * 30)
            # Show first 500 characters of the report
            preview = result["final_report"][:500]
            print(f"{preview}...")
            print("-" * 30)

            return True
        else:
            print("❌ FAILED: No final report generated")
            print(f"Result keys: {list(result.keys())}")
            return False

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False


async def test_api_key_retrieval():
    """Test that the get_api_key_for_model function works for Google models."""

    try:
        print("\n🔍 Testing API key retrieval...")

        # Mock config for testing
        mock_config = {"configurable": {}}

        # Test Google key retrieval
        api_key = get_api_key_for_model("google:gemini-2.0-flash-lite", mock_config)

        if api_key:
            print(f"✅ API key retrieval successful: {api_key[:10]}...")
            return True
        else:
            print("❌ API key retrieval failed - no key returned")
            return False

    except Exception as e:
        print(f"❌ API key retrieval error: {str(e)}")
        return False


async def main():
    """Run all Google AI Studio tests."""
    print("🚀 Starting Google AI Studio Integration Tests")
    print("=" * 50)

    # Test 1: API key retrieval
    key_success = await test_api_key_retrieval()

    # Test 2: Full integration test
    if key_success:
        integration_success = await test_google_ai_studio_integration()
    else:
        print("⏭️  Skipping integration test due to API key failure")
        integration_success = False

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"   API Key Retrieval: {'✅ PASS' if key_success else '❌ FAIL'}")
    print(f"   Full Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")

    if key_success and integration_success:
        print(
            "\n🎉 All tests passed! Google AI Studio integration is working correctly."
        )
    else:
        print("\n⚠️  Some tests failed. Check your Google AI Studio configuration.")
        print("\nTroubleshooting tips:")
        print("1. Ensure GOOGLE_API_KEY is set in your .env file")
        print("2. Verify your API key is valid and has sufficient quota")
        print("3. Check that you have access to Gemini 2.0 Flash-Lite model")
        print("4. Consider further reducing concurrent operations if rate limited")


if __name__ == "__main__":
    asyncio.run(main())
