"""Test file for OpenRouter integration with DeepSeek model."""  
  
import asyncio  
import uuid  
import os  
from dotenv import load_dotenv  
from open_deep_research.deep_researcher import deep_researcher_builder  
from langgraph.checkpoint.memory import MemorySaver  
  
load_dotenv()  
  
async def test_openrouter_integration():  
    """Test OpenRouter integration with DeepSeek model."""  
      
    # Configure OpenRouter with DeepSeek model  
    config = {  
        "configurable": {  
            "thread_id": str(uuid.uuid4()),  
            # OpenRouter configuration  
            "research_model": "openrouter:deepseek/deepseek-chat-v3-0324:free",  
            "compression_model": "openrouter:deepseek/deepseek-chat-v3-0324:free",   
            "final_report_model": "openrouter:deepseek/deepseek-chat-v3-0324:free",  
            "summarization_model": "openrouter:deepseek/deepseek-chat-v3-0324:free",  
              
            # Research parameters  
            "max_structured_output_retries": 3,  
            "allow_clarification": False,  
            "max_concurrent_research_units": 2,  # Keep low for testing  
            "search_api": "tavily",  
            "max_researcher_iterations": 2,  # Keep low for testing  
            "max_react_tool_calls": 5,  
              
            # Token limits  
            "research_model_max_tokens": 4000,  
            "compression_model_max_tokens": 4000,  
            "final_report_model_max_tokens": 4000,  
            "summarization_model_max_tokens": 2000,  
        }  
    }  
      
    # Simple test query  
    test_query = "What are the key benefits of renewable energy?"  
      
    try:  
        # Initialize the graph  
        graph = deep_researcher_builder.compile(checkpointer=MemorySaver())  
          
        print("🧪 Testing OpenRouter integration with DeepSeek...")  
        print(f"📝 Query: {test_query}")  
        print("⚙️  Model: deepseek/deepseek-chat-v3-0324:free")  
        print("-" * 50)  
          
        # Run the research  
        result = await graph.ainvoke(  
            {"messages": [{"role": "user", "content": test_query}]},  
            config  
        )  
          
        # Check results  
        if "final_report" in result and result["final_report"]:  
            print("✅ SUCCESS: OpenRouter integration working!")  
            print(f"📄 Report length: {len(result['final_report'])} characters")  
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
  
async def test_model_initialization():  
    """Test that the init_openrouter_model function works correctly."""  
    from open_deep_research.utils import init_openrouter_model  
      
    try:  
        print("\n🔧 Testing model initialization...")  
          
        # Get the API key for OpenRouter  
        api_key = os.getenv("OPENROUTER_API_KEY")  
        if not api_key:  
            print("❌ OPENROUTER_API_KEY not found in environment")  
            return False  
          
        print(f"🔑 Found API key: {api_key[:10]}...")  
          
        # Test OpenRouter model initialization with API key  
        model = init_openrouter_model(  
            "openrouter:openai/gpt-3.5-turbo",  
            max_tokens=1000,  
            api_key=api_key  # Pass the API key explicitly  
        )  
          
        # Simple test message  
        test_message = [{"role": "user", "content": "Hello, can you respond with 'OpenRouter test successful'?"}]  
          
        response = await model.ainvoke(test_message)  
          
        if response and hasattr(response, 'content'):  
            print("✅ Model initialization successful!")  
            print(f"📝 Response: {response.content[:100]}...")  
            return True  
        else:  
            print("❌ Model initialization failed - no response content")  
            return False  
              
    except Exception as e:  
        print(f"❌ Model initialization error: {str(e)}")  
        return False  
  
async def test_api_key_retrieval():  
    """Test that the get_api_key_for_model function works for OpenRouter."""  
    from open_deep_research.utils import get_api_key_for_model  
      
    try:  
        print("\n🔍 Testing API key retrieval...")  
          
        # Mock config for testing  
        mock_config = {"configurable": {}}  
          
        # Test OpenRouter key retrieval  
        api_key = get_api_key_for_model("openrouter:deepseek/deepseek-chat-v3-0324:free", mock_config)  
          
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
    """Run all OpenRouter tests."""  
    print("🚀 Starting OpenRouter Integration Tests")  
    print("=" * 50)  
      
    # Test 1: API key retrieval  
    key_success = await test_api_key_retrieval()  
      
    # Test 2: Model initialization  
    init_success = await test_model_initialization()  
      
    # Test 3: Full integration test  
    if init_success:  
        integration_success = await test_openrouter_integration()  
    else:  
        print("⏭️  Skipping integration test due to initialization failure")  
        integration_success = False  
      
    # Summary  
    print("\n" + "=" * 50)  
    print("📊 TEST SUMMARY:")  
    print(f"   API Key Retrieval: {'✅ PASS' if key_success else '❌ FAIL'}")  
    print(f"   Model Initialization: {'✅ PASS' if init_success else '❌ FAIL'}")  
    print(f"   Full Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")  
      
    if key_success and init_success and integration_success:  
        print("\n🎉 All tests passed! OpenRouter integration is working correctly.")  
    else:  
        print("\n⚠️  Some tests failed. Check your OpenRouter configuration.")  
        print("\nTroubleshooting tips:")  
        print("1. Ensure OPENROUTER_API_KEY is set in your .env file")  
        print("2. Verify your init_openrouter_model function is implemented in utils.py")  
        print("3. Check that get_api_key_for_model handles OpenRouter models")  
        print("4. Ensure your API key starts with 'sk-or-' and has sufficient credits")  
  
if __name__ == "__main__":  
    asyncio.run(main())
