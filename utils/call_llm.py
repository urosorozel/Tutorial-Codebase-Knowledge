from google import genai
import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log")

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"

# Default to Azure OpenAI
def call_llm(prompt: str, use_cache: bool = True, provider: str = "azure") -> str:
    """
    Call an LLM with the given prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        use_cache: Whether to use cached responses
        provider: The LLM provider to use ('azure' is currently the only fully supported option)
        
    Returns:
        The LLM response as a string
    """
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")
    
    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except:
                logger.warning(f"Failed to load cache, starting with empty cache")
        
        # Return from cache if exists
        if prompt in cache:
            logger.info(f"CACHE HIT: {cache[prompt][:100]}...")
            return cache[prompt]
    
    # Call the LLM if not in cache or cache disabled
    if provider == "azure":
        try:
            # Use Azure OpenAI
            client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://someendpoint.openai.azure.com/"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            )
            
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content
            
            # Log the response
            logger.info(f"AZURE RESPONSE: {response_text[:100]}...")
            
            # Update cache if enabled
            if use_cache:
                update_cache(prompt, response_text)
            
            return response_text
        except Exception as e:
            error_msg = f"Azure OpenAI error: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    elif provider == "gemini":
        logger.warning("Gemini support is currently disabled due to authentication issues.")
        raise NotImplementedError("Gemini support is currently disabled. Please use 'azure' provider.")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def update_cache(prompt, response_text):
    """Helper function to update the cache"""
    try:
        # Load cache
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except:
                pass
        
        # Add to cache and save
        cache[prompt] = response_text
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# Commented alternatives below can be uncommented and used if needed
# # Use Anthropic Claude 3.7 Sonnet Extended Thinking
# def call_llm(prompt, use_cache: bool = True):
#     from anthropic import Anthropic
#     client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key"))
#     response = client.messages.create(
#         model="claude-3-7-sonnet-20250219",
#         max_tokens=21000,
#         thinking={
#             "type": "enabled",
#             "budget_tokens": 20000
#         },
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.content[1].text

# # Use OpenAI o1
# def call_llm(prompt, use_cache: bool = True):    
#     from openai import OpenAI
#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
#     r = client.chat.completions.create(
#         model="o1",
#         messages=[{"role": "user", "content": prompt}],
#         response_format={
#             "type": "text"
#         },
#         reasoning_effort="medium",
#         store=False
#     )
#     return r.choices[0].message.content

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"
    
    # Test Azure OpenAI
    try:
        print("Making call to Azure OpenAI...")
        response = call_llm(test_prompt, use_cache=False)
        print(f"Azure Response: {response}")
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
    
    # Gemini test is disabled to avoid authentication errors
    print("\nNote: Gemini provider is currently disabled due to authentication issues.")
    
