import os
from dotenv import load_dotenv
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from datetime import datetime
from langgraph.graph import StateGraph
from typing import Dict, Any, TypedDict
import argparse
import uuid
import json
import re
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

# Define the state schema for LangGraph
class AgentState(TypedDict, total=False):
    user_query: str
    thread_id: str
    short_term: list
    long_term_candidates: list
    long_term: list
    context: dict
    assistant_response: str

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Embedding model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Redis setup
REDIS_URL = "redis://localhost:6379"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Define the index schema
schema = IndexSchema.from_dict({
    "index": {
        "name": "message_index",
        "prefix": "message",
        "key_separator": ":"
    },
    "fields": [
        {"name": "content", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 384,
                "algorithm": "FLAT",
                "distance_metric": "COSINE"
            }
        }
    ]
})

# Create the SearchIndex object
redis_index = SearchIndex(schema=schema, redis_url=REDIS_URL)

# LLM setup
gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")

def create_vector_index(force_recreate=False):
    """Creates a Redis search index using the redis-vl schema."""
    try:
        redis_index.create(overwrite=force_recreate)
        if force_recreate:
            print("✅ Dropped old and recreated Redis Stack vector index 'message_index'.")
        else:
            print("✅ Redis Stack vector index 'message_index' confirmed to exist.")
    except Exception as e:
        print(f"Error creating/confirming vector index: {e}")

# --- LangGraph Nodes ---
def user_input_node(state):
    user_query = input("You: ").strip()
    state['user_query'] = user_query
    return state

def short_term_memory_node(state):
    thread_id = state['thread_id']
    short_term_window = 5
    recent_message_ids = redis_client.zrange(f"thread:{thread_id}:messages", -short_term_window, -1)
    
    short_term_messages = []
    if recent_message_ids:
        for msg_key in recent_message_ids:
            try:
                full_key = f"{redis_index.prefix}{redis_index.key_separator}{msg_key}"
                content_json = redis_client.hget(full_key, "content")
                if content_json:
                    msg = json.loads(content_json)
                    short_term_messages.append(msg)
            except Exception as e:
                print(f"Error retrieving or parsing message {msg_key}: {e}")
    else:
        print(f"No recent messages found for thread {thread_id}.")
        
    state['short_term'] = short_term_messages
    return state

def long_term_memory_node(state):
    query = state['user_query']
    max_candidates = 15  # Increased to get more diverse results
    min_similarity = 0.0  # Set to 0 to see all results, we'll filter after
    
    try:
        # Generate embedding for the query
        query_embedding = st_model.encode(query).astype(np.float32).tolist()
        
        # Create vector query with increased limit
        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="embedding",
            return_fields=["content", "vector_distance", "thread_id"],
            num_results=max_candidates,
            return_score=True
        )
        
        # Execute the search
        results = redis_index.query(vector_query)
        
        # Process and filter results
        seen_messages = set()
        relevant_messages = []
        
        for doc in results:
            try:
                # Calculate similarity from distance
                distance = float(doc.get('vector_distance', 1.0))
                similarity = 1 - distance
                
                # Parse the content
                content_dict = json.loads(doc['content'])
                
                # Skip if we've seen this exact message content before
                msg_key = (content_dict.get('user_input', ''), 
                          content_dict.get('assistant_response', ''))
                if msg_key in seen_messages:
                    continue
                seen_messages.add(msg_key)
                
                # Add similarity and other metadata
                content_dict['similarity'] = round(similarity, 4)
                content_dict['thread_id'] = doc.get('thread_id', '')
                
                relevant_messages.append(content_dict)
                
            except Exception as e:
                print(f"Error processing search result: {e}")
        
        # Sort by similarity in descending order
        relevant_messages.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Debug output before filtering
        print("\n=== [1] VECTOR SEARCH: ALL RETRIEVED MESSAGES ===")
        for i, msg in enumerate(relevant_messages, 1):
            print(f"[{i}] Similarity: {msg.get('similarity', 0):.3f}")
            print(f"Thread: {msg.get('thread_id', 'N/A')}")
            print(f"User: {msg.get('user_input', 'N/A')}")
            print(f"Assistant: {msg.get('assistant_response', 'N/A')}")
            print(f"Full message content: {json.dumps(msg, indent=2, ensure_ascii=False)}")
            if 'tools_used' in msg and msg['tools_used']:
                print(f"Tools used: {msg['tools_used']}")
            print()
        
        # Filter out very low similarity results unless we don't have enough
        filtered_messages = [
            msg for msg in relevant_messages 
            if msg.get('similarity', 0) >= 0.1 or len(relevant_messages) < 5
        ][:15]  # Take top 15 after filtering
        
        state['long_term_candidates'] = filtered_messages
            
    except Exception as e:
        state['long_term_candidates'] = []
        print(f"Error during long-term memory retrieval: {e}")
        import traceback
        traceback.print_exc()
        
    return state

def llm_field_selection_node(state):
    messages = state['long_term_candidates']
    query = state['user_query']
    if not messages:
        state['long_term'] = []
        return state
    
    # Show messages before LLM filtering
    print("\n=== [2A] MESSAGES BEFORE LLM FILTERING ===")
    for i, msg in enumerate(messages, 1):
        print(f"[Before {i}] Similarity: {msg.get('similarity', 0):.3f}")
        print(f"Thread: {msg.get('thread_id', 'N/A')}")
        print(f"User: {msg.get('user_input', 'N/A')}")
        print(f"Assistant: {msg.get('assistant_response', 'N/A')}")
        if 'tools_used' in msg and msg['tools_used']:
            print(f"Tools used: {msg['tools_used']}")
        print(f"Full message: {json.dumps(msg, ensure_ascii=False, indent=2)}\n")
    
    prompt = f"""
You are an expert at extracting relevant fields from memory messages for an LLM agent.
Given the current user query and a list of past messages (as JSON), return ONLY a valid JSON list of objects.
For each message:
- Extract the field that represents the user's input (even if it is named 'user_input', 'question', 'query', or similar).
- Extract the field that represents the assistant's response (even if it is named 'assistant_response', 'reply', 'response', or similar).
- If present, also extract any field that lists tools used (e.g., 'tools_used', 'tools', etc.).
- Do NOT invent, omit, or rename fields. If a field is missing, leave it out for that message.
- Do NOT use markdown or code blocks. Do NOT add explanations.
- If you are unsure which field represents the user's input or assistant's response, use your best judgment based on the field's content.
Current user query: {query}
Past messages (as JSON):
{json.dumps(messages, ensure_ascii=False, indent=2)}
Return ONLY a valid JSON list of objects, with no extra text.
"""
    try:
        print("\n=== [2B] SENDING TO LLM FOR FILTERING ===")
        print(f"Prompt: {prompt[:500]}...")
        
        response = gemini_model.generate_content(prompt)
        llm_output = response.text if hasattr(response, 'text') else str(response)
        
        print("\n=== [2C] RAW LLM RESPONSE ===")
        print(llm_output)
        
        json_str = llm_output
        codeblock_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', llm_output)
        if codeblock_match:
            json_str = codeblock_match.group(1)
            
        parsed = json.loads(json_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            state['long_term'] = parsed
            print("\n=== [2D] AFTER LLM FIELD SELECTION ===")
            for i, msg in enumerate(parsed, 1):
                print(f"[After {i}] {json.dumps(msg, ensure_ascii=False, indent=2)}")
                print()
        else:
            state['long_term'] = []
            print("\n=== [2D] NO MESSAGES AFTER FILTERING ===")
    except Exception as e:
        print(f"\n=== [2E] ERROR IN LLM FIELD SELECTION ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        state['long_term'] = []
    return state

def context_construction_node(state):
    max_context_tokens = 4000
    def estimate_tokens(text):
        return int(len(text.split()) * 1.3)
    short_term = state.get('short_term', [])
    long_term = state.get('long_term', [])
    short_term_tokens = sum(estimate_tokens(f"{msg.get('user_input', '')} {msg.get('assistant_response', '')}") for msg in short_term)
    long_term_tokens = sum(estimate_tokens(f"{msg.get('user_input', '')} {msg.get('assistant_response', '')}") for msg in long_term)
    remaining_tokens = max_context_tokens - short_term_tokens
    if remaining_tokens < 0:
        short_term = short_term[-2:]
        short_term_tokens = sum(estimate_tokens(f"{msg.get('user_input', '')} {msg.get('assistant_response', '')}") for msg in short_term)
        remaining_tokens = max_context_tokens - short_term_tokens
    filtered_long_term = []
    long_term_tokens_used = 0
    for msg in long_term:
        msg_tokens = estimate_tokens(f"{msg.get('user_input', '')} {msg.get('assistant_response', '')}")
        if long_term_tokens_used + msg_tokens <= remaining_tokens:
            filtered_long_term.append(msg)
            long_term_tokens_used += msg_tokens
        else:
            break
    state['context'] = {
        'short_term': short_term,
        'long_term': filtered_long_term,
        'token_usage': {
            'short_term_tokens': short_term_tokens,
            'long_term_tokens': long_term_tokens_used,
            'total_tokens': short_term_tokens + long_term_tokens_used,
            'max_tokens': max_context_tokens
        },
        'memory_stats': {
            'short_term_count': len(short_term),
            'long_term_count': len(filtered_long_term)
        }
    }
    return state

def final_llm_node(state):
    context = state['context']
    user_query = state['user_query']
    formatted_context = "=== MEMORY CONTEXT ===\n\n"
    if context['short_term']:
        formatted_context += "📝 RECENT CONVERSATION:\n"
        for i, msg in enumerate(context['short_term'], 1):
            formatted_context += f"{i}. User: {msg.get('user_input', '')}\n   Assistant: {msg.get('assistant_response', '')}\n"
            if msg.get('tools_used'):
                formatted_context += f"   Tools: {msg.get('tools_used', [])}\n"
            formatted_context += "\n"
    if context['long_term']:
        formatted_context += "🧠 RELEVANT HISTORICAL CONTEXT:\n"
        for i, msg in enumerate(context['long_term'], 1):
            formatted_context += f"{i}. User: {msg.get('user_input', '')}\n   Assistant: {msg.get('assistant_response', '')}\n"
            if msg.get('tools_used'):
                formatted_context += f"   Tools: {msg.get('tools_used', [])}\n"
            formatted_context += "\n"
    stats = context['memory_stats']
    token_usage = context['token_usage']
    formatted_context += f"📊 MEMORY STATS: {stats['short_term_count']} recent, {stats['long_term_count']} historical messages "
    formatted_context += f"({token_usage['total_tokens']}/{token_usage['max_tokens']} tokens used)\n\n"
    enhanced_prompt = f"""
You are an AI assistant with advanced memory capabilities. Use the context below to provide relevant and contextual responses.
{formatted_context}
Current User Input: {user_query}
Provide a helpful, conversational response that leverages both recent conversation context and relevant historical information.
If the user asks for information that requires tools (time, calculations, web search), use the appropriate tool.
"""
    print("\n=== [3] CONTEXT PASSED TO FINAL LLM ===")
    print(enhanced_prompt)
    
    try:
        # Try to get a response from Gemini
        response = gemini_model.generate_content(enhanced_prompt)
        final_response = response.text if hasattr(response, 'text') else str(response)
        print("\n=== [4] FINAL LLM OUTPUT ===")
        print(final_response)
    except Exception as e:
        print(f"\n⚠️ Error calling Gemini API: {str(e)}")
        # Fallback response using the most relevant context
        if 'long_term' in state and state['long_term']:
            # Use the most relevant message from long-term memory
            most_relevant = state['long_term'][0]
            if 'assistant_response' in most_relevant and most_relevant['assistant_response']:
                final_response = most_relevant['assistant_response']
            else:
                final_response = "I'm having trouble accessing my full capabilities right now. " \
                              "Based on our conversation, here's what I know: " \
                              "To make a cake, remember to avoid overmixing the batter, use the correct oven temperature, " \
                              "and don't open the oven door too often. What type of cake would you like to make?"
        else:
            final_response = "I'm having trouble accessing my full capabilities right now. " \
                          "Here's a general tip: When making a cake, be careful not to overmix the batter, " \
                          "use the correct oven temperature, and avoid opening the oven door too often. " \
                          "What type of cake are you interested in making?"
    
    state['assistant_response'] = final_response
    return state

def store_message_node(state):
    thread_id = state['thread_id']
    user_input = state['user_query']
    assistant_response = state['assistant_response']
    timestamp = datetime.now().isoformat()
    message_id = f"{thread_id}:{timestamp}"
    
    message_data = {
        'id': message_id,
        'timestamp': timestamp,
        'thread_id': thread_id,
        'user_input': user_input,
        'assistant_response': assistant_response,
        'tools_used': []
    }
    
    content_str = json.dumps(message_data, ensure_ascii=False, sort_keys=True)
    embedding_np = st_model.encode(content_str)
    embedding_bytes = np.array(embedding_np, dtype=np.float32).tobytes()
    
    data_to_load = {
        "id": message_id,
        "content": content_str,
        "embedding": embedding_bytes
    }
    
    redis_index.load([data_to_load], id_field="id")
    
    redis_client.zadd(f"thread:{thread_id}:messages", {message_id: datetime.now().timestamp()})
    
    return state

# Build the LangGraph
graph = StateGraph(state_schema=AgentState)
graph.add_node("UserInput", user_input_node)
graph.add_node("ShortTermMemory", short_term_memory_node)
graph.add_node("LongTermMemory", long_term_memory_node)
graph.add_node("LLMFieldSelection", llm_field_selection_node)
graph.add_node("ContextConstruction", context_construction_node)
graph.add_node("FinalLLM", final_llm_node)
graph.add_node("StoreMessage", store_message_node)

graph.add_edge("UserInput", "ShortTermMemory")
graph.add_edge("ShortTermMemory", "LongTermMemory")
graph.add_edge("LongTermMemory", "LLMFieldSelection")
graph.add_edge("LLMFieldSelection", "ContextConstruction")
graph.add_edge("ContextConstruction", "FinalLLM")
graph.add_edge("FinalLLM", "StoreMessage")
graph.add_edge("StoreMessage", "UserInput")
graph.set_entry_point("UserInput")

app = graph.compile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangGraph agent in single-turn or continuous mode.")
    parser.add_argument('--single-turn', action='store_true', help='Run the agent for a single user input/response cycle and exit.')
    parser.add_argument('--recreate-index', action='store_true', help='Drop and recreate the Redis vector index before starting.')
    args = parser.parse_args()

    create_vector_index(force_recreate=args.recreate_index)

    session_thread_id = f"conversation_{uuid.uuid4().hex[:8]}"
    print(f"Starting new conversation with Thread ID: {session_thread_id}")
    state = {'thread_id': session_thread_id}

    if args.single_turn:
        state = app.invoke(state, config={"recursion_limit": 100})
        print(f"\n[Agent exited after single turn. Thread ID: {state.get('thread_id', 'N/A')}]")
    else:
        try:
            while True:
                state = app.invoke(state, config={"recursion_limit": 100})
                print(f"\n[Turn complete. Current thread_id: {state.get('thread_id', 'N/A')}]")
        except KeyboardInterrupt:
            print(f"\n[Agent stopped by user. Thread ID: {state.get('thread_id', 'N/A')}]")