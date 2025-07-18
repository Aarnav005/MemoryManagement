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
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
redis_client_bin = redis.Redis(host='localhost', port=6379, decode_responses=False)

# LLM setup
gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")
# --- Redis Stack Vector Index Creation ---
def create_vector_index(force_recreate=False):
    try:
        if force_recreate:
            try:
                redis_client_bin.execute_command('FT.DROPINDEX', 'message_index')
                print("✅ Dropped old Redis Stack vector index 'message_index'.")
            except redis.ResponseError as e:
                if 'Unknown Index name' in str(e):
                    pass
                else:
                    print(f"Error dropping vector index: {e}")
        redis_client_bin.execute_command(
            'FT.CREATE', 'message_index',
            'ON', 'HASH',
            'PREFIX', '1', 'message:',
            'SCHEMA',
            'content', 'TEXT',
            'embedding_bin', 'VECTOR', 'FLAT', '6', 'TYPE', 'FLOAT32', 'DIM', '384', 'DISTANCE_METRIC', 'COSINE'
        )
        print("✅ Redis Stack vector index 'message_index' created.")
    except redis.ResponseError as e:
        if 'Index already exists' in str(e):
            if force_recreate:
                print("Index already exists after forced recreation attempt.")
            pass
        else:
            print(f"Error creating vector index: {e}")

def decode_dict_keys_and_values(d):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            key = k.decode() if isinstance(k, bytes) else k
            # Remove binary fields not needed for LLM
            if key == 'embedding_bin':
                continue
            if isinstance(v, bytes):
                try:
                    v = v.decode(errors='replace')
                except Exception:
                    v = str(v)
            out[key] = decode_dict_keys_and_values(v)
        return out
    elif isinstance(d, list):
        return [decode_dict_keys_and_values(i) for i in d]
    else:
        return d

# --- LangGraph Nodes ---
def user_input_node(state):
    user_query = input("You: ").strip()
    state['user_query'] = user_query
    return state

def short_term_memory_node(state):
    # Always use the session's thread_id
    thread_id = state['thread_id']
    short_term_window = 5
    recent_message_ids = redis_client_bin.zrange(f"thread:{thread_id}:messages", -short_term_window, -1)
    short_term_messages = []
    for msg_id in recent_message_ids:
        try:
            hash_data = redis_client_bin.hgetall(f"message:{msg_id}")
            content_json = hash_data.get('content')
            if content_json:
                msg = json.loads(content_json)
                short_term_messages.append(msg)
        except Exception as e:
            print(f"Error retrieving message:{msg_id}: {e}")
    if not recent_message_ids:
        print(f"No recent messages found for thread {thread_id}.")
    state['short_term'] = short_term_messages
    return state

def long_term_memory_node(state):
    query = state['user_query']
    max_long_term_messages = 10
    similarity_threshold = 0.2
    distance_threshold = 1 - similarity_threshold

    query_embedding = np.array(st_model.encode(query), dtype=np.float32)
    query_embedding_bytes = query_embedding.tobytes()
    try:
        # Simplest KNN query. RediSearch will return [DocID, [Field1, Val1, ..., score, ScoreVal]]
        q = f"*=>[KNN {max_long_term_messages} @embedding_bin $vec as score]"
        results = redis_client_bin.execute_command(
            'FT.SEARCH', 'message_index', q,
            'PARAMS', '2', 'vec', query_embedding_bytes,
            'RETURN', '2', 'content', 'score',
            'SORTBY', 'score', 'ASC',
            'DIALECT', '2'
        )

        relevant_messages = []
        # results[0] is the total count.
        # results[1:] are the documents, typically [DocID, [field1, value1, ...]]
        for i in range(1, len(results), 2):
            fields_list = results[i+1] # This is the list like [b'content', b'{...}', b'score', b'0.123']

            msg = {}
            for j in range(0, len(fields_list), 2):
                k = fields_list[j]
                v = fields_list[j+1]

                key = k.decode() if isinstance(k, bytes) else k
                if isinstance(v, bytes):
                    try:
                        v = v.decode(errors='replace')
                    except Exception:
                        v = str(v)
                msg[key] = v

            score_val = msg.get('score')
            # Filter in Python based on the similarity threshold
            if score_val is not None:
                try:
                    current_distance = float(score_val)
                    similarity = 1 - current_distance
                    if similarity >= similarity_threshold:
                        content_json = msg.get('content')
                        if content_json:
                            try:
                                content_dict = json.loads(content_json)
                                content_dict['similarity'] = similarity
                                relevant_messages.append(content_dict)
                            except Exception as e:
                                print(f"Error decoding content JSON: {e}")
                except ValueError:
                    print(f"Warning: Could not convert score_val '{score_val}' to float.")
        state['long_term_candidates'] = relevant_messages
        print("\n=== [1] VECTOR SEARCH: ALL RETRIEVED MESSAGES ===")
        for i, msg in enumerate(relevant_messages, 1):
            print(f"[{i}] {json.dumps(msg, ensure_ascii=False, indent=2)}")
            print()
    except Exception as e:
        state['long_term_candidates'] = []
        print(f"Error during long-term memory retrieval: {e}")
    return state

def llm_field_selection_node(state):
    messages = state['long_term_candidates']
    query = state['user_query']
    if not messages:
        state['long_term'] = []
        return state
    messages_for_prompt = decode_dict_keys_and_values(messages)
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
{json.dumps(messages_for_prompt, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON list of objects, with no extra text.
"""
    try:
        response = gemini_model.generate_content(prompt)
        llm_output = response.text if hasattr(response, 'text') else str(response)
        # Extract JSON from code block if present
        json_str = llm_output
        codeblock_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', llm_output)
        if codeblock_match:
            json_str = codeblock_match.group(1)
        parsed = json.loads(json_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            state['long_term'] = parsed
            # --- PRINT FILTERED MESSAGES (READABLE) ---
            print("\n=== [2] AFTER LLM FIELD SELECTION: FILTERED MESSAGES ===")
            for i, msg in enumerate(parsed, 1):
                print(f"[{i}] {json.dumps(msg, ensure_ascii=False, indent=2)}")
                print()
        else:
            state['long_term'] = []
    except Exception as e:
        state['long_term'] = []
    return state

def context_construction_node(state):
    # Token management: fit context within max_context_tokens
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
    # --- PRINT CONTEXT PASSED TO FINAL LLM ---
    print("\n=== [3] CONTEXT PASSED TO FINAL LLM ===")
    print(enhanced_prompt)
    # --- CALL FINAL LLM ---
    response = gemini_model.generate_content(enhanced_prompt)
    final_response = response.text if hasattr(response, 'text') else str(response)
    # --- PRINT FINAL LLM OUTPUT ---
    print("\n=== [4] FINAL LLM OUTPUT ===")
    print(final_response)
    state['assistant_response'] = final_response
    return state

def store_message_node(state):
    thread_id = state['thread_id']
    user_input = state['user_query']
    assistant_response = state['assistant_response']
    tools_used = []
    timestamp = datetime.now().isoformat()
    message_id = f"{thread_id}:{timestamp}"
    message_data = {
        'id': message_id,
        'timestamp': timestamp,
        'thread_id': thread_id,
        'user_input': user_input,
        'assistant_response': assistant_response,
        'tools_used': tools_used
    }
    content_str = json.dumps(message_data, ensure_ascii=False, sort_keys=True)
    embedding = st_model.encode(content_str).tolist()
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    redis_client_bin.hset(f"message:{message_id}", mapping={
        'content': content_str,
        'embedding_bin': embedding_bytes
    })
    redis_client_bin.zadd(f"thread:{thread_id}:messages", {message_id: datetime.now().timestamp()})
    current_ids = redis_client_bin.zrange(f"thread:{thread_id}:messages", 0, -1)
    return state

# --- Build the LangGraph ---
create_vector_index()
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
graph.add_edge("StoreMessage", "UserInput")  # Loop for next turn
graph.set_entry_point("UserInput")  # Set entry point

app = graph.compile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangGraph agent in single-turn or continuous mode.")
    parser.add_argument('--single-turn', action='store_true', help='Run the agent for a single user input/response cycle and exit.')
    parser.add_argument('--recreate-index', action='store_true', help='Drop and recreate the Redis vector index before starting.')
    args = parser.parse_args()

    # Only recreate the index if requested
    create_vector_index(force_recreate=args.recreate_index)

    # Generate a session-unique thread_id
    session_thread_id = f"conversation_{uuid.uuid4().hex[:8]}"
    state = {'thread_id': session_thread_id}
    if args.single_turn:
        state = app.invoke(state)
        print(f"[Agent exited after single turn. Thread ID: {state.get('thread_id', 'N/A')}]" )
    else:
        while True:
            state = app.invoke(state)
            print(f"[Current thread_id: {state.get('thread_id', 'N/A')}]" ) 