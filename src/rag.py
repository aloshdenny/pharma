from pinecone import Pinecone
import os
import json
import logging
from groq import Groq
from decouple import config

_logger = logging.getLogger("rag")

# ==============================
# CONFIG
# ==============================
PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_HOST = config("PINECONE_HOST")
PINECONE_NAMESPACE = config("PINECONE_NAMESPACE")

GROQ_API_KEY = config("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

# ==============================
# INIT CLIENTS
# ==============================
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(host=PINECONE_HOST)

groq_client = Groq(api_key=GROQ_API_KEY)

def pinecone_search(query: str, top_k: int = 3):
    """Semantic search over Pinecone. Returns list of text snippets or [] on failure."""
    try:
        res = pinecone_index.search(
            namespace=PINECONE_NAMESPACE,
            query={
                "inputs": {"text": query},
                "top_k": top_k
            },
            fields=["text", *[]],  # you can include other fields if needed
        )
        hits = res.get("result", {}).get("hits", [])
        return [hit["fields"]["text"] for hit in hits if hit.get("fields", {}).get("text")]
    except KeyError as e:
        _logger.warning("Pinecone response structure unexpected: %s", e)
        return []
    except Exception as e:
        _logger.exception("Pinecone search failed: %s", e)
        raise

# ==============================
# CHAT HISTORY (TEMPORARY)
# ==============================
CHAT_HISTORY = []

def ask_groq_with_context(query: str, max_retries: int = 3):
    global CHAT_HISTORY
    
    # 1. Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "pinecone_search",
                "description": "Search for relevant pharmacy and PBM rejection call records from a vector database. Use this to find context about drug rejections, insurance plans, and PBM policies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant records.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of records to retrieve.",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # Add user message to history
    CHAT_HISTORY.append({"role": "user", "content": query})

    system_message = {
        "role": "system",
        "content": """
        You are a pharmacy call-center AI speaking to patients, doctors, or PBM reps over the phone.

        Your responses must sound like a real human on a call:
        - Speak in short sentences.
        - Do NOT give long explanations.
        - Do NOT structure answers with bullets or paragraphs.
        - Only say what is necessary for the current turn.
        - Ask one question at a time if information is missing.
        - Pause after asking a question.

        Use a calm, professional, conversational tone.

        You have access to a tool called `pinecone_search` that retrieves relevant past PBM rejection call records.
        - Only call the tool if it is clearly needed to answer the current question.
        - Do NOT mention the tool or database to the caller.
        - If relevant records are found, quietly use the logic to respond briefly.
        - If no relevant records are found, or the tool yields nothing, you can try again with a different query.
        - If more information is needed (like Insurance name or Pharmacy ID), ask the caller before calling the tool.

        Never give long medical, insurance, or policy explanations unless explicitly asked.
        Never summarize background reasoning.
        Never speak like documentation or an assistant.
        You are on a live phone call.
        """
    }

    empty_search_retries = 0
    max_total_steps = 10 # Safety limit to prevent infinite loops
    
    for step in range(max_total_steps):
        llm_messages = [system_message] + CHAT_HISTORY

        # Use streaming to handle both text and tool calls
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=llm_messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )

        full_content = ""
        tool_calls_buffer = {} # idx -> {id, name, arguments}

        print("\nAI Assistant: ", end="", flush=True)
        
        for chunk in chat_completion:
            delta = chunk.choices[0].delta
            
            # Handle tool calls in stream
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {"id": tc.id, "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments
            
            # Handle text content in stream
            if delta.content:
                print(delta.content, end="", flush=True)
                full_content += delta.content

        print() # New line after response

        if tool_calls_buffer:
            # Construct the tool_calls for history
            tool_calls = []
            for idx in sorted(tool_calls_buffer.keys()):
                tc = tool_calls_buffer[idx]
                tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    }
                })
            
            # Add assistant message with tool calls to history
            CHAT_HISTORY.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": tool_calls
            })
            
            # Execute tool calls
            has_new_results = False
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                
                if fn_name == "pinecone_search":
                    search_query = fn_args.get("query", "")
                    top_k = fn_args.get("top_k", 3)
                    
                    print(f"--- [Searching Database: \"{search_query}\"] ---")
                    hits = pinecone_search(search_query, top_k=top_k)
                    
                    if hits:
                        has_new_results = True
                        context = "\n---\n".join(hits)
                    else:
                        context = "No relevant records found for this specific query."
                    
                    CHAT_HISTORY.append({
                        "tool_call_id": tc["id"],
                        "role": "tool",
                        "name": fn_name,
                        "content": context,
                    })

            if not has_new_results:
                empty_search_retries += 1
                if empty_search_retries >= max_retries:
                    CHAT_HISTORY.append({
                        "role": "system",
                        "content": "Notice: Multiple searches have yielded no relevant records. Please inform the caller that you cannot find the requested information or ask for different details."
                    })
            
            # Continue the loop to let the model generate text based on tool results
            continue
        else:
            # No tool calls, this was just text. Conversation turn is complete.
            CHAT_HISTORY.append({"role": "assistant", "content": full_content})
            break

if __name__ == "__main__":
    print("--- Pharmacy AI Call Assistant (Session Started) ---")
    print("Type 'exit' to end the session.\n")
    
    while True:
        try:
            user_input = input("Caller: ")
            if not user_input.strip():
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("AI Assistant: Thank you, goodbye.")
                break
            
            ask_groq_with_context(user_input)
        except KeyboardInterrupt:
            print("\nAI Assistant: Session terminated by user.")
            break
        except Exception as e:
            print(f"\n[System Error]: {e}")
            break