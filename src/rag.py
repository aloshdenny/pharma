from pinecone import Pinecone
import os
import json
import logging
from groq import Groq
from decouple import config
from system_prompt import SYSTEM_PROMPT

PINECONE_API_KEY     = config("PINECONE_API_KEY")
PINECONE_HOST        = config("PINECONE_HOST")
PINECONE_NAMESPACE   = config("PINECONE_NAMESPACE")

GROQ_API_KEY  = config("GROQ_API_KEY")
GROQ_MODEL    = "openai/gpt-oss-120b"

pc             = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(host=PINECONE_HOST)
groq_client    = Groq(api_key=GROQ_API_KEY)

def pinecone_search(query: str, top_k: int = 5):
    """Semantic search over Pinecone. Returns list of text snippets or [] on failure."""
    try:
        res = pinecone_index.search(
            namespace=PINECONE_NAMESPACE,
            query={
                "inputs": {"text": query},
                "top_k": top_k,
            },
            fields=["text"],
        )
        hits = res.get("result", {}).get("hits", [])
        return [hit["fields"]["text"] for hit in hits if hit.get("fields", {}).get("text")]
    except KeyError as e:
        _logger.warning("Pinecone response structure unexpected: %s", e)
        return []
    except Exception as e:
        _logger.exception("Pinecone search failed: %s", e)
        raise

CHAT_HISTORY = []

def ask_groq_with_context(query: str, max_retries: int = 3):
    global CHAT_HISTORY

    tools = [
        {
            "type": "function",
            "function": {
                "name": "pinecone_search",
                "description": (
                    "Search the insurance and pharmacy database to retrieve patient records, "
                    "policy details, medication coverage, claim status, denial codes, "
                    "dispensing history, and alternative drug availability."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A natural language query to find the relevant patient or claim record. "
                                "Include identifiers like Emirates ID, policy number, patient name, "
                                "drug name, or claim ID when available."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of records to retrieve. Default is 3.",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    CHAT_HISTORY.append({"role": "user", "content": query})

    system_message = {"role": "system", "content": SYSTEM_PROMPT}

    empty_search_retries = 0
    max_total_steps = 10

    for step in range(max_total_steps):
        llm_messages = [system_message] + CHAT_HISTORY

        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=llm_messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )

        full_content      = ""
        tool_calls_buffer = {}

        print("\nAI Agent: ", end="", flush=True)

        for chunk in chat_completion:
            delta = chunk.choices[0].delta

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments

            if delta.content:
                print(delta.content, end="", flush=True)
                full_content += delta.content

        print()

        if tool_calls_buffer:
            tool_calls = [
                {
                    "id": tool_calls_buffer[idx]["id"],
                    "type": "function",
                    "function": {
                        "name": tool_calls_buffer[idx]["name"],
                        "arguments": tool_calls_buffer[idx]["arguments"],
                    },
                }
                for idx in sorted(tool_calls_buffer)
            ]

            CHAT_HISTORY.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": tool_calls,
            })

            has_new_results = False

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                if fn_name == "pinecone_search":
                    search_query = fn_args.get("query", "")
                    top_k        = fn_args.get("top_k", 3)

                    print(f"  [Database lookup: \"{search_query}\"]")
                    hits = pinecone_search(search_query, top_k=top_k)

                    if hits:
                        has_new_results = True
                        context = "\n---\n".join(hits)
                    else:
                        context = (
                            "No matching record found. "
                            "The patient may not be in the system, or the details provided may be incorrect."
                        )

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
                        "content": (
                            "Notice: Multiple database lookups returned no results. "
                            "Ask the caller to verify their Emirates ID, policy number, or date of birth, "
                            "or let them know the record cannot be located at this time."
                        ),
                    })

            continue

        else:
            CHAT_HISTORY.append({"role": "assistant", "content": full_content})
            break


if __name__ == "__main__":
    print("=" * 55)
    print("  UAE Health Insurance AI Agent — Call Session")
    print("=" * 55)
    print("Type 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("Caller: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nAI Agent: Thank you for calling. Take care, goodbye.")
                break
            ask_groq_with_context(user_input)
        except KeyboardInterrupt:
            print("\nAI Agent: Session ended. Goodbye.")
            break
        except Exception as e:
            print(f"\n[System Error]: {e}")
            break