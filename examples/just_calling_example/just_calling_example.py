from tagent import run_agent

# That's literally all you need to start
result = run_agent(
    goal="Translate 'Hello world' to Chinese",
    model="openrouter/google/gemini-2.5-flash-lite-preview-06-17",
    max_iterations=3
)

print(result.get("raw_data", {}).get("llm_direct_response"))