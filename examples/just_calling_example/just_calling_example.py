from tagent import run_agent

# That's literally all you need to start
result = run_agent(
    goal="Translate 'Hello world' to Chinese",
    model="openrouter/google/gemini-2.5-flash",
    max_iterations=3,
    verbose=False,
    tools={}
)

print(f"Goal achieved: {result.goal_achieved}")
print(f"Tasks completed: {result.completed_tasks}")
print(f"Planning cycles: {result.planning_cycles}")

if result.final_output:
    print(f"\n🎯 MAIN RESULT: {result.final_output.result}")
    print(f"📝 SUMMARY: {result.final_output.summary}")
    if result.final_output.achievements:
        print(f"✅ ACHIEVEMENTS: {', '.join(result.final_output.achievements)}")
    if result.final_output.challenges:
        print(f"⚠️  CHALLENGES: {', '.join(result.final_output.challenges)}")
else:
    print("❌ No final output generated")