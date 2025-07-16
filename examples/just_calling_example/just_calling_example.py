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

if result.output:
    print(f"\n🎯 MAIN RESULT: {result.output.result}")
    print(f"📝 SUMMARY: {result.output.summary}")
    if result.output.achievements:
        print(f"✅ ACHIEVEMENTS: {', '.join(result.output.achievements)}")
    if result.output.challenges:
        print(f"⚠️  CHALLENGES: {', '.join(result.output.challenges)}")
else:
    print("❌ No output generated")