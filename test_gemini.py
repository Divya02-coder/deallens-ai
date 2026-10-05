from ai_engine.llm import GeminiLLM

llm = GeminiLLM()

response = llm.generate(
    "Reply with exactly: DealLens Gemini connection successful."
)

print("\n==============================")
print("GEMINI RESPONSE")
print("==============================")
print(response)