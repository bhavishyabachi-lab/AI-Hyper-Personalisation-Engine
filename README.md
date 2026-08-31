# AI Hyper-Personalisation Engine — OpenRouter build

Live LLM provider: OpenRouter.
Primary free model: openai/gpt-oss-120b:free.
Fallbacks: meta-llama/llama-3.3-70b-instruct:free and openrouter/free.
Secret: OPENROUTER_API_KEY in Streamlit Secrets.

The app has Single Consumer, Batch / At Scale, Impact Analysis and Prompt Architecture.
If a free model is temporarily unavailable, the app can fall back to another model or
the user can switch to Demo / offline.
