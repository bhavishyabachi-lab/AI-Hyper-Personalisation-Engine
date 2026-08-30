# AI Hyper-Personalisation Engine — Final Prototype

## What it is
A working Streamlit web application for the Consumer Behavior Working with AI project:
"AI Driven Hyper-Personalisation: Hyper-Personalisation at Scale — A Generative AI Dynamic Content Engine."

## Features
- Single Consumer mode
- Batch / At Scale mode using CSV/Excel
- Actual LLM generation in Live LLM mode
- Demo/offline mode for presentations
- AI persona
- Consumer-behaviour interpretation
- Relevant signal selection
- Communication strategy
- Personalised message + mandatory CTA
- Generic control message
- Quality check
- Prompt architecture
- Respondent-based personalisation-impact analysis

## Run
1. Install Python 3.10+.
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   streamlit run app.py

## Live LLM
Select Live LLM in the sidebar and enter an OpenAI API key. The current default model is GPT-5.6 Luna, which is listed by OpenAI as a cost-sensitive workload model and is available through the Responses API. You can change the model in the sidebar.

## Deployment
The app can be deployed from GitHub to Streamlit Community Cloud.
Do not put the API key inside app.py or commit it to GitHub. Add it as a deployment secret.

## Prototype note
Demo mode contains a small set of benchmark scenarios for presentation. Live LLM mode is the mode to use when demonstrating arbitrary companies/consumers not hard-coded into the app.
