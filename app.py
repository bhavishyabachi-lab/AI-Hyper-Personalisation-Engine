import os
import json
import io
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="AI Hyper-Personalisation Engine",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
:root {
  --ink:#172033; --muted:#68748a; --line:#e5e9f2;
  --accent:#5b5ce2; --accent-soft:#efefff; --good:#eef8f1;
}
.block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem;}
.hero {padding: 28px 30px; border-radius: 18px; background: linear-gradient(135deg,#20234a,#5b5ce2); color:white; margin-bottom: 20px;}
.hero h1 {margin:0 0 6px 0; font-size: 2rem;}
.hero p {margin:0; opacity:.88; font-size:1rem;}
.section-card {padding:18px 18px 8px 18px; border:1px solid var(--line); border-radius:16px; background:#fff; margin-bottom:16px;}
.helper {color:var(--muted); font-size:.82rem; line-height:1.45;}
.output-card {border:1px solid var(--line); border-radius:16px; padding:18px; background:#fff; height:100%;}
.message-card {border:1px solid #d8d8f7; border-radius:16px; padding:20px; background:linear-gradient(135deg,#f8f8ff,#ffffff);}
.generic-card {border:1px solid var(--line); border-radius:16px; padding:20px; background:#fafbfe;}
.signal {display:inline-block; padding:6px 10px; border-radius:999px; margin:3px 4px 3px 0; font-size:.78rem; background:var(--accent-soft); color:#4b4dc0;}
.excluded {background:#f3f4f7; color:#6c7483;}
.strategy-row {padding:11px 0; border-bottom:1px solid var(--line);}
.strategy-key {color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.05em;}
.strategy-val {font-weight:700;}
.small-note {font-size:.75rem; color:var(--muted);}
div[data-testid="stMetric"] {border:1px solid var(--line); padding:10px; border-radius:12px; background:#fff;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants
# -----------------------------
MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

CHANNELS = ["WhatsApp", "SMS", "Email", "Push notification", "Social media", "Website", "In-app message"]
LIFECYCLES = ["Prospect", "New customer", "Active customer", "Loyal customer", "At-risk", "Inactive", "Churned / potentially churned"]
OBJECTIVES = ["Awareness", "Engagement", "Consideration", "Conversion", "Re-engagement", "Retention", "Feedback", "Win-back", "Cross-sell", "Upsell", "Loyalty", "Reminder / completion"]
MESSAGE_TYPES = [
    "AI chooses",
    "Promotional", "Awareness", "Consideration", "Product recommendation",
    "Behaviour-triggered", "Abandonment", "Re-engagement", "Retention",
    "Loyalty / reward", "Cross-sell", "Upsell", "Feedback", "Service / recovery",
    "Transactional / reminder", "Win-back", "Educational", "Occasion-based"
]
TONES = ["AI chooses", "Conversational", "Friendly", "Quirky", "Humorous", "Premium", "Emotional", "Urgent", "Empathetic", "Professional", "Playful", "Informative"]
LENGTHS = ["Very short", "Short", "Medium", "Long"]

SYSTEM_PROMPT = """
You are the core intelligence of a generic AI Hyper-Personalisation Engine for
consumer and marketing communication.

The engine must work across industries, companies, channels and consumer situations.
Your job is NOT simply to insert personal data into a template. You must interpret
the consumer situation and dynamically adapt the communication.

PRIORITY ORDER
1. Situation and lifecycle appropriateness
2. Consumer relevance
3. Brand authenticity
4. Clear communication objective
5. Creativity, only when appropriate
6. Actionability

CONSUMER-BEHAVIOUR PRINCIPLES
- Use needs, motivations, behaviour, preferences, lifecycle and context when relevant.
- Demographics are secondary signals unless they materially affect relevance.
- Do not invent facts about the consumer.
- Treat inferred motivations as hypotheses, not facts.
- Use only the minimum relevant signals needed for a natural message.
- Do not reveal raw tracking data unnecessarily.
- More personalisation is not automatically better; avoid intrusive wording.

COMMUNICATION PRINCIPLES
- Determine the most appropriate communication type from lifecycle + trigger + objective.
- Match tone to situation; do not make sensitive/transactional situations artificially quirky.
- Match length and style to the specified channel.
- Keep brand positioning consistent.
- The personalised message must contain a clear CTA.
- The CTA must align with the objective and situation.
- The generic message must not use consumer-specific information.
- Personalised and generic messages must be meaningfully distinguishable.

OUTPUT
Return only valid JSON matching the supplied schema.
Do not provide hidden chain-of-thought. Provide concise user-facing summaries only.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "persona": {"type": "string"},
        "consumer_insight": {"type": "string"},
        "signals_used": {"type": "array", "items": {"type": "string"}},
        "signals_excluded": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "signal": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["signal", "reason"]
            }
        },
        "strategy": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "message_type": {"type": "string"},
                "objective": {"type": "string"},
                "primary_appeal": {"type": "string"},
                "tone": {"type": "string"},
                "personalisation_level": {"type": "string"},
                "key_value_proposition": {"type": "string"},
                "cta": {"type": "string"}
            },
            "required": ["message_type", "objective", "primary_appeal", "tone", "personalisation_level", "key_value_proposition", "cta"]
        },
        "personalised_message": {"type": "string"},
        "generic_message": {"type": "string"},
        "quality_check": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relevance": {"type": "string"},
                "brand_fit": {"type": "string"},
                "creativity": {"type": "string"},
                "cta_fit": {"type": "string"},
                "intrusiveness_risk": {"type": "string"}
            },
            "required": ["relevance", "brand_fit", "creativity", "cta_fit", "intrusiveness_risk"]
        }
    },
    "required": ["persona", "consumer_insight", "signals_used", "signals_excluded", "strategy", "personalised_message", "generic_message", "quality_check"]
}

DEMO_SCENARIOS: Dict[str, Dict[str, str]] = {
    "Practo — app uninstall / feedback": {
        "company":"Practo","industry":"Healthcare / health-tech","product":"Practo mobile app",
        "positioning":"Convenient digital healthcare access","segment":"Existing app user",
        "age":"","occupation":"","location":"India",
        "behaviour":"Previously used the app and then uninstalled it",
        "preferences":"","motivation":"","price_sensitivity":"Unknown",
        "lifecycle_stage":"Churned / potentially churned","trigger":"App uninstallation",
        "context":"Shortly after uninstall","objective":"Feedback","channel":"WhatsApp",
        "message_type":"AI chooses","tone":"Empathetic","length":"Short"
    },
    "Domino's — Raksha Bandhan / promotion": {
        "company":"Domino's Pizza India","industry":"Food & beverage","product":"Pizza",
        "positioning":"Convenient, playful and value-focused","segment":"Occasion-oriented household",
        "age":"","occupation":"","location":"India",
        "behaviour":"","preferences":"Pizza; shared meals","motivation":"Celebrate Raksha Bandhan together",
        "price_sensitivity":"Medium","lifecycle_stage":"Active customer","trigger":"Raksha Bandhan",
        "context":"Festival occasion","objective":"Conversion","channel":"WhatsApp",
        "message_type":"AI chooses","tone":"Quirky","length":"Short"
    },
    "Myntra — repeated sneaker browsing": {
        "company":"Myntra","industry":"Fashion e-commerce","product":"Running shoes",
        "positioning":"Trendy, youthful and accessible","segment":"Budget-conscious fitness consumer",
        "age":"24","occupation":"Young professional","location":"Bengaluru",
        "behaviour":"Frequently browses running shoes and compares prices",
        "preferences":"Running; minimalist designs","motivation":"Fitness plus value for money",
        "price_sensitivity":"High","lifecycle_stage":"Active customer","trigger":"Viewed running shoes twice recently",
        "context":"Weekend","objective":"Conversion","channel":"WhatsApp",
        "message_type":"AI chooses","tone":"Quirky","length":"Short"
    },
    "CRED — payment reminder": {
        "company":"CRED","industry":"Fintech","product":"Credit card bill payment",
        "positioning":"Premium, frictionless financial utility","segment":"Existing credit-card user",
        "age":"","occupation":"","location":"India",
        "behaviour":"Has an outstanding credit-card balance","preferences":"","motivation":"Avoid interest charges and complete payment",
        "price_sensitivity":"","lifecycle_stage":"Active customer","trigger":"Credit-card payment due tomorrow",
        "context":"Payment deadline","objective":"Reminder / completion","channel":"WhatsApp",
        "message_type":"AI chooses","tone":"Professional","length":"Short"
    },
    "Agoda — communications feedback": {
        "company":"Agoda","industry":"Travel & hospitality","product":"Travel booking platform",
        "positioning":"Helpful and traveler-focused","segment":"Existing traveler",
        "age":"","occupation":"","location":"India",
        "behaviour":"Has interacted with Agoda communications","preferences":"Travel",
        "motivation":"Wants more relevant travel communication","price_sensitivity":"",
        "lifecycle_stage":"Active customer","trigger":"Feedback request",
        "context":"Post-interaction","objective":"Feedback","channel":"Email",
        "message_type":"AI chooses","tone":"Friendly","length":"Medium"
    }
}

def compose_user_input(data: Dict[str, Any]) -> str:
    def val(x): return str(x).strip() if x is not None and str(x).strip() else "Not provided"
    return f"""
BUSINESS
Company: {val(data.get('company'))}
Industry: {val(data.get('industry'))}
Product / Service: {val(data.get('product'))}
Brand positioning: {val(data.get('positioning'))}

CONSUMER
Segment / Persona: {val(data.get('segment'))}
Age: {val(data.get('age'))}
Occupation: {val(data.get('occupation'))}
Geography: {val(data.get('location'))}
Behaviour / previous interactions: {val(data.get('behaviour'))}
Preferences / interests: {val(data.get('preferences'))}
Needs / motivation: {val(data.get('motivation'))}
Price sensitivity: {val(data.get('price_sensitivity'))}

CONSUMER STATE
Lifecycle stage: {val(data.get('lifecycle_stage'))}
Trigger / recent event: {val(data.get('trigger'))}
Current context / occasion: {val(data.get('context'))}

CAMPAIGN
Objective: {val(data.get('objective'))}
Channel: {val(data.get('channel'))}
Requested message type: {val(data.get('message_type'))}
Requested tone/style: {val(data.get('tone'))}
Requested length: {val(data.get('length'))}

INSTRUCTIONS FOR THIS CASE
- First interpret the consumer and situation.
- Select only relevant signals.
- If message type is "AI chooses", choose it from the situation.
- If tone is "AI chooses", select an appropriate tone.
- Make the final message sound realistically sendable by the stated company/industry.
- Include a clear CTA inside the personalised message.
- Do not reveal tracking details more explicitly than necessary.
- Keep the generic version consumer-neutral.
"""

def make_client(api_key: str | None):
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def llm_generate(data: Dict[str, Any], client: OpenAI, model: str) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=compose_user_input(data),
        text={
            "format": {
                "type": "json_schema",
                "name": "hyper_personalisation_output",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)

def enforce_cta(result: Dict[str, Any]) -> Dict[str, Any]:
    # Guardrail: if the LLM forgot a CTA, append the generated CTA rather than leaving it absent.
    msg = result.get("personalised_message", "").strip()
    cta = result.get("strategy", {}).get("cta", "").strip()
    if cta and cta.lower() not in msg.lower():
        result["personalised_message"] = msg.rstrip() + f"\n\n👉 {cta}"
    return result

def demo_generate(data: Dict[str, Any]) -> Dict[str, Any]:
    company = data["company"].lower()
    trigger = data["trigger"].lower()
    objective = data["objective"].lower()
    channel = data["channel"]
    tone = data["tone"]

    if "practo" in company and "uninstall" in trigger:
        return {
            "persona":"Recently Churned Healthcare-App User",
            "consumer_insight":"The known signal is disengagement after an app uninstall. The reason for leaving is not known, so it is not assumed.",
            "signals_used":["Existing customer relationship","App uninstallation","Churned lifecycle stage","Feedback objective","WhatsApp channel"],
            "signals_excluded":[
                {"signal":"Age","reason":"Not provided and not necessary for a feedback request"},
                {"signal":"Occupation","reason":"Not relevant to the current communication objective"},
                {"signal":"Reason for uninstalling","reason":"Unknown; the engine must not invent it"}
            ],
            "strategy":{"message_type":"Feedback / recovery","objective":"Feedback","primary_appeal":"Help improve the experience",
                        "tone":"Friendly and empathetic","personalisation_level":"Behavioural + lifecycle + event-triggered",
                        "key_value_proposition":"The consumer's feedback can help improve the experience","cta":"Share your feedback"},
            "personalised_message":"Bhavishya Paila, can we ask you one quick question? 👀\n\nWe noticed you're no longer using the Practo App 📲 We'd really like to know what we could have done better.\n\n👉 Share your feedback",
            "generic_message":"We'd love your feedback on your experience with our app. Please share your thoughts with us.",
            "quality_check":{"relevance":"High","brand_fit":"High","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}
        }

    if "domino" in company:
        return {
            "persona":"Occasion-Oriented Pizza Sharer",
            "consumer_insight":"The immediate festival occasion creates a social, celebratory consumption context where togetherness and a simple offer can be more relevant than personal demographics.",
            "signals_used":["Raksha Bandhan occasion","Shared-meal context","Conversion objective","Quirky tone","WhatsApp"],
            "signals_excluded":[{"signal":"Age","reason":"No need to segment the family occasion by age in this message"},{"signal":"Occupation","reason":"Not relevant to the occasion"}],
            "strategy":{"message_type":"Occasion-based promotion","objective":"Conversion","primary_appeal":"Celebration + value",
                        "tone":"Quirky and festive","personalisation_level":"Contextual + occasion-based",
                        "key_value_proposition":"A simple offer makes a shared celebration more rewarding","cta":"Order now"},
            "personalised_message":"Raksha Bandhan plans? 🍕 No sibling fights over the last slice today. Celebrate together with a little extra pizza on the table.\n\n👉 Order now",
            "generic_message":"Enjoy great pizza deals today. Order now.",
            "quality_check":{"relevance":"High","brand_fit":"High","creativity":"High","cta_fit":"High","intrusiveness_risk":"Low"}
        }

    if "cred" in company and "due" in trigger:
        return {
            "persona":"Deadline-Driven Credit Card User",
            "consumer_insight":"The immediate payment deadline makes clarity, urgency and frictionless action more important than entertainment.",
            "signals_used":["Payment due trigger","Existing customer relationship","Reminder objective","WhatsApp channel"],
            "signals_excluded":[{"signal":"Age","reason":"Not relevant to a payment reminder"},{"signal":"Occupation","reason":"Not relevant to the action required"}],
            "strategy":{"message_type":"Transactional / reminder","objective":"Reminder / completion","primary_appeal":"Avoid unnecessary interest charges",
                        "tone":"Clear and professional","personalisation_level":"Transactional + contextual",
                        "key_value_proposition":"Complete the outstanding payment before the deadline","cta":"Pay now"},
            "personalised_message":"Your credit-card payment is due tomorrow. Complete your remaining payment to avoid interest charges on the outstanding amount.\n\n👉 Pay now",
            "generic_message":"Your credit-card payment is due soon. Please complete your payment.",
            "quality_check":{"relevance":"High","brand_fit":"High","creativity":"Low","cta_fit":"High","intrusiveness_risk":"Low"}
        }

    if "agoda" in company:
        return {
            "persona":"Travel-Engaged Feedback Seeker",
            "consumer_insight":"The consumer has interacted with travel communications and the stated objective is to improve relevance, so an empathetic feedback request is more appropriate than a promotional message.",
            "signals_used":["Travel interest","Existing customer relationship","Feedback objective","Email channel"],
            "signals_excluded":[{"signal":"Age","reason":"Not relevant to the feedback request"},{"signal":"Location","reason":"No location-specific content is required"}],
            "strategy":{"message_type":"Feedback / relationship","objective":"Feedback","primary_appeal":"Help tailor future communications",
                        "tone":"Friendly and appreciative","personalisation_level":"Relationship + preference-based",
                        "key_value_proposition":"Feedback can improve future travel communication","cta":"Start the survey"},
            "personalised_message":"Dear Bhavishya,\n\nWe'd love to make the travel messages you receive more useful and relevant to you. Take a minute to tell us what you'd like to see more of.\n\n👉 Start the survey",
            "generic_message":"We'd appreciate your feedback on our travel communications. Please take a short survey.",
            "quality_check":{"relevance":"High","brand_fit":"High","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}
        }

    # Generic demo fallback for arbitrary scenarios
    motivation = data.get("motivation","").lower()
    price = data.get("price_sensitivity","")
    behaviour = data.get("behaviour","")
    mtype = data.get("message_type","")
    if price == "High" or "price" in motivation or "saving" in motivation:
        appeal = "Value / affordability"
    elif "convenience" in motivation or "time" in motivation:
        appeal = "Convenience / time saving"
    elif "style" in motivation or "status" in motivation or "premium" in motivation:
        appeal = "Style / experience"
    else:
        appeal = "Relevant product benefit"

    if "feedback" in objective:
        cta = "Share your feedback"
        inferred_type = "Feedback"
    elif "retention" in objective:
        cta = "Keep exploring"
        inferred_type = "Retention"
    elif "re-engagement" in objective or data.get("lifecycle_stage","").lower() in ("inactive","at-risk"):
        cta = "Take another look"
        inferred_type = "Re-engagement"
    elif "conversion" in objective:
        cta = "Check it out now"
        inferred_type = "Promotional"
    elif "cross" in objective:
        cta = "Discover more"
        inferred_type = "Cross-sell"
    else:
        cta = "Learn more"
        inferred_type = "Awareness"

    inferred_tone = tone if tone != "AI chooses" else ("Quirky and conversational" if "quirky" in (tone+" "+str(behaviour)).lower() else "Clear and conversational")
    result = {
        "persona": data.get("segment") or "Context-Aware Consumer",
        "consumer_insight":"The supplied behaviour, motivation and lifecycle information is used to adapt the communication while avoiding unnecessary consumer detail.",
        "signals_used":[x for x in ["Behaviour","Motivation","Lifecycle stage","Trigger","Context","Price sensitivity"] if data.get({"Behaviour":"behaviour","Motivation":"motivation","Lifecycle stage":"lifecycle_stage","Trigger":"trigger","Context":"context","Price sensitivity":"price_sensitivity"}[x])],
        "signals_excluded":[
            {"signal":"Age","reason":"Not needed unless age materially changes relevance"},
            {"signal":"Occupation","reason":"Not needed unless occupation changes the communication context"}
        ],
        "strategy":{"message_type":mtype if mtype and mtype!="AI chooses" else inferred_type,"objective":data["objective"],
                    "primary_appeal":appeal,"tone":inferred_tone,"personalisation_level":"Behavioural + motivational + contextual",
                    "key_value_proposition":"A benefit aligned with the consumer's stated context","cta":cta},
        "personalised_message":f"{'Still thinking about ' + data['product'] + '? 👀' if behaviour else 'Here’s something that may fit what you’re looking for.'}\n\n{('Find an option that fits your needs and your budget.' if 'Value' in appeal else 'Make your next choice a little easier.' if 'Convenience' in appeal else 'Discover something that fits your preferences.')}\n\n👉 {cta}",
        "generic_message":f"Explore {data['product']} and see what’s available.\n\n👉 {cta}",
        "quality_check":{"relevance":"Medium","brand_fit":"Medium","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}
    }
    return result

# -----------------------------
# Sidebar / session
# -----------------------------
st.markdown("""
<div class="hero">
  <h1>✨ AI Hyper-Personalisation Engine</h1>
  <p>Hyper-personalisation at scale — dynamically adapt marketing communication to the consumer, situation and campaign objective.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Engine controls")
    mode = st.radio("Generation mode", ["Live LLM", "Demo / offline"], index=0)
    api_key = st.text_input("OpenAI API key", type="password", help="Used only for the current session and not displayed.")
    model = st.text_input("LLM model", value=MODEL_DEFAULT)
    st.caption("Live mode uses an OpenAI Responses API model with structured JSON output. Demo mode is available for presentations without an API connection.")

if mode == "Live LLM" and not api_key:
    st.warning("Live LLM mode needs an OpenAI API key. For presentation testing, switch the sidebar to Demo / offline.")

tabs = st.tabs(["Single Consumer", "Batch / At Scale", "Impact Analysis", "Prompt Architecture"])

# -----------------------------
# Single Consumer
# -----------------------------
with tabs[0]:
    st.subheader("Create a personalised marketing message")
    st.markdown('<div class="helper">Fill the fields you know. Optional fields can be left blank; the engine must never invent missing consumer facts.</div>', unsafe_allow_html=True)

    demo_name = st.selectbox("Load an example", ["None"] + list(DEMO_SCENARIOS.keys()))
    if demo_name != "None":
        defaults = DEMO_SCENARIOS[demo_name]
    else:
        defaults = {}

    with st.expander("1. Business context", expanded=True):
        c1,c2 = st.columns(2)
        company = c1.text_input("Company *", value=defaults.get("company",""), help="Brand/company whose communication is being created. Example: Myntra, Practo, Nike.")
        industry = c2.text_input("Industry *", value=defaults.get("industry",""), help="Business category. Example: fashion e-commerce, healthcare, food delivery.")
        c1,c2 = st.columns(2)
        product = c1.text_input("Product / Service *", value=defaults.get("product",""), help="Specific product, service, app, subscription or offer being communicated.")
        positioning = c2.text_input("Brand positioning", value=defaults.get("positioning",""), help="How the brand wants to be perceived, e.g. premium, playful, affordable. Optional.")

    with st.expander("2. Consumer context", expanded=True):
        segment = st.text_input("Consumer segment / persona *", value=defaults.get("segment",""), help="A segment label or short description. Example: budget-conscious fitness consumer. If enough information is supplied, the AI can structure the persona.")
        c1,c2,c3 = st.columns(3)
        age = c1.text_input("Age", value=defaults.get("age",""), help="Approximate age or age group. Do not enter sensitive personal details.")
        occupation = c2.text_input("Occupation", value=defaults.get("occupation",""), help="Life-stage/occupation if it matters to the context. Example: student, working professional.")
        location = c3.text_input("Geography", value=defaults.get("location",""), help="City/region/country. The AI should only use it when it adds meaningful relevance.")
        behaviour = st.text_area("Behaviour / previous interactions", value=defaults.get("behaviour",""), help="What the consumer has done: viewed product, abandoned cart, purchased repeatedly, stopped using app, etc.")
        preferences = st.text_area("Preferences / interests", value=defaults.get("preferences",""), help="Interests or preferences: running, skincare, sustainable products, travel, etc.")
        motivation = st.text_area("Needs / motivation", value=defaults.get("motivation",""), help="Why the consumer may want the product: convenience, saving money, quality, status, fitness, enjoyment, etc.")
        price = st.selectbox("Price sensitivity", ["Unknown","Low","Medium","High"], index=["Unknown","Low","Medium","High"].index(defaults.get("price_sensitivity","Unknown")) if defaults.get("price_sensitivity","Unknown") in ["Unknown","Low","Medium","High"] else 0)

    with st.expander("3. Consumer state", expanded=True):
        lifecycle = st.selectbox("Lifecycle stage *", LIFECYCLES, index=LIFECYCLES.index(defaults.get("lifecycle_stage","Active customer")) if defaults.get("lifecycle_stage","Active customer") in LIFECYCLES else 2, help="Where the consumer is in the relationship with the brand.")
        trigger = st.text_input("Trigger / recent event *", value=defaults.get("trigger",""), help="What happened that explains why communication is occurring now.")
        context = st.text_input("Current context / occasion", value=defaults.get("context",""), help="Situational context such as festival, weekend, after work, weather, travel season.")

    with st.expander("4. Campaign & communication", expanded=True):
        c1,c2 = st.columns(2)
        objective = c1.selectbox("Marketing objective *", OBJECTIVES, index=OBJECTIVES.index(defaults.get("objective","Conversion")) if defaults.get("objective","Conversion") in OBJECTIVES else 3, help="What should the consumer ideally do after receiving the message?")
        channel = c2.selectbox("Channel *", CHANNELS, index=CHANNELS.index(defaults.get("channel","WhatsApp")) if defaults.get("channel","WhatsApp") in CHANNELS else 0, help="Where the message is delivered. The writing must adapt to the channel.")
        c1,c2 = st.columns(2)
        mtype = c1.selectbox("Message type", MESSAGE_TYPES, index=MESSAGE_TYPES.index(defaults.get("message_type","AI chooses")) if defaults.get("message_type","AI chooses") in MESSAGE_TYPES else 0, help="Communication job. Choose AI chooses to let the engine infer it from lifecycle + trigger + objective.")
        tone = c2.selectbox("Tone / style", TONES, index=TONES.index(defaults.get("tone","AI chooses")) if defaults.get("tone","AI chooses") in TONES else 0, help="Use creativity only when it fits the situation.")
        length = st.selectbox("Message length", LENGTHS, index=LENGTHS.index(defaults.get("length","Short")) if defaults.get("length","Short") in LENGTHS else 1)

    input_data = {
        "company":company,"industry":industry,"product":product,"positioning":positioning,
        "segment":segment,"age":age,"occupation":occupation,"location":location,
        "behaviour":behaviour,"preferences":preferences,"motivation":motivation,
        "price_sensitivity":price,"lifecycle_stage":lifecycle,"trigger":trigger,"context":context,
        "objective":objective,"channel":channel,"message_type":mtype,"tone":tone,"length":length
    }

    required_missing = [label for label, value in [
        ("Company", company), ("Industry", industry), ("Product / Service", product),
        ("Consumer Segment / Persona", segment), ("Trigger / Recent Event", trigger),
        ("Marketing Objective", objective)
    ] if not str(value).strip()]

    st.divider()
    if st.button("🚀 Generate personalised content", type="primary", use_container_width=True):
        if required_missing:
            st.error("Please fill: " + ", ".join(required_missing))
        elif mode == "Live LLM" and not api_key:
            st.error("Add your OpenAI API key in the sidebar, or switch to Demo / offline mode.")
        else:
            try:
                with st.spinner("The engine is interpreting the consumer, selecting relevant signals, choosing the communication strategy and generating content..."):
                    if mode == "Live LLM":
                        client = make_client(api_key)
                        result = llm_generate(input_data, client, model)
                    else:
                        result = demo_generate(input_data)
                    result = enforce_cta(result)
                st.session_state["single_result"] = result
                st.session_state["single_input"] = input_data
                st.success("Generated successfully.")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    result = st.session_state.get("single_result")
    if result:
        st.divider()
        st.subheader("AI output")
        c1,c2 = st.columns([1.3,1])
        with c1:
            st.markdown("### AI Consumer Persona")
            st.markdown(f"**{result['persona']}**")
            st.write(result["consumer_insight"])
        with c2:
            st.markdown("### Quality Check")
            q = result["quality_check"]
            for k,v in q.items():
                st.write(f"**{k.replace('_',' ').title()}:** {v}")

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("### Personalisation signals used")
            for s in result["signals_used"]:
                st.markdown(f'<span class="signal">✓ {s}</span>', unsafe_allow_html=True)
        with c2:
            st.markdown("### Signals excluded / not used")
            for x in result["signals_excluded"]:
                st.markdown(f'<span class="signal excluded">{x["signal"]}</span> <span class="small-note">{x["reason"]}</span><br>', unsafe_allow_html=True)

        st.markdown("### Communication strategy")
        s = result["strategy"]
        scols = st.columns(4)
        for col, (key, label) in zip(scols, [
            ("message_type","Message type"),("objective","Objective"),("primary_appeal","Primary appeal"),("cta","CTA")
        ]):
            col.metric(label, s[key])
        c1,c2,c3 = st.columns(3)
        c1.metric("Tone", s["tone"])
        c2.metric("Personalisation", s["personalisation_level"])
        c3.metric("Key value proposition", s["key_value_proposition"])

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("### ✨ AI-Personalised Message")
            st.markdown(f'<div class="message-card"><div style="font-size:1.02rem;line-height:1.6;white-space:pre-wrap">{result["personalised_message"]}</div></div>', unsafe_allow_html=True)
            st.button("Generate another version", key="another_version")
        with c2:
            st.markdown("### Generic Control")
            st.markdown(f'<div class="generic-card"><div style="font-size:1.02rem;line-height:1.6;white-space:pre-wrap">{result["generic_message"]}</div></div>', unsafe_allow_html=True)
        st.markdown("#### Why this message is personalised")
        st.write("The personalised version is conditioned on the consumer and situation signals identified above; the generic control intentionally excludes consumer-specific information.")

# -----------------------------
# Batch
# -----------------------------
with tabs[1]:
    st.subheader("Batch / At Scale")
    st.markdown('<div class="helper">Upload a CSV or Excel file using the same fields as Single Consumer Mode. The same engine is applied to each row.</div>', unsafe_allow_html=True)

    template_cols = list(input_data.keys())
    template = pd.DataFrame([DEMO_SCENARIOS["Myntra — repeated sneaker browsing"]])[template_cols]
    st.download_button("Download sample CSV template", template.to_csv(index=False).encode("utf-8"), "hyper_personalisation_template.csv", "text/csv")

    upload = st.file_uploader("Upload consumer dataset", type=["csv","xlsx","xls"])
    batch_df = None
    if upload:
        try:
            if upload.name.lower().endswith(".csv"):
                batch_df = pd.read_csv(upload)
            else:
                batch_df = pd.read_excel(upload)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")

    if batch_df is not None:
        st.write(f"**{len(batch_df)} rows loaded.**")
        st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)
        if len(batch_df) > 100:
            st.warning("For this prototype, batch processing is capped at 100 rows.")
            batch_df = batch_df.head(100)

        if st.button("⚡ Generate for all consumers", type="primary", use_container_width=True):
            if mode == "Live LLM" and not api_key:
                st.error("Add your OpenAI API key in the sidebar for live batch generation.")
            else:
                outputs = []
                progress = st.progress(0)
                for idx, row in batch_df.fillna("").iterrows():
                    data = {k: str(row.get(k,"")) for k in template_cols}
                    # Default shared fields when blank
                    data["lifecycle_stage"] = data["lifecycle_stage"] or "Active customer"
                    data["objective"] = data["objective"] or "Conversion"
                    data["channel"] = data["channel"] or "WhatsApp"
                    data["message_type"] = data["message_type"] or "AI chooses"
                    data["tone"] = data["tone"] or "AI chooses"
                    data["length"] = data["length"] or "Short"
                    try:
                        if mode == "Live LLM":
                            res = llm_generate(data, make_client(api_key), model)
                        else:
                            res = demo_generate(data)
                        res = enforce_cta(res)
                        outputs.append({
                            "row": idx+1,
                            "company": data["company"],
                            "product": data["product"],
                            "persona": res["persona"],
                            "lifecycle_stage": data["lifecycle_stage"],
                            "trigger": data["trigger"],
                            "message_type": res["strategy"]["message_type"],
                            "cta": res["strategy"]["cta"],
                            "personalised_message": res["personalised_message"],
                            "generic_message": res["generic_message"],
                        })
                    except Exception as exc:
                        outputs.append({"row":idx+1,"company":data["company"],"product":data["product"],"persona":"ERROR","lifecycle_stage":data["lifecycle_stage"],"trigger":data["trigger"],"message_type":"ERROR","cta":str(exc),"personalised_message":"","generic_message":""})
                    progress.progress((len(outputs))/len(batch_df))
                st.session_state["batch_outputs"] = pd.DataFrame(outputs)
                st.success(f"Processed {len(outputs)} consumers.")

    batch_outputs = st.session_state.get("batch_outputs")
    if batch_outputs is not None:
        st.subheader("Batch results")
        st.dataframe(batch_outputs, use_container_width=True, hide_index=True)
        st.download_button("Download generated results", batch_outputs.to_csv(index=False).encode("utf-8"), "hyper_personalisation_results.csv", "text/csv")

# -----------------------------
# Impact Analysis
# -----------------------------
with tabs[2]:
    st.subheader("Personalisation Impact Analysis")
    st.markdown('<div class="helper">Rate the generic and personalised versions with actual respondents. The app calculates averages; it does not fabricate research findings.</div>', unsafe_allow_html=True)

    if "impact_rows" not in st.session_state:
        st.session_state["impact_rows"] = []

    st.markdown("### Add a message pair")
    current = st.session_state.get("single_result")
    if current:
        st.info("A message pair from Single Consumer Mode is available. You can add it to the study dataset.")
        if st.button("Add current pair"):
            pair = {
                "case": st.session_state["single_input"]["company"] + " — " + st.session_state["single_input"]["product"],
                "personalised": current["personalised_message"],
                "generic": current["generic_message"]
            }
            st.session_state["impact_rows"].append(pair)

    if st.session_state["impact_rows"]:
        imp = pd.DataFrame(st.session_state["impact_rows"])
        st.dataframe(imp, use_container_width=True, hide_index=True)

        st.markdown("### Enter respondent ratings")
        st.caption("1 = very low, 5 = very high. Intrusiveness should be interpreted in the opposite direction: lower is generally better.")
        dims = ["Relevance","Perceived personalisation","Persuasiveness","Emotional appeal","Purchase / engagement intention","Brand authenticity","Intrusiveness"]
        all_scores = []
        for i, pair in enumerate(st.session_state["impact_rows"]):
            with st.expander(f"{i+1}. {pair['case']}", expanded=False):
                st.markdown("**Personalised**")
                p_scores = {d: st.slider(f"P — {d}",1,5,3,key=f"p_{i}_{d}") for d in dims}
                st.markdown("**Generic**")
                g_scores = {d: st.slider(f"G — {d}",1,5,3,key=f"g_{i}_{d}") for d in dims}
                for d in dims:
                    all_scores.append({"case":pair["case"],"version":"Personalised","dimension":d,"score":p_scores[d]})
                    all_scores.append({"case":pair["case"],"version":"Generic","dimension":d,"score":g_scores[d]})

        if all_scores:
            scores = pd.DataFrame(all_scores)
            pivot = scores.groupby(["version","dimension"])["score"].mean().unstack(0)
            pivot["Difference (P-G)"] = pivot["Personalised"] - pivot["Generic"]
            st.markdown("### Average scores")
            st.dataframe(pivot.round(2), use_container_width=True)
            st.markdown("### Interpretation")
            st.write("Positive differences indicate higher average scores for personalised content. For intrusiveness, a negative difference is generally preferable.")
    else:
        st.info("Generate a message in Single Consumer Mode, then click “Add current pair” here.")

# -----------------------------
# Prompt Architecture
# -----------------------------
with tabs[3]:
    st.subheader("Prompt Architecture")
    st.markdown("""
**1. System instructions**  
Permanent rules defining the engine's role, consumer-behaviour principles, privacy/intrusiveness boundaries, CTA requirement and output discipline.

**2. Dynamic input variables**  
Company, industry, product, consumer profile, behaviour, motivation, geography, lifecycle, trigger, context, objective, channel, message type and tone.

**3. Consumer interpretation**  
The LLM creates/structures a concise persona and interprets relevant needs, motivations and decision drivers.

**4. Relevant signal selection**  
The LLM decides which supplied signals materially improve the communication and which should be ignored.

**5. Communication strategy**  
The LLM determines message type, objective, appeal, tone, personalisation level, value proposition and CTA.

**6. Dynamic content generation**  
The LLM writes the personalised message for the selected channel.

**7. Quality check**  
The output is checked for relevance, brand fit, creativity, CTA fit and intrusiveness risk.

**8. Generic control**  
A consumer-neutral version is generated for impact comparison.
""")
    st.markdown("### Core flow")
    st.code("Understand → Select relevant signals → Strategise → Generate → Check → Compare", language="text")
    st.markdown("### LLM structured output")
    st.json(OUTPUT_SCHEMA)
