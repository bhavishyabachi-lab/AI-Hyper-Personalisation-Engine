
import os
import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Hyper-Personalisation Engine", page_icon="✨", layout="wide")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.block-container{max-width:1240px;padding-top:1.8rem;padding-bottom:4rem}
.hero{padding:30px 32px;border-radius:20px;background:linear-gradient(135deg,#20234a,#5b5ce2);color:#fff;margin-bottom:22px}
.hero h1{margin:0 0 8px;font-size:2.15rem}.hero p{margin:0;opacity:.9;font-size:1rem}
.helper{font-size:.82rem;color:#68748a;line-height:1.5}
.output-card{border:1px solid #e4e8f1;border-radius:16px;padding:18px;background:#fff}
.message{border:1px solid #d8d8f7;border-radius:16px;padding:22px;background:linear-gradient(135deg,#f8f8ff,#fff);line-height:1.65;white-space:pre-wrap;font-size:1.03rem}
.generic{background:#fafbfe;border-color:#e4e8f1}
.chip{display:inline-block;padding:6px 10px;border-radius:999px;background:#efefff;color:#4b4dc0;margin:3px 4px 3px 0;font-size:.78rem}
.chip.gray{background:#f2f3f6;color:#68748a}
.insight{background:#f3f7ff;border-left:4px solid #5b5ce2;padding:12px 14px;border-radius:8px;line-height:1.55}
.warningbox{background:#fff7e8;border:1px solid #ead39a;padding:12px 14px;border-radius:10px;line-height:1.5}
.small{font-size:.75rem;color:#68748a}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants
# -----------------------------
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHANNELS = ["WhatsApp","SMS","Email","Push notification","Social media","Website","In-app message"]
LIFECYCLES = ["Prospect","New customer","Active customer","Loyal customer","At-risk","Inactive","Churned / potentially churned"]
OBJECTIVES = ["Awareness","Engagement","Consideration","Conversion","Re-engagement","Retention","Feedback","Win-back","Cross-sell","Upsell","Loyalty","Reminder / completion"]
MESSAGE_TYPES = ["AI chooses","Promotional","Awareness","Consideration","Product recommendation","Behaviour-triggered","Abandonment","Re-engagement","Retention","Loyalty / reward","Cross-sell","Upsell","Feedback","Service / recovery","Transactional / reminder","Win-back","Educational","Occasion-based"]
TONES = ["AI chooses","Conversational","Friendly","Quirky","Humorous","Premium","Emotional","Urgent","Empathetic","Professional","Playful","Informative"]
LENGTHS = ["Very short","Short","Medium","Long"]
DIMENSIONS = ["Relevance","Perceived personalisation","Persuasiveness","Emotional appeal","Purchase / engagement intention","Brand authenticity","Intrusiveness"]

SYSTEM_PROMPT = r"""
You are the intelligence layer of a generic AI Hyper-Personalisation Engine.

MISSION
Generate communication that feels realistically sendable by the stated brand while
being meaningfully adapted to the consumer, situation and campaign objective.

CORE PRIORITY ORDER
1. Situational appropriateness
2. Consumer relevance
3. Consumer-behaviour meaning
4. Brand authenticity
5. Objective clarity
6. Naturalness
7. Creativity
8. Channel and length fit

Do not sacrifice relevance, clarity or trust for cleverness.

CONSUMER-BEHAVIOUR LOGIC
Use behaviour, needs, motivations, preferences, lifecycle, trigger and context when
they materially improve the communication. Demographics are secondary.
Never invent facts. Clearly separate supplied facts from reasonable interpretation.
More attributes do not automatically mean better personalisation.

SIGNAL SELECTION
For every case, identify the few signals that deserve to influence the communication.
Do not repeat raw tracking data unnecessarily. A signal can be used without being
revealed literally to the consumer.

SIGNAL -> MEANING -> CREATIVE ANGLE
Translate an observed signal into a consumer-relevant communication implication.
Then choose one dominant creative angle that connects the consumer meaning, current
situation, product, brand and objective.

Possible creative angles: reassurance, value, convenience, discovery, urgency,
emotional connection, occasion/social context, recognition, problem-solution,
playful challenge, exclusivity, celebration.

COMMUNICATION STRATEGY
Determine message type, objective, primary appeal, tone, personalisation level,
key value proposition and CTA. If message type or tone is AI chooses, infer it.

MESSAGE
Write one final personalised message that:
- sounds natural and brand-authentic
- contains meaningful personalisation, not just a name
- reflects the strongest relevant signal(s)
- uses the chosen creative angle
- fits channel and length
- avoids explaining the personalisation inside the message
- avoids AI/corporate clichés
- avoids unnecessary tracking disclosure
- contains a clear, natural CTA

Do not use phrases like:
"Based on your behaviour..."
"As a price-sensitive customer..."
"Since you are 24..."

CTA
Always include a CTA in the personalised message.
Align the CTA with the desired consumer action.
Examples: conversion -> Shop/Order/Book; feedback -> Share feedback/Start survey;
re-engagement -> Take another look; win-back -> Come back; reminder -> Pay/Complete.

GENERIC CONTROL
Create a genuinely generic control for the same product/service and objective.
Do not use consumer identity, behaviour, motivation, lifecycle, preferences or trigger.

QUALITY CHECK
Internally test:
- Would this work for many unrelated consumers? If yes, strengthen personalisation.
- Does the creative angle actually reflect the consumer meaning?
- Does it sound like the stated brand/industry?
- Does tone fit the situation?
- Is CTA correct?
- Is any tracking disclosure unnecessary?
- Are facts invented?
If weak, rewrite before returning.

VARIATION MODE
When asked for another version, keep the same strategy and consumer logic but develop
a substantially different creative execution. Do not merely replace a few words.

Return only valid JSON matching the provided schema. Do not expose hidden chain-of-thought.
"""

SCHEMA = {
    "type":"object","additionalProperties":False,
    "properties":{
        "persona":{"type":"string"},
        "consumer_insight":{"type":"string"},
        "creative_angle":{"type":"string"},
        "signals_used":{"type":"array","items":{"type":"string"}},
        "signals_excluded":{"type":"array","items":{
            "type":"object","additionalProperties":False,
            "properties":{"signal":{"type":"string"},"reason":{"type":"string"}},
            "required":["signal","reason"]
        }},
        "strategy":{"type":"object","additionalProperties":False,
            "properties":{
                "message_type":{"type":"string"},"objective":{"type":"string"},
                "primary_appeal":{"type":"string"},"tone":{"type":"string"},
                "personalisation_level":{"type":"string"},
                "key_value_proposition":{"type":"string"},"cta":{"type":"string"}
            },
            "required":["message_type","objective","primary_appeal","tone","personalisation_level","key_value_proposition","cta"]
        },
        "personalised_message":{"type":"string"},
        "generic_message":{"type":"string"},
        "quality_check":{"type":"object","additionalProperties":False,
            "properties":{
                "relevance":{"type":"string"},"personalisation":{"type":"string"},
                "brand_fit":{"type":"string"},"creativity":{"type":"string"},
                "cta_fit":{"type":"string"},"intrusiveness_risk":{"type":"string"}
            },
            "required":["relevance","personalisation","brand_fit","creativity","cta_fit","intrusiveness_risk"]
        }
    },
    "required":["persona","consumer_insight","creative_angle","signals_used","signals_excluded","strategy","personalised_message","generic_message","quality_check"]
}

DEMO = {
"Practo — app uninstall / feedback":{
"company":"Practo","industry":"Healthcare / health-tech","product":"Practo mobile app","positioning":"Convenient digital healthcare access",
"segment":"Existing app user","age":"","occupation":"","location":"India","behaviour":"Previously used the app and then uninstalled it",
"preferences":"","motivation":"","price_sensitivity":"Unknown","lifecycle_stage":"Churned / potentially churned","trigger":"App uninstallation",
"context":"Shortly after uninstall","objective":"Feedback","channel":"WhatsApp","message_type":"AI chooses","tone":"Empathetic","length":"Short"},
"Domino's — Raksha Bandhan / promotion":{
"company":"Domino's Pizza India","industry":"Food & beverage","product":"Pizza","positioning":"Convenient, playful and value-focused",
"segment":"Occasion-oriented household","age":"","occupation":"","location":"India","behaviour":"",
"preferences":"Pizza; shared meals","motivation":"Celebrate Raksha Bandhan together","price_sensitivity":"Medium","lifecycle_stage":"Active customer",
"trigger":"Raksha Bandhan","context":"Festival occasion","objective":"Conversion","channel":"WhatsApp","message_type":"AI chooses","tone":"Quirky","length":"Short"},
"Myntra — repeated sneaker browsing":{
"company":"Myntra","industry":"Fashion e-commerce","product":"Running shoes","positioning":"Trendy, youthful and accessible",
"segment":"Budget-conscious fitness consumer","age":"24","occupation":"Young professional","location":"Bengaluru",
"behaviour":"Frequently browses running shoes and compares prices","preferences":"Running; minimalist designs","motivation":"Fitness plus value for money",
"price_sensitivity":"High","lifecycle_stage":"Active customer","trigger":"Viewed running shoes twice recently","context":"Weekend","objective":"Conversion",
"channel":"WhatsApp","message_type":"AI chooses","tone":"Quirky","length":"Short"},
"CRED — payment reminder":{
"company":"CRED","industry":"Fintech","product":"Credit card bill payment","positioning":"Premium, frictionless financial utility",
"segment":"Existing credit-card user","age":"","occupation":"","location":"India","behaviour":"Has an outstanding credit-card balance",
"preferences":"","motivation":"Avoid interest charges and complete payment","price_sensitivity":"Unknown","lifecycle_stage":"Active customer",
"trigger":"Credit-card payment due tomorrow","context":"Payment deadline","objective":"Reminder / completion","channel":"WhatsApp","message_type":"AI chooses","tone":"Professional","length":"Short"},
"Agoda — communications feedback":{
"company":"Agoda","industry":"Travel & hospitality","product":"Travel booking platform","positioning":"Helpful and traveler-focused",
"segment":"Existing traveler","age":"","occupation":"","location":"India","behaviour":"Has interacted with Agoda communications",
"preferences":"Travel","motivation":"Wants more relevant travel communication","price_sensitivity":"Unknown","lifecycle_stage":"Active customer",
"trigger":"Feedback request","context":"Post-interaction","objective":"Feedback","channel":"Email","message_type":"AI chooses","tone":"Friendly","length":"Medium"}
}

def get_api_key(user_key: str) -> str:
    if user_key:
        return user_key.strip()
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY","").strip()

def to_prompt(d: Dict[str,Any], variation: bool=False) -> str:
    ordered = ["company","industry","product","positioning","segment","age","occupation","location",
               "behaviour","preferences","motivation","price_sensitivity","lifecycle_stage","trigger",
               "context","objective","channel","message_type","tone","length"]
    body = "\n".join(f"{k}: {d.get(k) or 'Not provided'}" for k in ordered)
    mode = "\nVARIATION MODE: Create a genuinely different creative execution while preserving the same strategic logic." if variation else ""
    return "CASE DATA\n"+body+mode

def generate_live(d: Dict[str,Any], api_key: str, model: str, variation: bool=False) -> Dict[str,Any]:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=to_prompt(d, variation=variation),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=SCHEMA,
        ),
    )
    return json.loads(response.text)

def cta_guard(result: Dict[str,Any]) -> Dict[str,Any]:
    msg = result.get("personalised_message","").strip()
    cta = result.get("strategy",{}).get("cta","").strip()
    if cta and cta.lower() not in msg.lower():
        result["personalised_message"] = msg.rstrip() + f"\n\n👉 {cta}"
    return result

def demo_generate(d: Dict[str,Any]) -> Dict[str,Any]:
    c=d.get("company","").lower(); t=d.get("trigger","").lower()
    if "practo" in c and "uninstall" in t:
        return {"persona":"Recently Churned Healthcare-App User","consumer_insight":"The known signal is disengagement after an app uninstall. The reason for leaving is unknown and is not assumed.","creative_angle":"Invite the customer to improve the experience without pressure.","signals_used":["Existing relationship","App uninstallation","Churned lifecycle","Feedback objective","WhatsApp"],"signals_excluded":[{"signal":"Age","reason":"Not provided and unnecessary"},{"signal":"Occupation","reason":"Not relevant"},{"signal":"Reason for uninstalling","reason":"Unknown"}],"strategy":{"message_type":"Feedback / recovery","objective":"Feedback","primary_appeal":"Help improve the experience","tone":"Friendly and empathetic","personalisation_level":"Behavioural + lifecycle + trigger","key_value_proposition":"Feedback can help improve the experience","cta":"Share your feedback"},"personalised_message":"Bhavishya Paila, can we ask you one quick question? 👀\n\nWe noticed you're no longer using the Practo App. We'd really like to know what we could have done better.\n\n👉 Share your feedback","generic_message":"We'd love your feedback on your experience with our app. Please share your thoughts with us.","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}}
    if "domino" in c:
        return {"persona":"Occasion-Oriented Pizza Sharer","consumer_insight":"The festival creates a shared celebration moment where social context matters more than demographic targeting.","creative_angle":"Turn sibling rivalry into a playful pizza moment.","signals_used":["Raksha Bandhan","Shared-meal context","Conversion objective","Quirky tone","WhatsApp"],"signals_excluded":[{"signal":"Age","reason":"Not needed"},{"signal":"Occupation","reason":"Not relevant"}],"strategy":{"message_type":"Occasion-based promotion","objective":"Conversion","primary_appeal":"Celebration + value","tone":"Quirky and festive","personalisation_level":"Contextual + occasion-based","key_value_proposition":"Make the shared celebration more rewarding","cta":"Order now"},"personalised_message":"Raksha Bandhan plans? 🍕 No sibling fights over the last slice. Give everyone a favourite and make the celebration a little more delicious.\n\n👉 Order now","generic_message":"Enjoy our latest pizza offers today.\n\n👉 Order now","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"High","cta_fit":"High","intrusiveness_risk":"Low"}}
    if "cred" in c and "due" in t:
        return {"persona":"Deadline-Driven Credit Card User","consumer_insight":"An immediate payment deadline makes clarity and fast action more important than entertainment.","creative_angle":"Make the deadline and consequence unmistakably clear.","signals_used":["Payment due trigger","Outstanding balance","Reminder objective","WhatsApp"],"signals_excluded":[{"signal":"Age","reason":"Not relevant"},{"signal":"Occupation","reason":"Not relevant"}],"strategy":{"message_type":"Transactional / reminder","objective":"Reminder / completion","primary_appeal":"Avoid unnecessary interest charges","tone":"Clear and professional","personalisation_level":"Transactional + contextual","key_value_proposition":"Complete payment before the deadline","cta":"Pay now"},"personalised_message":"Your credit-card payment is due tomorrow. Clear the remaining payment today to avoid interest charges on the outstanding amount.\n\n👉 Pay now","generic_message":"Your credit-card payment is due soon. Please complete your payment.","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"Low","cta_fit":"High","intrusiveness_risk":"Low"}}
    if "agoda" in c:
        return {"persona":"Travel-Engaged Feedback Seeker","consumer_insight":"The customer already engages with travel communication and the objective is improving future relevance, so feedback is more appropriate than promotion.","creative_angle":"Give the traveler a voice in shaping future communication.","signals_used":["Travel interest","Existing relationship","Feedback objective","Email"],"signals_excluded":[{"signal":"Age","reason":"Not relevant"},{"signal":"Location","reason":"Not needed"}],"strategy":{"message_type":"Feedback / relationship","objective":"Feedback","primary_appeal":"Help tailor future communication","tone":"Friendly and appreciative","personalisation_level":"Relationship + preference","key_value_proposition":"Feedback can improve future travel communication","cta":"Start the survey"},"personalised_message":"Dear Bhavishya,\n\nHelp us make the travel messages you receive more useful to you. Tell us what you'd like to see more of—and what you'd rather skip.\n\n👉 Start the survey","generic_message":"We'd appreciate your feedback on our travel communications. Please take a short survey.","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}}
    # Generic fallback is deliberately labelled as demo logic.
    obj=d.get("objective","Conversion")
    if obj=="Feedback": cta="Share your feedback"; mt="Feedback"
    elif obj=="Re-engagement": cta="Take another look"; mt="Re-engagement"
    elif obj=="Reminder / completion": cta="Complete it now"; mt="Transactional / reminder"
    elif obj=="Cross-sell": cta="Discover more"; mt="Cross-sell"
    elif obj=="Retention" or obj=="Loyalty": cta="Keep exploring"; mt="Retention / loyalty"
    else: cta="Shop now"; mt="Promotional"
    return {"persona":d.get("segment") or "Context-Aware Consumer","consumer_insight":"Demo mode uses simplified logic and is provided only to demonstrate the interface. Live mode is required for arbitrary LLM-generated content.","creative_angle":"Connect the strongest supplied consumer motivation to the campaign objective.","signals_used":["Available relevant consumer/context signals"],"signals_excluded":[{"signal":"Unused optional details","reason":"They do not materially improve this demo message"}],"strategy":{"message_type":mt,"objective":obj,"primary_appeal":"Relevant benefit","tone":d.get("tone") if d.get("tone")!="AI chooses" else "Conversational","personalisation_level":"Contextual demo logic","key_value_proposition":"A relevant benefit for this scenario","cta":cta},"personalised_message":f"Here’s something relevant to your needs around {d.get('product','this offer')}.\n\n👉 {cta}","generic_message":f"Explore {d.get('product','this offer')}.\n\n👉 {cta}","quality_check":{"relevance":"Demo","personalisation":"Demo","brand_fit":"Demo","creativity":"Demo","cta_fit":"High","intrusiveness_risk":"Low"}}

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="hero"><h1>✨ AI Hyper-Personalisation Engine</h1><p>Dynamic communication conditioned on consumer behaviour, lifecycle, trigger, context and campaign objective.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Engine controls")
    mode = st.radio("Generation mode", ["Live LLM","Demo / offline"], index=0)
    user_key = st.text_input("Gemini API key", type="password", help="Optional when the deployment has GEMINI_API_KEY configured as a secret.")
    model = st.text_input("LLM model", MODEL)
    if mode=="Demo / offline":
        st.caption("Demo mode uses predefined logic. It is for testing the interface, not for proving LLM generation.")
    else:
        st.caption("Live LLM mode is the mode to use for the final prototype demonstration.")

tabs = st.tabs(["Single Consumer","Batch / At Scale","Impact Analysis","Prompt Architecture"])

# -----------------------------
# Single consumer
# -----------------------------
with tabs[0]:
    st.subheader("Single Consumer — deep personalisation")
    st.markdown('<div class="helper">Enter the information you actually know. The engine should not invent missing consumer facts. The segment/persona can be a short description rather than a polished persona.</div>', unsafe_allow_html=True)

    demo_name = st.selectbox("Load benchmark scenario", ["None"] + list(DEMO.keys()))
    d0 = DEMO.get(demo_name,{})
    with st.expander("1. Business context", True):
        c1,c2=st.columns(2)
        company=c1.text_input("Company *",d0.get("company",""),help="Brand/company. Example: Nike, Myntra, Practo.")
        industry=c2.text_input("Industry *",d0.get("industry",""),help="Business category. Example: fashion e-commerce, healthcare, food delivery.")
        c1,c2=st.columns(2)
        product=c1.text_input("Product / Service *",d0.get("product",""),help="The specific product, service, subscription, app or offer.")
        positioning=c2.text_input("Brand positioning",d0.get("positioning",""),help="How the brand wants to be perceived. Optional.")
    with st.expander("2. Consumer context", True):
        segment=st.text_input("Consumer segment / persona",d0.get("segment",""),help="Optional when the other consumer details are enough for the AI to construct a persona.")
        c1,c2,c3=st.columns(3)
        age=c1.text_input("Age",d0.get("age",""),help="Optional.")
        occupation=c2.text_input("Occupation",d0.get("occupation",""),help="Optional.")
        location=c3.text_input("Geography",d0.get("location",""),help="Optional; use only when it can add meaningful relevance.")
        behaviour=st.text_area("Behaviour / previous interactions",d0.get("behaviour",""),help="What the consumer has actually done. Examples: viewed product, abandoned cart, purchased repeatedly, stopped using app.")
        preferences=st.text_area("Preferences / interests",d0.get("preferences",""),help="What they like or care about. Examples: running, minimalist designs, travel.")
        motivation=st.text_area("Needs / motivation",d0.get("motivation",""),help="Why they may want the offering. Examples: convenience, fitness, status, saving money.")
        price=st.selectbox("Price sensitivity",["Unknown","Low","Medium","High"],index=["Unknown","Low","Medium","High"].index(d0.get("price_sensitivity","Unknown")) if d0.get("price_sensitivity","Unknown") in ["Unknown","Low","Medium","High"] else 0)
    with st.expander("3. Consumer state", True):
        lifecycle=st.selectbox("Lifecycle stage *",LIFECYCLES,index=LIFECYCLES.index(d0.get("lifecycle_stage","Active customer")) if d0.get("lifecycle_stage","Active customer") in LIFECYCLES else 2)
        trigger=st.text_input("Trigger / recent event *",d0.get("trigger",""),help="The event that explains why this communication is happening now.")
        context=st.text_input("Current context / occasion",d0.get("context",""),help="Festival, weekend, after work, deadline, travel season, etc.")
    with st.expander("4. Campaign & communication", True):
        c1,c2=st.columns(2)
        objective=c1.selectbox("Marketing objective *",OBJECTIVES,index=OBJECTIVES.index(d0.get("objective","Conversion")) if d0.get("objective","Conversion") in OBJECTIVES else 3)
        channel=c2.selectbox("Channel *",CHANNELS,index=CHANNELS.index(d0.get("channel","WhatsApp")) if d0.get("channel","WhatsApp") in CHANNELS else 0)
        c1,c2=st.columns(2)
        mtype=c1.selectbox("Message type",MESSAGE_TYPES,index=MESSAGE_TYPES.index(d0.get("message_type","AI chooses")) if d0.get("message_type","AI chooses") in MESSAGE_TYPES else 0)
        tone=c2.selectbox("Tone / style",TONES,index=TONES.index(d0.get("tone","AI chooses")) if d0.get("tone","AI chooses") in TONES else 0)
        length=st.selectbox("Message length",LENGTHS,index=LENGTHS.index(d0.get("length","Short")) if d0.get("length","Short") in LENGTHS else 1)

    data={"company":company,"industry":industry,"product":product,"positioning":positioning,"segment":segment,"age":age,"occupation":occupation,"location":location,"behaviour":behaviour,"preferences":preferences,"motivation":motivation,"price_sensitivity":price,"lifecycle_stage":lifecycle,"trigger":trigger,"context":context,"objective":objective,"channel":channel,"message_type":mtype,"tone":tone,"length":length}

    c1,c2=st.columns([3,1])
    with c1:
        generate=st.button("🚀 Generate personalised content",type="primary",use_container_width=True)
    with c2:
        variant=st.button("↻ Another creative version",use_container_width=True)

    if generate or variant:
        missing=[n for n,v in [("Company",company),("Industry",industry),("Product / Service",product),("Trigger / Recent Event",trigger),("Marketing Objective",objective)] if not v.strip()]
        if missing:
            st.error("Please fill: "+", ".join(missing))
        elif mode=="Live LLM" and not get_api_key(user_key):
            st.error("Live LLM mode needs a Gemini API key. Add it in the sidebar or configure GEMINI_API_KEY in the deployment secrets.")
        else:
            try:
                with st.spinner("Understanding consumer → selecting signals → deciding communication strategy → choosing creative angle → generating → checking..."):
                    result = generate_live(data,get_api_key(user_key),model,variation=variant) if mode=="Live LLM" else demo_generate(data)
                    result=cta_guard(result)
                st.session_state["single_result"]=result
                st.session_state["single_data"]=data
                st.session_state["single_mode"]=mode
            except Exception as e:
                st.error(f"Generation failed: {e}")

    r=st.session_state.get("single_result")
    if r:
        st.divider(); st.subheader("AI output")
        if st.session_state.get("single_mode")=="Demo / offline":
            st.warning("Demo/offline result. Switch to Live LLM for genuine generative output.")
        c1,c2=st.columns([1.3,1])
        with c1:
            st.markdown("### AI Consumer Persona"); st.markdown(f"**{r['persona']}**")
            st.write(r["consumer_insight"])
            st.markdown("### Creative angle"); st.markdown(f'<div class="insight">{r["creative_angle"]}</div>',unsafe_allow_html=True)
        with c2:
            st.markdown("### AI quality check")
            for k,v in r["quality_check"].items(): st.write(f"**{k.replace('_',' ').title()}:** {v}")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("### Personalisation signals used")
            for s in r["signals_used"]: st.markdown(f'<span class="chip">✓ {s}</span>',unsafe_allow_html=True)
        with c2:
            st.markdown("### Signals excluded")
            for x in r["signals_excluded"]: st.markdown(f'<span class="chip gray">{x["signal"]}</span> — {x["reason"]}',unsafe_allow_html=True)
        st.markdown("### Communication strategy")
        s=r["strategy"]
        rows=[("Message type",s["message_type"]),("Objective",s["objective"]),("Primary appeal",s["primary_appeal"]),("Tone",s["tone"]),("Personalisation",s["personalisation_level"]),("Key value proposition",s["key_value_proposition"]),("CTA",s["cta"])]
        st.dataframe(pd.DataFrame(rows,columns=["Decision","Selected approach"]),use_container_width=True,hide_index=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("### ✨ AI-Personalised Message")
            st.markdown(f'<div class="message">{r["personalised_message"]}</div>',unsafe_allow_html=True)
        with c2:
            st.markdown("### Generic Control")
            st.markdown(f'<div class="message generic">{r["generic_message"]}</div>',unsafe_allow_html=True)
        st.markdown("### Why this is personalised")
        st.write("The personalised version is conditioned on the selected consumer and situation signals. The generic control is deliberately consumer-neutral.")
        if st.button("Add this pair to Impact Analysis",key="add_pair"):
            st.session_state["impact_pair"]={"case":data["company"]+" — "+data["product"],"personalised":r["personalised_message"],"generic":r["generic_message"]}
            st.success("Message pair added to Impact Analysis.")

# -----------------------------
# Batch
# -----------------------------
with tabs[1]:
    st.subheader("Batch / At Scale")
    st.markdown('<div class="helper">Upload a CSV or Excel file containing the same consumer/business fields. One engine is applied independently to every row. For the final project, use Live LLM mode.</div>',unsafe_allow_html=True)
    template_cols=list(data.keys())
    template=pd.DataFrame([{k:"" for k in template_cols}])
    st.download_button("Download empty batch template",template.to_csv(index=False).encode("utf-8"),"hyper_personalisation_batch_template.csv","text/csv")
    up=st.file_uploader("Upload consumer dataset",type=["csv","xlsx","xls"],key="batch_upload")
    if up:
        try:
            df=pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            st.write(f"**{len(df)} consumer rows loaded.**")
            st.dataframe(df.head(10),use_container_width=True,hide_index=True)
            if st.button("⚡ Generate for all consumers",type="primary",use_container_width=True):
                if mode=="Live LLM" and not get_api_key(user_key):
                    st.error("Live LLM batch generation needs an API key.")
                else:
                    required=["company","industry","product","trigger","objective"]
                    missing=[c for c in required if c not in df.columns]
                    if missing:
                        st.error("Missing required columns: "+", ".join(missing))
                    else:
                        outputs=[]; bar=st.progress(0)
                        keys=["company","industry","product","positioning","segment","age","occupation","location","behaviour","preferences","motivation","price_sensitivity","lifecycle_stage","trigger","context","objective","channel","message_type","tone","length"]
                        for i,row in df.fillna("").iterrows():
                            d={k:str(row.get(k,"")) for k in keys}
                            d["lifecycle_stage"]=d["lifecycle_stage"] or "Active customer"
                            d["objective"]=d["objective"] or "Conversion"
                            d["channel"]=d["channel"] or "WhatsApp"
                            d["message_type"]=d["message_type"] or "AI chooses"
                            d["tone"]=d["tone"] or "AI chooses"
                            d["length"]=d["length"] or "Short"
                            try:
                                z=generate_live(d,get_api_key(user_key),model) if mode=="Live LLM" else demo_generate(d)
                                z=cta_guard(z)
                                outputs.append({
                                    "Consumer #":i+1,"Company":d["company"],"Product":d["product"],
                                    "Persona":z["persona"],"Lifecycle":d["lifecycle_stage"],"Trigger":d["trigger"],
                                    "Message Type":z["strategy"]["message_type"],"Creative Angle":z["creative_angle"],
                                    "Primary Appeal":z["strategy"]["primary_appeal"],"Tone":z["strategy"]["tone"],"CTA":z["strategy"]["cta"],
                                    "Personalised Message":z["personalised_message"],"Generic Message":z["generic_message"]
                                })
                            except Exception as e:
                                outputs.append({"Consumer #":i+1,"Company":d["company"],"Product":d["product"],"Persona":"ERROR","Lifecycle":d["lifecycle_stage"],"Trigger":d["trigger"],"Message Type":"ERROR","Creative Angle":"","Primary Appeal":"","Tone":"","CTA":str(e),"Personalised Message":"","Generic Message":""})
                            bar.progress((i+1)/len(df))
                        st.session_state["batch_outputs"]=pd.DataFrame(outputs)
                        st.session_state["batch_mode"]=mode
                        st.success("Batch processing complete.")
        except Exception as e:
            st.error(f"Could not read the dataset: {e}")

    bo=st.session_state.get("batch_outputs")
    if bo is not None:
        if st.session_state.get("batch_mode")=="Demo / offline":
            st.warning("These are demo outputs. They are not evidence of LLM generation quality.")
        st.markdown("### Batch results")
        st.dataframe(bo,use_container_width=True,hide_index=True)
        st.download_button("Download generated results",bo.to_csv(index=False).encode(),"hyper_personalisation_results.csv","text/csv")
        if len(bo)>0:
            selected=st.selectbox("Inspect one consumer in detail",list(bo["Consumer #"].astype(str)))
            row=bo[bo["Consumer #"].astype(str)==selected].iloc[0]
            c1,c2=st.columns(2)
            with c1:
                st.markdown("### Personalised"); st.markdown(f'<div class="message">{row["Personalised Message"] or "No output"}</div>',unsafe_allow_html=True)
            with c2:
                st.markdown("### Generic"); st.markdown(f'<div class="message generic">{row["Generic Message"] or "No output"}</div>',unsafe_allow_html=True)
            st.write(f"**Creative angle:** {row['Creative Angle']}")
            st.write(f"**Message type:** {row['Message Type']}  |  **CTA:** {row['CTA']}")

# -----------------------------
# Impact analysis
# -----------------------------
with tabs[2]:
    st.subheader("Personalisation Impact Analysis")
    st.markdown('<div class="helper">This module is for the actual respondent study. Enter real respondent ratings for the same personalised/generic message pair. Do not use the tool-generated numbers below as research findings.</div>',unsafe_allow_html=True)

    pair=st.session_state.get("impact_pair")
    if pair:
        st.markdown(f"**Current study case:** {pair['case']}")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Personalised message**"); st.markdown(f'<div class="message">{pair["personalised"]}</div>',unsafe_allow_html=True)
        with c2:
            st.markdown("**Generic message**"); st.markdown(f'<div class="message generic">{pair["generic"]}</div>',unsafe_allow_html=True)
        st.divider()
        n=st.number_input("Number of respondents",min_value=1,max_value=200,value=10,step=1)
        st.markdown("**Rating scale:** 1 = very low, 5 = very high. For Intrusiveness, lower is generally preferable.")
        ids=[f"R{i:03d}" for i in range(1,int(n)+1)]
        cols=[]
        for d in DIMENSIONS:
            cols.extend([f"P — {d}",f"G — {d}"])
        if st.button("Create / reset rating table",key="create_rating"):
            st.session_state["rating_df"]=pd.DataFrame([[rid]+[3]*(len(cols)) for rid in ids],columns=["Respondent"]+cols)
        rating_df=st.session_state.get("rating_df")
        if rating_df is not None:
            edited=st.data_editor(rating_df,use_container_width=True,num_rows="fixed",column_config={
                c: st.column_config.NumberColumn(c,min_value=1,max_value=5,step=1) for c in cols
            })
            st.session_state["rating_df"]=edited
            records=[]
            for _,rr in edited.iterrows():
                for d in DIMENSIONS:
                    records.append({"Respondent":rr["Respondent"],"Dimension":d,"Personalised":rr[f"P — {d}"],"Generic":rr[f"G — {d}"]})
            scores=pd.DataFrame(records)
            summary=scores.groupby("Dimension")[["Personalised","Generic"]].mean().round(2)
            summary["Difference (P-G)"]=(summary["Personalised"]-summary["Generic"]).round(2)
            st.markdown("### Results")
            st.dataframe(summary,use_container_width=True)
            st.markdown("### Personalised vs generic")
            st.bar_chart(summary[["Personalised","Generic"]])
            st.markdown("### Interpretation guide")
            st.write("Positive Difference (P-G) means personalised content scored higher on that dimension. For Intrusiveness, a negative difference is generally preferable.")
            st.download_button("Download respondent ratings",edited.to_csv(index=False).encode(),"personalisation_impact_ratings.csv","text/csv")
            st.download_button("Download impact summary",summary.reset_index().to_csv(index=False).encode(),"personalisation_impact_summary.csv","text/csv")
    else:
        st.info("Generate a single-consumer message first, then click “Add this pair to Impact Analysis”.")

# -----------------------------
# Prompt architecture
# -----------------------------
with tabs[3]:
    st.subheader("Prompt Architecture")
    st.markdown("""
### Engine flow

**Business + Consumer + Lifecycle + Trigger + Context + Campaign**
→ **Consumer interpretation**
→ **Relevant signal selection**
→ **Signal meaning**
→ **Communication strategy**
→ **Creative angle**
→ **LLM generation**
→ **CTA**
→ **Quality check**
→ **Generic control**

### Core generation principle

> **Signal → Meaning → Creative Angle → Message**

The LLM is not instructed to merely insert consumer attributes into a template. It must determine what the signals mean for the communication and then create a natural execution.
""")
    st.code("Understand → Select signals → Interpret meaning → Strategise → Choose creative angle → Generate → Check",language="text")
