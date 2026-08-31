
import os, json, re, time
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AI Hyper-Personalisation Engine", page_icon="✨", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1240px;padding-top:1.6rem;padding-bottom:4rem}
.hero{padding:30px 32px;border-radius:20px;background:linear-gradient(135deg,#20234a,#5b5ce2);color:#fff;margin-bottom:22px}
.hero h1{margin:0 0 8px;font-size:2.1rem}.hero p{margin:0;opacity:.9}
.helper{font-size:.82rem;color:#68748a;line-height:1.5}
.message{border:1px solid #d8d8f7;border-radius:16px;padding:22px;background:#f8f8ff;line-height:1.65;white-space:pre-wrap;font-size:1.03rem}
.generic{background:#fafbfe;border-color:#e4e8f1}
.chip{display:inline-block;padding:6px 10px;border-radius:999px;background:#efefff;color:#4b4dc0;margin:3px 4px 3px 0;font-size:.78rem}
.chip.gray{background:#f2f3f6;color:#68748a}
.insight{background:#f3f7ff;border-left:4px solid #5b5ce2;padding:12px 14px;border-radius:8px;line-height:1.55}
.status{padding:9px 12px;border-radius:9px;background:#eef8f1;border:1px solid #cbe6d3;font-size:.82rem}
.warn{background:#fff7e8;border:1px solid #ead39a;padding:12px 14px;border-radius:10px}
</style>
""", unsafe_allow_html=True)

PRIMARY_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free"
]

CHANNELS=["WhatsApp","SMS","Email","Push notification","Social media","Website","In-app message"]
LIFECYCLES=["Prospect","New customer","Active customer","Loyal customer","At-risk","Inactive","Churned / potentially churned"]
OBJECTIVES=["Awareness","Engagement","Consideration","Conversion","Re-engagement","Retention","Feedback","Win-back","Cross-sell","Upsell","Loyalty","Reminder / completion"]
MESSAGE_TYPES=["AI chooses","Promotional","Awareness","Consideration","Product recommendation","Behaviour-triggered","Abandonment","Re-engagement","Retention","Loyalty / reward","Cross-sell","Upsell","Feedback","Service / recovery","Transactional / reminder","Win-back","Educational","Occasion-based"]
TONES=["AI chooses","Conversational","Friendly","Quirky","Humorous","Premium","Emotional","Urgent","Empathetic","Professional","Playful","Informative"]
LENGTHS=["Very short","Short","Medium","Long"]
DIMENSIONS=["Relevance","Perceived personalisation","Persuasiveness","Emotional appeal","Purchase / engagement intention","Brand authenticity","Intrusiveness"]

SYSTEM_PROMPT = """
You are the content intelligence of a generic AI Hyper-Personalisation Engine.

GOAL
Create marketing or customer communication that a real brand could plausibly send,
while making the message meaningfully relevant to the consumer and current situation.

PRIORITY
1. Situational appropriateness
2. Consumer relevance
3. Consumer-behaviour meaning
4. Brand authenticity
5. Objective clarity
6. Naturalness
7. Creativity
8. Channel/length fit

CORE PROCESS
Signal -> consumer meaning -> communication strategy -> creative angle -> message.

CONSUMER BEHAVIOUR
Use behaviour, needs, motivation, preferences, lifecycle, trigger and context only when
they materially improve relevance. Demographics are secondary.
Never invent facts. Do not pretend an inference is a known fact.
Do not use every available field simply because it exists.

SIGNAL SELECTION
Choose the few signals that materially affect the communication. Report what was used
and what was excluded, with concise reasons.

COMMUNICATION STRATEGY
Determine the best message type, primary appeal, tone, personalisation level,
key value proposition and CTA from the situation.
Lifecycle + trigger + objective should influence the communication type.

CREATIVE ANGLE
Choose ONE clear creative angle, such as reassurance, value, convenience, discovery,
urgency, emotional connection, occasion/social context, recognition, problem-solution,
playful challenge, exclusivity or celebration.
Do not force humour into sensitive or transactional situations.

PERSONALISED MESSAGE
The message must:
- sound genuinely sendable by the stated brand/industry
- meaningfully reflect relevant consumer/context information
- use the chosen creative angle
- fit channel and requested length
- avoid mentioning raw tracking details unnecessarily
- avoid "AI copy" language
- avoid inventing facts
- contain a natural CTA

Do NOT write phrases like:
"Based on your behaviour..."
"As a price-sensitive customer..."
"Since you are 24..."

CTA
Always include a clear CTA in the personalised message and make it appropriate to the
objective.

GENERIC CONTROL
Create a plausible generic message using only the company/product/basic objective.
Do NOT use consumer-specific identity, behaviour, motivation, lifecycle, preference,
trigger or demographic information.

QUALITY CHECK
Before returning, ask:
- Is the personalisation meaningful?
- Does the creative angle emerge naturally from the consumer meaning?
- Does this sound like the brand/industry?
- Is the tone appropriate?
- Is the CTA appropriate?
- Is unnecessary behavioural tracking exposed?
- Are facts invented?
Rewrite if weak.

Return JSON only using this structure:
{
"persona":"...",
"consumer_insight":"...",
"creative_angle":"...",
"signals_used":["..."],
"signals_excluded":[{"signal":"...","reason":"..."}],
"strategy":{"message_type":"...","objective":"...","primary_appeal":"...","tone":"...","personalisation_level":"...","key_value_proposition":"...","cta":"..."},
"personalised_message":"...",
"generic_message":"...",
"quality_check":{"relevance":"High/Medium/Low","personalisation":"High/Medium/Low","brand_fit":"High/Medium/Low","creativity":"High/Medium/Low","cta_fit":"High/Medium/Low","intrusiveness_risk":"High/Medium/Low"}
}
Do not include markdown fences.
"""

DEMO = {
"Practo — app uninstall / feedback":{
"company":"Practo","industry":"Healthcare / health-tech","product":"Practo mobile app","positioning":"Convenient digital healthcare access","segment":"Existing app user","age":"","occupation":"","location":"India","behaviour":"Previously used the app and then uninstalled it","preferences":"","motivation":"","price_sensitivity":"Unknown","lifecycle_stage":"Churned / potentially churned","trigger":"App uninstallation","context":"Shortly after uninstall","objective":"Feedback","channel":"WhatsApp","message_type":"AI chooses","tone":"Empathetic","length":"Short"},
"Domino's — Raksha Bandhan / promotion":{
"company":"Domino's Pizza India","industry":"Food & beverage","product":"Pizza","positioning":"Convenient, playful and value-focused","segment":"Occasion-oriented household","age":"","occupation":"","location":"India","behaviour":"","preferences":"Pizza; shared meals","motivation":"Celebrate Raksha Bandhan together","price_sensitivity":"Medium","lifecycle_stage":"Active customer","trigger":"Raksha Bandhan","context":"Festival occasion","objective":"Conversion","channel":"WhatsApp","message_type":"AI chooses","tone":"Quirky","length":"Short"},
"Myntra — repeated sneaker browsing":{
"company":"Myntra","industry":"Fashion e-commerce","product":"Running shoes","positioning":"Trendy, youthful and accessible","segment":"Budget-conscious fitness consumer","age":"24","occupation":"Young professional","location":"Bengaluru","behaviour":"Frequently browses running shoes and compares prices","preferences":"Running; minimalist designs","motivation":"Fitness plus value for money","price_sensitivity":"High","lifecycle_stage":"Active customer","trigger":"Viewed running shoes twice recently","context":"Weekend","objective":"Conversion","channel":"WhatsApp","message_type":"AI chooses","tone":"Quirky","length":"Short"},
"CRED — payment reminder":{
"company":"CRED","industry":"Fintech","product":"Credit card bill payment","positioning":"Premium, frictionless financial utility","segment":"Existing credit-card user","age":"","occupation":"","location":"India","behaviour":"Has an outstanding credit-card balance","preferences":"","motivation":"Avoid interest charges and complete payment","price_sensitivity":"Unknown","lifecycle_stage":"Active customer","trigger":"Credit-card payment due tomorrow","context":"Payment deadline","objective":"Reminder / completion","channel":"WhatsApp","message_type":"AI chooses","tone":"Professional","length":"Short"},
"Agoda — communications feedback":{
"company":"Agoda","industry":"Travel & hospitality","product":"Travel booking platform","positioning":"Helpful and traveler-focused","segment":"Existing traveler","age":"","occupation":"","location":"India","behaviour":"Has interacted with Agoda communications","preferences":"Travel","motivation":"Wants more relevant travel communication","price_sensitivity":"Unknown","lifecycle_stage":"Active customer","trigger":"Feedback request","context":"Post-interaction","objective":"Feedback","channel":"Email","message_type":"AI chooses","tone":"Friendly","length":"Medium"}
}

def get_key(user_key):
    if user_key.strip():
        return user_key.strip()
    try:
        k=st.secrets.get("OPENROUTER_API_KEY","")
        if k: return str(k).strip()
    except Exception: pass
    return os.getenv("OPENROUTER_API_KEY","").strip()

def parse_json(raw):
    raw=raw.strip()
    raw=re.sub(r"^```json\s*","",raw,flags=re.I)
    raw=re.sub(r"^```\s*","",raw)
    raw=re.sub(r"\s*```$","",raw)
    try: return json.loads(raw)
    except Exception:
        m=re.search(r"\{.*\}",raw,re.S)
        if m: return json.loads(m.group(0))
        raise ValueError("The model did not return readable JSON.")

def openrouter_call(data,key,model,variation=False):
    prompt=SYSTEM_PROMPT+"\n\nCASE DATA:\n"+json.dumps(data,ensure_ascii=False,indent=2)
    if variation:
        prompt += "\n\nGenerate a substantially different creative execution while keeping the same consumer insight, strategy, channel, objective and CTA logic."
    payload={
        "model":model,
        "messages":[
            {"role":"system","content":"Return only valid JSON. Never use markdown fences."},
            {"role":"user","content":prompt}
        ],
        "response_format":{"type":"json_object"}
    }
    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json","HTTP-Referer":"https://streamlit.io","X-Title":"AI Hyper-Personalisation Engine"}
    r=requests.post("https://openrouter.ai/api/v1/chat/completions",headers=headers,json=payload,timeout=75)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:700]}")
    body=r.json()
    return parse_json(body["choices"][0]["message"]["content"])

def live_generate(data,key,model,variation=False):
    models=[]
    for m in [model,PRIMARY_MODEL]+FALLBACK_MODELS:
        if m and m not in models: models.append(m)
    errors=[]
    for m in models:
        for attempt in range(2):
            try:
                result=openrouter_call(data,key,m,variation)
                result["_model_used"]=m
                return ensure_cta(result)
            except Exception as e:
                errors.append(f"{m}: {e}")
                time.sleep(1.5*(attempt+1))
    raise RuntimeError("Live LLM temporarily unavailable. "+ " | ".join(errors[:3]))

def ensure_cta(r):
    c=str(r.get("strategy",{}).get("cta","")).strip()
    m=str(r.get("personalised_message","")).strip()
    if c and c.lower() not in m.lower():
        r["personalised_message"]=m+"\n\n👉 "+c
    return r

def demo_generate(d):
    # Strong deterministic fallback. It is intentionally transparent: it is not presented as LLM output.
    company=d.get("company","").lower()
    trigger=d.get("trigger","").lower()
    obj=d.get("objective","Conversion")
    if "practo" in company and "uninstall" in trigger:
        return {"persona":"Recently Churned Healthcare-App User","consumer_insight":"The known signal is disengagement after an app uninstall. The reason for leaving is unknown.","creative_angle":"Invite the customer to improve the experience without pressure.","signals_used":["Existing relationship","App uninstallation","Churned lifecycle","Feedback objective","WhatsApp"],"signals_excluded":[{"signal":"Age","reason":"Not provided and unnecessary"},{"signal":"Occupation","reason":"Not relevant"},{"signal":"Reason for uninstalling","reason":"Unknown"}],"strategy":{"message_type":"Feedback / recovery","objective":"Feedback","primary_appeal":"Help improve the experience","tone":"Friendly and empathetic","personalisation_level":"Behavioural + lifecycle + trigger","key_value_proposition":"Feedback can help improve the experience","cta":"Share your feedback"},"personalised_message":"We noticed you're no longer using the Practo App. We'd really like to know what we could have done better.\n\n👉 Share your feedback","generic_message":"We'd love your feedback on your experience with our app. Please share your thoughts with us.","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}}
    if "domino" in company:
        return {"persona":"Occasion-Oriented Pizza Sharer","consumer_insight":"The festival creates a shared celebration moment where social context matters more than demographics.","creative_angle":"Turn sibling rivalry into a playful pizza moment.","signals_used":["Raksha Bandhan","Shared-meal context","Conversion objective","Quirky tone","WhatsApp"],"signals_excluded":[{"signal":"Age","reason":"Not needed"},{"signal":"Occupation","reason":"Not relevant"}],"strategy":{"message_type":"Occasion-based promotion","objective":"Conversion","primary_appeal":"Celebration + value","tone":"Quirky and festive","personalisation_level":"Contextual + occasion-based","key_value_proposition":"Make the shared celebration more rewarding","cta":"Order now"},"personalised_message":"Raksha Bandhan plans? 🍕 No sibling fights over the last slice. Give everyone a favourite and make the celebration a little more delicious.\n\n👉 Order now","generic_message":"Enjoy our latest pizza offers today.\n\n👉 Order now","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"High","cta_fit":"High","intrusiveness_risk":"Low"}}
    if "cred" in company:
        return {"persona":"Deadline-Driven Credit Card User","consumer_insight":"An immediate payment deadline makes clarity and fast action more important than entertainment.","creative_angle":"Make the deadline and consequence unmistakably clear.","signals_used":["Payment due trigger","Outstanding balance","Reminder objective","WhatsApp"],"signals_excluded":[{"signal":"Age","reason":"Not relevant"},{"signal":"Occupation","reason":"Not relevant"}],"strategy":{"message_type":"Transactional / reminder","objective":"Reminder / completion","primary_appeal":"Avoid unnecessary interest charges","tone":"Clear and professional","personalisation_level":"Transactional + contextual","key_value_proposition":"Complete payment before the deadline","cta":"Pay now"},"personalised_message":"Your credit-card payment is due tomorrow. Clear the remaining payment today to avoid interest charges on the outstanding amount.\n\n👉 Pay now","generic_message":"Your credit-card payment is due soon. Please complete your payment.","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"Low","cta_fit":"High","intrusiveness_risk":"Low"}}
    if "agoda" in company:
        return {"persona":"Travel-Engaged Feedback Seeker","consumer_insight":"The customer already engages with travel communication and the objective is improving future relevance.","creative_angle":"Give the traveler a voice in shaping future communication.","signals_used":["Travel interest","Existing relationship","Feedback objective","Email"],"signals_excluded":[{"signal":"Age","reason":"Not relevant"},{"signal":"Location","reason":"Not needed"}],"strategy":{"message_type":"Feedback / relationship","objective":"Feedback","primary_appeal":"Help tailor future communication","tone":"Friendly and appreciative","personalisation_level":"Relationship + preference","key_value_proposition":"Feedback can improve future travel communication","cta":"Start the survey"},"personalised_message":"Dear Bhavishya,\n\nHelp us make the travel messages you receive more useful to you. Tell us what you'd like to see more of—and what you'd rather skip.\n\n👉 Start the survey","generic_message":"We'd appreciate your feedback on our travel communications. Please take a short survey.","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}}
    # Generic fallback
    motivation=d.get("motivation","").lower()
    if d.get("price_sensitivity")=="High" or "save" in motivation or "budget" in motivation:
        appeal="Value / affordability"; body="Find an option that fits what you need without overspending."
    elif "convenience" in motivation or "time" in motivation:
        appeal="Convenience"; body="Make your next choice easier and fit it into your day."
    else:
        appeal="Relevant benefit"; body="Discover an option that fits what you're looking for."
    if "feedback" in obj.lower(): cta="Share your feedback"; mt="Feedback"
    elif "re-engagement" in obj.lower(): cta="Take another look"; mt="Re-engagement"
    elif "reminder" in obj.lower(): cta="Complete it now"; mt="Transactional / reminder"
    elif "cross" in obj.lower(): cta="Discover more"; mt="Cross-sell"
    elif "retention" in obj.lower() or "loyalty" in obj.lower(): cta="Keep exploring"; mt="Retention / loyalty"
    else: cta="Shop now"; mt="Promotional"
    hook=f"Still thinking about {d.get('product','this')}? 👀" if d.get("behaviour") else "A little something for your next move."
    return {"persona":d.get("segment") or "Context-Aware Consumer","consumer_insight":"Demo fallback uses the available consumer and situation inputs without presenting them as facts beyond what was supplied.","creative_angle":"Connect the strongest available motivation to the campaign objective.","signals_used":["Relevant supplied consumer/context signals"],"signals_excluded":[{"signal":"Unneeded optional details","reason":"They do not materially improve this message"}],"strategy":{"message_type":mt,"objective":obj,"primary_appeal":appeal,"tone":d.get("tone") if d.get("tone")!="AI chooses" else "Conversational","personalisation_level":"Contextual fallback logic","key_value_proposition":body,"cta":cta},"personalised_message":hook+"\n\n"+body+"\n\n👉 "+cta,"generic_message":f"Explore {d.get('product','this offering')} and see what's available.\n\n👉 {cta}","quality_check":{"relevance":"Medium","personalisation":"Medium","brand_fit":"Medium","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}}

# Header
st.markdown('<div class="hero"><h1>✨ AI Hyper-Personalisation Engine</h1><p>Dynamic communication conditioned on consumer behaviour, lifecycle, trigger, context and campaign objective.</p></div>',unsafe_allow_html=True)
with st.sidebar:
    mode=st.radio("Generation mode",["Live LLM","Demo / offline"],index=0)
    user_key=st.text_input("OpenRouter API key",type="password",help="Optional if OPENROUTER_API_KEY is saved in Streamlit Secrets.")
    model=st.text_input("Live model",PRIMARY_MODEL)
    st.caption("Live mode uses OpenRouter. Demo/offline is the fallback so the prototype remains usable if a free model is temporarily unavailable.")
tabs=st.tabs(["Single Consumer","Batch / At Scale","Impact Analysis","Prompt Architecture"])

with tabs[0]:
    st.subheader("Single Consumer — deep personalisation")
    st.markdown('<div class="helper">Fill what you actually know. Optional fields can be blank. The engine must not invent consumer facts.</div>',unsafe_allow_html=True)
    demo_name=st.selectbox("Load benchmark scenario",["None"]+list(DEMO.keys()))
    d0=DEMO.get(demo_name,{})
    with st.expander("1. Business context",True):
        c1,c2=st.columns(2)
        company=c1.text_input("Company *",d0.get("company",""),help="Brand/company.")
        industry=c2.text_input("Industry *",d0.get("industry",""),help="Business category.")
        c1,c2=st.columns(2)
        product=c1.text_input("Product / Service *",d0.get("product",""),help="Specific product/service/app/subscription.")
        positioning=c2.text_input("Brand positioning",d0.get("positioning",""),help="How the brand wants to be perceived. Optional.")
    with st.expander("2. Consumer context",True):
        segment=st.text_input("Consumer segment / persona",d0.get("segment",""),help="Optional if the other consumer inputs are sufficient.")
        c1,c2,c3=st.columns(3)
        age=c1.text_input("Age",d0.get("age","")); occupation=c2.text_input("Occupation",d0.get("occupation","")); location=c3.text_input("Geography",d0.get("location",""))
        behaviour=st.text_area("Behaviour / previous interactions",d0.get("behaviour",""))
        preferences=st.text_area("Preferences / interests",d0.get("preferences",""))
        motivation=st.text_area("Needs / motivation",d0.get("motivation",""))
        price=st.selectbox("Price sensitivity",["Unknown","Low","Medium","High"],index=["Unknown","Low","Medium","High"].index(d0.get("price_sensitivity","Unknown")) if d0.get("price_sensitivity","Unknown") in ["Unknown","Low","Medium","High"] else 0)
    with st.expander("3. Consumer state",True):
        lifecycle=st.selectbox("Lifecycle stage *",LIFECYCLES,index=LIFECYCLES.index(d0.get("lifecycle_stage","Active customer")) if d0.get("lifecycle_stage","Active customer") in LIFECYCLES else 2)
        trigger=st.text_input("Trigger / recent event *",d0.get("trigger",""))
        context=st.text_input("Current context / occasion",d0.get("context",""))
    with st.expander("4. Campaign & communication",True):
        c1,c2=st.columns(2)
        objective=c1.selectbox("Marketing objective *",OBJECTIVES,index=OBJECTIVES.index(d0.get("objective","Conversion")) if d0.get("objective","Conversion") in OBJECTIVES else 3)
        channel=c2.selectbox("Channel *",CHANNELS,index=CHANNELS.index(d0.get("channel","WhatsApp")) if d0.get("channel","WhatsApp") in CHANNELS else 0)
        c1,c2=st.columns(2)
        mtype=c1.selectbox("Message type",MESSAGE_TYPES,index=MESSAGE_TYPES.index(d0.get("message_type","AI chooses")) if d0.get("message_type","AI chooses") in MESSAGE_TYPES else 0)
        tone=c2.selectbox("Tone / style",TONES,index=TONES.index(d0.get("tone","AI chooses")) if d0.get("tone","AI chooses") in TONES else 0)
        length=st.selectbox("Message length",LENGTHS,index=LENGTHS.index(d0.get("length","Short")) if d0.get("length","Short") in LENGTHS else 1)
    data={"company":company,"industry":industry,"product":product,"positioning":positioning,"segment":segment,"age":age,"occupation":occupation,"location":location,"behaviour":behaviour,"preferences":preferences,"motivation":motivation,"price_sensitivity":price,"lifecycle_stage":lifecycle,"trigger":trigger,"context":context,"objective":objective,"channel":channel,"message_type":mtype,"tone":tone,"length":length}
    c1,c2=st.columns(2)
    generate=c1.button("🚀 Generate personalised content",type="primary",use_container_width=True)
    variant=c2.button("↻ Another creative version",use_container_width=True)
    if generate or variant:
        missing=[n for n,v in [("Company",company),("Industry",industry),("Product / Service",product),("Trigger / Recent Event",trigger)] if not v.strip()]
        key=get_key(user_key)
        if missing: st.error("Please fill: "+", ".join(missing))
        elif mode=="Live LLM" and not key: st.error("OpenRouter API key not found. Add it to Streamlit Secrets or the sidebar.")
        else:
            try:
                with st.spinner("Creating the consumer interpretation and message..."):
                    r=live_generate(data,key,model,variant) if mode=="Live LLM" else demo_generate(data)
                st.session_state["single"]=r; st.session_state["single_data"]=data; st.session_state["single_mode"]=mode
                st.success("Generated.")
            except Exception as e:
                st.error(f"{e}\n\nSwitch to Demo / offline if the free model is temporarily unavailable.")
    r=st.session_state.get("single")
    if r:
        st.divider(); st.subheader("AI output")
        if st.session_state.get("single_mode")=="Demo / offline": st.warning("Demo/offline output — not a live LLM result.")
        elif r.get("_model_used"): st.markdown(f'<div class="status">Live model used: <strong>{r["_model_used"]}</strong></div>',unsafe_allow_html=True)
        c1,c2=st.columns([1.35,1])
        with c1:
            st.markdown("### AI Consumer Persona"); st.markdown("**"+str(r["persona"])+"**"); st.write(r["consumer_insight"])
            st.markdown("### Creative angle"); st.markdown(f'<div class="insight">{r["creative_angle"]}</div>',unsafe_allow_html=True)
        with c2:
            st.markdown("### Quality check")
            for k,v in r["quality_check"].items(): st.write(f"**{k.replace('_',' ').title()}:** {v}")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("### Signals used")
            for s in r["signals_used"]: st.markdown(f'<span class="chip">✓ {s}</span>',unsafe_allow_html=True)
        with c2:
            st.markdown("### Signals excluded")
            for x in r["signals_excluded"]: st.markdown(f'<span class="chip gray">{x["signal"]}</span> — {x["reason"]}',unsafe_allow_html=True)
        st.markdown("### Communication strategy")
        s=r["strategy"]
        st.dataframe(pd.DataFrame([["Message type",s["message_type"]],["Objective",s["objective"]],["Primary appeal",s["primary_appeal"]],["Tone",s["tone"]],["Personalisation",s["personalisation_level"]],["Key value proposition",s["key_value_proposition"]],["CTA",s["cta"]]],columns=["Decision","Selected approach"]),use_container_width=True,hide_index=True)
        c1,c2=st.columns(2)
        with c1: st.markdown("### ✨ AI-Personalised Message"); st.markdown(f'<div class="message">{r["personalised_message"]}</div>',unsafe_allow_html=True)
        with c2: st.markdown("### Generic Control"); st.markdown(f'<div class="message generic">{r["generic_message"]}</div>',unsafe_allow_html=True)
        st.markdown("### Why this is personalised"); st.write("The personalised version uses relevant consumer and situation signals to change the communication. The generic control intentionally excludes consumer-specific information.")
        if st.button("Add this pair to Impact Analysis",key="impact_add"):
            st.session_state["impact_pair"]={"case":data["company"]+" — "+data["product"],"personalised":r["personalised_message"],"generic":r["generic_message"]}
            st.success("Added.")

with tabs[1]:
    st.subheader("Batch / At Scale")
    st.markdown('<div class="helper">Upload a CSV or Excel with the same field names as the Single Consumer form. The engine processes each row independently.</div>',unsafe_allow_html=True)
    cols=["company","industry","product","positioning","segment","age","occupation","location","behaviour","preferences","motivation","price_sensitivity","lifecycle_stage","trigger","context","objective","channel","message_type","tone","length"]
    st.download_button("Download batch template",pd.DataFrame([{c:"" for c in cols}]).to_csv(index=False).encode(),"hyper_personalisation_batch_template.csv","text/csv")
    up=st.file_uploader("Upload consumer dataset",type=["csv","xlsx","xls"],key="batch")
    if up:
        try:
            df=pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            st.write(f"**{len(df)} consumers loaded.**")
            st.dataframe(df.head(10),use_container_width=True,hide_index=True)
            if st.button("⚡ Generate for all consumers",type="primary",use_container_width=True):
                key=get_key(user_key)
                missing=[c for c in ["company","industry","product","trigger","objective"] if c not in df.columns]
                if missing: st.error("Missing columns: "+", ".join(missing))
                elif mode=="Live LLM" and not key: st.error("OpenRouter API key not found.")
                else:
                    out=[]; bar=st.progress(0)
                    for i,row in df.fillna("").iterrows():
                        d={c:str(row.get(c,"")) for c in cols}
                        d["lifecycle_stage"]=d["lifecycle_stage"] or "Active customer"; d["objective"]=d["objective"] or "Conversion"; d["channel"]=d["channel"] or "WhatsApp"; d["message_type"]=d["message_type"] or "AI chooses"; d["tone"]=d["tone"] or "AI chooses"; d["length"]=d["length"] or "Short"
                        try:
                            z=live_generate(d,key,model) if mode=="Live LLM" else demo_generate(d)
                            out.append({"Consumer #":i+1,"Company":d["company"],"Product":d["product"],"Persona":z["persona"],"Lifecycle":d["lifecycle_stage"],"Trigger":d["trigger"],"Message Type":z["strategy"]["message_type"],"Creative Angle":z["creative_angle"],"Primary Appeal":z["strategy"]["primary_appeal"],"Tone":z["strategy"]["tone"],"CTA":z["strategy"]["cta"],"Personalised Message":z["personalised_message"],"Generic Message":z["generic_message"]})
                        except Exception as e:
                            out.append({"Consumer #":i+1,"Company":d["company"],"Product":d["product"],"Persona":"ERROR","Lifecycle":d["lifecycle_stage"],"Trigger":d["trigger"],"Message Type":"ERROR","Creative Angle":"","Primary Appeal":"","Tone":"","CTA":str(e),"Personalised Message":"","Generic Message":""})
                        bar.progress((i+1)/len(df))
                    st.session_state["batch_outputs"]=pd.DataFrame(out); st.session_state["batch_mode"]=mode
                    st.success("Batch processing complete.")
        except Exception as e: st.error(f"Could not read dataset: {e}")
    bo=st.session_state.get("batch_outputs")
    if bo is not None:
        st.dataframe(bo,use_container_width=True,hide_index=True)
        st.download_button("Download generated results",bo.to_csv(index=False).encode(),"hyper_personalisation_results.csv","text/csv")
        if len(bo):
            pick=st.selectbox("Inspect one consumer",bo["Consumer #"].astype(str).tolist())
            rr=bo[bo["Consumer #"].astype(str)==pick].iloc[0]
            c1,c2=st.columns(2)
            with c1: st.markdown("### Personalised"); st.markdown(f'<div class="message">{rr["Personalised Message"] or "No output"}</div>',unsafe_allow_html=True)
            with c2: st.markdown("### Generic"); st.markdown(f'<div class="message generic">{rr["Generic Message"] or "No output"}</div>',unsafe_allow_html=True)
            st.write(f"**Creative angle:** {rr['Creative Angle']}")
            st.write(f"**Message type:** {rr['Message Type']}  | **CTA:** {rr['CTA']}")

with tabs[2]:
    st.subheader("Personalisation Impact Analysis")
    st.markdown('<div class="helper">This is the actual research module. Use real respondent ratings for the generic vs personalised pair.</div>',unsafe_allow_html=True)
    pair=st.session_state.get("impact_pair")
    if pair:
        st.write(f"**Study case:** {pair['case']}")
        c1,c2=st.columns(2)
        with c1: st.markdown("**Personalised**"); st.markdown(f'<div class="message">{pair["personalised"]}</div>',unsafe_allow_html=True)
        with c2: st.markdown("**Generic**"); st.markdown(f'<div class="message generic">{pair["generic"]}</div>',unsafe_allow_html=True)
        n=st.number_input("Number of respondents",1,100,10,1)
        if st.button("Create / reset rating table"):
            cc=["Respondent"]
            for dim in DIMENSIONS: cc += [f"P — {dim}",f"G — {dim}"]
            st.session_state["ratings"]=pd.DataFrame([[f"R{i:03d}"]+[3]*(len(cc)-1) for i in range(1,int(n)+1)],columns=cc)
        ratings=st.session_state.get("ratings")
        if ratings is not None:
            edited=st.data_editor(ratings,use_container_width=True,num_rows="fixed")
            st.session_state["ratings"]=edited
            rows=[]
            for dim in DIMENSIONS:
                p=pd.to_numeric(edited[f"P — {dim}"],errors="coerce").mean()
                g=pd.to_numeric(edited[f"G — {dim}"],errors="coerce").mean()
                rows.append([dim,round(p,2),round(g,2),round(p-g,2)])
            summary=pd.DataFrame(rows,columns=["Dimension","Personalised mean","Generic mean","Difference (P-G)"])
            st.markdown("### Results")
            st.dataframe(summary,use_container_width=True,hide_index=True)
            st.bar_chart(summary.set_index("Dimension")[["Personalised mean","Generic mean"]])
            st.write("Positive P-G means personalised content scored higher. For Intrusiveness, a negative P-G is generally preferable.")
            st.download_button("Download ratings",edited.to_csv(index=False).encode(),"personalisation_impact_ratings.csv","text/csv")
            st.download_button("Download summary",summary.to_csv(index=False).encode(),"personalisation_impact_summary.csv","text/csv")
    else:
        st.info("Generate a message in Single Consumer Mode and click “Add this pair to Impact Analysis”.")

with tabs[3]:
    st.subheader("Prompt Architecture")
    st.markdown("""
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
""")
    st.code("Signal → Meaning → Strategy → Creative angle → Message → CTA → Quality check",language="text")
