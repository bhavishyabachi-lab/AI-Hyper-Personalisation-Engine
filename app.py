
import os, json
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI Hyper-Personalisation Engine", page_icon="✨", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1220px;padding-top:2rem;padding-bottom:4rem}
.hero{padding:28px 30px;border-radius:18px;background:linear-gradient(135deg,#20234a,#5b5ce2);color:white;margin-bottom:22px}
.hero h1{margin:0 0 7px;font-size:2.1rem}.hero p{margin:0;opacity:.9}
.card{border:1px solid #e5e9f2;border-radius:16px;padding:18px;background:#fff;margin-bottom:16px}
.message{border:1px solid #d8d8f7;border-radius:15px;padding:20px;background:#f8f8ff;line-height:1.65;white-space:pre-wrap}
.generic{border-color:#e5e9f2;background:#fafbfe}
.chip{display:inline-block;padding:6px 10px;border-radius:999px;background:#efefff;color:#4b4dc0;margin:3px 4px 3px 0;font-size:.78rem}
.chip.gray{background:#f2f3f6;color:#68748a}.helper{font-size:.82rem;color:#68748a;line-height:1.45}
</style>
""", unsafe_allow_html=True)

MODEL=os.getenv("OPENAI_MODEL","gpt-5.6-luna")
CHANNELS=["WhatsApp","SMS","Email","Push notification","Social media","Website","In-app message"]
LIFECYCLES=["Prospect","New customer","Active customer","Loyal customer","At-risk","Inactive","Churned / potentially churned"]
OBJECTIVES=["Awareness","Engagement","Consideration","Conversion","Re-engagement","Retention","Feedback","Win-back","Cross-sell","Upsell","Loyalty","Reminder / completion"]
MESSAGE_TYPES=["AI chooses","Promotional","Awareness","Consideration","Product recommendation","Behaviour-triggered","Abandonment","Re-engagement","Retention","Loyalty / reward","Cross-sell","Upsell","Feedback","Service / recovery","Transactional / reminder","Win-back","Educational","Occasion-based"]
TONES=["AI chooses","Conversational","Friendly","Quirky","Humorous","Premium","Emotional","Urgent","Empathetic","Professional","Playful","Informative"]
LENGTHS=["Very short","Short","Medium","Long"]

SYSTEM_PROMPT=r"""
You are the generation intelligence inside a generic AI Hyper-Personalisation Engine.

Your goal is to create communication that feels like something a real brand would
actually send, while being meaningfully adapted to the consumer and current situation.

PRIORITY
1. Situational appropriateness
2. Consumer relevance
3. Consumer-behaviour meaning
4. Brand authenticity
5. Objective clarity
6. Naturalness
7. Creativity
8. Channel/length fit

Do not optimise for creativity at the expense of relevance, trust or clarity.

CONSUMER BEHAVIOUR
Use behaviour, needs, motivation, preferences, lifecycle, trigger and context only when
they materially help. Demographics are secondary unless they change relevance.
Never invent consumer facts. Treat inferred motivations as hypotheses.
More data is not automatically better personalisation.

SIGNAL SELECTION
Review all supplied signals and use the minimum useful set. Do not echo raw tracking
data unnecessarily.

SIGNAL -> MEANING
Translate signals into a communication implication before writing.
Example: repeated browsing can indicate interest plus unresolved decision friction.
Do not merely repeat "you viewed this product twice."

STRATEGY
Determine message type, objective, primary appeal, desired response, tone, personalisation
level, key value proposition and CTA. If type/tone is "AI chooses", infer it.

CREATIVE ANGLE
Select one dominant creative angle: reassurance, value, convenience, discovery, urgency,
emotional connection, occasion/social context, recognition, problem-solution,
playful challenge, exclusivity, or celebration. Do not force humour into sensitive or
transactional situations.

MESSAGE
The personalised message must:
- be plausibly sendable by the stated brand/industry
- reflect at least one meaningful consumer signal
- use the creative angle
- fit the channel and requested length
- respect brand positioning
- avoid unnecessary tracking disclosure
- avoid invented facts
- contain a natural CTA
- avoid generic "AI copy" language

Never write phrases such as "Based on your behaviour..." or "As a price-sensitive customer..."

CTA
Always include an explicit CTA inside the personalised message. The CTA must match the
objective: conversion -> shop/order/book; feedback -> share feedback/start survey;
awareness -> learn more; re-engagement -> take another look; retention -> keep exploring;
win-back -> come back; cross-sell -> discover more; reminder -> pay/complete.

GENERIC CONTROL
Generate a plausible generic version using only company/product/basic objective.
Do not use consumer identity, behaviour, motivation, lifecycle, preferences or trigger.

QUALITY REVIEW
Internally check:
- meaningful personalisation
- brand realism
- clear creative idea
- CTA fit
- appropriate tone
- low unnecessary intrusiveness
- no invented facts
Rewrite if weak.

Do not reveal hidden chain-of-thought. Return only concise user-facing analysis and the final output.
"""

SCHEMA={"type":"object","additionalProperties":False,"properties":{
"persona":{"type":"string"},"consumer_insight":{"type":"string"},"creative_angle":{"type":"string"},
"signals_used":{"type":"array","items":{"type":"string"}},
"signals_excluded":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{"signal":{"type":"string"},"reason":{"type":"string"}},"required":["signal","reason"]}},
"strategy":{"type":"object","additionalProperties":False,"properties":{
"message_type":{"type":"string"},"objective":{"type":"string"},"primary_appeal":{"type":"string"},
"tone":{"type":"string"},"personalisation_level":{"type":"string"},"key_value_proposition":{"type":"string"},"cta":{"type":"string"}},
"required":["message_type","objective","primary_appeal","tone","personalisation_level","key_value_proposition","cta"]},
"personalised_message":{"type":"string"},"generic_message":{"type":"string"},
"quality_check":{"type":"object","additionalProperties":False,"properties":{
"relevance":{"type":"string"},"personalisation":{"type":"string"},"brand_fit":{"type":"string"},
"creativity":{"type":"string"},"cta_fit":{"type":"string"},"intrusiveness_risk":{"type":"string"}},
"required":["relevance","personalisation","brand_fit","creativity","cta_fit","intrusiveness_risk"]}},
"required":["persona","consumer_insight","creative_angle","signals_used","signals_excluded","strategy","personalised_message","generic_message","quality_check"]}

DEMOS={
"Practo — app uninstall / feedback":{
"company":"Practo","industry":"Healthcare / health-tech","product":"Practo mobile app","positioning":"Convenient digital healthcare access","segment":"Existing app user","age":"","occupation":"","location":"India","behaviour":"Previously used the app and then uninstalled it","preferences":"","motivation":"","price_sensitivity":"Unknown","lifecycle_stage":"Churned / potentially churned","trigger":"App uninstallation","context":"Shortly after uninstall","objective":"Feedback","channel":"WhatsApp","message_type":"AI chooses","tone":"Empathetic","length":"Short"},
"Domino's — Raksha Bandhan / promotion":{
"company":"Domino's Pizza India","industry":"Food & beverage","product":"Pizza","positioning":"Convenient, playful and value-focused","segment":"Occasion-oriented household","age":"","occupation":"","location":"India","behaviour":"","preferences":"Pizza; shared meals","motivation":"Celebrate Raksha Bandhan together","price_sensitivity":"Medium","lifecycle_stage":"Active customer","trigger":"Raksha Bandhan","context":"Festival occasion","objective":"Conversion","channel":"WhatsApp","message_type":"AI chooses","tone":"Quirky","length":"Short"},
"Myntra — repeated sneaker browsing":{
"company":"Myntra","industry":"Fashion e-commerce","product":"Running shoes","positioning":"Trendy, youthful and accessible","segment":"Budget-conscious fitness consumer","age":"24","occupation":"Young professional","location":"Bengaluru","behaviour":"Frequently browses running shoes and compares prices","preferences":"Running; minimalist designs","motivation":"Fitness plus value for money","price_sensitivity":"High","lifecycle_stage":"Active customer","trigger":"Viewed running shoes twice recently","context":"Weekend","objective":"Conversion","channel":"WhatsApp","message_type":"AI chooses","tone":"Quirky","length":"Short"},
"CRED — payment reminder":{
"company":"CRED","industry":"Fintech","product":"Credit card bill payment","positioning":"Premium, frictionless financial utility","segment":"Existing credit-card user","age":"","occupation":"","location":"India","behaviour":"Has an outstanding credit-card balance","preferences":"","motivation":"Avoid interest charges and complete payment","price_sensitivity":"Unknown","lifecycle_stage":"Active customer","trigger":"Credit-card payment due tomorrow","context":"Payment deadline","objective":"Reminder / completion","channel":"WhatsApp","message_type":"AI chooses","tone":"Professional","length":"Short"},
"Agoda — communications feedback":{
"company":"Agoda","industry":"Travel & hospitality","product":"Travel booking platform","positioning":"Helpful and traveler-focused","segment":"Existing traveler","age":"","occupation":"","location":"India","behaviour":"Has interacted with Agoda communications","preferences":"Travel","motivation":"Wants more relevant travel communication","price_sensitivity":"Unknown","lifecycle_stage":"Active customer","trigger":"Feedback request","context":"Post-interaction","objective":"Feedback","channel":"Email","message_type":"AI chooses","tone":"Friendly","length":"Medium"}}

def make_demo(d):
    c=d["company"].lower(); t=d["trigger"].lower()
    if "practo" in c and "uninstall" in t:
        return {"persona":"Recently Churned Healthcare-App User","consumer_insight":"The strongest known signal is disengagement after uninstall. The reason for leaving is unknown and must not be invented.","creative_angle":"Invite the customer to help improve the experience without pressure.","signals_used":["Existing relationship","App uninstallation","Churned lifecycle","Feedback objective","WhatsApp"],"signals_excluded":[{"signal":"Age","reason":"Not provided and not needed"},{"signal":"Occupation","reason":"Not relevant"},{"signal":"Reason for uninstalling","reason":"Unknown"}],"strategy":{"message_type":"Feedback / recovery","objective":"Feedback","primary_appeal":"Help improve the experience","tone":"Friendly and empathetic","personalisation_level":"Behavioural + lifecycle + trigger","key_value_proposition":"Feedback can improve the experience","cta":"Share your feedback"},"personalised_message":"Bhavishya Paila, can we ask you one quick question? 👀\n\nWe noticed you're no longer using the Practo App. We'd really like to know what we could have done better.\n\n👉 Share your feedback","generic_message":"We'd love your feedback on your experience with our app. Please share your thoughts with us.","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}}
    if "domino" in c:
        return {"persona":"Occasion-Oriented Pizza Sharer","consumer_insight":"The festival occasion creates a social consumption moment where celebration and sharing are more relevant than demographic targeting.","creative_angle":"Turn sibling rivalry into a playful pizza moment.","signals_used":["Raksha Bandhan occasion","Shared-meal context","Conversion objective","Quirky tone","WhatsApp"],"signals_excluded":[{"signal":"Age","reason":"Not needed for an occasion-led message"},{"signal":"Occupation","reason":"Not relevant to the occasion"}],"strategy":{"message_type":"Occasion-based promotion","objective":"Conversion","primary_appeal":"Celebration + value","tone":"Quirky and festive","personalisation_level":"Contextual + occasion-based","key_value_proposition":"Make the shared celebration more rewarding","cta":"Order now"},"personalised_message":"Raksha Bandhan plans? 🍕 No sibling fights over the last slice today. Give everyone a favourite and make the celebration a little more delicious.\n\n👉 Order now","generic_message":"Enjoy our latest pizza offers today.\n\n👉 Order now","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"High","cta_fit":"High","intrusiveness_risk":"Low"}}
    if "cred" in c and "due" in t:
        return {"persona":"Deadline-Driven Credit Card User","consumer_insight":"An immediate payment deadline makes clarity and fast action more important than entertainment.","creative_angle":"Make the deadline and consequence unmistakably clear.","signals_used":["Payment due trigger","Outstanding balance","Reminder objective","WhatsApp"],"signals_excluded":[{"signal":"Age","reason":"Not relevant"},{"signal":"Occupation","reason":"Not relevant"}],"strategy":{"message_type":"Transactional / reminder","objective":"Reminder / completion","primary_appeal":"Avoid unnecessary interest charges","tone":"Clear and professional","personalisation_level":"Transactional + contextual","key_value_proposition":"Complete payment before the deadline","cta":"Pay now"},"personalised_message":"Your credit-card payment is due tomorrow. Clear the remaining payment today to avoid interest charges on the outstanding amount.\n\n👉 Pay now","generic_message":"Your credit-card payment is due soon. Please complete your payment.","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"Low","cta_fit":"High","intrusiveness_risk":"Low"}}
    if "agoda" in c:
        return {"persona":"Travel-Engaged Feedback Seeker","consumer_insight":"The customer already interacts with travel communication and the objective is improving relevance, so feedback is more appropriate than promotion.","creative_angle":"Give the traveler a voice in shaping future communication.","signals_used":["Travel interest","Existing relationship","Feedback objective","Email"],"signals_excluded":[{"signal":"Age","reason":"Not relevant"},{"signal":"Location","reason":"No location-specific content is needed"}],"strategy":{"message_type":"Feedback / relationship","objective":"Feedback","primary_appeal":"Help tailor future communication","tone":"Friendly and appreciative","personalisation_level":"Relationship + preference","key_value_proposition":"Feedback can improve future travel communication","cta":"Start the survey"},"personalised_message":"Dear Bhavishya,\n\nHelp us make the travel messages you receive more useful to you. Tell us what you'd like to see more of—and what you'd rather skip.\n\n👉 Start the survey","generic_message":"We'd appreciate your feedback on our travel communications. Please take a short survey.","quality_check":{"relevance":"High","personalisation":"High","brand_fit":"High","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}}
    appeal="Value / affordability" if d.get("price_sensitivity")=="High" else "Convenience" if "convenience" in d.get("motivation","").lower() else "Relevant product benefit"
    cta="Share your feedback" if "feedback" in d["objective"].lower() else "Shop now" if "conversion" in d["objective"].lower() else "Learn more"
    hook=f"Still thinking about {d['product']}? 👀" if d.get("behaviour") else "Here’s something worth exploring."
    body="Find an option that fits what you need without overspending." if "Value" in appeal else "Make your next choice a little easier." if appeal=="Convenience" else "Discover something that fits what you're looking for."
    return {"persona":d["segment"],"consumer_insight":"The supplied behaviour, motivation and lifecycle information is used to adapt the communication without forcing every field into the copy.","creative_angle":"Connect the most relevant consumer motivation to the campaign objective naturally.","signals_used":[x for x, key in [("Behaviour","behaviour"),("Motivation","motivation"),("Price sensitivity","price_sensitivity"),("Lifecycle stage","lifecycle_stage"),("Trigger","trigger"),("Context","context")] if d.get(key)],"signals_excluded":[{"signal":"Age","reason":"Only useful if age materially changes relevance"},{"signal":"Occupation","reason":"Only useful if it changes the communication context"}],"strategy":{"message_type":d["message_type"] if d["message_type"]!="AI chooses" else "Behaviour-triggered promotional","objective":d["objective"],"primary_appeal":appeal,"tone":d["tone"] if d["tone"]!="AI chooses" else "Conversational","personalisation_level":"Behavioural + motivational + contextual","key_value_proposition":body,"cta":cta},"personalised_message":hook+"\n\n"+body+"\n\n👉 "+cta,"generic_message":f"Explore {d['product']} and see what's available.\n\n👉 {cta}","quality_check":{"relevance":"Medium","personalisation":"Medium","brand_fit":"Medium","creativity":"Medium","cta_fit":"High","intrusiveness_risk":"Low"}}

def llm(d,key,model):
    client=OpenAI(api_key=key)
    r=client.responses.create(model=model,instructions=SYSTEM_PROMPT,input="CASE DATA\n"+json.dumps(d,ensure_ascii=False),text={"format":{"type":"json_schema","name":"hyper_personalisation_output","strict":True,"schema":SCHEMA}})
    return enforce(json.loads(r.output_text))

def enforce(r):
    c=r.get("strategy",{}).get("cta","").strip()
    if c and c.lower() not in r.get("personalised_message","").lower():
        r["personalised_message"]=r["personalised_message"].rstrip()+f"\n\n👉 {c}"
    return r

st.markdown('<div class="hero"><h1>✨ AI Hyper-Personalisation Engine</h1><p>One generic engine. Dynamic, consumer-behaviour-conditioned communication across brands, situations and scale.</p></div>',unsafe_allow_html=True)
with st.sidebar:
    mode=st.radio("Generation mode",["Live LLM","Demo / offline"],index=0)
    api_key=st.text_input("OpenAI API key",type="password")
    model=st.text_input("Model",MODEL)
    st.caption("Live mode uses the LLM. Demo mode is for interface testing without an API connection.")

tabs=st.tabs(["Single Consumer","Batch / At Scale","Impact Analysis","Prompt Architecture"])

with tabs[0]:
    st.subheader("Create one communication")
    demo=st.selectbox("Load benchmark scenario",["None"]+list(DEMOS.keys()))
    d0=DEMOS.get(demo,{})
    with st.expander("1. Business context",True):
        c1,c2=st.columns(2)
        company=c1.text_input("Company *",d0.get("company",""),help="Brand/company. Example: Myntra, Practo, Nike.")
        industry=c2.text_input("Industry *",d0.get("industry",""),help="Business category.")
        c1,c2=st.columns(2)
        product=c1.text_input("Product / Service *",d0.get("product",""),help="Specific product, service, app, subscription or offer.")
        positioning=c2.text_input("Brand positioning",d0.get("positioning",""),help="How the brand wants to be perceived. Optional.")
    with st.expander("2. Consumer context",True):
        segment=st.text_input("Consumer segment / persona *",d0.get("segment",""),help="Describe the target consumer or segment.")
        c1,c2,c3=st.columns(3)
        age=c1.text_input("Age",d0.get("age","")); occupation=c2.text_input("Occupation",d0.get("occupation","")); location=c3.text_input("Geography",d0.get("location",""))
        behaviour=st.text_area("Behaviour / previous interactions",d0.get("behaviour",""),help="What the consumer actually did.")
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
    if st.button("🚀 Generate personalised content",type="primary",use_container_width=True):
        missing=[n for n,v in [("Company",company),("Industry",industry),("Product / Service",product),("Consumer Segment / Persona",segment),("Trigger / Recent Event",trigger)] if not v.strip()]
        if missing: st.error("Please fill: "+", ".join(missing))
        elif mode=="Live LLM" and not api_key: st.error("Add an OpenAI API key in the sidebar, or switch to Demo / offline.")
        else:
            try:
                with st.spinner("Interpreting → selecting signals → choosing creative angle → generating → checking..."):
                    result=enforce(llm(data,api_key,model) if mode=="Live LLM" else make_demo(data))
                st.session_state["single_result"]=result; st.session_state["single_data"]=data
            except Exception as e: st.error(f"Generation failed: {e}")
    r=st.session_state.get("single_result")
    if r:
        st.divider(); st.subheader("AI output")
        c1,c2=st.columns([1.35,1])
        with c1:
            st.markdown("### AI Consumer Persona"); st.markdown("**"+r["persona"]+"**"); st.write(r["consumer_insight"])
            st.markdown("### Creative angle"); st.info(r["creative_angle"])
        with c2:
            st.markdown("### AI quality check")
            for k,v in r["quality_check"].items(): st.write(f"**{k.replace('_',' ').title()}:** {v}")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("### Signals used")
            for s in r["signals_used"]: st.markdown(f'<span class="chip">✓ {s}</span>',unsafe_allow_html=True)
        with c2:
            st.markdown("### Signals excluded")
            for x in r["signals_excluded"]: st.markdown(f'<span class="chip gray">{x["signal"]}</span> — {x["reason"]}',unsafe_allow_html=True)
        st.markdown("### Communication strategy")
        s=r["strategy"]; cols=st.columns(4)
        for col,key,label in zip(cols,["message_type","objective","primary_appeal","cta"],["Message type","Objective","Primary appeal","CTA"]): col.metric(label,s[key])
        c1,c2,c3=st.columns(3); c1.metric("Tone",s["tone"]); c2.metric("Personalisation",s["personalisation_level"]); c3.metric("Key value proposition",s["key_value_proposition"])
        c1,c2=st.columns(2)
        with c1: st.markdown("### ✨ AI-Personalised Message"); st.markdown(f'<div class="message">{r["personalised_message"]}</div>',unsafe_allow_html=True)
        with c2: st.markdown("### Generic Control"); st.markdown(f'<div class="message generic">{r["generic_message"]}</div>',unsafe_allow_html=True)
        st.markdown("### Why this is personalised"); st.write("The personalised version uses the relevant consumer and situation signals to change the communication approach; the generic control excludes consumer-specific information.")

with tabs[1]:
    st.subheader("Batch / At Scale")
    st.markdown('<div class="helper">Upload CSV or Excel using the same columns as Single Consumer Mode. One engine is applied to every row.</div>',unsafe_allow_html=True)
    template=pd.DataFrame([DEMOS["Myntra — repeated sneaker browsing"]])
    st.download_button("Download sample CSV template",template.to_csv(index=False).encode(),"hyper_personalisation_template.csv","text/csv")
    up=st.file_uploader("Upload consumer dataset",type=["csv","xlsx","xls"])
    if up:
        try:
            df=pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            st.write(f"**{len(df)} consumers loaded.**"); st.dataframe(df.head(10),use_container_width=True,hide_index=True)
            if st.button("⚡ Generate for all consumers",type="primary",use_container_width=True):
                if mode=="Live LLM" and not api_key: st.error("Add an OpenAI API key for live generation.")
                else:
                    required_cols=["company","industry","product","segment","trigger","objective"]
                    missing=[c for c in required_cols if c not in df.columns]
                    if missing: st.error("Missing columns: "+", ".join(missing))
                    else:
                        outs=[]; bar=st.progress(0)
                        for i,row in df.fillna("").iterrows():
                            d={k:str(row.get(k,"")) for k in ["company","industry","product","positioning","segment","age","occupation","location","behaviour","preferences","motivation","price_sensitivity","lifecycle_stage","trigger","context","objective","channel","message_type","tone","length"]}
                            d["lifecycle_stage"]=d["lifecycle_stage"] or "Active customer"; d["objective"]=d["objective"] or "Conversion"; d["channel"]=d["channel"] or "WhatsApp"; d["message_type"]=d["message_type"] or "AI chooses"; d["tone"]=d["tone"] or "AI chooses"; d["length"]=d["length"] or "Short"
                            try:
                                z=enforce(llm(d,api_key,model) if mode=="Live LLM" else make_demo(d))
                                outs.append({"Consumer #":i+1,"Company":d["company"],"Product":d["product"],"Persona":z["persona"],"Lifecycle":d["lifecycle_stage"],"Trigger":d["trigger"],"Message Type":z["strategy"]["message_type"],"CTA":z["strategy"]["cta"],"Personalised Message":z["personalised_message"],"Generic Message":z["generic_message"]})
                            except Exception as e:
                                outs.append({"Consumer #":i+1,"Company":d["company"],"Product":d["product"],"Persona":"ERROR","Lifecycle":d["lifecycle_stage"],"Trigger":d["trigger"],"Message Type":"ERROR","CTA":str(e),"Personalised Message":"","Generic Message":""})
                            bar.progress((i+1)/len(df))
                        st.session_state["batch"]=pd.DataFrame(outs)
                        st.success("Batch processing complete.")
        except Exception as e: st.error(f"Could not read the file: {e}")
    if "batch" in st.session_state:
        st.dataframe(st.session_state["batch"],use_container_width=True,hide_index=True)
        st.download_button("Download generated results",st.session_state["batch"].to_csv(index=False).encode(),"hyper_personalisation_results.csv","text/csv")

with tabs[2]:
    st.subheader("Personalisation Impact Analysis")
    st.markdown("Use actual respondents for the final analysis. Suggested dimensions: Relevance, Perceived personalisation, Persuasiveness, Emotional appeal, Purchase/engagement intention, Brand authenticity and Intrusiveness.")
    st.info("The final study should compare the same personalised and generic message pairs with real respondent ratings. Do not present illustrative values as actual findings.")

with tabs[3]:
    st.subheader("Prompt Architecture")
    st.markdown("""
**Inputs → Consumer interpretation → Relevant signal selection → Signal meaning → Communication strategy → Creative angle → LLM generation → CTA → Quality check → Generic control**

### Key generation principle
**Signal → Meaning → Creative angle → Message**

That is the main distinction between this engine and simple template-based personalisation.
""")
    st.code("Understand → Select → Interpret → Strategise → Create → Generate → Check",language="text")
