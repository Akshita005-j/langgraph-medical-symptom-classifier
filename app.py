import streamlit as st
from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# -------------------------------
# Streamlit configuration
# -------------------------------
st.set_page_config(
    page_title="AI Medical Symptom Classifier",
    layout="centered"
)

st.title("🏥 AI Medical Symptom Classifier")
st.write("Enter your symptom and get guidance.")

# -------------------------------
# Initialize LLM
# -------------------------------
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

llm = get_llm()

# -------------------------------
# Define Graph State
# -------------------------------
class GraphState(TypedDict, total=False):
    symptom: str
    category: str
    answer: str

# -------------------------------
# Node: Classify symptom
# -------------------------------
def classify_symptom(state: GraphState) -> dict:
    symptom = state.get("symptom", "").strip()

    prompt = f"""
Classify the following symptom into exactly one category:
General
Emergency
Mental Health

Return only the category name.

Symptom: {symptom}
"""

    response = llm.invoke([
        SystemMessage(
            content="You are a cautious medical symptom classification assistant."
        ),
        HumanMessage(content=prompt)
    ])

    category = response.content.strip().lower()

    # Normalize the model's response
    if "emergency" in category:
        category = "emergency"
    elif "mental health" in category or "mental_health" in category:
        category = "mental health"
    else:
        category = "general"

    return {"category": category}

# -------------------------------
# Nodes: Guidance
# -------------------------------
def general(state: GraphState) -> dict:
    return {
        "answer": (
            "This may be a non-emergency issue. Consider consulting "
            "a qualified healthcare professional for an assessment. "
            "Seek urgent care if symptoms are severe or worsening."
        )
    }

def emergency(state: GraphState) -> dict:
    return {
        "answer": (
            "Your symptoms may require urgent assessment. Contact "
            "local emergency services or seek emergency medical care "
            "immediately if you may be experiencing a medical emergency."
        )
    }

def mental_health(state: GraphState) -> dict:
    return {
        "answer": (
            "Consider speaking with a qualified mental health professional. "
            "If you are in immediate danger or may harm yourself, contact "
            "local emergency services or a crisis support service."
        )
    }

# -------------------------------
# Build LangGraph
# -------------------------------
builder = StateGraph(GraphState)

builder.add_node("classify", classify_symptom)
builder.add_node("general", general)
builder.add_node("emergency", emergency)
builder.add_node("mental_health", mental_health)

builder.set_entry_point("classify")

def route_category(state: GraphState) -> Literal[
    "general", "emergency", "mental_health"
]:
    category = state.get("category", "general").lower().strip()

    if category == "emergency":
        return "emergency"
    elif category in ("mental health", "mental_health"):
        return "mental_health"
    else:
        return "general"

builder.add_conditional_edges(
    "classify",
    route_category,
    {
        "general": "general",
        "emergency": "emergency",
        "mental_health": "mental_health"
    }
)

builder.add_edge("general", END)
builder.add_edge("emergency", END)
builder.add_edge("mental_health", END)

graph = builder.compile()

# -------------------------------
# Streamlit UI
# -------------------------------
symptom = st.text_input("Enter your symptom")

if st.button("Analyze"):
    if not symptom.strip():
        st.warning("Please enter a symptom.")
    else:
        try:
            with st.spinner("Analyzing your symptom..."):
                result = graph.invoke({"symptom": symptom.strip()})

            st.success(f"Category: {result.get('category', 'General').title()}")
            st.info(result.get("answer", "Please consult a healthcare professional."))

            st.caption(
                "This tool provides general guidance, not a diagnosis. "
                "It cannot reliably rule out emergencies. Seek urgent care "
                "for severe or rapidly worsening symptoms."
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
```
