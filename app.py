import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, END

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="AI Medical Symptom Classifier",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 AI Medical Symptom Classifier")
st.write("Enter your symptom to receive general guidance.")

# -------------------------------
# Define Graph State
# -------------------------------
class GraphState(TypedDict, total=False):
    symptom: str
    category: str
    answer: str

# -------------------------------
# Classify symptom without an API
# -------------------------------
def classify_symptom(state: GraphState) -> dict:
    symptom = state.get("symptom", "").lower().strip()

    emergency_keywords = [
        "chest pain",
        "difficulty breathing",
        "can't breathe",
        "cannot breathe",
        "severe bleeding",
        "unconscious",
        "stroke",
        "overdose",
        "severe allergic reaction"
    ]

    mental_health_keywords = [
        "anxiety",
        "panic attack",
        "depressed",
        "depression",
        "suicidal",
        "self harm",
        "self-harm",
        "mental stress",
        "feeling hopeless"
    ]

    if any(word in symptom for word in emergency_keywords):
        category = "Emergency"

    elif any(word in symptom for word in mental_health_keywords):
        category = "Mental Health"

    else:
        category = "General"

    return {"category": category}


# -------------------------------
# Guidance Nodes
# -------------------------------
def general(state: GraphState) -> dict:
    return {
        "answer": (
            "Consider consulting a qualified healthcare professional "
            "for appropriate assessment. Seek urgent care if your "
            "symptoms are severe or worsening."
        )
    }


def emergency(state: GraphState) -> dict:
    return {
        "answer": (
            "Your symptoms may indicate a medical emergency. "
            "Contact local emergency services or seek emergency "
            "medical care immediately. Do not wait for this app "
            "to assess your symptoms."
        )
    }


def mental_health(state: GraphState) -> dict:
    return {
        "answer": (
            "Consider speaking with a qualified mental health "
            "professional. If you may harm yourself or are in "
            "immediate danger, contact emergency services or "
            "a crisis support service now."
        )
    }


# -------------------------------
# Routing
# -------------------------------
def route_category(state: GraphState) -> str:
    category = state.get("category", "General")

    if category == "Emergency":
        return "emergency"

    if category == "Mental Health":
        return "mental_health"

    return "general"


# -------------------------------
# Build LangGraph
# -------------------------------
builder = StateGraph(GraphState)

builder.add_node("classify", classify_symptom)
builder.add_node("general", general)
builder.add_node("emergency", emergency)
builder.add_node("mental_health", mental_health)

builder.set_entry_point("classify")

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
# Run the app
# -------------------------------
symptom = st.text_input(
    "Enter your symptom",
    placeholder="e.g., headache, cough, or anxiety"
)

if st.button("Analyze", type="primary"):
    if not symptom.strip():
        st.warning("Please enter a symptom.")
    else:
        result = graph.invoke({"symptom": symptom.strip()})

        st.success(f"Category: {result['category']}")
        st.info(result["answer"])

        st.caption(
            "Educational demonstration only. This tool does not "
            "provide a diagnosis and cannot reliably rule out emergencies."
        )
