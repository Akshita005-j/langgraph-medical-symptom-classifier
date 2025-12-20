import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# App UI
st.set_page_config(page_title="AI Medical Symptom Classifier", layout="centered")
st.title("🏥 AI Medical Symptom Classifier")
st.write("Enter your symptom and get guidance.")

# State
class GraphState(dict):
    pass

# Node: classify symptom
def classify_symptom(state):
    prompt = f"""
    You are a medical assistant.
    Classify the symptom into:
    - General
    - Emergency
    - Mental Health

    Symptom: {state['symptom']}
    Respond with only one word.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state["category"] = response.content.strip()
    return state

# Node: general
def general(state):
    state["answer"] = "This seems like a general issue. Please consult a general physician."
    return state

# Node: emergency
def emergency(state):
    state["answer"] = "This is an emergency. Seek immediate medical attention."
    return state

# Node: mental_health(state)
def mental_health(state):
    state["answer"] = "This may relate to mental health. Consider talking to a mental health professional."
    return state

# Build graph
builder = StateGraph(GraphState)
builder.add_node("classify", classify_symptom)
builder.add_node("general", general)
builder.add_node("emergency", emergency)
builder.add_node("mental_health", mental_health)

builder.set_entry_point("classify")

builder.add_conditional_edges(
    "classify",
    lambda x: x["category"].lower(),
    {
        "general": "general",
        "emergency": "emergency",
        "mental health": "mental_health"
    }
)

builder.add_edge("general", END)
builder.add_edge("emergency", END)
builder.add_edge("mental_health", END)

graph = builder.compile()

# UI interaction
symptom = st.text_input("Enter your symptom")

if st.button("Analyze"):
    if symptom:
        result = graph.invoke({"symptom": symptom})
        st.success(f"Category: {result['category']}")
        st.info(result["answer"])
    else:
        st.warning("Please enter a symptom.")
