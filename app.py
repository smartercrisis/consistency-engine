import json
import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util

ST_CONFIG_FILE = "style_anchor.json"

# Default style anchor
DEFAULT_ANCHOR = {
    "core_aesthetic": "Cinematic Radeon-tech, razor-sharp edge details, multi-layer microchip geometry, matte black and brushed titanium substrates, volumetric lighting.",
    "color_palette": "Crimson Red (#C50000), Royal Gold (#D4AF37), Deep Carbon (#111111)",
    "negative_constraints": "No cartoonish illustration, no flat lighting, no pastel colors, no chaotic or cluttered backgrounds."
}

def load_anchor():
    if os.path.exists(ST_CONFIG_FILE):
        with open(ST_CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        with open(ST_CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_ANCHOR, f, indent=4)
        return DEFAULT_ANCHOR

anchor_data = load_anchor()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.set_page_config(page_title="CIAT25 Visual Consistency Engine (LITE)", layout="wide")

st.title("⚡ CIAT25.1: Visual Consistency Engine (LITE)")
st.markdown("*Gate 0 Perimeter Ingress & Structured Prompt Realignment*")

# Sidebar: Manage Style Anchor
with st.sidebar:
    st.header("1. Style Anchor Vault")
    new_aesthetic = st.text_area("Core Aesthetic", value=anchor_data["core_aesthetic"], height=100)
    new_palette = st.text_input("Color Palette", value=anchor_data["color_palette"])
    new_negative = st.text_area("Negative Constraints", value=anchor_data["negative_constraints"], height=80)
    
    if st.button("Save Style Anchor"):
        updated_anchor = {
            "core_aesthetic": new_aesthetic,
            "color_palette": new_palette,
            "negative_constraints": new_negative
        }
        with open(ST_CONFIG_FILE, "w") as f:
            json.dump(updated_anchor, f, indent=4)
        st.success("Style Anchor updated successfully!")
        anchor_data = updated_anchor

# Main Panel: Prompt Ingress & Realignment
st.header("2. Structural Prompt Realignment Gate")
raw_prompt = st.text_input("Enter your messy or casual prompt:", placeholder="e.g., make a server rack with red lights")

if st.button("Realign & Generate Anchored Prompt"):
    if not raw_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Evaluating semantic drift and structuring schema..."):
            anchor_embedding = model.encode(anchor_data["core_aesthetic"], convert_to_tensor=True)
            prompt_embedding = model.encode(raw_prompt, convert_to_tensor=True)
            
            similarity_score = util.cos_sim(prompt_embedding, anchor_embedding).item()
            
            # DETERMINISTIC STRUCTURAL INJECTION (Fixing the string-glitch)
            realigned_prompt = (
                f"{raw_prompt}. "
                f"[Aesthetic Directive: {anchor_data['core_aesthetic']}]. "
                f"[Color Schema: {anchor_data['color_palette']}]. "
                f"High-end production render, 8k resolution, crisp structural focus."
            )
            
            negative_prompt_output = anchor_data['negative_constraints']
            
            # Display Metrics & Clean Separation
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Semantic Similarity Score", value=f"{similarity_score:.2f}")
            with col2:
                st.markdown("🟢 **Structured Schema Enforced**")
            
            st.markdown("---")
            st.subheader("📋 Positive Anchored Prompt (Copy to Image Generator)")
            st.code(realigned_prompt, language="text")
            
            st.subheader("🚫 Negative Prompt Constraints (Optional)")
            st.code(negative_prompt_output, language="text")
