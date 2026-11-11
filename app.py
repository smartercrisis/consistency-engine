import json
import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION & SETUP ---
ST_CONFIG_FILE = "style_anchor.json"

# Default style anchor if none exists
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

# Load lightweight embedding model for similarity check (cached so it only loads once)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="CIAT25 Visual Consistency Engine (LITE)", layout="wide")

st.title("⚡ CIAT25.1: Visual Consistency Engine (LITE)")
st.markdown("*Gate 0 Perimeter Ingress & Prompt Realignment Pipeline*")

# Sidebar: Manage Style Anchor
with st.sidebar:
    st.header("1. Style Anchor Vault")
    st.markdown("Define your master visual DNA. Messy prompts will be dynamically rewritten toward this anchor.")
    
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
st.header("2. Prompt Realignment Gate (Gate 0 Rewrite)")
raw_prompt = st.text_input("Enter your messy or casual prompt:", placeholder="e.g., make a cool server room with red lights")

if st.button("Realign & Generate Anchored Prompt"):
    if not raw_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Evaluating semantic drift and realigning prompt..."):
            # Compute embeddings for distance evaluation
            anchor_embedding = model.encode(anchor_data["core_aesthetic"], convert_to_tensor=True)
            prompt_embedding = model.encode(raw_prompt, convert_to_tensor=True)
            
            similarity_score = util.cos_sim(prompt_embedding, anchor_embedding).item()
            
            # Realignment Logic (Deterministic rewriting rules based on similarity)
            if similarity_score >= 0.65:
                status_msg = "🟢 **Direct Alignment:** Prompt is close to brand parameters."
                realigned_prompt = f"{raw_prompt}, rendered in {anchor_data['color_palette']}, {anchor_data['core_aesthetic']}, highly detailed, 8k resolution"
            else:
                status_msg = f"🟡 **Drift Detected (Score: {similarity_score:.2f}):** Rewriting messy prompt toward anchor."
                realigned_prompt = (
                    f"A technical schematic visualization of {raw_prompt}, "
                    f"styled with {anchor_data['core_aesthetic']}, "
                    f"incorporating color palette {anchor_data['color_palette']}, "
                    f"avoiding {anchor_data['negative_constraints']}, "
                    f"octane render, volumetric lighting, crisp details."
                )
            
            # Display Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Semantic Similarity Score", value=f"{similarity_score:.2f}")
            with col2:
                st.markdown(status_msg)
            
            st.markdown("---")
            st.subheader("📋 Final Anchored Prompt (Ready to Copy)")
            st.code(realigned_prompt, language="text")
            st.info("Copy this clean, aligned prompt above and paste it manually into your image generator.")
