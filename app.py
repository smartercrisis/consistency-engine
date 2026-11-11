import json
import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util

ST_CONFIG_FILE = "style_anchor.json"

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
st.markdown("*Gate 0 Perimeter Ingress & Fused Schema Synthesis*")

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

if st.button("Realign & Synthesize Anchored Output"):
    if not raw_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Evaluating semantic drift and synthesizing payload..."):
            anchor_embedding = model.encode(anchor_data["core_aesthetic"], convert_to_tensor=True)
            prompt_embedding = model.encode(raw_prompt, convert_to_tensor=True)
            
            similarity_score = util.cos_sim(prompt_embedding, anchor_embedding).item()
            
            # TRUE SYNTHESIS: Cleanly weaving subject and aesthetic into a unified string
            clean_subject = raw_prompt.strip().rstrip('.')
            fused_prompt_string = (
                f"{clean_subject}, rendered with {anchor_data['core_aesthetic']} "
                f"incorporating a primary color palette of {anchor_data['color_palette']}, "
                f"high-end production render, 8k resolution, volumetric lighting."
            )
            
            # JSON Payload for machine governance & audit
            structured_payload = {
                "ingress_metadata": {
                    "gate": "Gate-0-Perimeter",
                    "similarity_score": round(similarity_score, 4),
                    "status": "APPROVED" if similarity_score >= 0.4 else "RE-ANCHORED"
                },
                "fused_prompt": fused_prompt_string,
                "negative_constraints": anchor_data['negative_constraints']
            }
            
            # Display Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Semantic Similarity Score", value=f"{similarity_score:.2f}")
            with col2:
                st.markdown("🟢 **Synthesis Complete**")
            
            st.markdown("---")
            st.subheader("📋 Fused Prompt (Ready for Image Generator)")
            st.code(fused_prompt_string, language="text")
            
            st.subheader("📋 Machine Audit JSON Payload")
            st.code(json.dumps(structured_payload, indent=4), language="json")
