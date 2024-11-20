import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Visualize Attention Patterns",
    layout="wide",
)


def load_doc(fname):
    with open(fname, "r") as f:
        text = f.read()
    return text.split()


# Plots and interactive plotly graph to visualize the attention pattern for a single document.
def plot_attn_for_doc(attn_pattern, doc_tokens):

    fig = go.Figure(data=go.Heatmap(z=attn_pattern, colorscale="RdBu", zmin=-1, zmax=1))
    fig.update_layout(
        title="Attention Pattern",
        xaxis=dict(
            tickvals=np.arange(len(doc_tokens)), ticktext=doc_tokens, title="Source"
        ),
        yaxis=dict(
            tickvals=np.arange(len(doc_tokens)),
            ticktext=doc_tokens,
            title="Destination",
        ),
    )

    return fig


__all__ = ["plot_attn_for_doc", "load_doc"]

###################### Start page #########################

if __name__ == "__main__":
    st.markdown("# Visualize Attention Patterns")

    # Minimal example on how to visualize attention patterns for a single document

    # (1) Load attention pattern
    attn_pattern_fname = ""  # FILL IN WITH THE PATH TO THE ATTENTION PATTERN FILE
    data = np.load(attn_pattern_fname)

    # (2) Load document text
    doc_fname = ""  # FILL IN WITH THE PATH TO YOUR FILE
    doc_tokens = load_doc(doc_fname)

    # (3) Plot
    fig = plot_attn_for_doc(data, doc_tokens)
    st.plotly_chart(fig)
