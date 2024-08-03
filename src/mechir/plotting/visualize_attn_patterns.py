import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Visualize Attention Patterns",
    layout="wide",
)


def load_doc(fname):

    return []


'''
Plots and interactive plotly graph to visualize the attention pattern for a single document.
'''
def plot_attn_for_doc(attn_pattern, doc_tokens):

    fig = go.Figure(data=go.Heatmap(z=attn_pattern,colorscale='RdBu', zmin=-1, zmax=1))
    fig.update_layout(
            title='Attention Pattern',
            xaxis=dict(tickvals=np.arange(len(doc_tokens)),ticktext=doc_tokens, title="Source"),
            yaxis=dict(tickvals=np.arange(len(doc_tokens)),ticktext=doc_tokens[::-1], title="Destination"),
        )

    return fig



###################### Start page #########################

st.markdown("# Homepage")
st.sidebar.markdown("# Homepage")


# Minimal example on how to visualize attention patterns for a single document

# (1) Load data
attn_pattern_fname = ""
data = np.load(attn_pattern_fname)

# TODO: load document tokens
doc_fname = ""
doc_tokens = load_doc(doc_fname)

# (2) Plot
fig = plot_attn_for_doc(data, doc_tokens)
st.plotly_chart(fig)

