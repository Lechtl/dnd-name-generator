"""Web app for the D&D name generator. Start it with:

    streamlit run app.py

Streamlit runs this whole file from top to bottom every time the user clicks or
moves a slider. Anything slow, like loading the model, is therefore cached.
"""

import numpy as np
import streamlit as st

from slm.config import RACES, Config
from slm.name_filter import generate_until_acceptable
from slm.numpy_model import NumpyNameModel
from slm.tokenizer import CharTokenizer

GITHUB_URL = "https://github.com/Lechtl/dnd-name-generator"

config = Config()


@st.cache_resource
def load_model() -> tuple[NumpyNameModel, CharTokenizer]:
    # Runs once per server start. Later reruns reuse the same model.
    return NumpyNameModel.load(config.weights_path), CharTokenizer.load(config.tokenizer_path)


def race_label(race: str) -> str:
    # "half-elf" -> "Half-Elf" for display; the model still gets "half-elf".
    return "-".join(part.capitalize() for part in race.split("-"))


st.set_page_config(page_title="D&D Name Generator", page_icon="🎲")
st.title("D&D Name Generator")
st.write(
    "Invents fantasy names for the nine races of the D&D 5e Player's Handbook. "
    "Behind it is a small character-level language model that writes one letter at a time."
)

model, tokenizer = load_model()

race = st.selectbox("Race", RACES, index=RACES.index("dwarf"), format_func=race_label)
count = st.slider("Number of names", min_value=1, max_value=20, value=10)
temperature = st.slider(
    "Temperature",
    min_value=0.3,
    max_value=2.0,
    value=config.default_temperature,
    step=0.1,
    help="Low values give safe, familiar-sounding names. High values give stranger ones.",
)

if st.button("Generate names", type="primary"):
    rng = np.random.default_rng()
    names = [
        generate_until_acceptable(
            lambda: model.generate_name(
                tokenizer, race, temperature, rng, config.max_generated_length
            )
        )
        for _ in range(count)
    ]
    st.subheader(f"{race_label(race)} names")
    # Two columns keep a long list compact.
    left, right = st.columns(2)
    for i, name in enumerate(names):
        (left if i % 2 == 0 else right).markdown(f"- {name}")

with st.expander("How does it work?"):
    st.markdown(
        """
- The model reads a name letter by letter and learns to predict the next letter.
  It was trained on about 1,900 names from D&D, Tolkien and other fantasy sources.
- The race sets the model's starting memory, so each race begins a name differently.
- A GRU, a small recurrent neural network, carries that memory from letter to letter.
- **Temperature** changes how boldly the next letter is picked.
- A filter rejects names with three identical letters in a row and a list of
  banned names, such as dictators. Rejected names are generated again.
- The model was trained with Keras. This app runs the same weights in plain NumPy,
  which starts in well under a second.
"""
    )
    st.markdown(f"Source code and details: [GitHub]({GITHUB_URL})")
