import cv2 as cv
import numpy as np
import streamlit as st
import json


def log_(x):
    return np.log10(x + 1)


def exp_(x):
    return 10**x - 1


def model(x, a, b, c):
    return a * x**b + c


def reverse_model(y, a, b, c):
    return ((y - c) / a) ** (1 / b)


def format_ion(ion) -> str:
    base, charge = ion.split(" ")
    return r"$\text{" + base + "}^{" + charge + "}$"


@st.cache_data
def get_params():
    return json.load(open("model-params.json"))


params = get_params()

with st.form("Analysis"):
    ion = st.selectbox("Ion type", list(params.keys()))
    part = st.selectbox("Part type", list(params[ion].keys()))
    image = st.file_uploader(
        "Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )
    if st.form_submit_button("Process"):
        st.session_state["ion"] = ion
        st.session_state["part"] = part
        st.session_state["image"] = cv.cvtColor(
            cv.imdecode(
                np.frombuffer(
                    image.read(),  # type: ignore
                    np.uint8,
                ),
                1,
            ),
            cv.COLOR_BGR2RGB,
        )
if "image" not in st.session_state:
    st.stop()

a = params[st.session_state["ion"]][st.session_state["part"]]["a"]
b = params[st.session_state["ion"]][st.session_state["part"]]["b"]
c = params[st.session_state["ion"]][st.session_state["part"]]["c"]

img = cv.cvtColor(st.session_state["image"], cv.COLOR_RGB2GRAY)
threshold = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
mask = cv.bitwise_not(threshold)
img = cv.bitwise_and(img, img, mask=mask)
pixels = np.sum(img > 0)  # type: ignore

st.title("Result")
st.image(st.session_state["image"], caption="Original image", use_column_width=True)
st.metric(
    f"Predicted {format_ion(ion)} concentration",
    f"{reverse_model(log_(pixels), a, b, c)}",
)
