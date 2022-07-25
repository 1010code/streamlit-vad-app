import io
import base64
import numpy as np
import scipy.io.wavfile
from typing import Text
import streamlit as st
from utils.io import Audio

import streamlit.components.v1 as components

import torch
def to_base64(waveform: np.ndarray, sample_rate: int = 16000) -> Text:
    """Convert waveform to base64 data"""
    waveform /= np.max(np.abs(waveform)) + 1e-8
    with io.BytesIO() as content:
        scipy.io.wavfile.write(content, sample_rate, waveform)
        content.seek(0)
        b64 = base64.b64encode(content.read()).decode()
        b64 = f"data:audio/x-wav;base64,{b64}"
    return b64

st.markdown("""### 🎹 語音活性檢測 (Voice activity detection) """)

# 初始化聲音資訊
audio = Audio(sample_rate=16000, mono=True)

@st.cache(allow_output_mutation=True)
def load_model():
    # 載入模型
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

    return model, utils
model, utils = load_model()
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# 上傳檔案
uploaded_file = st.file_uploader("Choose an audio file")
if uploaded_file is not None:

    try:
        duration = audio.get_duration(uploaded_file)
    except RuntimeError as e:
        st.error(e)
        st.stop()

#     waveform, sample_rate = torchaudio.load(uploaded_file)
    waveform, sample_rate = audio(uploaded_file)
    uri = "".join(uploaded_file.name.split())
    file = {"waveform": waveform, "sample_rate": sample_rate, "uri": uri}

    with st.spinner(f"Processing first {duration:g} seconds..."):
        speech_timestamps = get_speech_timestamps(waveform, model, sampling_rate=sample_rate)

    with open('assets/template.html') as html, open('assets/style.css') as css:
        html_template = html.read()
        st.markdown('<style>{}</style>'.format(css.read()), unsafe_allow_html=True)

    colors = [
        "#ffd70033",
        "#00ffff33",
        "#ff00ff33",
        "#00ff0033",
        "#9932cc33",
        "#00bfff33",
        "#ff7f5033",
        "#66cdaa33",
    ]
    num_colors = len(colors)

    label2color = {'SPEECH': '#ffd70033'}
    BASE64 = to_base64(waveform.numpy().T)

    REGIONS = ""
    LEGENDS = ""
    labels=[]
    for i in range(len(speech_timestamps)):
        start=speech_timestamps[i]['start']/sample_rate
        end=speech_timestamps[i]['end']/sample_rate
        label='SPEECH'
        REGIONS += f"var re = wavesurfer.addRegion({{start: {start:g}, end: {end:g}, color: '{label2color[label]}', resize : false, drag : false}});"
        if not label in labels:
            LEGENDS += f"<li><span style='background-color:{label2color[label]}'></span>{label}</li>"
            labels.append(label)
    print(BASE64)
    html = html_template.replace("BASE64", BASE64).replace("REGIONS", REGIONS)
    components.html(html, height=250, scrolling=True)
    st.markdown("<div style='overflow : auto'><ul class='legend'>"+LEGENDS+"</ul></div>", unsafe_allow_html=True)
    st.markdown("---")
