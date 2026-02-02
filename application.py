import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
from gtts import gTTS
import pytesseract
from PIL import Image
from datetime import datetime
import speech_recognition as sr

# ===============================
# CONFIGURATION
# ===============================
DetectorFactory.seed = 0
st.set_page_config(
    page_title="🌐 Traducteur & Chatbot IA Multimodal",
    layout="wide"
)

# ===============================
# STYLE SIMPLE & PRO
# ===============================
st.markdown("""
<style>
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #1e40af;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# CHARGEMENT DES MODÈLES
# ===============================
@st.cache_resource
def load_chatbot():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

@st.cache_resource
def load_translator():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

chatbot = load_chatbot()
tokenizer, model = load_translator()

# ===============================
# ÉTATS
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "history" not in st.session_state:
    st.session_state.history = []

if "current_text" not in st.session_state:
    st.session_state.current_text = ""

# ===============================
# LANGUES
# ===============================
LANG_MAP = {
    "Français": "fra_Latn",
    "Anglais": "eng_Latn",
    "Arabe": "ary_Arab",
    "Espagnol": "spa_Latn"
}

NLLB_TO_GTTS = {
    "fra_Latn": "fr",
    "eng_Latn": "en",
    "ary_Arab": "ar",
    "spa_Latn": "es"
}

DETECT_TO_NLLB = {
    "fr": ("Français", "fra_Latn"),
    "en": ("Anglais", "eng_Latn"),
    "ar": ("Arabe", "ary_Arab"),
    "es": ("Espagnol", "spa_Latn")
}

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("📜 Historique traductions")
    if st.session_state.history:
        for h in reversed(st.session_state.history):
            st.markdown(f"**{h['time']}**  \n{h['src'][:30]} → {h['tr'][:30]}")
            st.divider()
    else:
        st.info("Aucune traduction")

# ===============================
# INTERFACE PRINCIPALE
# ===============================
st.title("🌐 Traducteur & Chatbot IA Tout-en-Un")
tab1, tab2 = st.tabs(["💬 Chatbot IA", "🌍 Traducteur IA"])

# ======================================================
# ONGLET 1 : CHATBOT IA
# ======================================================
with tab1:
    st.subheader("💬 Chat avec l’IA")

    # Affichage historique
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrée utilisateur
    user_input = st.chat_input("Pose ta question...")

    if user_input:
        # Message utilisateur
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # PROMPT POUR FLAN-T5
        prompt = f"""
Question: {user_input}
Answer in a clear, helpful and concise way.
"""

        # Réponse IA
        with st.chat_message("assistant"):
            with st.spinner("🤖 L'IA réfléchit..."):
                response = chatbot(
                    prompt,
                    max_length=200,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9
                )[0]["generated_text"]

                st.markdown(response)

        # Sauvegarde réponse
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

    if st.button("🗑️ Effacer le chat"):
        st.session_state.chat_history = []
        st.rerun()

# ======================================================
# ONGLET 2 : TRADUCTEUR IA MULTIMODAL
# ======================================================
with tab2:
    col1, col2 = st.columns(2)

    # ---------- SAISIE ----------
    with col1:
        st.subheader("📥 Saisie")
        mode = st.radio(
            "Méthode :",
            ["Clavier", "Image (OCR)", "Vocal (Micro)", "Fichier"],
            horizontal=True
        )

        if mode == "Clavier":
            st.session_state.current_text = st.text_area(
                "Texte",
                st.session_state.current_text,
                height=150
            )

        elif mode == "Image (OCR)":
            img_file = st.file_uploader("Image", type=["png", "jpg"])
            if img_file:
                img = Image.open(img_file)
                st.image(img, width=200)
                st.session_state.current_text = pytesseract.image_to_string(img)

        elif mode == "Vocal (Micro)":
            audio = st.audio_input("Enregistrer")
            if audio:
                recog = sr.Recognizer()
                with sr.AudioFile(audio) as src:
                    data = recog.record(src)
                    try:
                        st.session_state.current_text = recog.recognize_google(
                            data, language="fr-FR"
                        )
                        st.success("Texte reconnu")
                    except:
                        st.error("Erreur reconnaissance vocale")

        elif mode == "Fichier":
            f = st.file_uploader("Fichier .txt", type=["txt"])
            if f:
                st.session_state.current_text = f.read().decode("utf-8")

    # ---------- DÉTECTION DE LANGUE ----------
    detected_lang_name = "Inconnue"
    detected_nllb = "fra_Latn"

    if st.session_state.current_text.strip() != "":
        try:
            det_code = detect(st.session_state.current_text)
            if det_code in DETECT_TO_NLLB:
                detected_lang_name, detected_nllb = DETECT_TO_NLLB[det_code]
        except:
            pass

    st.info(f"🧠 Langue détectée : **{detected_lang_name}**")

    # ---------- ÉCOUTER LE TEXTE SAISI ----------
    if st.button("🔊 Écouter le texte saisi"):
        if st.session_state.current_text.strip() != "":
            try:
                tts_input = gTTS(
                    st.session_state.current_text,
                    lang=NLLB_TO_GTTS.get(detected_nllb, "fr")
                )
                tts_input.save("input.mp3")
                st.audio("input.mp3")
            except:
                st.error("Impossible de lire le texte")
        else:
            st.warning("Aucun texte à écouter")

    # ---------- TRADUCTION ----------
    with col2:
        st.subheader("📤 Traduction")
        target_lang = st.selectbox("Langue cible", list(LANG_MAP.keys()))

        if st.button("✨ Traduire", use_container_width=True):
            if st.session_state.current_text.strip() != "":
                with st.spinner("Traduction..."):
                    pipe = pipeline(
                        "translation",
                        model=model,
                        tokenizer=tokenizer,
                        src_lang=detected_nllb,
                        tgt_lang=LANG_MAP[target_lang]
                    )

                    result = pipe(
                        st.session_state.current_text,
                        max_length=500
                    )[0]["translation_text"]

                    st.session_state.last_result = result
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M"),
                        "src": st.session_state.current_text,
                        "tr": result
                    })
            else:
                st.warning("Aucun texte")

        # ---------- AFFICHAGE + ÉCOUTE + TÉLÉCHARGEMENT ----------
        if "last_result" in st.session_state:
            st.success(st.session_state.last_result)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔊 Écouter traduction"):
                    tts = gTTS(
                        st.session_state.last_result,
                        lang=NLLB_TO_GTTS[LANG_MAP[target_lang]]
                    )
                    tts.save("tr.mp3")
                    st.audio("tr.mp3")

            with c2:
                st.download_button(
                    "📥 Télécharger",
                    st.session_state.last_result,
                    "traduction.txt"
                )

