import base64
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
HF_USERNAME = "Sricharan451706"
HF_MODEL_REPO = f"{HF_USERNAME}/Tourism-Package-Predictor"
HF_DATASET_REPO = f"{HF_USERNAME}/Tourism"
LOG_FILENAME = "prediction_logs.csv"
MODEL_PATH = Path("models/model.joblib")

st.set_page_config(
    page_title="Wellness Tourism | Purchase Predictor",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #f1fbfc;
        --muted: #b4cbd0;
        --teal: #39d4c0;
        --teal-dark: #13a99e;
        --mint: #123b45;
        --line: rgba(139, 226, 220, .25);
        --paper: #061c25;
        --white: rgba(7, 37, 47, .76);
    }

    .stApp {
        background: #061c25;
        color: #f4fbfc;
        font-family: 'DM Sans', sans-serif;
        isolation: isolate;
    }
    .stApp:before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: -2;
        background-image:
            linear-gradient(180deg, rgba(3,22,31,.70), rgba(3,30,38,.80)),
            url("data:image/jpeg;base64,__TOURISM_BG__");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        transform: scale(1.04);
        animation: scenicDrift 34s ease-in-out infinite alternate;
        will-change: transform;
    }
    .stApp:after {
        content: "";
        position: fixed;
        inset: -20%;
        z-index: -1;
        pointer-events: none;
        background:
            radial-gradient(ellipse at 18% 18%, rgba(39,213,192,.18), transparent 34%),
            radial-gradient(ellipse at 82% 72%, rgba(31,155,203,.15), transparent 38%);
        animation: ambientFloat 18s ease-in-out infinite alternate;
    }
    @keyframes scenicDrift {
        from { transform: scale(1.04) translate3d(-.5%, 0, 0); }
        to { transform: scale(1.13) translate3d(.8%, -1%, 0); }
    }
    @keyframes ambientFloat {
        from { opacity: .55; transform: translate3d(-1%, 0, 0) rotate(0deg); }
        to { opacity: 1; transform: translate3d(1%, -1%, 0) rotate(2deg); }
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.035em;
    }

    .eyebrow {
        color: var(--teal);
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .72rem;
        font-weight: 800;
        margin-bottom: .7rem;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.25rem);
        font-weight: 800;
        line-height: 1.08;
        letter-spacing: -.055em;
        max-width: 780px;
        margin: 0;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.04rem;
        line-height: 1.7;
        max-width: 700px;
        margin-top: .9rem;
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        border: 1px solid #cce5e1;
        background: rgba(255,255,255,.75);
        color: var(--teal-dark);
        padding: .42rem .78rem;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 700;
        margin-top: 1.25rem;
    }

    .section-kicker {
        color: var(--teal);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: .15rem 0 .25rem;
    }

    .section-title {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -.025em;
        margin: 0 0 .25rem;
    }

    .section-copy {
        color: var(--muted);
        font-size: .88rem;
        margin: 0 0 1.15rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, rgba(10, 55, 66, .82), rgba(5, 31, 42, .76));
        border: 1px solid rgba(111, 225, 217, .36);
        border-radius: 20px;
        box-shadow: 0 14px 38px rgba(0, 12, 20, .22), inset 0 1px 0 rgba(255,255,255,.06);
        backdrop-filter: blur(18px);
    }

    [data-testid="stWidgetLabel"] p {
        color: #e5f4f5;
        font-weight: 600;
        font-size: .88rem;
    }

    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        background: rgba(3, 25, 35, .72);
        border-color: rgba(174, 224, 226, .45);
        color: #f4fbfc;
        border-radius: 10px;
        min-height: 2.8rem;
    }
    [data-baseweb="select"] *, [data-testid="stNumberInput"] input {
        color: #f4fbfc !important;
    }
    [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] p {
        color: #c0d5d9;
    }
    .section-kicker { color: #55dfd0; }
    .section-title { color: #f4fbfc; }
    .section-copy { color: #b4cbd0; }
    .result-panel {
        background: rgba(5, 35, 46, .88);
        border-color: rgba(115, 223, 216, .32);
        color: #f4fbfc;
        backdrop-filter: blur(16px);
    }
    .result-label, .result-note { color: #bfd5d9; }
    .footnote { color: #b5cbd0; }

    [data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: .35rem;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--teal);
        color: white;
        border: 0;
        border-radius: 11px;
        min-height: 3.1rem;
        padding: .65rem 1.4rem;
        font-weight: 800;
        transition: all .18s ease;
        box-shadow: 0 7px 16px rgba(8,127,140,.18);
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: var(--teal-dark);
        color: white;
        border: 0;
        transform: translateY(-1px);
    }

    .result-panel {
        border-radius: 18px;
        padding: 1.35rem 1.5rem;
        border: 1px solid var(--line);
        background: white;
        margin-top: 1rem;
    }

    .result-label {
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .72rem;
        font-weight: 800;
        color: var(--muted);
    }

    .result-value {
        font-family: 'Manrope', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -.04em;
        margin: .35rem 0;
    }

    .result-note {
        color: var(--muted);
        font-size: .9rem;
    }

    .footnote {
        color: #788990;
        font-size: .78rem;
        line-height: 1.6;
        margin-top: 1.4rem;
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.2rem 1rem 3rem;
        }
        .hero-title { font-size: 2.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use the banner already included in this app folder as the hero background.
BANNER_PATH = Path(__file__).with_name("tourism_hero.jpg")
try:
    banner_base64 = base64.b64encode(BANNER_PATH.read_bytes()).decode("utf-8")
except OSError:
    banner_base64 = ""

# Inject the same seasonal travel artwork into the animated page background.
st.markdown(
    """<style>
    .stApp:before { background-image:
        linear-gradient(180deg, rgba(3,22,31,.70), rgba(3,30,38,.80)),
        url("data:image/jpeg;base64,__TOURISM_BG__"); }
    </style>""".replace("__TOURISM_BG__", banner_base64),
    unsafe_allow_html=True,
)

st.markdown(
    """<style>

    .hero-banner {
        position: relative; overflow: hidden; isolation: isolate;
        min-height: 340px; display: flex; align-items: center;
        padding: clamp(2rem, 5vw, 4.2rem); border-radius: 26px;
        background-image: linear-gradient(90deg, rgba(5,26,34,.94) 0%, rgba(5,36,42,.78) 48%, rgba(5,36,42,.18) 100%), url("data:image/png;base64,__BANNER_BASE64__");
        background-size: cover; background-position: center 48%;
        box-shadow: 0 22px 55px rgba(8,49,57,.19);
        animation: heroReveal .85s ease both;
    }
    .hero-banner:before, .hero-banner:after {
        content: ""; position: absolute; z-index: -1; border-radius: 50%; pointer-events: none;
    }
    .hero-banner:before {
        width: 290px; height: 290px; right: 8%; top: -120px;
        border: 1px solid rgba(255,255,255,.24);
        box-shadow: 0 0 0 35px rgba(255,255,255,.035), 0 0 0 75px rgba(255,255,255,.025);
        animation: slowFloat 9s ease-in-out infinite;
    }
    .hero-banner:after {
        width: 170px; height: 170px; right: 27%; bottom: -115px;
        background: rgba(108,220,194,.16); filter: blur(8px);
        animation: slowFloat 7s ease-in-out infinite reverse;
    }
    .hero-content { max-width: 680px; position: relative; z-index: 1; }
    .hero-banner .eyebrow { color: #8de0d1; }
    .hero-banner .hero-title { color: #fff !important; max-width: 650px; }
    .hero-banner .hero-copy { color: rgba(255,255,255,.83); max-width: 590px; }
    .hero-banner .hero-chip {
        color: #eafffa; border-color: rgba(190,255,239,.34);
        background: rgba(255,255,255,.12); backdrop-filter: blur(12px);
    }
    .hero-glow {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #8de0d1; box-shadow: 0 0 0 6px rgba(141,224,209,.15), 0 0 18px #8de0d1;
        margin-right: .35rem;
    }
    .section-kicker, .section-title, .section-copy { animation: fadeUp .55s ease both; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px); border-color: #c5dfda;
        box-shadow: 0 14px 34px rgba(28,61,68,.075);
    }
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(110deg, #087f8c, #0b9c91, #087f8c);
        background-size: 200% 100%; animation: buttonShimmer 5s ease infinite;
    }
    .result-panel { animation: fadeUp .55s ease both; }
    @keyframes heroReveal {
        from { opacity: 0; transform: translateY(14px) scale(.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slowFloat {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(18px) rotate(7deg); }
    }
    @keyframes buttonShimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *:before, *:after {
            animation-duration: .01ms !important; animation-iteration-count: 1 !important;
            scroll-behavior: auto !important; transition-duration: .01ms !important;
        }
    }

</style>""".replace("__BANNER_BASE64__", banner_base64),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <section class="hero-banner">
      <div class="hero-content">
        <div class="eyebrow"><span class="hero-glow"></span> Customer intelligence · Predictive analytics</div>
        <div class="hero-title">Wellness journeys.<br>Smarter predictions.</div>
        <div class="hero-copy">
          Understand the customer behind every enquiry. Explore travel preferences,
          customer details and sales interactions to estimate purchase intent.
        </div>
        <div class="hero-chip">✦ &nbsp; Wellness tourism purchase predictor</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------
# Model and Hugging Face helpers
# ---------------------------------------------------------
def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as exc:
            st.error(f"I couldn't load the local model: {exc}")

    try:
        model_file = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="model.joblib",
            token=get_hf_token(),
        )
        return joblib.load(model_file)
    except Exception as exc:
        st.error("The prediction model could not be loaded.")
        st.caption(
            "Check that the model exists in the configured Hugging Face repository "
            "and that HF_TOKEN has access if the repository is private."
        )
        with st.expander("Technical details"):
            st.code(str(exc))
        return None


def save_prediction(input_data, prediction, probability):
    token = get_hf_token()
    if not token:
        return False

    try:
        api = HfApi(token=token)
        log_entry = input_data.copy()
        log_entry["Prediction"] = int(prediction)
        log_entry["Probability"] = float(probability)
        log_entry["Timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()

        try:
            log_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=LOG_FILENAME,
                repo_type="dataset",
                token=token,
            )
            existing_log = pd.read_csv(log_path)
            updated_log = pd.concat([existing_log, log_entry], ignore_index=True)
        except Exception:
            updated_log = log_entry

        temp_path = Path(LOG_FILENAME)
        updated_log.to_csv(temp_path, index=False)
        api.upload_file(
            path_or_fileobj=str(temp_path),
            path_in_repo=LOG_FILENAME,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
        )
        return True
    except Exception as exc:
        print(f"Prediction logging failed: {exc}")
        return False


model = load_model()

if model is not None:
    st.markdown(
        '<div class="section-kicker">01 / Customer information</div>'
        '<div class="section-title">Who is the customer?</div>'
        '<div class="section-copy">Basic profile and household details.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        with st.container(border=True):
            profile_left, profile_right = st.columns(2, gap="large")
            with profile_left:
                age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
                gender = st.selectbox("Gender", ["Male", "Female"])
                marital_status = st.selectbox(
                    "Marital status", ["Married", "Divorced", "Single", "Unmarried"]
                )
                occupation = st.selectbox(
                    "Occupation",
                    ["Salaried", "Small Business", "Large Business", "Free Lancer"],
                )
            with profile_right:
                designation = st.selectbox(
                    "Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
                )
                monthly_income = st.number_input(
                    "Monthly income", min_value=0, value=20000, step=1000
                )
                passport = st.selectbox(
                    "Passport", [0, 1], format_func=lambda value: "Yes" if value else "No"
                )
                own_car = st.selectbox(
                    "Own car", [0, 1], format_func=lambda value: "Yes" if value else "No"
                )

        st.write("")
        st.markdown(
            '<div class="section-kicker">02 / Travel preferences</div>'
            '<div class="section-title">What does their trip look like?</div>'
            '<div class="section-copy">Travel party, destination preference and past travel activity.</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            travel_left, travel_right = st.columns(2, gap="large")
            with travel_left:
                type_of_contact = st.selectbox(
                    "Type of contact", ["Company Invited", "Self Inquiry"]
                )
                city_tier = st.selectbox("City tier", [1, 2, 3])
                number_of_person_visiting = st.number_input(
                    "People visiting", min_value=1, value=2, step=1
                )
                number_of_children_visiting = st.number_input(
                    "Children visiting", min_value=0, value=0, step=1
                )
            with travel_right:
                preferred_property_star = st.selectbox(
                    "Preferred property rating", [3.0, 4.0, 5.0], format_func=lambda x: f"{x:.0f} star"
                )
                number_of_trips = st.number_input(
                    "Trips in the past year", min_value=0, value=1, step=1
                )
                product_pitched = st.selectbox(
                    "Package pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
                )

        st.write("")
        st.markdown(
            '<div class="section-kicker">03 / Sales interaction</div>'
            '<div class="section-title">How did the pitch go?</div>'
            '<div class="section-copy">A few details about the sales conversation.</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            sales_left, sales_right = st.columns(2, gap="large")
            with sales_left:
                pitch_satisfaction_score = st.slider(
                    "Pitch satisfaction score", min_value=1, max_value=5, value=3,
                    help="Rate the customer's satisfaction with the sales pitch."
                )
                number_of_followups = st.number_input(
                    "Number of follow-ups", min_value=0, value=3, step=1
                )
            with sales_right:
                duration_of_pitch = st.number_input(
                    "Pitch duration (minutes)", min_value=0, value=10, step=1
                )
                st.caption("Use the information recorded during the customer interaction.")

        st.write("")
        submitted = st.form_submit_button(
            "Generate purchase prediction  →", use_container_width=True
        )

    if submitted:
        if number_of_children_visiting > number_of_person_visiting:
            st.error("Children visiting cannot be greater than the total number of people visiting.")
        else:
            input_data = pd.DataFrame(
                {
                    "Age": [age],
                    "TypeofContact": [type_of_contact],
                    "CityTier": [city_tier],
                    "Occupation": [occupation],
                    "Gender": [gender],
                    "NumberOfPersonVisiting": [number_of_person_visiting],
                    "PreferredPropertyStar": [preferred_property_star],
                    "MaritalStatus": [marital_status],
                    "NumberOfTrips": [number_of_trips],
                    "Passport": [passport],
                    "OwnCar": [own_car],
                    "NumberOfChildrenVisiting": [number_of_children_visiting],
                    "Designation": [designation],
                    "MonthlyIncome": [monthly_income],
                    "PitchSatisfactionScore": [pitch_satisfaction_score],
                    "ProductPitched": [product_pitched],
                    "NumberOfFollowups": [number_of_followups],
                    "DurationOfPitch": [duration_of_pitch],
                }
            )

            try:
                prediction = int(model.predict(input_data)[0])
                probabilities = model.predict_proba(input_data)[0]
                class_labels = list(model.classes_)
                positive_index = class_labels.index(1) if 1 in class_labels else 1
                probability = float(probabilities[positive_index])

                st.write("")
                st.markdown(
                    '<div class="section-kicker">Prediction result</div>'
                    '<div class="section-title">Model assessment</div>',
                    unsafe_allow_html=True,
                )

                if prediction == 1:
                    result_title = "Likely to purchase"
                    result_description = "The model predicts that this customer is likely to purchase the wellness package."
                    result_color = "#087f8c"
                else:
                    result_title = "Unlikely to purchase"
                    result_description = "The model predicts that this customer is less likely to purchase the wellness package."
                    result_color = "#b66b36"

                st.markdown(
                    f"""
                    <div class="result-panel">
                        <div class="result-label">Predicted outcome</div>
                        <div class="result-value" style="color:{result_color}">{result_title}</div>
                        <div class="result-note">{result_description}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                metric_left, metric_right = st.columns([1, 2])
                with metric_left:
                    st.metric("Purchase probability", f"{probability:.1%}")
                with metric_right:
                    st.progress(probability, text="Estimated probability of purchase")

                if save_prediction(input_data, prediction, probability):
                    st.caption("Prediction saved to the configured Hugging Face dataset.")
                else:
                    st.caption("Prediction generated. Logging was skipped because storage access is not configured.")

            except Exception as exc:
                st.error(f"The model could not generate a prediction: {exc}")
                st.caption("Check that the input fields match the features used when the model was trained.")

st.markdown(
    """
    <div class="footnote">
        This tool is a machine learning demonstration. Its output is an estimate based on
        the training data and should support, not replace, human judgement.
    </div>
    """,
    unsafe_allow_html=True,
)
