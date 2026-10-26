import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Mobile Price Prediction System",
    page_icon="📱",
    layout="wide"
)

# Dark interface with orange accents, inspired by the supplied screenshot.
st.markdown("""
<style>
.stApp { background-color: #101010; color: #f5f5f5; }
h1 { text-align: center; color: #f7f7f7; }
div[data-testid="stForm"] {
    background: #252629; padding: 22px; border: 1px solid #393a3f;
    border-radius: 8px;
}
div.stButton > button, div[data-testid="stFormSubmitButton"] button {
    background: #ff7918; color: white; border: none; border-radius: 7px;
    font-weight: 600; min-height: 44px; width: 100%;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    background: #e9690d; color: white;
}
div[data-testid="stMetric"] {
    background: #252629; border: 1px solid #393a3f;
    padding: 18px; border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    classifier = joblib.load("best_model.pkl")
    class_scaler = joblib.load("scaler.pkl")
    regressor = joblib.load("best_reg_model.pkl")
    reg_scaler = joblib.load("reg_scaler.pkl")
    return classifier, class_scaler, regressor, reg_scaler

try:
    model, scaler, price_model, price_scaler = load_models()
except Exception as e:
    st.error("Model files load nahi ho paaye. Check karo ki chaaron .pkl files app.py ke same folder mein hain.")
    st.exception(e)
    st.stop()

FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc",
    "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores",
    "pc", "px_height", "px_width", "ram", "sc_h", "sc_w",
    "talk_time", "three_g", "touch_screen", "wifi"
]

st.title("📱 Mobile Price Prediction System")
st.write("Enter the phone specifications to view the Price Range (0–3) and predicted price.")

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("Phone Specifications")
    with st.form("phone_specs"):
        a, b = st.columns(2)
        with a:
            battery_power = st.slider("Battery Power (mAh)", 500, 6000, 1767)
            bluetooth = st.checkbox("Bluetooth", value=False)
            clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5, 0.1)
            dual_sim = st.checkbox("Dual SIM", value=False)
            front_camera = st.slider("Front Camera (MP)", 0, 19, 8)
            four_g = st.checkbox("4G", value=True)
            internal_memory = st.selectbox("Internal Memory (GB)", [8, 16, 32, 64, 128, 256, 512], index=3)
            thickness = st.slider("Thickness (cm)", 0.1, 1.0, 0.5, 0.1)
            weight = st.slider("Weight (g)", 80, 200, 111)
            processor_cores = st.slider("Processor Cores", 1, 8, 7)
        with b:
            primary_camera = st.slider("Primary Camera (MP)", 0, 48, 30)
            pixel_height = st.slider("Pixel Height", 0, 1999, 1000)
            pixel_width = st.slider("Pixel Width", 500, 1999, 1200)
            ram = st.slider("RAM (MB)", 256, 8192, 4096, 256)
            screen_height = st.slider("Screen Height (cm)", 5, 19, 12)
            screen_width = st.slider("Screen Width (cm)", 0, 17, 6)
            talk_time = st.slider("Talk Time (hours)", 2, 20, 10)
            three_g = st.checkbox("3G", value=True)
            touch_screen = st.checkbox("Touch Screen", value=True)
            wifi = st.checkbox("WiFi", value=False)

        submitted = st.form_submit_button("Predict Mobile Price")

with right:
    st.subheader("Output")
    if submitted:
        values = [[
            battery_power, int(bluetooth), clock_speed, int(dual_sim), front_camera,
            int(four_g), internal_memory, thickness, weight, processor_cores,
            primary_camera, pixel_height, pixel_width, ram, screen_height,
            screen_width, talk_time, int(three_g), int(touch_screen), int(wifi)
        ]]
        phone = pd.DataFrame(values, columns=FEATURES)
        try:
            price_range = int(model.predict(scaler.transform(phone))[0])
            predicted_price = int(round(float(price_model.predict(price_scaler.transform(phone))[0])))
            st.success("Prediction completed!")
            st.metric("Predicted Price Range", f"{price_range} / 3")
            st.metric("Predicted Price", f"₹ {predicted_price:,}")
            st.caption("Price Range 0–3 is the model's classification output. The price in rupees is the regression model's estimate.")
        except Exception as e:
            st.error("Prediction mein problem aayi. Error details:")
            st.exception(e)
    else:
        st.info("Phone specifications select karke **Predict Mobile Price** par click karo.")
        st.markdown("**Price Range guide**")
        st.write("0 = Low cost · 1 = Medium-low · 2 = Medium-high · 3 = High cost")
        st.caption("The range labels are a simple guide; actual classification follows the trained model.")
