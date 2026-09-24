import streamlit as st

from src.predict import predict_churn


st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📊",
    layout="wide",
)


st.title("Telecom Customer Churn Prediction")

st.write(
    "Enter customer information below to estimate the probability "
    "that the customer will churn."
)


with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    # ---------------------------------------------------------
    # Column 1
    # ---------------------------------------------------------

    with col1:

        st.subheader("Customer Information")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1,
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

    # ---------------------------------------------------------
    # Column 2
    # ---------------------------------------------------------

    with col2:

        st.subheader("Services")

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service",
            ],
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No",
            ],
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

    # ---------------------------------------------------------
    # Column 3
    # ---------------------------------------------------------

    with col3:

        st.subheader("Contract & Billing")

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk"
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    result = predict_churn(customer)

    probability = result["churn_probability"]
    threshold = result["decision_threshold"]
    prediction = result["prediction"]

    st.divider()

    st.subheader("Prediction Result")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Churn Probability",
        f"{probability:.2%}",
    )

    metric2.metric(
        "Decision Threshold",
        f"{threshold:.0%}",
    )

    metric3.metric(
        "Prediction",
        prediction,
    )

    if prediction == "Churn":
        st.warning(
            "This customer is classified as having elevated churn risk."
        )
    else:
        st.success(
            "This customer is classified as lower churn risk."
        )