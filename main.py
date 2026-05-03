from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st
st.session_state['clicked'] = True
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = None
st.title("Yamaan Faraz YF predictor.")
uploaded_file = st.file_uploader("upload .csv or .txt file")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    st.table(df)
    X = st.text_input(f"Give name of col for Year..", key="1")
    X1 = st.text_input(f"Give name of col for Month..", key="2")
    Y = st.text_input(f"Give name for target.", key="3")
    if X in df.columns and Y in df.columns:
        try:
            rX = df[[X, X1]].values
            rY = df[Y].values
            # 1. Initialize the model
            model = LinearRegression()

            # 2. Train it
            model.fit(rX, rY)

            # 3. Save the actual model object to session state
            st.session_state['model_trained'] = model
        except Exception as e:
            st.error(e)
        if st.session_state.get('model_trained') is not None:
            col1, col2 = st.columns(2)
            with col1:
                year_in = st.text_input("Year (e.g., 2026):")
            with col2:
                month_in = st.text_input("Month (1-12) can expand if your file contains from 12 to 24, so write 25 if want to know for January:")

            if year_in and month_in:
                try:
                    # Convert to float/int
                    year_val = float(year_in)
                    month_val = float(month_in)

                    # Model expects [[Year, Month]] because of our new CSV structure
                    result = st.session_state['model_trained'].predict([[year_val, month_val]])
                    st.divider()
                    st.success(f"Predicted Result: {result[0]:.2f}")
                except ValueError:
                    st.error("Please enter valid numbers for both Year and Month.")