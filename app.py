import streamlit as st

st.title("Activation Functions")
activation_function = st.selectbox('Choose an activation function', ['None', 'Sigmoid', 'Tanh', ])

if activation_function == 'Sigmoid':
    st.write('Sigmoid')