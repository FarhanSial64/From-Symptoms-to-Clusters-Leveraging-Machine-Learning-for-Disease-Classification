import streamlit as st

# Title of your app
st.title("Data Science Concepts App")

# Example input
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")

# Example button
if st.button("Say Hello"):
    st.success("You clicked the button!")