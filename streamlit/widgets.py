import streamlit as st

st.write('Enter name: ')
name=st.text_input("Enter Name: ")

if name:
    st.write(f'Your name is : {name}')