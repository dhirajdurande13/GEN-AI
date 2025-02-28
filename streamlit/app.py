import streamlit as st
import pandas as pd
import numpy as np
st.title("Streamlit page")
st.write("simple text!!!!!")



df=pd.DataFrame({
    'first Col':[1,2,3,4,5],
    'second Col':[10,20,30,40,50]
})
# st.write(f'Dataframe: {df}')
st.write(df)


# create line chart
chart_data=pd.DataFrame(
    np.random.randn(20,3),columns=['a','b','c']
)
st.line_chart(chart_data)
# st.line_chart(df)

# slider
age=st.slider("select your age:",0,100,25)
st.write(f"Age is: {age}")

options=["P","j","C"]
let=st.selectbox("Select Leter: ",options)
print(f"{let}")