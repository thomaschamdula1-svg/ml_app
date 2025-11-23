#!/usr/bin/env python
# coding: utf-8

# In[4]:


# IMPORT ESSENTIAL LIBRIES FOR DATA ANAYSIS,MACHINE LEARNING,DEEP LEARNING AND CAS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import sklearn #machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import openpyxl
# In[6]:
#SETTING

st.set_page_config(page_title="EBDAT ML APP", layout="centered")
st.markdown("""
<style>
.big-title{
font-size:38px;
color:#CAF50;
font-weight:700;
text-align:center;
}
.card{

padding:20px;
boarder-radius:15px;
background:#f5f5f5;
text-align:center;
box-shadow:4px 10px rgba(0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'> superstore profit prediction App</div><br>", unsafe_allow_html=True)

st.sidebar.header("Navigation")
page=st.sidebar.radio("Go to:",["Home","Model Training", "Visualizations","predictions"])
    
st.title("superstore profit prediction app with charts(excel supported)")

                   

uploaded_file = st.file_uploader("superstore.xlsx", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("### dataset preview")
    st.dataframe(df.head())

    if "sales" not in df.columns or "profit" not in df.columns:
        st.error("your excel file must contain 'sales' and 'profit' columns!")
    else:
        x=df[["sales"]]
        y=df[["profit"]]
        x_train,x_test,y_train,y_test=train_test_split(
            x,y, test_size=0.2,random_state=42
        )

        model=LinearRegression()
        model.fit(x_train,y_train)
        st.success("model trained successfully!")

        #show accuracy

        score=model.score(x_test,y_test)
        st.write(f"### model R^2 score:{score:.4f}")
#home page
        if page == "Home":
            st.subheader("welcome")
            st.write("this application predict ** profit** using **sales** from your superstore excel file powered by edbat consultancy.")

        st.write("## sales vs profit")

        fig,ax=plt.subplots()
        ax.scatter(df["sales"], df["profit"])

        x_range = np.linspace(df["sales"].min(), df["sales"].max(), 100)
        y_pred_line = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred_line)

        ax.set_xlabel("sales")
        ax.set_ylabel("profit")
        ax.set_title("sales vs profit with regression line")
        st.pyplot(fig)

        #histo

        st.write("## profit distribution")
        fig2,ax2=plt.subplots()
        ax2.hist(df["profit"], bins=20)

        ax2.set_xlabel("profit")
        ax2.set_ylabel("frequency")
        ax2.set_title("profit histogram")

        st.pyplot(fig2)
#prediction section
        st.write("## predict profit")

        sales_input = st.number_input("enter sales value:", min_value=0.0)
        if st.button("predict profit"):
            predicted_profit = float(model.predict([[sales_input]])[0])
            st.write(f"### predicted profit: {predicted_profit:.2f}") 


# In[ ]:




