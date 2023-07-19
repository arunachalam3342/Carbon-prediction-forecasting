import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import base64
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

@st.cache(allow_output_mutation=True)

def preprocess():
    pass

#prediction
df=pd.read_csv('CO2 Emissions_Canada.csv')
new_df=pd.read_csv('CO2 Emissions_Canada.csv')
new_df=new_df.drop(index=df[df['Fuel Type']=='N'].index[0])
le=LabelEncoder()
le_1=LabelEncoder()
le_2=LabelEncoder()
le_3=LabelEncoder()
df['Make']=le.fit_transform(df['Make'])
df['Make']=df['Make'].apply(lambda x:x+1)
df['Model']=le_1.fit_transform(df['Model'])
df['Model']=df['Model'].apply(lambda x:x+1)
df['Vehicle Class']=le_2.fit_transform(df['Vehicle Class'])
df['Vehicle Class']=df['Vehicle Class'].apply(lambda x:x+1)
df['Transmission']=le_3.fit_transform(df['Transmission'])
df['Transmission']=df['Transmission'].apply(lambda x:x+1)
df=pd.get_dummies(df,columns=['Fuel Type'])
fuel=["Z","D","X","E","N"]
fuel=pd.DataFrame(fuel,index=fuel,columns=['Fuel'])
fuel=pd.get_dummies(fuel,columns=['Fuel'])

#time series forecast
df_time=pd.read_csv('CO2 dataset.csv')
df_time.dropna(inplace=True)
df_time['Year']=pd.to_datetime(df_time['Year'],format="%Y")
df_time.set_index(['Year'],inplace=True)
df_time.index.freq='YS'
train_df=df_time.loc[:'1990-01-01']

#streamlit section
im = Image.open('App_Icon.png')
st.set_page_config(page_title="CO2 Emmision Prediction App",page_icon = im)
app_mode=st.sidebar.selectbox('Select Page',['Home','Prediction','Timeseries'])
page_bg_img='''
    <style>
    .stApp{
    background-size:cover;
    background-image:url('https://wallpaperaccess.com/full/776490.jpg')
    }
    </style>
    '''
st.markdown(page_bg_img, unsafe_allow_html=True)
hide_default_format = """
       <style>
       
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


    
    
    
if app_mode=='Home':
    
    st.markdown(page_bg_img,unsafe_allow_html=True)
    components.html("""<h1><p class='font-family:Arial'><u>Welcome to CO2 Emmision Prediction Web App</u></p></h1>""")
    st.subheader("Prediction Page   :   Predict the emmision of the Carbon Dioxide")
    st.subheader("Timeseries Page   :   Display the timeseries of the Carbon Dioxide")


    
elif app_mode=='Prediction':
    st.title('Welcome to prediction page')
    st.image('img.jpg')
    st.header('Fill all necessary information in order to calculate amount of Carbon Dioxide emitted')
    st.sidebar.header("Enter the values")
    company=st.sidebar.selectbox("Car company",list(new_df['Make'].unique()))
    model=st.sidebar.selectbox("Car model",list(new_df['Model'].unique()))
    car_type=st.sidebar.selectbox("Car type",list(new_df['Vehicle Class'].unique()))
    engine=st.sidebar.number_input("Engine size (L)")
    cylinder=st.sidebar.number_input("Cylinders")
    transmission=st.sidebar.selectbox("Transmission",list(new_df['Transmission'].unique()))
    fuel_class=st.sidebar.selectbox("Fuel Class",['X','Z','D','E'])
    st.subheader("Fuel Consumption of the car in different areas")
    fcmp1=st.number_input("Fuel Consumption city(L/100km)")
    fcmp2=st.number_input("Fuel Consumption Hwy (L/100 km)")
    fcmp3=st.number_input("Fuel Consumption Comb (L/100 km)")
    fcmp4=st.number_input("Fuel Consumption Comb (mpg)")
    l=[company,model,car_type,engine,cylinder,transmission,fuel_class,fcmp1,fcmp2,fcmp3,fcmp4]
    l[0]=(le.transform(np.array([l[0]]))+1)[0]
    l[1]=(le_1.transform(np.array([l[1]]))+1)[0]
    l[2]=(le_2.transform(np.array([l[2]]))+1)[0]
    l[5]=(le_3.transform(np.array([l[5]]))+1)[0]
    pred=[l[0],l[1],l[2],l[3],l[4],l[5]]
    fuel_arr=list(fuel[fuel.index==l[6]].values.flatten())
    for i in range(7,len(l)):
        pred.append(l[i])
    pred=pred+fuel_arr
    pred=np.array(pred)
    predict=st.button("Predict")
    if predict:
        model=tf.keras.models.load_model('dnn_model')
        val=model.predict(pred)
        st.info("CO2 Emitted By Vehicle is " + str(val[0][0]) + "(g/km)")
        
elif app_mode=='Timeseries':
    components.html("""<h1><p style='font-family:Arial;font-size:35px;padding-bottom:10px;'><u>Atmosphere CO2 Emission Forecasting</u></p></h1>""")
    st.sidebar.header("CO2 Emission time series")
    year=int(st.sidebar.slider("Years to forecast",0,100))
    predict=st.sidebar.button("Forecast")
    if predict:
        robjects.r.assign('year',year)
        result=robjects.r('''
        load<-readRDS("arima_model.rda")
        pred<-predict(load,n.ahead=year)
        ''')
        end_date="01/01/" + str(1991+year)
        date_rng=pd.date_range(start='01/01/1991',end=end_date,freq='Y')
        time_pred=pd.DataFrame(result[0],index=date_rng,columns=["CO2"])
        col1,col2=st.columns(2,gap="medium")
        with col1:
            original_style='<p style="font-family:sans-serif;color:Green;font-size:20px;">Forecasted CO2 Emission Dataframe</p>'
            st.markdown(original_style,unsafe_allow_html=True)
            st.dataframe(time_pred)
        with col2:
            original_style='<p style="font-family:sans-serif;color:blue;font-size:23px;">Time Series Graph</p>'
            st.markdown(original_style,unsafe_allow_html=True)
            fig,ax=plt.subplots()
            train_df['CO2'].plot(style="--",color="blue",legend=True,label="known")
            time_pred['CO2'].plot(color="green",label="Prediction",legend=True)
            st.pyplot(fig)
    