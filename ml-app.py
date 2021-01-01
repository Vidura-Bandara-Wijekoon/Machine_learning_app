import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import base64
import matplotlib.pyplot as plt 
import matplotlib
# Import the required module for text  
# to speech conversion 
from gtts import gTTS 
# This module is imported so that we can  
# play the converted audio 
import os 
  

from load_css import local_css

# Model building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X.shape)
    st.write('Test set')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

#---------------------------------#

local_css("style.css")


st.write("""
          <p style="color:yellow" align="center" ><font face = "Comic sans MS" size =" 5"><i>
        For maching learning enthusiastics around the world ! Welcome Vidura!</i></font>
    </p>
    """, unsafe_allow_html=True)



st.write("""
          <p style="color:red" align="center" ><font face = "Comic sans MS" size =" 5"><i>
        Hello visitor!</i></font>
    </p>
    """, unsafe_allow_html=True)
    
st.write("""
         
# *Machine Learning Wizard* :sunglasses:
""")

st.write("""
         <p style="color:blue" align="center" ><font face = "Comic sans MS" size =" 5"><i>
<strong>RandomForestRegressor()</strong> </font></p><p style="color:white" align="center" ><font face = "Comic sans MS" size =" 5">function is used in this app for build a regression model using the </font></p><p style="color:blue" align="center" ><font face = "Comic sans MS" size =" 5"><strong>Random Forest</strong> </font></p><p style="color:white" align="center" ><font face = "Comic sans MS" size =" 5">algorithm.
You can change the hyperparameters according to your desire!
</i></font></p>
    """, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:D:/2021-1-1/ml-app-main/ml-app-main/ml-app-main/1567833.jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('D:/2021-1-1/ml-app-main/ml-app-main/ml-app-main/1567833.jpg')
#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. DataSet Uploading Location'):
    uploaded_file = st.sidebar.file_uploader("Upload your input files with csv extention", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://github.com/Vidura-Wijekoon/machine-learning-1)
""")


# Sidebar - Specify parameter settings
with st.sidebar.header('2. Parameters Modification'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Modification of Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. Modification of General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        #diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #Y = pd.Series(diabetes.target, name='response')
        #df = pd.concat( [X,Y], axis=1 )

        #st.markdown('The Diabetes dataset is used as the example.')
        #st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)


	# Pie Chart
if st.checkbox("Pie Plot"):
	all_columns_names = df.columns.tolist()
	if st.button("Generate Pie Plot"):
		st.success("Generating A Pie Plot")
		st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
		st.pyplot()


st.set_option('deprecation.showPyplotGlobalUse', False)