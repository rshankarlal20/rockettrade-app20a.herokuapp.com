import streamlit as st
from datetime import date

import yfinance as yf
#from fbprophet import Prophet
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

def main():

    html_temp = """
            <div style="background-color:royalblue;padding:10px;border-radius:10px">
            <h1 style="color:white;text-align:center;">Stock Predictions App using fbprophet</h1>
            </div>
            """
    st.markdown(html_temp, unsafe_allow_html = True)
    
page_choice=st.sidebar.radio("Pages",["Stock Prediction Algo","Plots","Codes"])


def create_space(number_of_row):
    for i in range(number_of_row):

        st.sidebar.markdown("&nbsp;")

create_space(5)

st.sidebar.markdown("**Info**\n \nCopyright : Rama Lal \ rshankarlal20@gmail.com")


st.sidebar.markdown(""" 
                        <a href="https://www.linkedin.com/in/rama-lal/" target="blank"><img align="center" 
                        src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg" alt="rama-lal" height="30" width="30" /></a><a
                        href="https://github.com/rshankarlal20" target="blank"><img align="center"
                        src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/github.svg" alt="rshankarlal20" height="30" width="30" /></a></p>

                    """, unsafe_allow_html = True)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App - Algorithm Trading')

stocks = ('GOOG', 'AAPL', 'MSFT', 'AMZN', 'RPOWER.NS', 'TATAMOTORS.NS', 'GE', 'RELIANCE.NS', 'CAPLIPOINT.NS', 'TATACHEM.NS', 'TATAPOWER.NS', 'MOTHERSUMI.NS', 'ICICIBANK.NS', 'ZOMATO.NS', 'SBIN.NS', 'TATAELXSI.NS', 'AMARAJABAT.NS', 'EXIDEIND.NS', 'DLF.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

    
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading real life data from Web... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)