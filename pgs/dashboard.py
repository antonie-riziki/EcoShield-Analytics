import streamlit as st
import pandas as pd
import seaborn as sb
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import gzip
import pickle
import glob 
import warnings 

from prophet import Prophet

warnings.filterwarnings('ignore')


sys.path.insert(1, 'modules')

from func import clean_dataframe, plot_dist_plot_for_mean_temp, heatmap_for_mean_temp, boxplot_for_mean_temp, rolling_avg_for_mean_temp, lineplot_for_mean_temp, region_group_for_mean_temp, mean_temp_for_all_regions


## Page Configurations

st.set_page_config(
    page_title="Uganda Mean Temperature Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


st.markdown(
    """
    <div class=title>
        <div style=" justify-content: center;">
            <h1 style="text-align: center; margin-top: -50px; color: #007B8A;"> Temperature Trend In Uganda 🌍</h1>
        </div>
    </div> 
    """,
    unsafe_allow_html=True,
)



# df = pd.read_csv('src/Uganda_daily_mean_temperature_with_coordinates_final.csv')
df = pd.read_csv("src/Uganda_daily_mean_temperature_with_coordinates_final.zip")

clean_dataframe(df)
# df.drop(columns=['Unnamed: 0', 'index'], inplace=True)

# st.dataframe(df.head())



with st.sidebar:
    st.title('🗺️ Uganda Temperature Trend Analysis ')
    
    year_list = list(df.year.unique())[::-1]
    
    selected_year = st.selectbox('Select a year', year_list, index=len(year_list)-1)
    df_selected_year = df[df.year == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="region", ascending=False)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)



def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None,
                             scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=1200
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        ) 
    # height=300
    return heatmap


def make_choropleth(input_df, input_id, input_column, input_color_theme, geojson_file):
    choropleth = px.choropleth(
        input_df,
        geojson=geojson_file,
        locations=input_id,
        featureidkey="properties.shapeName", 
        color=input_column,
        color_continuous_scale=input_color_theme,
        range_color=(0, input_df[input_column].max()),
        labels={input_column: input_column}
    )
    choropleth.update_geos(fitbounds="locations", visible=False)
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth


def get_yearly_temperature(df, year):
    
    get_year = df[df['year'] == year]

    get_year_df = (
        get_year.groupby('region', as_index=False)['mean_temperature']
        .mean().sort_values(by="mean_temperature", ascending=False)
        .rename(columns={'mean_temperature': 'avg mean'})
    )

    # get_year_df = get_year_df.reset_index(inplace=True)

    return get_year_df


def load_prophet_model(data):
	with gzip.open("models/predict_future_temp.pkl.gz", "rb") as f:
		model = pickle.load(f)


	data.reset_index(inplace=True)
	data.rename(columns={'date': 'ds', 'mean_temperature': 'y'}, inplace=True)

	data['ds'] = data['ds'].dt.tz_localize(None)

	future = model.make_future_dataframe(periods=90)  # Predict 1 year into the future
	forecast = model.predict(future)

	fig = go.Figure()


	fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual'))

	fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))

	# prediction intervals
	fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line_color='lightgrey', name='Lower Confidence Interval'))
	fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line_color='lightgrey', name='Upper Confidence Interval'))

	fig.update_layout(
	    title=f'Future Mean Temperatures',
	    xaxis_title='date',
	    yaxis_title='mean_temperature',
	    legend_title='Legend'
	)

	# Display the plot in Streamlit
	# st.plotly_chart(fig)

	
	return st.plotly_chart(fig, use_container_width=True)

























# Dashboard Main Panel

with open("src/geoBoundaries-UGA-ADM1_simplified.geojson") as f:
    uganda_geojson = json.load(f)


col = st.columns((1.5, 6.5, 2), gap='medium')

with col[0]:

    max_temp = df["mean_temperature"].max()
    min_temp = df["mean_temperature"].min()


    max_data = pd.DataFrame({
    	"Category": ["max temp", ""],
    	"Value": [max_temp, (max_temp * 0.2)] 
    })

    fig_max = px.pie(
    	max_data,
    	values="Value",
    	names="Category",
    	hole=0.6,
    	color_discrete_sequence=["red", "lightgrey"]

    )

    fig_max.update_layout(
    	title_text=f"max temp ({max_temp:.2f}°C)",
    	annotations=[dict(text=f"{max_temp:.2f}°C", x=0.5, y=0.5, font_size=12, showarrow=False)]
    )

    min_data = pd.DataFrame({
    	"Category": ["min temp", ""],
    	"Value": [min_temp, (min_temp * 0.2)]

    })

    fig_min = px.pie(
    	min_data,
    	values="Value",
    	names="Category",
    	hole=0.6,
    	color_discrete_sequence=["blue", "lightgrey"]
    )

    fig_min.update_layout(
    	title_text=f"min temp ({min_temp:.2f}°C)",
    	annotations=[dict(text=f"{min_temp:.2f}°C", x=0.5, y=0.5, font_size=12, showarrow=False)]
    )

    st.plotly_chart(fig_max, use_container_width=True)
    st.plotly_chart(fig_min, use_container_width=True)


with col[1]:

    fig = px.scatter_mapbox(
    	df.head(100000),
    	lat="lat",
    	lon="lon",
    	color="mean_temperature",
    	size="mean_temperature",
    	hover_name="region",
    	hover_data=["date", "year"],
    	color_continuous_scale="thermal",
    	size_max=20,
    	zoom=6,
    	mapbox_style="open-street-map"

    )

    # fig.update_layout(title="Mean Temperature by Region (Interactive Map) ")
    
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================== #

    fig = px.scatter_mapbox(
    	df,
    	lat="lat",
    	lon="lon",
    	size="mean_temperature",         
    	color="mean_temperature",        
    	hover_name="region",             
    	animation_frame="year",          
    	mapbox_style="open-street-map",
    	zoom=6,
    	height=600

    )



    fig.update_layout(
    	# title="Animated Mean Temperature Over Time",
    	margin={"r":0,"t":40,"l":0,"b":0}

    )

    st.plotly_chart(fig, use_container_width=True)


    
    # choropleth = make_choropleth(df_selected_year, 'region', 'mean_temperature', selected_color_theme, geojson_file=uganda_geojson)
    # st.plotly_chart(choropleth, use_container_width=True)
    
    
    

with col[2]:
    st.markdown(f'#### Avg Mean Temp {selected_year}')


    yearly_dataframe = get_yearly_temperature(df, selected_year)


    st.data_editor(
	    yearly_dataframe,
	    column_config={
	        "avg mean": st.column_config.ProgressColumn(
	            "Mean Temp (°C)",
	            format="%.2f",
	            min_value=float(yearly_dataframe["avg mean"].min()),
	            max_value=float(yearly_dataframe["avg mean"].max()),
	            
	        )
	    },
	    hide_index=True,
	    use_container_width=True
	)
    
    styled_df = yearly_dataframe.style.bar(subset=["avg mean"], color='#ff4b4b')
    

    with st.expander('About', expanded=True):
        st.write('''
            - Data: [Africa Agriculture Watch](https://www.aagwa.org/Uganda/data?p=Uganda%2Ftemperature_trend_analysis%2Fdaily_mean_temperatures).
            - :orange[**About**]: interactive dashboard that visualizes community-level vulnerability by combining climate exposure, food security, and nutrition adequacy data.
            - :orange[**What we aim to achieve**]: to be a leading seamless communication tools that visualizes vulnerability indicators interactively and a go-to platform ensuring clear and accessible data for local policymakers, NGOs, and community planners.

            ''') 




heatmap = make_heatmap(df, 'year', 'region', 'mean_temperature', selected_color_theme)
st.altair_chart(heatmap, use_container_width=True)
    


col1, col2 = st.columns([5, 5])

with col1:
	plot_dist_plot_for_mean_temp(df)


# with col2:
selected_region = st.selectbox(label="region", options=df['region'].unique())
boxplot_for_mean_temp(df, selected_region)


rolling_avg_for_mean_temp(df, selected_region)



load_prophet_model(df)



