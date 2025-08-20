import africastalking
import streamlit as st 
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import os
import glob 
import time
import math
import warnings 
import plotly.graph_objects as go
import plotly.express as px


import google.generativeai as genai

from geopy.geocoders import Nominatim
from prophet import Prophet

warnings.filterwarnings('ignore')


africastalking.initialize(
    username='EMID',
    api_key = os.getenv("AT_API_KEY")
)

sms = africastalking.SMS


df = pd.read_csv("src/Uganda_daily_mean_temperature_with_coordinates_final.zip")



def welcome_message(first_name, phone_number):

    recipients = [f"+254{str(phone_number)}"]

    print(recipients)
    print(phone_number)

    # Set your message
    message = f"Hi {first_name}, Youre now part of our smart climate & data insights platform. Stay updated with climate alerts that matter to you. ";

    # Set your shortCode or senderId
    sender = 20880

    try:
        response = sms.send(message, recipients, sender)

        print(response)

    except Exception as e:
        print(f'Houston, we have a problem: {e}')

    st.toast(f"Account Created Successfully")



def clean_dataframe(df):
	df['date'] = pd.to_datetime(df['date'],  format="%d/%m/%Y", errors='coerce')
	df.drop(columns=['Unnamed: 0', 'index'], inplace=True)


	return df


def plot_dist_plot_for_mean_temp(df):
	fig = px.histogram(
		df, 
		x="mean_temperature",
		nbins=30, 
		title="Distribution Plot For Mean Temperature (°C)"

	)

	fig.update_layout(
		title=dict(x=0.5, font=dict(size=18)),
		xaxis_title="Mean Temperature (°C)",
		yaxis_title="Count"

	)

	return st.plotly_chart(fig, use_container_width=True)


def heatmap_for_mean_temp(df):
	df['month'] = pd.to_datetime(df['date']).dt.month
	# df['year'] = pd.to_datetime(df['date']).dt.year
	pivot = df.pivot_table(values='mean_temperature', index='year', columns='month')

	plt.figure(figsize=(14,10))
	fig = sb.heatmap(pivot, cmap='coolwarm', annot=True, linewidths=0.5)
	plt.xlabel(None)
	plt.ylabel(None)
	plt.title("Monthly Mean Temperature (°C) by Year")
	# plt.show()

	return fig



def boxplot_for_mean_temp(df, selected_region):
	region_grp = df.groupby('region')
	get_region = region_grp.get_group(selected_region)

	fig = px.box(
		get_region,
		x="year",
		y="mean_temperature",
		points="all",  
		title=f"Yearly Distribution of Mean Temperature (°C) for \n{selected_region}",
	)

	fig.update_layout(
		xaxis_title="Year",
		yaxis_title="Mean Temperature (°C)",
		title=dict(x=0.5, font=dict(size=18)),
		xaxis=dict(tickangle=90),
		height=600,
		width=1000
	)


	return st.plotly_chart(fig, use_container_width=True)


def rolling_avg_for_mean_temp(df, selected_region):
	# reduce noise and highlight long-term patterns

	region_grp = df.groupby('region')
	get_region = region_grp.get_group(selected_region)

	# 30-day rolling average
	get_region['rolling_mean'] = get_region['mean_temperature'].rolling(window=30).mean()


	fig = go.Figure()

	fig.add_trace(go.Scatter(
		x=get_region['date'].tail(1000),
		y=get_region['mean_temperature'].tail(1000),
		mode='lines',
		name='Daily Temperature',
		line=dict(color='blue'),
		opacity=0.4

	))


	fig.add_trace(go.Scatter(
		x=get_region['date'].tail(1000),
		y=get_region['rolling_mean'].tail(1000),
		mode='lines',
		name='30-Day Rolling Average',
		line=dict(color='green', width=3)

	))

	fig.update_layout(
		title=f'Mean Temperature with 30-Day Rolling Average for {selected_region}',
		xaxis_title='Date',
		yaxis_title='Mean Temperature (°C)',
		template='plotly_white'

	)


	return st.plotly_chart(fig, use_container_width=True)


def lineplot_for_mean_temp(df):
	plt.figure(figsize=(12,6))
	fig = plt.plot(df['date'], df['mean_temperature'], label='Mean Temperature', linewidth=2)
	plt.xlabel("Date")
	plt.ylabel("Mean Temperature (°C)")
	plt.title("Mean Temperature Over Time")
	plt.legend()
	plt.show()

	return fig


def region_group_for_mean_temp(df, selected_region):
	region_group = df.groupby('region')
	get_region = region_group.get_group([selected_region])
	get_region[['date', 'mean_temperature']].value_counts().to_frame()

	fig = plt.plot(get_region['date'], get_region['mean_temperature'], label='Mean Temperature', linewidth=2)
	plt.title(f"Mean Temperature for {selected_region}", fontdict={'size': 12})

	return fig


def mean_temp_for_all_regions(df):
	df['date'] = pd.to_datetime(df['date'])

	# Unique regions
	regions = df['region'].unique()

	# Define subplot grid size
	n_regions = len(regions)
	n_cols = 5 
	n_rows = math.ceil(n_regions / n_cols)

	# Create figure and axes
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 25), sharex=False, sharey=False)
	axes = axes.flatten()  

	# Plot each region
	for i, region in enumerate(regions):
		region_data = df[df['region'] == region]
		axes[i].plot(region_data['date'], region_data['mean_temperature'], color='tab:blue')
		axes[i].set_title(region, fontsize=12)
		axes[i].tick_params(axis='x', labelrotation=45, labelsize=7)
		axes[i].tick_params(axis='y', labelsize=7)

	# Remove unused subplots if regions < n_rows*n_cols
	for j in range(i+1, len(axes)):
		fig.delaxes(axes[j])

	# Add global labels
	fig.suptitle("Mean Temperature (°C) Over Time by Region", fontsize=16)
	# fig.supxlabel("Date")
	fig.supylabel("Mean Temperature (°C)")

	plt.tight_layout(rect=[0, 0, 1, 0.96])
	plt.show()

	return plt.show()





def the_explainer(prompt):

	model = genai.GenerativeModel("gemini-2.0-flash", 
		system_instruction = '''
					You are an intelligent data analysis assistant designed to help users understand insights derived from grouped datasets. 
					Your primary objective is to provide clear, concise, and engaging explanations of visualized data based on the user's selected country or region, the specific series being analyzed, and key insights.

					Your responsibilities include:
					1. Explaining the purpose of the graph and its relevance to the selected parameters.
					2. Highlighting key insights in a structured and easy-to-understand manner.
					3. Encouraging users to interpret trends, disparities, or patterns observed in the graph.
					4. Using a professional yet approachable tone to ensure the explanation is interactive and user-friendly.

					Make sure your explanations are tailored to the user's selections and provide actionable insights wherever applicable also summarize and quantify the results and possible as you can.
					
					Note: Make it short and also implement Eli5 to be able to favor non technical users therefore avoid technical jargons
					''')

	response = model.generate_content(
    prompt,
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)
	
	return response.text