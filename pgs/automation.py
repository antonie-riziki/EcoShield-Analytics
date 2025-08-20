import streamlit as st
import seaborn as sb 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import io
import africastalking
import google.generativeai as genai

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.lib.utils import ImageReader
from PIL import Image


from sklearn.impute import SimpleImputer

sys.path.insert(1, './pages')
# print(sys.path.insert(1, '../pages/'))

from func import the_explainer

from dotenv import load_dotenv

load_dotenv()


st.markdown(
    """
    <div class=title>
        <div style=" justify-content: center;">
            <h1 style="text-align: center; margin-top: -50px; color: #007B8A;"> EcoShield IntelliBoard 📊</h1>
            <p style="text-align: center;">From Data to Insights Seamlessly</p>
        </div>
    </div> 
    """,
    unsafe_allow_html=True,
)

st.divider()



st.info("""
        **About:**
        IntelliBoard is an automated data visualization engine that takes CSV files and generates an interactive dashboard. It highlights key metrics and patterns to help you focus on what matters most.
        
        """)


st.info("""
        **What to Expect:**
        
        EcoShield’s automated dashboard analyzer is designed to take away the complexity of working with raw datasets. 
        With just a simple CSV upload, the system automatically:

        - Cleans and organizes your data.
        - Generates interactive charts, tables, and summaries.
        - Highlights patterns, anomalies, and long-term trends.
        - Provides customizable filters and drill-downs for deeper exploration.
        - No manual setup—just instant clarity from your data.
        
        """)



st.success("""
        **Value to Your Data:**

        - Preservation of Accuracy: The system minimizes human error by applying automated statistical checks.
        - Contextual Insights: Transforms numbers into narratives, giving meaning to otherwise hidden trends.
        - Scalability: Works across small datasets and large-scale CSV files with equal efficiency..
        
        """)



st.info("""
        **Value to Decision Making:**

        - Evidence-Based Choices: Instead of relying on gut feeling, decisions are grounded in real data patterns.
        - Real-Time Responsiveness: With instant dashboards, users can adapt strategies quickly as new data arrives.
        - Prioritization: Automatically surfaces the most impactful variables so users know what matters most.
        """)



st.info("""
        **Value to Policy & Strategy:**

        - Policy Shaping: By exposing long-term trends, anomalies, and correlations, the analyzer guides effective policy-making.
        - Transparency & Accountability: Decision-makers can present clean, easy-to-understand dashboards to stakeholders, fostering trust.
        - Forward Planning: Predictive insights allow policymakers to anticipate risks and opportunities, supporting proactive rather than reactive strategies

        """)


# Store file paths
uploaded_files = {}



def load_data(file):
    try:
        df = pd.read_csv(file, encoding='latin-1')
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def clean_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)  # Remove duplicates
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill missing numeric values
    df.fillna("Unknown", inplace=True)  # Fill missing categorical values
    return df

def generate_report(df):
    # report = {
    #     "Total Rows": len(df),
    #     "Total Columns": len(df.columns),
    #     "Missing Values": df.isnull().sum().sum(),
    #     "Duplicate Rows": df.duplicated().sum(),
    #     "Column Data Types": df.dtypes.to_dict(),
    #     "Basic Statistics": df.describe().to_dict()
    # }
    report = df.describe()
 
    return report


def get_df_info(df):
     buffer = io.StringIO ()
     df.info (buf=buffer)
     lines = buffer.getvalue ().split ('\n')
     # lines to print directly
     lines_to_print = [0, 1, 2, -2, -3]
     for i in lines_to_print:
         st.write (lines [i])
     # lines to arrange in a df
     list_of_list = []
     for x in lines [0:-3]:
         list = x.split ()
         list_of_list.append (list)
     info_df = pd.DataFrame (list_of_list)
     # info_df.drop (columns=['index', 'null'], axis=1, inplace=True)
     st.dataframe(info_df)


simple_impute = SimpleImputer()

def get_categorical_series(df):
    categories = []
    simple_impute = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for i in df.select_dtypes(include=['object']):
        categories.append(i)
    df[categories] = simple_impute.fit_transform(df[categories])
    return df.head()


def get_quantitative_series(df):
    numericals = []
    simple_impute = SimpleImputer(missing_values=np.nan, strategy='mean')
    for i in df.select_dtypes(include=['float64', 'int64']):
        numericals.append(i)
    df[numericals] = simple_impute.fit_transform(df[numericals])
    return df.head()


def get_heatmap(df):
    cmap_options = ['coolwarm', 'viridis', 'mako', 'cubehelix', 'rocket', 'flare', 'magma', 'Greens', 'Reds_r', 'BuGn_r', 'terrain_r']
    cmap_selection = st.pills('color map', options=cmap_options)

    numerical_categories = []
    for i in df.select_dtypes(include=['float64','int64']):
        numerical_categories.append(i)
        fig, ax = plt.subplots(figsize=(10, 6))
        sb.heatmap(df[numerical_categories].corr(), annot=True, fmt=".2f", cmap=cmap_selection, ax=ax)
        plt.title(f'Pearsons correlation of columns', fontdict={'size':14})
    return st.pyplot(fig, use_container_width=True)



def get_gemini_insights(df, selected_columns):
    
    prompt = f"""
    Perform exploratory data analysis on the following dataset columns:
    {selected_columns}

    Provide a concise, precise, and quantitative explanation of trends, distributions, 
    and correlations found in the data. Focus on key statistical insights.
    """
    
    model = genai.GenerativeModel("gemini-2.0-flash", 
        system_instruction = '''
            Role: You are an experienced data analyst with deep expertise in exploratory data analysis (EDA), statistical reasoning, and business intelligence. 
            Your task is to analyze datasets, extract key patterns, and generate concise, quantitative, and insightful summaries for decision-making.

            ### Guidelines for Analysis:
            1️. Be precise and quantitative – Use statistics, trends, and patterns to explain insights (e.g., mean, median, standard deviation, skewness).
            2️. Identify key distributions – Determine if the dataset exhibits normality, skewness, or outliers. Mention the spread, peaks, and anomalies.
            3️. Highlight correlations – Describe potential relationships within the data (e.g., positive correlation between income and age).
            4️. Explain business impact – Where relevant, connect insights to real-world implications (e.g., high attrition rate in employees over 40 suggests retention issues).
            5️. Avoid generic statements – Focus on actionable insights backed by data-driven reasoning.

            ### Handling General Questions:
            - If the user asks a **general question** (e.g., "What is the mean sample size across all studies?"), analyze the dataset holistically.
            - Identify relevant information based on the context of the question and infer the best approach to compute the answer.
            - If multiple interpretations exist, provide the most meaningful one along with reasoning.
            - If additional clarification is needed, ask the user a follow-up question.

            ### Expected Output Format:
            - **Summary:** Provide a concise, data-backed statement (e.g., "The dataset shows a right-skewed distribution with a median value of 52.")
            - **Key Observations:** Highlight significant patterns or anomalies (e.g., "There is an unexpected spike in sales during Q4, suggesting seasonal demand.")
            - **Correlations & Trends:** Identify relationships within the data (e.g., "An increase in customer retention correlates with higher engagement levels (+0.72).")
            - **Actionable Insights:** Provide practical recommendations (e.g., "The high standard deviation in expenditure suggests unstable spending habits among customers.")
            - **General Questions Response:** Answer based on available data (e.g., "The average sample size across all studies in the dataset is 450, with a standard deviation of 85. The distribution suggests that most studies have a sample size between 365 and 535.")

        ''')


    response = model.generate_content(
    prompt,
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)
    return st.write(response.text)

# Function to generate and download PDF report using ReportLab
def generate_pdf(df, selected_columns, insights):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    
    # Ensure insights is not None
    insights = insights if insights else "No insights available."

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(200, 750, "Exploratory Data Analysis Report")
    
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 720, f"Selected Columns: {', '.join(selected_columns)}")

    pdf.setFont("Helvetica", 11)
    
    # **Format and wrap insights text**
    y_position = 690  # Start drawing from this Y position
    max_width = 500   # Max width for text wrapping
    lines = simpleSplit(insights, pdf._fontname, pdf._fontsize, max_width)
    
    for line in lines:
        pdf.drawString(50, y_position, line)
        y_position -= 20  # Move down for next line
        
        # Add new page if needed
        if y_position < 100:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y_position = 750  # Reset position for new page

    pdf.showPage()  # Move to next page for plots

    # Save plots for selected columns
    for col in selected_columns:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sb.histplot(df[col], kde=True, ax=ax)
            plt.title(f"Distribution of {col}")
            
            img_path = f"{col}.png"
            plt.savefig(img_path, format="png")
            plt.close()

            pdf.drawImage(img_path, 100, 400, width=400, height=300)
            pdf.showPage()

    pdf.save()
    
    buffer.seek(0)
    return buffer





# def group_items(df):

#     grp_cols = []

#     for i in df.select_dtypes(include=['object']):
#         grp_cols.append(i)

#     grpby_columns = st.multiselect('select series to group', grp_cols)


#     if grpby_columns is not None:
#         group_by_cols = df.groupby(grpby_columns)

#         st.write(group_by_cols.head())

#     else: 
#         st.markdown('select column(s) to group')

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])



if uploaded_file:
    file_name = uploaded_file.name
    uploaded_files[file_name] = uploaded_file

    df = load_data(uploaded_file)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    if df is not None:
        st.info("""
        **Seamless Data Import:**

        The system allows users to upload CSV files or datasets with ease. Once uploaded, the raw data is immediately displayed in an 
        organized tabular format. This provides users with a quick overview of the dataset’s structure, including columns, data types, 
        and sample records.

        """)

        st.success("""
        **Key Value:**

        Users gain instant visibility into the dataset, enabling them to confirm that the right data has been loaded before proceeding 
        to analysis.

        """)
        
        
        st.dataframe(df.head())
        
        cleaned_df = clean_data(df)

        st.info("""
                **Descriptive Statistics & Key Metrics:**

                The system automatically generates statistical summaries of the dataset. These include measures of central tendency 
                (mean, median, mode), measures of dispersion (standard deviation, variance, range), and frequency distributions..

                """)

        st.success("""
                **Key Value:**

                This step helps users understand the underlying characteristics of the dataset, highlighting patterns, outliers, 
                and general trends before diving deeper into visualization or modeling.

                """)
        
        col1, col2 = st.columns(2)

        with col1:
            report = generate_report(cleaned_df)

            st.dataframe(report)
            st.json(report)
            st.dataframe(df.isnull().sum())

        with col2:
            st.write(df.shape)
            get_df_info(df)


        st.info("""
                **Visual Insights (Charts & Graphs):**

                The system converts raw numbers into visual stories by plotting graphs and charts. Users can interact with bar plots, 
                histograms, scatter plots, line graphs, and boxplots to uncover relationships, trends, and anomalies in the data.
                
                """)

        st.success("""
                **Key Value:**

                Graphical outputs simplify complex datasets, making patterns clearer and insights easier to communicate to both technical
                and non-technical audiences.

                """)
        
        vis1, vis2 = st.columns(2)

        with vis1:
            get_categorical_series(df)

            st.write('For categorical analysis')

            category_choice = st.selectbox(label="select series", options=[i for i in df.select_dtypes(include="object")])

            
            entry_options = [20, 50, 100, 200]
            st.write(len(df[category_choice].value_counts()))
            category_entries = st.pills('select entries', options=entry_options, key='cat_entries')

            st.bar_chart(df[category_choice].value_counts().head(category_entries))

            plt.title(f"Categorical analysis for the {category_choice}")

        with vis2:
            get_quantitative_series(df)

            ('For Quantitative analysis')

            quantitative_choice = st.selectbox(label="select series", options=[i for i in df.select_dtypes(include=["float64", "int64"])])

            # entry_options = [20, 50, 100, 200]
            st.write(len(df[category_choice].value_counts()))
            quantitative_entries = st.pills('select entries', options=entry_options, key='qty_entry')

            # df = px.data.tips()
            # fig = px.histogram(df[quantitative_choice])
            # fig.show()

            st.bar_chart(df[quantitative_choice].head(quantitative_entries))

        get_heatmap(df)
        
        st.info("""
                **Refining the Dataset:**

                The system identifies and resolves data quality issues such as missing values, duplicates, or inconsistencies. 
                It applies transformations like standardization, normalization, or removal of noisy data, then presents the cleaned dataset 
                in an updated tabular format.
                
                """)


        st.success("""
                **Key Value:**

                Clean data ensures reliable analysis, reduces biases, and builds confidence in the insights generated.

                """)

        st.dataframe(cleaned_df.head())
        
        # Option to download cleaned data
        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned Data", csv, "cleaned_data.csv", "text/csv", type="primary")



# Allow user to select previously uploaded files
# if uploaded_files:
#     selected_file = st.selectbox("Select a previously uploaded file:", list(uploaded_files.keys()))
#     if selected_file:
#         df = load_data(uploaded_files[selected_file])
#         if df is not None:
#             st.write("### Selected File Data Preview")
#             st.dataframe(df.head())
#         else:
#         	pass




# Grouping Data and Visualization
if uploaded_file:
    
    st.info("""
                **Aggragation & Comparative Analysis:**
                The system empowers users to group data series by categories, time intervals, or numerical ranges. 
                Aggregated outputs (e.g., sum, mean, median per group) are calculated and visualized in dynamic plots summaries.
                
                """)

    st.success("""
                **Key Value:**
                This functionality supports comparative analysis across dimensions (e.g., regions, time periods, categories), 
                helping users answer higher-level analytical and policy-driven questions.

                """)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        # User selects columns to group by
        selected_columns = st.multiselect("Select columns to group by:", df.columns)
    
    with col2:
        # User selects aggregation method
        aggregation_method = st.selectbox("Select aggregation method:", ["Min", "Max", "Sum", "Mean", "Median", "Std Dev", "Var", "Mean absolute dev", "Product"])
    
    if selected_columns:
        if aggregation_method == "Count":
            grouped_df = df.groupby(selected_columns).size().rename({'count': 'agg_column'})#.reset_index(name='agg_column')
        else:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            with col3:
                agg_column = st.selectbox("Select numerical column for aggregation:", numerical_columns)
            
                if aggregation_method == "Sum":
                    grouped_df = df.groupby(selected_columns)[agg_column].sum().reset_index()
                elif aggregation_method == "Min":
                    grouped_df = df.groupby(selected_columns)[agg_column].min().reset_index()
                elif aggregation_method == "Max":
                    grouped_df = df.groupby(selected_columns)[agg_column].max().reset_index()
                elif aggregation_method == "Mean":
                    grouped_df = df.groupby(selected_columns)[agg_column].mean().reset_index()
                elif aggregation_method == "Median":
                    grouped_df = df.groupby(selected_columns)[agg_column].median().reset_index()
                elif aggregation_method == "Std Dev":
                    # params = st.select
                    grouped_df = df.groupby(selected_columns)[agg_column].std().reset_index()
                elif aggregation_method == "Var":
                    grouped_df = df.groupby(selected_columns)[agg_column].var().reset_index()
                elif aggregation_method == "Mean absolute dev":
                    grouped_df = df.groupby(selected_columns)[agg_column].median().reset_index()
                elif aggregation_method == "Product":
                    grouped_df = df.groupby(selected_columns)[agg_column].prod().reset_index()
        

        st.data_editor(
            grouped_df,
            column_config={
                agg_column: st.column_config.ProgressColumn(
                    f"{agg_column} ({aggregation_method})",
                    format="%.2f",
                    min_value=float(grouped_df[agg_column].min()),
                    max_value=float(grouped_df[agg_column].max()),
                    
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        styled_df = grouped_df.style.bar(subset=[agg_column], color='#ff4b4b')
        
        st.info("""
                **The Explainer:**

                The Explainer is the intelligence core of the system. Beyond basic statistics and visualizations, it leverages AI to 
                interpret the dataset and generate human-like explanations. It automatically identifies correlations, trends, and anomalies, 
                then communicates them in plain, professional language..

                """)

            # st.success("""
            #     **Key Value:**
                
            #     The Explainer bridges the gap between data and decisions. It allows non-technical users to grasp insights quickly while 
            #     giving analysts and policymakers deeper context for evidence-based strategies. Instead of simply showing what the data is, 
            #     The Explainer communicates what the data means.

            #     """)

        col1, col2 = st.columns(2)

        with col1:
            # User selects type of plot
            plot_type = st.selectbox("Select plot type:", ["Bar", "Line", "Pie"])
        

        with col2:
            # User selects visualization library
            library_choice = st.selectbox("Select visualization library:", ["Matplotlib", "Seaborn"])
        
        # Generate plot
        fig, ax = plt.subplots()


        col_chart, col_exp = st.columns(2)

        with col_chart:

            ent, srt = st.columns(2)
            
            with ent:
                grp_entries = st.pills('select entries', options=entry_options, key='grp_entries')
                grouped_df = grouped_df.head(grp_entries)

            with srt:
                sorting_data = ['a-z', 'z-a']
                sort_data = st.pills('sort data', options=sorting_data, key='sort_data')
                
                if sort_data == 'a-z':
                    grouped_df = grouped_df.head(grp_entries).sort_values(by=agg_column, ascending=True)
                elif sort_data == 'z-a': 
                    grouped_df = grouped_df.head(grp_entries).sort_values(by=agg_column, ascending=False)

            
        
            if plot_type == "Bar":
                if library_choice == "Matplotlib":
                    plt.figure(figsize=(15, 10))
                    plt.title(f'Bar Graph for the {selected_columns} grouped into {agg_column}')
                    ax.barh(grouped_df[selected_columns[0]], grouped_df[agg_column])
                else:
                    plt.figure(figsize=(18, 15))
                    sb.barplot(data=grouped_df.head(grp_entries), x=selected_columns[0], y=agg_column, ax=ax)
                    plt.xticks(rotation=-90)
            elif plot_type == "Line":
                if library_choice == "Matplotlib":
                    ax.plot(grouped_df[selected_columns[0]], grouped_df[agg_column])
                else:
                    sb.lineplot(data=grouped_df, x=selected_columns[0], y=agg_column, ax=ax)
                    
            elif plot_type == "Pie":
                ax.pie(grouped_df[agg_column], labels=grouped_df[selected_columns[0]], autopct='%1.1f%%')
            
            st.pyplot(fig)

       

        with col_exp:

            st.markdown("""
                <style>
                .scroll-box {
                    max-height: 450px;
                    overflow-y: scroll;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    
                }
                </style>
            """, unsafe_allow_html=True)

            prompt = (
                f"You have selected the '{selected_columns}' and are analyzing the '{aggregation_method}' aggregation method. "
                f"This graph provides insights into how '{grouped_df[agg_column]}' varies within '{selected_columns}'.\n\n"
                f"Key Insights:\n"
            )

            # st.markdown(f'<div class="scroll-box">{the_explainer(prompt)}</div>', unsafe_allow_html=True)


            st.success(f'{the_explainer(prompt)}')
            



    selected_columns = st.multiselect("Select columns for EDA:", df.columns)

    if selected_columns:
        # Generate insights using Gemini LLM
        with st.spinner("Generating insights..."):
            insights = get_gemini_insights(df, selected_columns)
        
        
        st.markdown(insights)

        # Generate plots
        for col in selected_columns:
            plt.figure(figsize=(6, 8))
            fig, ax = plt.subplots()
            sb.histplot(df[col], kde=True, ax=ax)
            plt.title(f"Distribution of {col}")
            st.pyplot(fig)

        # Generate and provide PDF download
        pdf_output = generate_pdf(df, selected_columns, insights)
        st.download_button(label="Download EDA Report as PDF", data=pdf_output, file_name="EDA_Report.pdf", mime="application/pdf", type="primary")

