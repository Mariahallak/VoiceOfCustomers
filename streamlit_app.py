import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Sets the page to wide layout.
st.set_page_config(layout="wide")

# Sidebar for dashboard selection
view = st.sidebar.radio("Select Dashboard", ('Summary Dashboard', 
                                             'Section 1: Employee Experience',
                                             'Section 2: Recruiting & Onboarding',
                                             'Section 3: Performance & Talent',
                                             'Section 4: Learning',
                                             'Section 5: Compensation',
                                             'Section 6: Payroll',
                                             'Section 7: Time Management',
                                             'Section 8: User Experience',
                                             'Chatbot'
                                             ))

# Apply filters and display sections
if view == 'Global Overview':
    st.markdown("""
    <style>
        h1 {
            text-align: center;
            color: black;
            margin-bottom: 0px;  /* Remove bottom margin */
        }
        h3 {
            text-align: center;
            color: black;
            margin-top: 0.2px;  /* Remove top margin */
        }
    </style>
    <h1>Comprehensive Exploration Of Global Health and Demographic Dataset</h1>
    <h3>A deep dive into the complex interplay between health expenditures, immunization rates, socioeconomic factors, and their collective influence on life expectancy across the globe.</h3>
    """, unsafe_allow_html=True)


# Load and clean data
@st.cache_data
def load_data():
    data = pd.read_csv('/Users/mariahallak/Desktop/M2-T2/DataViz/Final Project/Life_Expectancy_with_Continent.csv')
    data['Year'] = data['Year'].astype(int)
    data.columns = data.columns.str.strip()  # Remove any trailing spaces in column names
    # Filter data to only include years from 2004 to 2014
    data = data[(data['Year'] >= 2004) & (data['Year'] <= 2014)]
    continent_map = {
        'Russian Federation': 'Europe',
        'Swaziland': 'Africa',
        'Syrian Arab Republic': 'Asia',
        'Turkey': 'Asia'
    }
    data['Continent'] = data['Country'].map(continent_map).fillna(data['Continent'])
    data['GDP per Capita'] = data['GDP'] / data['Population']
    #data['% Health Expenditure'] = (data['percentage expenditure'] / data['GDP']) * 100
    data['Health Expenditure (% of GDP)'] = data['Total expenditure']
    return data

data = load_data()

# Custom CSS for boxes with black background and white font
st.markdown(
    """
    <style>
    .box {
        background-color: #000000; /* Black background */
        color: #ffffff; /* White text */
        border-radius: 7px;
        padding: 10px;
        margin: 10px 0px;
        text-align: center; /* Center text */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


#Geolocation Data for Implementing the World Map
@st.cache_data
def load_geolocation_data():
    geo_data_url = 'https://raw.githubusercontent.com/IreneSunny96/Global-Health-and-Demographic-Dataset-Dashboard/main/data/Country_Continent_Mapping.csv'
    geo_data = pd.read_csv(geo_data_url)
    return geo_data

# Load geolocation data
geo_data = load_geolocation_data()

# Calculating averages for each country based on the selected years
def calculate_country_averages(data, year_range):
    #filtered_data = data[(data['Year'] >= max(2004, year_range[0])) & (data['Year'] <= min(2014, year_range[1]))]
    averages = data.groupby(['Country','Status']).agg({
        'Life expectancy': 'mean',
        'Population': 'mean',
        'GDP': 'mean',  # Ensure 'GDP' is aggregated if not done before
        'percentage expenditure': 'mean',
        'Total expenditure' : 'mean'
    }).reset_index()

    # Convert population from total to millions for better readability
    averages['Population'] = (averages['Population'] / 1e6).round(2)
    
    # Calculate GDP per Capita and round to two decimals
    averages['GDP per Capita'] = averages['GDP'].round(2)

    # If Total expenditure is already in percentage of GDP, rename or convert as required
    averages['Health Expenditure (% of GDP)'] = averages['Total expenditure'].round(2)

    # Rename columns for clarity in the map
    #averages.rename(columns={'percentage expenditure': '% Health Expenditure'}, inplace=True)
    return averages

# Function to create choropleth map
def create_choropleth(data, geo_data):
    # Merge the averaged data with geolocation data
    map_data = pd.merge(data, geo_data, how="left", left_on='Country', right_on='Country')
    
    # Round the columns to two decimal points before plotting
    map_data['Life expectancy'] = map_data['Life expectancy'].round(2)
    map_data['Population'] = map_data['Population'].round(2).astype(str) + ' Million'
    map_data['Health Expenditure (% of GDP)'] = map_data['Health Expenditure (% of GDP)'].round(2).astype(str) + '%'
    map_data['GDP per Capita'] = '$' + map_data['GDP per Capita'].round(2).astype(str)

    fig = px.choropleth(map_data,
                        locations="Country",  # DataFrame column with locations
                        color="Life expectancy",  # DataFrame column with color values
                        hover_name="Country",  # DataFrame column hover info
                        hover_data=["Life expectancy", "Population", "Health Expenditure (% of GDP)", "GDP per Capita"],
                        locationmode='country names',  # Set to use country names
                        color_continuous_scale=px.colors.diverging.RdYlGn,  # Color scale
                        title='Life Expectancy Distribution Among Countries',  # Title
                        projection="equirectangular",  # Use flat map projection
                        )
    fig.update_geos(showcoastlines=True, coastlinecolor="Black",
                    showland=True, landcolor="White",
                    showocean=True, oceancolor="LightBlue",
                    showlakes=True, lakecolor="LightBlue",
                    showcountries=True, countrycolor="Black")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})  # Remove the padding around the map
    fig.update_layout(coloraxis_colorbar=dict(
        title="Life Expectancy",
        tickvals=[50, 60, 70, 80, 90],
        ticktext=["50 years", "60 years", "70 years", "80 years", "90 years"],
        lenmode="pixels", len=200,
    ))
    return fig


# Function to create a box plot below the Global Map with customizations
def create_box_plot(data):
    data['Life expectancy'] = data['Life expectancy'].map(lambda x: f"{x:.2f}")
    # Create the box plot
    fig = px.box(data, x='Life expectancy', 
                 points="all",  # Display all points
                 labels={'Life expectancy': 'Average Life Expectancy (years)'},
                 hover_data=['Country', 'Status'],  # Show country name on hover
                 color = 'Status',
                )

    # Update layout to remove legend (if not needed) and refine other aesthetic elements
    #fig.update_layout(margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
     # Adjusting the layout
    fig.update_layout(
        height=300,  # Reducing the height
        margin=dict(l=0, r=0, t=25, b=0),  # Reducing the top margin to bring plot closer to title
        plot_bgcolor="rgba(0,0,0,0)"  # Transparent background
    )
    fig.update_traces(
        fillcolor='rgba(0,0,0,0)',  # Transparent fill
        #line=dict(width=0),  # No line around the box
        #marker=dict(line=dict(width=0))  # No line around the markers
    )
    return fig


#Line Charts
# Function for Life Expectancy vs. GDP Line Chart
def plot_life_expectancy_vs_gdp(filtered_data, year_range):
    #filtered_data = data[(data['Year'] >= year_range[0]) & (data['Year'] <= year_range[1])]
    grouped_data = filtered_data.groupby('Year').agg({
        'Life expectancy': 'mean',
        'GDP': 'mean'
    }).reset_index()

    # Round off the values to two decimals
    grouped_data['Life expectancy'] = grouped_data['Life expectancy'].round(2)
    grouped_data['GDP'] = grouped_data['GDP'].round(2)

    fig = px.line(grouped_data, x='Year', y='Life expectancy', title='Life Expectancy vs GDP per Capita ($)')
    fig.add_scatter(x=grouped_data['Year'], y=grouped_data['GDP'], mode='lines', name='GDP per Capita', yaxis='y2')
    fig.update_layout(yaxis={'title': 'Avg Life Expectancy (years)'}, yaxis2={'title': 'Avg GDP Per Capita ($)', 'overlaying': 'y', 'side': 'right'},
                      legend=dict(
                      orientation="h",
                      yanchor="bottom",
                      y=-0.5,  # Pushing the legend further down
                      xanchor="center",
                      x=0.5
        ),
                      margin=dict(t=60),  # Increase top margin
                      height=400  # Increase height
                      )
    return fig

# Function for Life Expectancy vs. Health Expenditure
def plot_life_expectancy_vs_health_expenditure(filtered_data, year_range):
    # Filter data based on year range
    #filtered_data = data[(data['Year'] >= year_range[0]) & (data['Year'] <= year_range[1])]
    # Group by year and calculate averages
    grouped_data = filtered_data.groupby('Year').agg({
        'Life expectancy': 'mean',
        'Health Expenditure (% of GDP)': 'mean'
    }).reset_index()

    # Round off the values to two decimals
    grouped_data['Life expectancy'] = grouped_data['Life expectancy'].round(2)
    grouped_data['Health Expenditure (% of GDP)'] = grouped_data['Health Expenditure (% of GDP)'].round(2)

    # Create the plot using Plotly
    fig = px.line(grouped_data, x='Year', y='Life expectancy', title='Life Expectancy vs % Health Expenditure')
    fig.add_scatter(x=grouped_data['Year'], y=grouped_data['Health Expenditure (% of GDP)'], mode='lines', name='Health Expenditure (% of GDP)', yaxis='y2')
    # Customizing the layout
    fig.update_layout(
        yaxis={'title': 'Avg Life Expectancy (years)'},
        yaxis2={'title': 'Health Expenditure (% of GDP)', 'overlaying': 'y', 'side': 'right'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,  # Pushing the legend further down
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=60),  # Increase top margin
        height=400  # Increase heighte
    )
    return fig



# Function to calculate statistics and create a table
def create_life_expectancy_table(filtered_data, year_range):
    # Filter data based on year range and calculate averages
    grouped_data = filtered_data.groupby(['Country', 'Continent', 'Status']).agg({
        'Population': 'mean',
        'Life expectancy': 'mean',
        'Adult Mortality': 'mean',
        'infant deaths': 'mean',
        'under-five deaths': 'mean',
        'HIV/AIDS': 'mean',
    }).reset_index()

    # Convert the population from total to millions and ensure it's rounded to two decimal places
    grouped_data['Population'] = (grouped_data['Population'] / 1e6)

    # Round other columns to two decimal places
    for col in ['Population','Life expectancy', 'Adult Mortality', 'infant deaths', 'under-five deaths', 'HIV/AIDS']:
        grouped_data[col] = grouped_data[col].round(2)

    # Rename columns to include units and descriptions
    grouped_data.rename(columns={
        'Country': 'Country',
        'Status': 'Status',
        'Continent': 'Continent',
        'Population': 'Population (Million)',
        'Life expectancy': 'Life Expectancy (Years)',
        'Adult Mortality': 'Adult Mortality (probability of death b/w ages 15 and 60 per 1,000 people)',
        'infant deaths': 'Infant Deaths (per 1,000 live births)',
        'under-five deaths': 'Under Age Five Deaths (per 1,000 live births)',
        'HIV/AIDS': 'HIV/AIDS (% of population)'
    }, inplace=True)

    # Sort the data based on life expectancy and select the top and bottom countries
    top_countries = grouped_data.nlargest(5, 'Life Expectancy (Years)')
    bottom_countries = grouped_data.nsmallest(5, 'Life Expectancy (Years)')
    
    # Combine the top and bottom countries into one DataFrame
    combined_data = pd.concat([top_countries, bottom_countries]).sort_values('Life Expectancy (Years)', ascending=False)

    # Ensure combined_data has a unique index before styling
    combined_data = combined_data.reset_index(drop=True)

    # Apply coloring to the country names based on their life expectancy ranking
    def apply_color(val):
        color = 'green' if val in top_countries['Country'].values else 'red' if val in bottom_countries['Country'].values else 'black'
        return f'color: {color}'

    # Style the table and hide the index
    styled_table = (combined_data.style
                    .applymap(apply_color, subset=['Country'])
                    .format('{:.2f}', subset=['Population (Million)', 
                                              'Life Expectancy (Years)', 
                                              'Adult Mortality (probability of death b/w ages 15 and 60 per 1,000 people)',
                                              'Infant Deaths (per 1,000 live births)', 
                                              'Under Age Five Deaths (per 1,000 live births)',
                                              'HIV/AIDS (% of population)'])
                    .set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center')]},
                        {'selector': '.row_heading, .blank', 'props': [('display', 'none')]},  # Hide the row headers and blank headers
                        {'selector': '.index_name', 'props': [('display', 'none')]},  # Hide index name
                        ])
                        #.hide_index()  # Hide the index
                        #{'selector': '.row0', 'props': [('display', 'none')]}  # Attempt to hide the first row if needed
                    )

    st.write(styled_table.to_html(index=False), unsafe_allow_html=True)

#VIEW 2
def plot_correlation_matrix(filtered_data_2, year_range, status):
    # Filter the data based on the year range and the status
    filtered_data_2 = filtered_data_2[(filtered_data_2['Year'] >= year_range[0]) & (filtered_data_2['Year'] <= year_range[1]) & (filtered_data_2['Status'] == status)]
    
    # Calculate the mean values for the required metrics for each country
    metrics = ['BMI', 'Alcohol', 'Hepatitis B', 'Polio', 'Diphtheria', 'thinness  1-19 years', 'Life expectancy']
    country_metrics_mean = filtered_data_2.groupby('Country')[metrics].mean()
    
    # Compute the correlation matrix
    corr_matrix = country_metrics_mean.corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # Return the figure
    return fig

#SCATTER PLOT
# Function to calculate the averages for the scatter plot
def calculate_averages_for_scatter(filtered_data_2, year_range, factor):
       # Filter the data based on the year range
    #filtered_data = data[(data['Year'] >= year_range[0]) & (data['Year'] <= year_range[1])]
    factor_column = factor

    # Calculate the mean values for life expectancy and the selected factor for each country
    country_averages = filtered_data_2.groupby(['Country', 'Status']).agg({
        'Life expectancy': 'mean',
        factor_column: 'mean'  # Use the adjusted column name
    }).reset_index()
    return country_averages

# Function to create a scatter plot
def create_scatter_plot(data, factor):
    fig = px.scatter(data, x=factor, y='Life expectancy', color='Status',
                     labels={'Life expectancy': 'Average Life Expectancy', factor: f'Average {factor}'},
                     title=f'Relationship Between {factor} and Life Expectancy')
    # Middle-align the title and increase font size
    fig.update_layout(
        title_x=0.08,  # This centers the title
        title_font_size=28  # Increase the font size as needed
    )

    # Update y-axis to show a wider range of values
    # Here, you can define 'range' as [min_value, max_value]
    fig.update_yaxes(range=[data['Life expectancy'].min() - 5, data['Life expectancy'].max() + 5])
    fig.update_yaxes(dtick=2)
    fig.update_layout(height=600)
    return fig


# Summary Information
    st.markdown("""
    - **Period**: 2004 to 2014 | Period with consistent data for most countries
    - **Countries**: 133 countries | 101 countries with data for all years
    - **Status**: 19 Developed vs 114 Developing
    - **Filter**: All visualisations in a view are based on the selections made in the sidebar  
    - **Reporting**: Data available at Year, Country level. Hence, metrics are averaged at required levels
    - **Raw data here**: [GitHub Repository](https://github.com/IreneSunny96/Global-Health-and-Demographic-Dataset-Dashboard/tree/main/data)
        """, unsafe_allow_html=True)

    year_range = st.sidebar.slider('Select Year Range', int(data['Year'].min()), int(data['Year'].max()), (int(data['Year'].min()), int(data['Year'].max())))
    selected_status = st.sidebar.multiselect('Select Status', options=data['Status'].unique(), default=data['Status'].unique())
    selected_continent = st.sidebar.multiselect('Select Continent', options=data['Continent'].unique(), default=data['Continent'].unique())
    filtered_data = data[(data['Year'] >= max(2004, year_range[0])) & (data['Year'] <= min(2014, year_range[1])) & (data['Status'].isin(selected_status)) & (data['Continent'].isin(selected_continent))]
    #filtered_data = data[(data['Year'] >= max(2004, year_range[0])) & (data['Year'] <= min(2014, year_range[1]))]
    
    st.markdown(
    """
    <style>
        /* Styling for the header */
        .header {
            text-align: center;
            text-decoration: underline;
            font-size: 30px;  /* Adjust size as needed */
        }
        /* Styling for the subheader */
        .subheader {
            text-align: center;
            font-size: 16px;  /* Smaller font size */
        }
    </style>
    <div class="header">A. Global Overview</div>
    <div class="subheader">Offers a visual analysis of life expectancy trends and their associated factors globally, across a spectrum of countries and years</div>
    """,
    unsafe_allow_html=True
)
    
    #Summary Statistics
    # Title
#     st.markdown(
#     """
#     <div style="text-align: center; font-size: 20px;">
#         <strong> Summary Statistics</strong>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
    st.subheader('Summary Statistics')
    # Calculate averages for the selected filters
    country_averages = calculate_country_averages(filtered_data, year_range)
    print(country_averages['Life expectancy'].isna().any())
    
    #Summary Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='box'>No of Countries<br>{filtered_data['Country'].nunique()}</div>", unsafe_allow_html=True)
    with col2:
        avg_life_expectancy = country_averages['Life expectancy'].mean()
        st.markdown(f"<div class='box'>Avg Life Expectancy<br>{avg_life_expectancy:.2f} years</div>", unsafe_allow_html=True)
    with col3:
        avg_gdp_per_capita = country_averages['GDP'].mean()
        st.markdown(f"<div class='box'>Avg GDP per Capita<br>${avg_gdp_per_capita:.2f}</div>", unsafe_allow_html=True)
    with col4:
        avg_health_expenditure = country_averages['Total expenditure'].mean()
        st.markdown(f"<div class='box'>Health Expenditure as % of GDP<br>{avg_health_expenditure:.2f}%</div>", unsafe_allow_html=True)
    
    # Global Map
    # Create and display the choropleth map
    st.subheader('Average Life Expectancy Across Countries')
    life_expectancy_map = create_choropleth(country_averages, geo_data)
    st.plotly_chart(life_expectancy_map, use_container_width=True)

    # Box Plot
    # Add a header for the box plot
    st.subheader('Distribution of Life Expectancy Across Countries')
    # Create and display the box plot below the map
    box_plot = create_box_plot(country_averages)
    st.plotly_chart(box_plot, use_container_width=True)

    #Table
    st.subheader('Best and Worst Countries by Average Life Expectancy')
    st.markdown("""
    <style>
        .small-font {
            font-size: 14px; /* Adjust the size as needed */
            text-align: left;
        }
    </style>
    <p class="small-font">Contrasting presentation of five countries with the highest and lowest life expectancies, offering insights into their health landscapes.</p>
    """, unsafe_allow_html=True)

    #create_life_expectancy_table(data, year_range)
    create_life_expectancy_table(filtered_data, year_range)

    st.subheader('Trends of Life Expectancy in Relation to Economic Prosperity and Health Investments')

    # Line Charts
    col5, col6 = st.columns([0.5, 0.5], gap="large")  # Adjust the gap as needed
    with col5:
        #st.subheader('Trends in Life Expectancy and GDP per Capita')
        fig1 = plot_life_expectancy_vs_gdp(filtered_data, year_range)
        fig1.update_layout(width=500, height=400)  # Adjust the size as needed
        st.plotly_chart(fig1, use_container_width=True)  # Set to True for responsive width
    with col6:
        #st.subheader('Trends in Life Expectancy and Health Expenditure')
        fig2 = plot_life_expectancy_vs_health_expenditure(filtered_data, year_range)
        fig2.update_layout(width=500, height=400)  # Adjust the size as needed
        st.plotly_chart(fig2, use_container_width=True)  # Set to True for responsive width
 
#View 2
elif view == 'Deep Dive : Socio Economic Influencers':
    st.markdown(
    """
    <style>
        /* Styling for the header */
        .header {
            text-align: center;
            text-decoration: underline;
            font-size: 30px;  /* Adjust size as needed */
        }
        /* Styling for the subheader */
        .subheader {
            text-align: center;
            font-size: 16px;  /* Smaller font size */
        }
    </style>
    <div class="header">B. Deep Dive : Socio Economic Influencers</div>
    <div class="subheader">Delves into the intricate relationship between health metrics and life expectancy, and how socioeconomic factors impact global health outcomes</div>
    """,
    unsafe_allow_html=True
)
        
    #st.header('Deep Dive : Socio Economic Influencers')
    # Implement additional views or visualizations
    # Custom CSS to center-align text within Streamlit's markdown
    st.markdown("""
    <style>
    .subheader-center {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Now using markdown to create a subheader with the same look
    st.markdown('<h2 class="subheader-center">Correlation Between Health Metrics and Life Expectancy</h2>', unsafe_allow_html=True)
    year_range = st.sidebar.slider('Select Year Range', int(data['Year'].min()), int(data['Year'].max()), (int(data['Year'].min()), int(data['Year'].max())))
    filtered_data_2 = data[(data['Year'] >= max(2004, year_range[0])) & (data['Year'] <= min(2014, year_range[1]))]
    # Display correlation matrices for Developed and Developing countries. 
    #Create columns for side-by-side correlation matrices
    col1, col2 = st.columns(2)
    
    # Apply custom CSS to center-align the subheaders
    st.markdown("""
    <style>
    .subheader-center {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

    with col1:
        st.markdown('<h3 class="subheader-center">Developed Countries</h3>', unsafe_allow_html=True)
        fig_developed = plot_correlation_matrix(filtered_data_2, year_range, 'Developed')
        st.pyplot(fig_developed)

    with col2:
        st.markdown('<h3 class="subheader-center">Developing Countries</h3>', unsafe_allow_html=True)
        fig_developing = plot_correlation_matrix(filtered_data_2, year_range, 'Developing')
        st.pyplot(fig_developing)

    st.markdown("""
    - **BMI**: Average Body Mass Index | **Alcohol**: Average alcohol consumption in liters per capita | **Hepatitis B**: Immunization coverage for Hepatitis B | **Polio**: Immunization coverage for Polio |  **Diphetheria**: % of population immunized for Diphtheria | **Thinness 1-19 Years**: prevalence of thinness among children and adolescents aged 1-19 in % | **Life Expectancy**: average years over the selected period 
        """, unsafe_allow_html=True)    

    # Scatter Plot
    # Radio button for factor selection
    #st.subheader('Social and Economic Factors vs Life Expectancy')
    st.markdown('<h2 class="subheader-center">Relationship between Socio Economic Factors and Life Expectancy</h2>', unsafe_allow_html=True)
    factor_options = {
    'Income Composition Of Resources (how well a country is utlising its resources, higher the better)': 'Income composition of resources',
    'Average Years of Schooling': 'Schooling',
    'Health Expenditure (% of GDP)': 'Total expenditure'
}
    factor = st.radio('Select a factor to compare with Life Expectancy', list(factor_options.keys()))
    
    # Calculate averages for the selected factor
    country_averages = calculate_averages_for_scatter(filtered_data_2, year_range, factor_options[factor])
    
    # Create and display the scatter plot
    scatter_plot = create_scatter_plot(country_averages, factor_options[factor])
    st.plotly_chart(scatter_plot, use_container_width=True)
