# XpertBot_advanced_data_science_Internship_tasks

## Chatbot Analytics Dashboard - Internship Project
## 📋 Project Overview
This project was completed during my internship at Abidjan.ai, a company specializing in AI-powered chatbot solutions. The main objective was to develop a comprehensive analytics dashboard to analyze chatbot performance, user engagement, and conversation patterns across multiple client deployments.

## 🎯 Project Objectives
The project was structured into five phases to systematically analyze and visualize chatbot data:

# PHASE 1: DATABASE UNDERSTANDING & EXPLORATION
1- Restore and explore the SQL dump - Import into PostgreSQL/MySQL/SQLite and generate an ERD  

please check: taks1_ERD diagram.pdf 

Key Tables (based on relationships):

bots - Central table containing bot configurations and metadata

chats - Stores conversation data between users and bots

clients - User/client information

action_events - Tracks user interactions and bot responses 

Various supporting tables for categories, menus, orders, etc.



2- Data profiling per table - Count records, check nulls, data types, distributions

3- Document the schema - Describe tables, keys, and relationships

# 🔧 Technical Implementation
Data Cleaning & Preparation
Used regex expressions to filter and clean data rows for example:

-- I used here regular expressions to clean the names and keep only the real names.
-- DELETE 
-- FROM clients
-- WHERE client_id > 0
--   AND name NOT REGEXP '^[A-Za-z]+([ -][A-Za-z]+)*$';


Implemented SQL queries for data validation and transformation

Handled null values and inconsistent data formats (I dropped many empty unsufull tables which We cannot get insights from such as: Posts and Posts_photo they were both empty)

Performed comprehensive data profiling for each table

# PHASE 2: DATA ANALYSIS (this phase is done 2 ways first one using SQL in a document called: task 4,5,6,7 queries second way using pandas as we can see in the first page of the dashboard)
Conversation volume analysis - Messages per day/week/topic/user
<img width="1578" height="711" alt="ChatVolumeTrends" src="https://github.com/user-attachments/assets/aa0e4b61-c758-4fe2-bd71-a71a716caa2a" />

User engagement analysis - Most active users, avg session length, drop-off rates

Topic performance - Most frequently used topics and satisfaction scores

Response efficiency - Average response time, number of turns per conversation

## Analysis Methods
SQL Implementation: All analysis queries (tasks 4-7) were first implemented in SQL for database-level processing

Python/Pandas Implementation: Recreated the same analyses using pandas for the dashboard application, including:

Time series analysis for chat volume trends

User engagement metrics calculation

Topic classification and performance analysis

Response time efficiency measurements



# PHASE 3: ADVANCED INSIGHTS & MODELING
Sentiment analysis of messages

Topic clustering using NLP to detect latent topics

Intent recognition improvement

User segmentation by usage behavior

# PHASE 4: VISUALIZATION & REPORTING
Create Streamlit dashboard with chat volume trends

# PHASE 5: PREDICTIVE/RECOMMENDATION FEATURES
Predict future chat volume using time series forecasting

Recommend new chatbot topics based on user query clusters

Build feedback classifier for user satisfaction

## 🗄️ Database Structure
The database centers around the bots table, which connects to most other tables, reflecting the chatbot-focused nature of the application:





# 🚀 Key Findings
Data Insights:
450 action events with specific null value patterns across columns

1200 chat interactions focused primarily on flight information for Beirut Airport

Multilingual support (English, French, Arabic) evident in conversation data

6 active users with diverse engagement patterns

10 configured bots with varying capabilities and integrations

Performance Metrics:
Calculated average session lengths and drop-off rates

Identified most frequently discussed topics

Measured bot response efficiency

Analyzed user satisfaction through conversation patterns

# 🛠️ Technologies Used
Database:MySQL

Backend: Python, Pandas, NumPy

ML/NLP: Scikit-learn, NLTK, Sentence Transformers

Visualization: Streamlit, Plotly

Analysis: SQL, Regex data cleaning

# 📈 Business Impact
This dashboard enables Abidjan.ai to:

Monitor chatbot performance in real-time

Identify popular topics and user needs

Optimize response strategies based on engagement patterns

Predict future chat volumes for resource planning

Improve user satisfaction through data-driven insights

# 🎓 Learning Outcomes
Through this project, I gained hands-on experience with:

Database design and ERD creation

Large-scale data analysis and cleaning

SQL and pandas for data manipulation

Machine learning applications for NLP tasks

Dashboard development and data visualization

End-to-end data analytics project lifecycle

This project demonstrates the application of data science techniques to solve real-world business problems in the chatbot industry, providing actionable insights for performance optimization and user experience improvement.

# to run this project just uploe the data to the database change connect the database in file called db_ustils to you database and run the app.py using streamlit run app.py
