# XpertBot_advanced_data_science_Internship_tasks

## Chatbot Analytics Dashboard - Internship Project
## 📋 Project Overview
This project was completed during my internship at XpertBot, in collaboration with Abidjan.ai, a company specializing in AI-powered chatbot solutions. The main objective was to develop a comprehensive analytics dashboard to evaluate chatbot performance, user engagement, and conversation patterns across multiple client deployments.

## 🎯 Project Objectives
The project was structured into five phases to systematically analyze and visualize chatbot data:

# PHASE 1: DATABASE UNDERSTANDING & EXPLORATION
1- Restore and explore the SQL dump - Import into MySQL and generate an ERD  

please check: task1_ERD diagram.pdf 

Key Tables (based on relationships):

bots - Central table containing bot configurations and metadata

chats - Stores conversation data between users and bots

clients - User/client information

action_events - Tracks user interactions and bot responses 

Various supporting tables for categories, menus, orders, etc.



2- Data profiling per table - Count records, check nulls, data types, distributions

please check: task2.docx

3- Document the schema - Describe tables, keys, and relationships

please check: task3.docx
## 🔧 Technical Implementation
Data Cleaning & Preparation
Used regex expressions to filter and clean data rows for example:

DELETE FROM clients WHERE client_id > 0 AND name NOT REGEXP '^[A-Za-z]+([ -][A-Za-z]+)*$';

Implemented SQL queries for data validation and transformation

Handled null values and inconsistent data formats (several empty tables were dropped, non-useful tables that provided no insights, such as Posts and Posts_photo, since both were empty.)

Performed comprehensive data profiling for each table

# PHASE 2: DATA ANALYSIS

## This phase was implemented in two ways: first, using SQL (document titled Task 4, 5, 6, 7 Queries), and second, using Python/Pandas, as shown on the first page of the dashboard.

Conversation volume analysis: Messages per day/week/topic/user


<img width="1578" height="711" alt="ChatVolumeTrends" src="https://github.com/user-attachments/assets/aa0e4b61-c758-4fe2-bd71-a71a716caa2a" />

User engagement analysis: Most active users, avg session length, drop-off rates 


<img width="1581" height="626" alt="User Engagement" src="https://github.com/user-attachments/assets/fd7da664-8d7d-4928-b528-feada63d0bdf" />

Note: A drop-off rate of 0 indicates that the chatbot is working effectively.

Topic performance: Most frequently used topics and satisfaction scores

<img width="1591" height="631" alt="Topic Performance" src="https://github.com/user-attachments/assets/3dc56e6e-9a97-431b-a8a9-1043e765a789" />

Response efficiency: Average response time, number of turns per conversation

<img width="1212" height="620" alt="Response Efficiency" src="https://github.com/user-attachments/assets/ff9e536c-3ea4-49b3-9f0f-33655b20ff2a" />

Note: We can conclude that there is a significant delay, which in this case may indicate an error in the provided data, as the response time was unusually large.



# PHASE 3: ADVANCED INSIGHTS & MODELING
## This phase leveraged machine learning and natural language processing to extract deeper, more sophisticated insights from the conversation data.

8. Sentiment Analysis of Messages
To gauge user satisfaction and emotional tone, we implemented VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis.

Process: Each user question and bot answer was scored on a polarity scale from -1 (negative) to +1 (positive) and categorized into negative, neutral, or positive segments.

Value: This analysis provides a direct measure of user sentiment throughout interactions, helping to identify frustrating experiences, positive engagements, and the emotional impact of the bot's responses.

9. Topic Clustering using NLP
We moved beyond predefined categories to discover latent, emerging themes in user conversations using unsupervised learning.

Process: Employed TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization, followed by K-Means clustering to group semantically similar questions. PCA was used for dimensionality reduction to visualize the clusters in a 2D space.

Value: This technique uncovered hidden topics and user needs that were not initially anticipated, allowing for proactive improvements to the chatbot's knowledge base and flow design.

10. Intent Recognition Improvement
To enhance the chatbot's ability to understand user goals, we built a hybrid intent recognition system.

Process: Combined a rule-based matcher (using keyword patterns for intents like booking, pricing, hours) with a machine learning classifier (Random Forest). The model was trained on labeled data to accurately predict the intent behind new, unseen queries.

Value: This results in a more robust and accurate intent detection system, reducing misinterpretations and ensuring users are routed to the correct information faster.

11. User Segmentation by Usage Behavior
We segmented users into distinct behavioral profiles to enable personalized engagement strategies.

Process: Users were clustered based on key behavioral features such as message frequency, session length, response time, and topic diversity using K-Means clustering.

Value: This allows for the identification of different user archetypes (e.g., Power Users, Casual Browsers, At-Risk Users). This segmentation enables targeted interventions, such as creating specialized flows for highly engaged users or identifying and assisting users who show signs of frustration.


<img width="1594" height="729" alt="Sentiment Analysis" src="https://github.com/user-attachments/assets/f2088caf-2394-4c39-a656-5f5e41612de7" />

# PHASE 4: VISUALIZATION & REPORTING
Create Streamlit dashboard with chat volume trends

# PHASE 5: PREDICTIVE/RECOMMENDATION FEATURES
Predict future chat volume using time series forecasting

Recommend new chatbot topics based on user query clusters


<img width="1577" height="727" alt="ML" src="https://github.com/user-attachments/assets/5a2b4257-c959-4c25-a325-2d70fb47c5f0" />



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

# To run this project, upload the data to your database, update the connection settings in the db_utils file, and then launch the app with the command: streamlit run app.py
