import sqlite3
import pandas as pd
import streamlit as st
import nltk
import random
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Page Configuration
st.set_page_config(
    page_title="Customer Satisfaction CRM Classifier",
    page_icon="📊",
    layout="wide"
)

DB_NAME = "crm_satisfaction.db"

# ---------------------------------------------------------
# 1. DATABASE SETUP & 100 SURVEY REVIEWS SEEDING
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            customer_name TEXT,
            tier TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS surveys (
            survey_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            comment TEXT,
            sentiment_score REAL,
            classification TEXT,
            date TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
        )
    ''')
    conn.commit()
    
    # Check if surveys already populated
    cursor.execute("SELECT COUNT(*) FROM surveys")
    count = cursor.fetchone()[0]
    
    if count == 0:
        seed_data(conn)
    
    conn.close()

def seed_data(conn):
    cursor = conn.cursor()
    
    names = [
        "Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince", "Evan Wright",
        "Fiona Gallagher", "George Clark", "Hannah Abbott", "Ian Malcolm", "Julia Roberts",
        "Kevin Hart", "Laura Croft", "Michael Scott", "Nancy Wheeler", "Oscar Martinez",
        "Pam Beesly", "Quentin Coldwater", "Rachel Green", "Steve Rogers", "Tina Belcher",
        "Ulysses Klaue", "Victor Stone", "Wanda Maximoff", "Xander Cage", "Yara Greyjoy",
        "Zack Snyder", "Aaron Burr", "Bella Swan", "Carl Grimes", "David Rose",
        "El Eleven", "Finn Hudson", "Geller Ross", "Harley Quinn", "Isabelle Lightwood",
        "Jack Sparrow", "Katniss Everdeen", "Logan Paul", "Mila Kunis", "Newt Scamander",
        "Owen Grady", "Phoebe Buffay", "Queenie Goldstein", "Rick Grimes", "Saul Goodman",
        "Tony Stark", "Ursula Buffay", "Vito Corleone", "Walter White", "Xena Warrior"
    ] * 2  # 100 people total
    
    tiers = ["Enterprise", "Pro", "Standard", "Free"]
    
    sample_comments = [
        ("Absolute game changer! Outstanding support and blazing fast software.", "Satisfied"),
        ("I am thoroughly disappointed. The application crashes constantly.", "Dissatisfied"),
        ("It's okay, nothing special. Does the job but lacks advanced reporting.", "Neutral"),
        ("Extremely happy with the recent updates. Keep up the great work!", "Satisfied"),
        ("Terrible customer service. Waited 3 days for a simple bug fix response.", "Dissatisfied"),
        ("Average product. Pricing is a bit high for the features provided.", "Neutral"),
        ("Phenomenal experience! Seamless onboarding and wonderful interface.", "Satisfied"),
        ("Worse tool I have ever used. Total waste of money and time.", "Dissatisfied"),
        ("It meets my basic requirements, but the UI could be much cleaner.", "Neutral"),
        ("Exceeded all my expectations! Highly recommend to everyone.", "Satisfied")
    ]
    
    sia = SentimentIntensityAnalyzer()
    
    for i in range(1, 101):
        cust_id = f"CUST-{1000 + i}"
        cust_name = names[i - 1]
        tier = random.choice(tiers)
        
        cursor.execute("INSERT OR IGNORE INTO customers VALUES (?, ?, ?)", (cust_id, cust_name, tier))
        
        base_text, expected_type = random.choice(sample_comments)
        comment = f"{base_text} (Ref #{i})"
        
        score = sia.polarity_scores(comment)['compound']
        if score >= 0.05:
            classification = "Satisfied"
        elif score <= -0.05:
            classification = "Dissatisfied"
        else:
            classification = "Neutral"
            
        date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        
        cursor.execute('''
            INSERT INTO surveys (customer_id, comment, sentiment_score, classification, date)
            VALUES (?, ?, ?, ?, ?)
        ''', (cust_id, comment, score, classification, date))
        
    conn.commit()

init_db()

# ---------------------------------------------------------
# 2. STREAMLIT USER INTERFACE
# ---------------------------------------------------------
st.title("📊 Customer Satisfaction CRM Dashboard")
st.markdown("NLP-powered system tracking survey comments, classifications, and customer profiles.")

# Load Data from SQLite
conn = sqlite3.connect(DB_NAME)
df = pd.read_sql('''
    SELECT s.survey_id, c.customer_id, c.customer_name, c.tier, 
           s.comment, s.sentiment_score, s.classification, s.date
    FROM surveys s
    JOIN customers c ON s.customer_id = c.customer_id
''', conn)
conn.close()

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
selected_tier = st.sidebar.multiselect("Select Customer Tier", options=df['tier'].unique(), default=df['tier'].unique())
selected_class = st.sidebar.multiselect("Satisfaction Filter", options=["Satisfied", "Neutral", "Dissatisfied"], default=["Satisfied", "Neutral", "Dissatisfied"])

filtered_df = df[df['tier'].isin(selected_tier) & df['classification'].isin(selected_class)]

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
total_surveys = len(filtered_df)
satisfied_count = len(filtered_df[filtered_df['classification'] == 'Satisfied'])
neutral_count = len(filtered_df[filtered_df['classification'] == 'Neutral'])
dissatisfied_count = len(filtered_df[filtered_df['classification'] == 'Dissatisfied'])

sat_percentage = (satisfied_count / total_surveys * 100) if total_surveys > 0 else 0

col1.metric("Total Surveys Analyzed", total_surveys)
col2.metric("Satisfied Rate", f"{sat_percentage:.1f}%")
col3.metric("Neutral Feedback", neutral_count)
col4.metric("Dissatisfied (At Risk)", dissatisfied_count, delta_color="inverse")

st.markdown("---")

# Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview Metrics", "📝 Survey Database", "👤 Customer Profiles", "➕ Add New Feedback"])

with tab1:
    st.subheader("Satisfaction Distribution")
    col_a, col_b = st.columns(2)
    
    with col_a:
        class_counts = filtered_df['classification'].value_counts()
        st.bar_chart(class_counts)
        
    with col_b:
        st.markdown("### Breakdown by Subscription Tier")
        tier_breakdown = pd.crosstab(filtered_df['tier'], filtered_df['classification'])
        st.dataframe(tier_breakdown, use_container_width=True)

with tab2:
    st.subheader("Filtered Survey Comments Record")
    st.dataframe(filtered_df[['survey_id', 'customer_name', 'tier', 'comment', 'sentiment_score', 'classification', 'date']], use_container_width=True)

with tab3:
    st.subheader("Individual Customer Profile Explorer")
    selected_customer = st.selectbox("Select Customer Name", options=df['customer_name'].unique())
    
    cust_data = df[df['customer_name'] == selected_customer].iloc[0]
    cust_surveys = df[df['customer_name'] == selected_customer]
    
    st.info(f"**Customer ID:** {cust_data['customer_id']} | **Tier:** {cust_data['tier']}")
    st.write("### Survey History for User:")
    st.dataframe(cust_surveys[['survey_id', 'comment', 'sentiment_score', 'classification', 'date']], use_container_width=True)

with tab4:
    st.subheader("Submit New Live Survey Comment")
    with st.form("feedback_form"):
        new_name = st.text_input("Customer Name")
        new_tier = st.selectbox("Customer Tier", ["Enterprise", "Pro", "Standard", "Free"])
        new_comment = st.text_area("Survey Comment")
        submitted = st.form_submit_button("Classify & Save")
        
        if submitted and new_name and new_comment:
            sia = SentimentIntensityAnalyzer()
            score = sia.polarity_scores(new_comment)['compound']
            if score >= 0.05:
                cl = "Satisfied"
            elif score <= -0.05:
                cl = "Dissatisfied"
            else:
                cl = "Neutral"
                
            c_id = f"CUST-{random.randint(5000, 9999)}"
            cur_date = datetime.now().strftime("%Y-%m-%d")
            
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("INSERT OR IGNORE INTO customers VALUES (?, ?, ?)", (c_id, new_name, new_tier))
            cur.execute("INSERT INTO surveys (customer_id, comment, sentiment_score, classification, date) VALUES (?, ?, ?, ?, ?)",
                        (c_id, new_comment, score, cl, cur_date))
            conn.commit()
            conn.close()
            
            st.success(f"Successfully processed! Classification Result: **{cl}** (Compound Score: {score})")
            st.rerun()
