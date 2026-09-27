# 🛒 Amazon Customer Voice Intelligence

## 📌 CodeAlpha Internship – Task 4: Sentiment Analysis

An interactive **Sentiment Analysis and Customer Feedback Intelligence** project built using **Python, NLP, VADER, Pandas, Matplotlib, and Streamlit**.

This project analyzes the **Amazon Fine Food Reviews dataset** to understand customer sentiment, review patterns, product-related themes, customer pain points, and helpfulness signals.

---

## 🎯 Project Objective

Customer reviews contain valuable information about:

- ⭐ Product satisfaction
- 🏆 Product quality
- 📦 Delivery and purchase experience
- 💰 Price and value
- 😋 Taste and flavor
- 👥 Customer engagement

The objective of this project is to transform large volumes of customer reviews into meaningful insights that can support **business intelligence and customer-experience analysis**.

### 🔑 Key Objectives

- 🧠 Analyze customer review sentiment
- 🔄 Compare rating-based sentiment with NLP-based sentiment
- 🔍 Identify positive and negative customer themes
- 👍 Analyze review helpfulness
- ⚠️ Detect potential customer pain points
- 📊 Explore relationships between ratings, sentiment, and review characteristics
- 💻 Build an interactive Streamlit dashboard
- 💡 Generate business-oriented insights from customer feedback

---

## 📊 Dataset

The project uses the **Amazon Fine Food Reviews dataset**.

The original dataset contains:

- 📝 **568,454 reviews**
- 📋 **10 columns**
- ⭐ Customer ratings from **1 to 5**

### 📌 Important Columns

| Column | Description |
|---|---|
| `Score` | ⭐ Customer rating from 1 to 5 |
| `Summary` | 📝 Short review summary |
| `Text` | 📄 Full customer review |
| `ProductId` | 🆔 Product identifier |
| `UserId` | 👤 User identifier |
| `HelpfulnessNumerator` | 👍 Number of helpful votes |
| `HelpfulnessDenominator` | 👥 Total helpfulness votes |
| `Time` | 🕒 Review timestamp |

---

## 🧠 Sentiment Analysis

Two different approaches were used to analyze customer sentiment.

### 1️⃣ Rating-Based Sentiment

Customer ratings were converted into sentiment categories:

| ⭐ Rating | 🧠 Sentiment |
|---|---|
| 1–2 | 🔴 Negative |
| 3 | 🟡 Neutral |
| 4–5 | 🟢 Positive |

---

### 2️⃣ VADER Sentiment Analysis

The **VADER Sentiment Analyzer** was applied to the review text.

VADER generates a compound sentiment score between **-1 and +1**.

The classification rules used were:

| VADER Score | Classification |
|---|---|
| `>= 0.05` | 🟢 Positive |
| `<= -0.05` | 🔴 Negative |
| Otherwise | 🟡 Neutral |

### 🔄 Sentiment Method Comparison

The agreement between rating-based sentiment and VADER sentiment was approximately:

> **📈 79.70% Agreement**

⚠️ This represents agreement between two different labeling methods. It should **not** be interpreted as supervised machine-learning accuracy.

---

## 🔎 Key Findings

### 😊 Overall Sentiment

Based on the complete dataset:

| Sentiment | Percentage |
|---|---:|
| 🟢 Positive | **78.07%** |
| 🟡 Neutral | **7.50%** |
| 🔴 Negative | **14.43%** |

---

### ⭐ Rating Distribution

The dataset contains a large proportion of **5-star reviews**, showing that highly positive ratings are common within this review collection.

---

## 👍 Helpfulness Analysis

Review helpfulness was analyzed using:

- 👍 Helpful votes
- 👥 Total votes
- 📊 Helpfulness ratio
- 🔎 Reviews with higher engagement

Additional analysis examined how helpfulness varies across different sentiment categories.

---

## 🧩 Customer Theme Analysis

The project identifies major customer-related themes from review text.

### 🔍 Main Themes

| Theme | Focus |
|---|---|
| 😋 **Taste & Flavor** | Taste, flavor, coffee, tea, sweetness |
| 🏆 **Product Quality** | Quality, freshness, product condition |
| 💰 **Price & Value** | Price, cost, value, affordability |
| 📦 **Purchase Experience** | Delivery, shipping, packaging, orders |
| 🐾 **Pet & Food Experience** | Pets, food, treats, pet products |
| ⚡ **Convenience** | Easy, quick, simple, convenience |

---

## 📈 Theme Sentiment Gaps

The analysis identified the following sentiment differences between positive and negative theme mentions:

| 🧩 Theme | 📊 Sentiment Gap |
|---|---:|
| 🏆 Product Quality | **+14.86 pp** |
| 😋 Taste & Flavor | **+4.41 pp** |
| 🐾 Pet & Food Experience | **-1.56 pp** |
| 💰 Price & Value | **-2.30 pp** |
| ⚡ Convenience | **-2.98 pp** |
| 📦 Purchase Experience | **-9.82 pp** |

> ⚠️ These theme results are based on keyword associations and should not be interpreted as causal relationships.

---

## 💻 Interactive Streamlit Dashboard

The project includes an interactive dashboard named:

# 🛒 Amazon Customer Voice Intelligence

### ✨ Dashboard Features

- 📊 **Sentiment Overview**
- ⭐ **Rating Distribution**
- 👥 **Customer Insights**
- 🔍 **Theme Explorer**
- 📋 **Related Customer Reviews**
- 📥 **Filtered Review Download**
- 🧠 **Individual Review Analyzer**
- 😊 **Positive Signal Detection**
- 😞 **Negative Signal Detection**
- 🧩 **Customer Theme Detection**
- 💼 **Business Interpretation**

The dashboard allows users to explore customer feedback interactively rather than only viewing static analysis results.

---

## 🧠 Individual Review Analyzer

The dashboard also provides an **Individual Review Analyzer**.

Users can enter a customer review and receive an instant analysis.

### 📝 Example Review

> The product tastes great and the quality is excellent, but the delivery was late and the packaging was poor.

### 🔍 The Analyzer Identifies

- 🧠 Predicted sentiment
- 😊 Positive signals
- 😞 Negative signals
- 🧩 Customer themes
- 💼 Business interpretation

The analyzer also supports:

> 🟡 **Mixed Sentiment**

when both positive and negative signals are detected in the same review.

---

## 🔄 Project Workflow

```text
📥 Amazon Fine Food Reviews
            ↓
🧹 Data Cleaning & Preparation
            ↓
⭐ Rating-Based Sentiment
            ↓
🧠 VADER NLP Sentiment Analysis
            ↓
🔍 Theme & Keyword Analysis
            ↓
👍 Helpfulness Analysis
            ↓
📊 Customer Insights
            ↓
💻 Interactive Streamlit Dashboard
            ↓
💡 Business Insights

| Technology              | Purpose                     |
| ----------------------- | --------------------------- |
| 🐍 **Python**           | Core programming            |
| 🐼 **Pandas**           | Data manipulation           |
| 🔢 **NumPy**            | Numerical operations        |
| 📊 **Matplotlib**       | Data visualization          |
| 🎨 **Seaborn**          | Statistical visualization   |
| 🧠 **NLTK**             | Natural Language Processing |
| 💭 **VADER**            | Sentiment analysis          |
| 💻 **Streamlit**        | Interactive dashboard       |
| 📓 **Jupyter Notebook** | Data analysis               |
| 🔧 **Git**              | Version control             |
| 🌐 **GitHub**           | Project hosting             |

Amazon_Sentiment_Analysis/
│
├── 📂 app/
│   └── 🐍 app.py
│
├── 📂 data/
│   ├── 📄 review_sample.csv
│   ├── 📄 dashboard_reviews.csv
│   ├── 📄 sentiment_summary.csv
│   └── 📄 rating_summary.csv
│
├── 📂 notebooks/
│   └── 📓 sentiment_analysis.ipynb
│
├── 📄 README.md
└── 📄 requirements.txt


The notebook covers:

📥 Data loading
🧹 Data preprocessing
📊 Exploratory Data Analysis
⭐ Rating analysis
🧠 VADER sentiment analysis
🔍 Theme analysis
👍 Helpfulness analysis
📈 Review-length analysis
🔄 Sentiment comparison
💡 Business insights


💼 Business Value

The project demonstrates how customer feedback can be transformed into actionable information.

🏢 Potential Business Applications
🏆 Product quality monitoring
📦 Delivery and packaging issue detection
💰 Price and value perception analysis
😋 Product taste and satisfaction analysis
⚠️ Customer pain-point identification
📊 Customer feedback monitoring
💡 Data-driven customer experience analysis


🔮 Future Improvements

Possible future enhancements include:

🤖 Train a machine-learning sentiment classifier
🧠 Use transformer models such as BERT
🔍 Improve topic and theme extraction using NLP techniques
📈 Add sentiment trends over time
🛍️ Add product-level sentiment analysis
📊 Add advanced Power BI-style analytics
☁️ Deploy the dashboard online
🔄 Add automated review monitoring


🎓 Internship Context

This project was developed as part of:

🏢 CodeAlpha Internship

Task 4 – Sentiment Analysis

The project demonstrates practical experience in:

🐍 Python
📊 Data Analysis
🧠 Natural Language Processing
💭 Sentiment Analysis
📈 Data Visualization
💻 Streamlit Development
🌐 Git & GitHub


👨‍💻 Author
Vijaya Patil

🎓 B.Tech – Artificial Intelligence & Data Science

💡 Interested in:

📊 Data Analytics
🤖 Artificial Intelligence
🧠 Machine Learning
💻 Data Science

🚀 Built with Python, NLP & Data Analytics

Turning customer reviews into meaningful insights. 🛒📊🧠


### One important correction

Your original structure says:

```text
├── dashboard_reviews.csv


📂 data/
├── 📄 review_sample.csv
├── 📄 sentiment_summary.csv
└── 📄 rating_summary.csv