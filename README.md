# 🚀 Nebula Finance Lab

**Nebula Finance Lab** is a multi-themed, interactive financial machine learning application built using [Streamlit](https://streamlit.io/). Developed for the **AF3005 Group Project (BSFT Batch 22, Sections A, B, C)** at _[Your University Name]_, this app allows users to fetch real-time stock market data, upload financial datasets, and apply machine learning models with visually engaging themes—each designed by a team member.

---

## 🎯 Project Objective

To collaboratively design and build a creative, real-time, and interactive financial ML application. The project integrates:

- 📊 Live stock data from Yahoo Finance
- 📂 Dataset uploads (CSV/Excel)
- 🤖 Four machine learning models
- 🎨 Four uniquely styled themes for enhanced user experience

---

## 🛠️ Features

### 📡 Data Sources

- **Yahoo Finance API (`yfinance`)**: Fetch live stock data using ticker symbols (e.g., `AAPL`, `GOOG`)
- **File Upload**: Import financial datasets (CSV/Excel) from Kaggle or local files

### 🤖 Machine Learning Models

| Model               | Theme              | Description                                |
|--------------------|--------------------|--------------------------------------------|
| **Linear Regression**   | 🧟 Zombie Theme       | Predictive analysis with horror visuals     |
| **Logistic Regression** | 👨‍🚀 Futuristic Theme  | Binary classification with sci-fi style     |
| **K-Means Clustering**  | 🐉 Game of Thrones    | Unsupervised learning with medieval design  |
| **XGBoost**             | 🌌 Nebula Pulse Theme | Advanced classification with pixel effects  |

### 🎨 Visual Themes

- 🧟 **Zombie**: Dark, eerie aesthetics with Creepster font and horror GIFs
- 👨‍🚀 **Futuristic**: Neon lights, Exo 2 font, space animations
- 🐉 **Game of Thrones**: Fire/ice graphics, Cinzel font, medieval UI
- 🌌 **Nebula Pulse**: Vibrant gaming style with Poppins font and pixel art

### 🧠 Interactivity

- 🎛️ Dynamic theme switching
- 📈 Interactive charts (Plotly, Matplotlib)
- 🔄 Buttons for live data fetching and model training
- ⚠️ Error handling with user-friendly messages
- 💾 Cache management to improve performance

---

## 👥 Team Members & Contributions

| Name              | Contribution                                                             | Theme             | ML Model             |
|-------------------|--------------------------------------------------------------------------|-------------------|----------------------|
| **Muhammad Arsalan** | Designed Zombie Theme, implemented Linear Regression, data integration | 🧟 Zombie          | Linear Regression     |
| **Hassaan Asif**     | Developed Futuristic Theme, implemented Logistic Regression, added RSI | 👨‍🚀 Futuristic      | Logistic Regression   |
| **Tallal Zubair**    | Created Game of Thrones Theme, implemented K-Means Clustering, UI lead | 🐉 Game of Thrones | K-Means Clustering    |
| **Tallal Zubair**    | Built Nebula Pulse Theme, implemented XGBoost, coordinated deployment  | 🌌 Nebula Pulse    | XGBoost               |


---

## 📸 Screenshots

> Add your project screenshots in the `screenshots/` folder.

- 📍 Welcome Page  
- 📊 Dashboard  
- 🔍 Analysis View  

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8+
- Git
- Streamlit Cloud account (for deployment)

### 🧪 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nebula-finance-lab.git
cd nebula-finance-lab

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
