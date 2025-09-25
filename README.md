# 💳 CredGuard - Real-Time Credit Card Fraud Detection

CredGuard is a **Streamlit-based web application** for detecting fraudulent credit card transactions in real-time.  
It leverages a **deep learning transformer model (SAINT)** and includes features like probability-based risk scoring, contextual metadata, and report generation.  

---

## ✨ Features
- 🔎 **Transaction Prediction**: Classify transactions as Legit or Fraud.  
- 📊 **Risk Analysis**: Fraud probability scoring with tiered alerts (Low / Medium / High).  
- 📄 **Report Generation**: Downloadable fraud detection reports for audit and compliance.  
- 🌍 **Metadata-Aware**: Includes transaction **Country, Device, and Time of Day** context.  
- 🧠 **Deep Learning Model**: Powered by SAINT (PyTorch).  
- 🎨 **Interactive UI**: Built with Streamlit for easy and responsive use.  

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
Create and activate virtual environment

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the app

bash
Copy code
streamlit run main.py
Open your browser → http://localhost:8501 🎉

📂 Project Structure
bash
Copy code
credit-card-fraud-detection/
├── main.py             # Streamlit app entry point
├── model.py            # Model architecture (SimpleSAINT)
├── scaler.joblib       # Pre-trained scaler for preprocessing
├── requirements.txt    # Dependencies
├── Procfile            # For Render deployment
