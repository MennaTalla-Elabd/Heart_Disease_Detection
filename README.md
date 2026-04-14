# Heart Disease Prediction System

### Hybrid Intelligent System (Expert System + Machine Learning )

## Overview

This project presents a **hybrid intelligent system** for heart disease prediction by integrating:

* A **Machine Learning model** trained on medical data
* A **rule-based Expert System** that encodes domain knowledge

This combination enhances both:

*  **Prediction accuracy**
*  **Model interpretability**
##  System Architecture

The system consists of three main components:

1. **Machine Learning Module**

   * Predicts the probability of heart disease
   * Captures hidden patterns in the dataset

2. **Expert System Module**

   * Applies predefined medical rules
   * Provides logical reasoning and explanations

3. **Integration Layer**

   * Combines outputs from both systems
   * Produces a final, reliable decision

---

## 🧪 Technologies

* **Programming Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Expert System:** Experta
* **Web Interface:** Streamlit
* **Model Persistence:** Joblib

---

## 📂 Project Structure

```id="7dj3h1"
heart_project/
├── data/
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│
├── ml_model/
│   ├── train_model.py
│   ├── predict.py
│   ├── decision_tree_model.pkl
│
├── expert_system/
│   ├── rules.py
│   ├── expert_system.py
│
├── utils/
│   ├── data_processing.py
│
├── comparison/
│   ├── compare_models.py
│   ├── accuracy_comparison.md
│
├── app/
│   ├── streamlit_app.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Installation

### 1. Clone Repository

```bash id="r3j8f2"
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies

```bash id="q8k2z1"
pip install -r requirements.txt
```

---

## Usage

Run the Streamlit application:

```bash id="p1x8c9"
streamlit run app.py
```

Access the app via:

```id="k2m9d3"
http://localhost:8501
```

---

##  Features

* Hybrid prediction (Expert System + ML)
* Explainable decision-making
* Interactive user interface
* Modular and scalable architecture

---

## 📊 Results

* The Decision Tree model provides fast and interpretable predictions
* The Expert System enhances trust through rule-based explanations
* The hybrid approach improves overall system reliability

*(You can add accuracy, confusion matrix, or evaluation metrics here)*

---

##  Future Work

* Integrate additional ML models (Random Forest, SVM, Neural Networks)
* Deploy the application to cloud platforms
* Expand the expert rule base
* Improve dataset size and quality

---

## 👩‍💻 Author
 Team Triple M+N
---

## License

This project is intended for academic and educational use.

---
