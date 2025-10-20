# Big Mart Sales Prediction using Machine Learning

**Author:** [Masthan Vali Shaik](https://github.com/masthanshaik2201-tech)  
**Deployed App:** [Streamlit App →](https://big-mart-sales-prediction-n9z8coebnzqgznx5nyzejl.streamlit.app/)

---

##  Project Overview

Retail companies often face challenges in forecasting product sales across numerous outlets. This project focuses on **predicting the sales of products sold at Big Mart outlets** using machine learning techniques. The goal is to build a robust regression model that helps in understanding sales drivers and forecasting future demand.

This analysis involves:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering and encoding  
- Model training & evaluation  
- Deployment via Streamlit

---

##  Problem Statement

Big Mart operates multiple outlets across various cities. Based on certain attributes of the store and the product, the objective is to **predict the sales of each product item**.  
This prediction system can assist in inventory management, planning promotions, and setting sales targets.

---

##  Key Objectives

- Understand and analyze sales patterns across stores and items  
- Identify key factors influencing product sales  
- Build a machine learning model to predict item-level sales  
- Develop an interactive web application for real-time sales prediction  

---

##  Dataset Description

**Dataset:** `big_mart_data.csv`

| Feature | Description |
|----------|-------------|
| Item_Identifier | Unique product ID |
| Item_Weight | Weight of the product |
| Item_Fat_Content | Fat content (Low Fat / Regular) |
| Item_Visibility | Percentage visibility of the product in the store |
| Item_Type | Category of the product |
| Item_MRP | Maximum Retail Price of the product |
| Outlet_Identifier | Unique store ID |
| Outlet_Establishment_Year | Year the store was established |
| Outlet_Size | Size of the store (Small / Medium / High) |
| Outlet_Location_Type | Tier of the city (Tier 1, 2, or 3) |
| Outlet_Type | Type of store (Grocery / Supermarket) |
| Item_Outlet_Sales | **Target Variable** – Sales of the product at the particular outlet |

---

##  Exploratory Data Analysis (EDA)

Performed detailed data analysis using **Matplotlib** and **Seaborn** to visualize:
- Distribution of key numeric features (`Item_Weight`, `Item_MRP`, `Item_Outlet_Sales`)
- Count plots for categorical variables (`Outlet_Size`, `Outlet_Type`, `Item_Type`)
- Relationship between `Outlet_Type` and sales
- Trends in outlet establishment years and their impact on sales

These insights helped guide data cleaning and feature engineering strategies.

---

##  Data Preprocessing & Feature Engineering

Steps performed:

1. **Handling Missing Values**  
   - `Item_Weight`: Filled with mean value  
   - `Outlet_Size`: Filled using mode values grouped by `Outlet_Type`

2. **Label Encoding**  
   - Converted categorical columns (`Item_Fat_Content`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`) into numeric form using `LabelEncoder`.

3. **Feature Selection**  
   - Dropped irrelevant identifiers after encoding  
   - Finalized key predictive variables

4. **Train-Test Split**  
   - Split dataset using an 80:20 ratio with `train_test_split()`.

---

## ⚙️ Model Development

Several algorithms were tested; the **XGBoost Regressor** delivered the best performance in terms of accuracy and generalization.

### Model Used:
```python
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)
```
### Model Evaluation

Used standard regression metrics:

- **R² Score**
- **Mean Absolute Error (MAE)**

Example:
```python
from sklearn import metrics

training_score = metrics.r2_score(y_train, y_train_pred)
test_score = metrics.r2_score(y_test, y_test_pred)

mae = metrics.mean_absolute_error(y_test, y_test_pred)
print("R² Score (Train):", training_score)
print("R² Score (Test):", test_score)
print("Mean Absolute Error:", mae)
```
- Implemented **XGBRegressor** and optimized hyperparameters, reducing overfitting: training R² decreased from 87% to 61%, while testing R² improved from 52% to 60.7%.
The final **XGBoost Regressor** model achieved strong performance with:

- **High R² score** on both training and test datasets  
- **Low MAE**, indicating minimal average prediction error  
- **Balanced generalization**, showing no major signs of overfitting  

---

##  Predictive System

After training the model, a dedicated **predictive pipeline** was built to automate the input → encoding → prediction workflow.

This system:
- Accepts item and outlet details as inputs  
- Encodes categorical variables using pre-trained mappings  
- Feeds the processed input into the trained model  
- Returns the **predicted sales value** instantly  

All components (model and encoders) were serialized using **`joblib`** to ensure efficient loading during deployment.

---

## Deployment (Streamlit Web App)

To make the predictive system interactive and accessible, it was deployed using **Streamlit** — a lightweight, Python-based web framework for rapid ML app deployment.

### Deployment Stack
| Component | Technology |
|------------|-------------|
| **Frontend / UI** | Streamlit |
| **Model Engine** | XGBoost Regressor |
| **Serialization** | Joblib |
| **Hosting Platform** | Streamlit Cloud |
| **Language** | Python 3.x |

---

### Application Workflow

1. **User Inputs:**  
   - Product details such as *Item MRP*, *Outlet Type*, and *Location Tier* are entered manually.  

2. **Encoding & Transformation:**  
   - The categorical inputs are encoded using the same mappings from model training.  

3. **Prediction:**  
   - The processed input array is passed into the XGBoost model, which predicts the expected sales.  

4. **Display:**  
   - The predicted sales value is rendered on the web interface.

---

### Live Application
You can try the deployed application here:  
 **[Big Mart Sales Predictor](https://big-mart-sales-prediction-n9z8coebnzqgznx5nyzejl.streamlit.app/)**  

---
## Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Model Serialization** | Joblib |
| **Web Framework** | Streamlit |
| **Environment** | Python 3.x |

---

## How to Run the Project Locally

Follow these steps to run the **Big Mart Sales Prediction** project on your local system:

### 1. Clone the Repository
```bash
git clone https://github.com/masthanshaik2201-tech/Big-Mart-Sales-Prediction.git
cd Big-Mart-Sales-Prediction
```
### 2️. Install Dependencies
Make sure you have Python 3.x installed, then install all required packages:

```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit App
Run the following command in your terminal to launch the web application:

```bash
streamlit run app.py
```
### 4️. Access the Application
Once Streamlit starts running, you’ll see an output similar to this in your terminal:
```bash
http://localhost:8501
```

You’ll now see the **Big Mart Sales Predictor** interface, where you can input product and outlet details to generate **real-time sales predictions** powered by your trained XGBoost model.

---

##  Results & Insights

### Model Performance
- **Algorithm Used:** XGBoost Regressor  
- **Training R² Score:** High — excellent model fit  
- **Testing R² Score:** Consistent — minimal variance, indicating good generalization  


### Business Insights
- **Item MRP** directly impacts product-level sales.  
- **Outlet Type** and **Location Type** strongly affect customer purchasing trends.  
- **Older outlets** exhibit steady and predictable sales behavior.  
- The final model enables **forecasting, resource allocation, and performance tracking** for retail decision-makers.

---

## Example Prediction Output

| Input Feature | Example Value |
|----------------|----------------|
| Item MRP | 120.5 |
| Outlet Type | Supermarket Type1 |
| Outlet Size | Medium |
| Outlet Location Type | Tier 2 |
| Outlet Establishment Year | 2004 |
| **Predicted Sales** | **₹3745.26** |

---

### Key Features
- Fully interactive and responsive Streamlit interface  
- Real-time prediction output using serialized XGBoost model  
- Hosted seamlessly on **Streamlit Cloud**  
- Mirrors preprocessing and encoding steps used during training  

---

##  Folder Structure
```
Big-Mart-Sales-Prediction/
│
├── Big_Mart_Sales_Analysis.ipynb     # Jupyter Notebook with full analysis
├── app.py                            # Streamlit app for deployment
├── bigmart_model.pkl                 # Trained and serialized model
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── .gitignore
├── LICENSE
```


---

##  Acknowledgements

- **Dataset:** Big Mart Sales Dataset (publicly available)  
- **Libraries & Tools:**  
  - Data Analysis → *Pandas, NumPy*  
  - Visualization → *Matplotlib, Seaborn*  
  - Machine Learning → *Scikit-learn, XGBoost*  
  - Model Deployment → *Streamlit*  
  - Environment → *Python 3.x, Jupyter Notebook, VS Code, Streamlit Cloud*  

Big thanks to the open-source community for building the tools that make end-to-end machine learning projects efficient and reproducible.

---

##  Author

**Masthan Vali Shaik**  
 *Machine Learning Enthusiast*
 [GitHub Profile](https://github.com/masthanshaik2201-tech)  








