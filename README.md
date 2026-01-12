### **Customer Churn Prediction Using Simple Artificial Neural Network (ANN)**

This application predicts whether a bank customer is likely to **churn** (i.e., leave the bank) based on their profile and account information.

### Skills / Technologies Used

**Programming Languages**
  
[![Python](https://skillicons.dev/icons?i=python)](https://www.python.org/)

**Machine Learning & AI**

[![TensorFlow](https://skillicons.dev/icons?i=tensorflow)](https://www.tensorflow.org/) 
[![Scikit-learn](https://skillicons.dev/icons?i=scikitlearn)](https://scikit-learn.org/)

**Web App & Deployment**

Streamlit
[![Streamlit](https://skillicons.dev/icons?i=streamlit)](https://streamlit.io/)

#### **Model Overview:**

* **Type:** Artificial Neural Network (ANN)
* **Purpose:** Binary classification (Churn / Not Churn)
  
* **Input Features:**

  * **Geography:** Customer's country
  * **Gender:** Male / Female
  * **Age:** Customer’s age (18 - 95 years)
  * **Credit Score:** Bank-provided credit rating
  * **Balance:** Account balance
  * **Estimated Salary:** Annual income
  * **Tenure:** Number of years as a customer
  * **Number of Products:** Number of bank products used
  * **Has Credit Card:** Whether the customer owns a credit card
  * **Is Active Member:** Customer activity status

#### **Model Architecture:**

* **Input Layer:** Accepts 10 features
* **Hidden Layers:** Multiple dense layers with ReLU activation
* **Output Layer:** Single neuron with Sigmoid activation (probability of churn)
* **Training:** Optimized using backpropagation with a suitable optimizer (e.g., Adam)

#### **Predictions:**

* The model outputs a **probability score** between 0 and 1.
* Customers with a probability > 0.5 are classified as likely to churn.
* Users can interactively enter customer details and instantly get predictions.

### This streamlit app is live:
https://customer-churn-prediction-meg.streamlit.app/

#### **Please don't forget to star this repo if you liked it!**
