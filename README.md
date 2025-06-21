# Customer-churn



import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import pickle

# === Load and preprocess data ===
df = pd.read_csv("Downloads/customer churn sample.csv")
df.drop(['CustomerID', 'TotalCharges'], axis=1, inplace=True)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})
df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
df.dropna(inplace=True)

def tenure_bucket(t):
    if t <= 12:
        return 0
    elif t <= 36:
        return 1
    else:
        return 2

df['TenureBucket'] = df['Tenure'].apply(tenure_bucket)
df['ChargesPerMonth'] = df['MonthlyCharges'] * df['Tenure']

features = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService', 
            'InternetService', 'MonthlyCharges', 'TenureBucket', 'ChargesPerMonth']
X = df[features]
y = df['Churn']

df_majority = df[df['Churn'] == 0]
df_minority = df[df['Churn'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced[features]
y = df_balanced['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# === GUI Functions ===
def predict_churn():
    try:
        gender = 0 if gender_var.get() == "Male" else 1
        senior = 0 if senior_var.get() == "No" else 1
        partner = 0 if partner_var.get() == "No" else 1
        dependents = 0 if dep_var.get() == "No" else 1
        tenure = int(tenure_var.get())
        phone = 0 if phone_var.get() == "No" else 1
        internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
        internet = internet_map[internet_var.get()]
        monthly = float(monthly_var.get())

        tenure_buck = tenure_bucket(tenure)
        charges_per_month = monthly * tenure

        input_features = np.array([[gender, senior, partner, dependents, tenure,
                                    phone, internet, monthly, tenure_buck, charges_per_month]])
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1] * 100

        if prediction == 1:
            messagebox.showinfo("Prediction Result", f"⚠️ Customer is likely to CHURN.\nProbability: {probability:.2f}%")
        else:
            messagebox.showinfo("Prediction Result", f"✅ Customer is likely to STAY.\nProbability: {probability:.2f}%")
    except Exception as e:
        messagebox.showerror("Input Error", f"Please check your inputs.\n\nError: {e}")

# === Menu Bar Functions ===
def new_file():
    messagebox.showinfo("New", "Start a new prediction session.")

def open_file():
    file = filedialog.askopenfilename(title="Open File")
    if file:
        messagebox.showinfo("Open", f"Selected file: {file}")

def save_file():
    file = filedialog.asksaveasfilename(defaultextension=".txt")
    if file:
        messagebox.showinfo("Save", f"Saved to: {file}")

def cut_text():
    app.focus_get().event_generate("<<Cut>>")

def copy_text():
    app.focus_get().event_generate("<<Copy>>")

def delete_text():
    widget = app.focus_get()
    if isinstance(widget, tk.Entry):
        widget.delete(0, tk.END)

# === GUI Design ===
app = tk.Tk()
app.title("Customer Churn Predictor")
app.state("zoomed")  # Fullscreen on Windows
app.configure(bg="#f2f2f2")

# ==== Menu Bar ====
menu_bar = tk.Menu(app)

file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="New", command=new_file)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Save", command=save_file)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=app.quit)
menu_bar.add_cascade(label="File", menu=file_menu)

edit_menu = tk.Menu(menu_bar, tearoff=0)
edit_menu.add_command(label="Copy", command=copy_text)
edit_menu.add_command(label="Cut", command=cut_text)
edit_menu.add_command(label="Delete", command=delete_text)
menu_bar.add_cascade(label="Edit", menu=edit_menu)

app.config(menu=menu_bar)

# === UI Layout ===
form_frame = tk.Frame(app, bg="#f2f2f2")
form_frame.pack(pady=30)

tk.Label(form_frame, text="📊 Customer Churn Prediction Tool", font=("Helvetica", 16, "bold"), bg="#f2f2f2", fg="#34495e").pack(pady=(0, 10))
tk.Label(form_frame, text="Enter customer details below and click 'Predict Churn'", font=("Helvetica", 11), bg="#f2f2f2", fg="#2c3e50").pack(pady=(0, 20))

# Form Variables
gender_var = tk.StringVar(value="Male")
senior_var = tk.StringVar(value="No")
partner_var = tk.StringVar(value="No")
dep_var = tk.StringVar(value="No")
phone_var = tk.StringVar(value="No")
internet_var = tk.StringVar(value="DSL")
tenure_var = tk.StringVar()
monthly_var = tk.StringVar()

def add_label_dropdown(frame, text, variable, options):
    tk.Label(frame, text=text, bg="#f2f2f2", fg="#2c3e50", font=("Helvetica", 10, "bold")).pack(pady=(10, 2))
    tk.OptionMenu(frame, variable, *options).pack()

# Input Fields
add_label_dropdown(form_frame, "Gender", gender_var, ["Male", "Female"])
add_label_dropdown(form_frame, "Senior Citizen", senior_var, ["No", "Yes"])
add_label_dropdown(form_frame, "Partner", partner_var, ["No", "Yes"])
add_label_dropdown(form_frame, "Dependents", dep_var, ["No", "Yes"])

tk.Label(form_frame, text="Tenure (months)", bg="#f2f2f2", fg="#2c3e50", font=("Helvetica", 10, "bold")).pack(pady=(10, 2))
tk.Entry(form_frame, textvariable=tenure_var).pack()

add_label_dropdown(form_frame, "Phone Service", phone_var, ["No", "Yes"])
add_label_dropdown(form_frame, "Internet Service", internet_var, ["DSL", "Fiber optic", "No"])

tk.Label(form_frame, text="Monthly Charges", bg="#f2f2f2", fg="#2c3e50", font=("Helvetica", 10, "bold")).pack(pady=(10, 2))
tk.Entry(form_frame, textvariable=monthly_var).pack()

# Predict Button
tk.Button(form_frame, text="Predict Churn", command=predict_churn, bg="#2980b9", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5).pack(pady=25)

# Run the app
app.mainloop()
