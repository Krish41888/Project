import gradio as gr
import joblib
import pandas as pd

# Load model
model = joblib.load("bagging_best_model.pkl")

def predict(age, duration, monthly_income, family_members):
    data = pd.DataFrame([[age, duration, monthly_income, family_members]],
                        columns=["Age", "DurationOfPitch", "MonthlyIncome", "NumberOfFamilyMembers"])
    prediction = model.predict(data)[0]
    return "Product Taken ✅" if prediction == 1 else "No Product ❌"

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label="Age"),
            gr.Number(label="Duration of Pitch"),
            gr.Number(label="Monthly Income"),
            gr.Number(label="Number of Family Members")],
    outputs="text",
    title="Tourism Prediction Model",
    description="Predict whether a customer will take the tourism product."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
