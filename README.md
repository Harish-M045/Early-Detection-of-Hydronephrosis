Hydronephrosis Detection using Deep Learning
This project detects Hydronephrosis from ultrasound kidney images using Deep Learning models.

Overview
Hydronephrosis is a condition where the kidney becomes swollen due to urine buildup. Early detection is important especially in pregnant women.

This project uses transfer learning with MobileNet to classify ultrasound images into:

Normal
Hydronephrosis
Technologies Used
Python
TensorFlow
Streamlit
MobileNetV2
OpenCV
Project Workflow
Collect ultrasound dataset
Preprocess images
Train CNN model
Save trained model
Deploy model using Streamlit
Run the Project
Install dependencies: pip install -r requirements.txt

Run Streamlit app: streamlit run app.py

Open browser: http://localhost:8501

Upload an ultrasound image to see the prediction.
