import os
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load models
diabetes_model_path = os.path.join("saved_models", "model_diabetes.pkl")
hypertension_model_path = os.path.join("saved_models", "model_BloodPressure.pkl")
obesity_model_path = os.path.join("saved_models", "model_Obesity.pkl")
cholestrol_model_path = os.path.join("saved_models", "model_Cholestrol.pkl")
model = load_model('model1_vgg19.h5')

# loading the saved models
diabetes_model = joblib.load(diabetes_model_path)
hypertension_model = joblib.load(hypertension_model_path)
Obesity_model = joblib.load(obesity_model_path)
cholestrol_model = joblib.load(cholestrol_model_path)

# Function to display the prediction page
def prediction_page():  

    # sidebar for navigation
    with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',

                            ['Diabetes Prediction',
                                'Blood Pressure Prediction',
                                'Obesity Prediction',
                                'Cholestrol Prediction',
                                'Malaria Prediction'],
                            menu_icon='hospital-fill',
                            icons=['droplet', 'heart-pulse', 'person', 'heart', "droplet-fill"],
                            default_index=0)
        

    # Diabetes Prediction Page
    if selected == 'Diabetes Prediction':

        # page title
        st.title('Diabetes Prediction using ML')

        # getting the input data from the user
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            age = st.text_input('Age')

        with col2:
            FBS = st.text_input('FBS')

        with col3:
            RBS = st.text_input('RBS value')

        with col4:
            HbA1c = st.text_input('HbA1c value')


        # code for Prediction
        diab_diagnosis = ''
        precautions = ''

        # creating a button for Prediction

        if st.button('Diabetes Test Result'):
            if age.strip() == '' or FBS.strip() == '' or RBS.strip() == '' or HbA1c.strip() == '':
                st.warning("All fields are required!")
            else:
                user_input = [age, FBS, RBS, HbA1c]
                user_input = [float(x) for x in user_input]
                diab_prediction = diabetes_model.predict([user_input])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                    precautions = """
                    **Precautions for Diabetic Patients:**
                    1. Follow a healthy diet plan rich in fruits, vegetables, whole grains, and lean proteins.
                    2. Monitor blood sugar levels regularly as advised by your healthcare provider.
                    3. Engage in regular physical activity such as walking, swimming, or cycling.
                    4. Take prescribed medications and insulin injections as directed.
                    5. Avoid sugary beverages and high-carbohydrate foods.
                    6. Maintain a healthy weight and strive to achieve a BMI within the recommended range.
                    7. Schedule regular check-ups with your doctor to monitor diabetes-related complications.
                    """
                else:
                    diab_diagnosis = 'The person is not diabetic'
                    precautions = """
                    **Precautions for Non-Diabetic Individuals:**
                    1. Maintain a balanced diet to prevent the risk of developing diabetes.
                    2. Stay physically active and exercise regularly to help regulate blood sugar levels.
                    3. Monitor your weight and aim to maintain a healthy BMI.
                    4. Limit the consumption of sugary and processed foods to reduce the risk of insulin resistance.
                    5. Get regular health check-ups to monitor blood glucose levels and overall health.
                    6. Stay hydrated and drink plenty of water throughout the day.
                    7. Reduce stress levels through relaxation techniques such as meditation or yoga.
                    """

        


        st.success(diab_diagnosis)
        st.markdown(precautions)

    # Blood Pressure Prediction Page
    if selected == 'Blood Pressure Prediction':

        # page title
        st.title('Blood Pressure Prediction using ML')

        col1, col2, col3 = st.columns(3)

        with col1:
            person_age = st.text_input("Person's age")

        with col2:
            systolic = st.text_input('Systolic')

        with col3:
            diastolic = st.text_input('Diastolic')

        with col1:
            sodium = st.text_input('Sodium')

        with col2:
            triglyceride = st.text_input('Triglyceride')


        # code for Prediction
        bloodPressure = ''
        blood_pressure_advice = ''

        # creating a button for Prediction

        if st.button('Blood Pressure Prediction'):
            if person_age.strip() == '' or systolic.strip() == '' or diastolic.strip() == '' or sodium.strip() == '' or triglyceride.strip() == '':
                st.warning("All fields are required!")
            else:
                user_input = [person_age, systolic, diastolic, sodium, triglyceride]
                user_input = [float(x) for x in user_input]
                bloodPressure_prediction = hypertension_model.predict([user_input])

                if bloodPressure_prediction[0] == 0:
                    bloodPressure = 'Blood Pressure level LOW'
                    blood_pressure_advice = """
                    **Precautions for Low Blood Pressure:**
                    1. Increase your salt intake to help raise blood pressure.
                    2. Stay hydrated by drinking plenty of fluids.
                    3. Avoid standing up quickly to prevent dizziness.
                    4. Eat small, frequent meals to prevent drops in blood pressure.
                    5. Avoid hot showers or baths, which can lower blood pressure further.
                    6. Exercise regularly to improve cardiovascular health.
                    7. Consult your doctor for personalized advice and treatment options.
                    """
                elif bloodPressure_prediction[0] == 1:
                    bloodPressure = 'Blood Pressure level NORMAL'
                    blood_pressure_advice = """
                    **Precautions for Normal Blood Pressure:**
                    1. Maintain a healthy diet rich in fruits, vegetables, and whole grains.
                    2. Limit salt and sodium intake to help maintain normal blood pressure levels.
                    3. Engage in regular physical activity, such as brisk walking or cycling.
                    4. Manage stress through relaxation techniques, such as meditation or deep breathing.
                    5. Monitor blood pressure regularly and seek medical advice if it consistently falls outside the normal range.
                    6. Avoid smoking and limit alcohol consumption to promote heart health.
                    7. Follow your doctor's recommendations for preventive healthcare.
                    """
                elif bloodPressure_prediction[0] == 2:
                    bloodPressure = 'Blood Pressure level HIGH'
                    blood_pressure_advice = """
                    **Precautions for High Blood Pressure:**
                    1. Follow a low-sodium diet to help lower blood pressure.
                    2. Engage in regular aerobic exercise, such as jogging or swimming.
                    3. Maintain a healthy weight through a balanced diet and exercise.
                    4. Limit alcohol consumption and avoid smoking to reduce cardiovascular risk.
                    5. Monitor blood pressure regularly and take prescribed medications as directed.
                    6. Manage stress through relaxation techniques and mindfulness practices.
                    7. Consult your doctor for personalized treatment and lifestyle recommendations.
                    """

        st.success(bloodPressure)
        st.markdown(blood_pressure_advice)

    # Obesity Prediction Page
    if selected == "Obesity Prediction":

        # page title
        st.title("Obesity Prediction using ML")

        col1, col2, col3 = st.columns(3)

        with col1:
            patient_age = st.text_input("Patient's age")

        with col2:
            Gender = st.text_input('Gender')

        with col3:
            waist_circum = st.text_input('Waist circumference')

        with col1:
            skin_fold = st.text_input('Skin fold')

        with col2:
            BMI = st.text_input('BMI')


    # Initialize flag for input validation
        valid_input = True

        # Code for Prediction
        Obesity = ''
        obesity_advice = ''

        # Creating a button for Prediction    
        if st.button("Obesity Test Result"):
            if patient_age.strip() == '' or waist_circum.strip() == '' or skin_fold.strip() == '' or BMI.strip() == '':
                st.warning("All fields are required!")
                valid_input = False
            else:
                if isinstance(Gender, str):
                    Gender = Gender.strip().lower()
                    if Gender in ['m', 'male']:
                        Gender = 1
                    elif Gender in ['f', 'female']:
                        Gender = 0
                    else:
                        st.warning("Invalid input for gender. Please enter 'm' or 'male' for male, or 'f' or 'female' for female.")
                        valid_input = False

                if valid_input:
                    user_input = [patient_age, Gender, waist_circum, skin_fold, BMI]
                    user_input = [float(x) for x in user_input]
                    obesity_prediction = Obesity_model.predict([user_input])

                    if obesity_prediction[0] == 1:
                        Obesity = "The person is Obese"
                        obesity_advice = """
                        **Precautions for Obesity:**
                        1. Adopt a balanced diet rich in fruits, vegetables, lean proteins, and whole grains.
                        2. Limit consumption of high-calorie, processed foods and sugary beverages.
                        3. Engage in regular physical activity, such as brisk walking, jogging, or swimming.
                        4. Set realistic weight loss goals and aim to achieve gradual, sustainable weight loss.
                        5. Monitor portion sizes and practice mindful eating to avoid overeating.
                        6. Seek support from healthcare professionals, such as dietitians or nutritionists, for personalized guidance.
                        7. Prioritize adequate sleep and manage stress levels to support overall well-being.
                        """
                    else:
                        Obesity = "The person is not Obese"
                        obesity_advice = """
                        **Precautions for Non-Obese Individuals:**
                        1. Maintain a healthy weight by following a balanced diet and staying physically active.
                        2. Monitor calorie intake and aim to consume a variety of nutrient-dense foods.
                        3. Engage in regular exercise to support cardiovascular health and muscle strength.
                        4. Practice portion control and mindful eating to prevent weight gain.
                        5. Stay hydrated by drinking plenty of water throughout the day.
                        6. Avoid excessive consumption of processed foods, sugary snacks, and high-fat meals.
                        7. Incorporate stress-reduction techniques, such as meditation or yoga, into your daily routine.
                        """

        # Display final output only if input is valid
        if valid_input:
            st.success(Obesity)
            st.markdown(obesity_advice)

    #Cholestrol prediction
    if selected == 'Cholestrol Prediction':

        # page title
        st.title('Cholestrol Prediction using ML')

        col1, col2, col3 = st.columns(3)

        with col1:
            Patient_age = st.text_input("Person's age")

        with col2:
            HDL = st.text_input('HDL')

        with col3:
            LDL = st.text_input('LDL')

        with col1:
            TG = st.text_input('TG')

        with col2:
            totalCholestrol = st.text_input('Total Cholestrol')


        # code for Prediction
        cholestrol = ''
        cholestrol_advice = ''

        # creating a button for Prediction

        if st.button('Cholestrol Prediction'):
            if Patient_age.strip() == '' or HDL.strip() == '' or LDL.strip() == '' or TG.strip() == '' or totalCholestrol.strip() == '':
                st.warning("All fields are required!")
            else:
                user_input = [Patient_age, HDL, LDL, TG, totalCholestrol]
                user_input = [float(x) for x in user_input]
                cholestrolPrediction = cholestrol_model.predict([user_input])

                if cholestrolPrediction[0] == 0:
                    cholestrol = 'Cholestrol level LOW'
                    cholestrol_advice = """
                    **Precautions for Low Cholesterol Levels:**
                    1. Consume healthy fats, such as those found in avocados, nuts, and olive oil, to help raise cholesterol levels.
                    2. Include foods rich in omega-3 fatty acids, such as fatty fish (salmon, mackerel, sardines), flaxseeds, and chia seeds, in your diet.
                    3. Eat cholesterol-rich foods in moderation, such as eggs and shellfish, to support healthy cholesterol levels.
                    4. Limit consumption of processed foods and refined carbohydrates, which may contribute to low cholesterol levels.
                    5. Consider consulting with a healthcare professional to determine the underlying cause of low cholesterol and appropriate treatment options.
                    """
                elif cholestrolPrediction[0] == 1:
                    cholestrol = 'Cholestrol level NORMAL'
                    cholestrol_advice = """
                    **Precautions for Normal Cholesterol Levels:**
                    1. Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.
                    2. Limit intake of saturated and trans fats, found in red meat, processed foods, and fried foods, to help maintain normal cholesterol levels.
                    3. Engage in regular physical activity, such as walking, cycling, or swimming, to support heart health and cholesterol metabolism.
                    4. Monitor cholesterol levels regularly and consult with a healthcare professional if levels change significantly.
                    5. Avoid smoking and limit alcohol consumption to reduce cardiovascular risk factors.
                    """
                elif cholestrolPrediction[0] == 2:
                    cholestrol = 'Cholestrol level HIGH'
                    cholestrol_advice = """
                    **Precautions for High Cholesterol Levels:**
                    1. Follow a heart-healthy diet low in saturated and trans fats, cholesterol, and refined sugars.
                    2. Increase consumption of soluble fiber-rich foods, such as oats, beans, fruits, and vegetables, to help lower cholesterol levels.
                    3. Incorporate regular aerobic exercise into your routine to improve cholesterol metabolism and overall cardiovascular health.
                    4. Take prescribed cholesterol-lowering medications, such as statins, as directed by your healthcare provider.
                    5. Maintain a healthy weight and aim to achieve a BMI within the recommended range.
                    6. Limit alcohol consumption and avoid smoking to reduce additional cardiovascular risk factors.
                    7. Schedule regular follow-up appointments with your doctor to monitor cholesterol levels and adjust treatment as needed.
                    """

        st.success(cholestrol)
        st.markdown(cholestrol_advice)

    #Malaria Prediction
    if selected == 'Malaria Prediction':
        st.title('Malaria Classification')

        # Upload an image
        uploaded_file = st.file_uploader("Choose an image...", type="png")

        if uploaded_file is not None:
            # Display the uploaded image
            image_display = st.image(uploaded_file, caption='Uploaded Image', width=200)

            # Preprocess the uploaded image
            img = image.load_img(uploaded_file, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make prediction
            prediction = model.predict(img_array)
            class_label = 'Person is Malaria Uninfected' if prediction[0][0] > 0.5 else 'Person is Malaria Infected'

            # Display prediction result
            st.write(f'Prediction: {class_label}')

            # Provide advice based on prediction
            if prediction[0][0] > 0.5:
                st.write("Precautions for Malaria Uninfected:")
                st.write("""
                1. Continue to practice preventive measures to avoid mosquito bites, such as using insect repellents and mosquito nets, particularly in malaria-endemic areas.
                2. Monitor your health for any symptoms of malaria, especially if you have recently traveled to or live in regions where malaria is prevalent.
                3. If you experience symptoms suggestive of malaria, such as fever, seek medical attention promptly for evaluation and testing.
                4. Stay informed about malaria risks in your area and follow local health advisories and recommendations.
                5. Consider taking antimalarial medications as prescribed if you plan to travel to malaria-endemic regions.
                """)
                
            else:
                st.write("Precautions for Malaria Infected:")
                st.write("""
                1. Seek immediate medical attention if you experience symptoms such as fever, chills, sweats, headache, body aches, nausea, or vomiting.
                2. Take prescribed antimalarial medications as directed by your healthcare provider.
                3. Use insect repellents and mosquito nets to prevent mosquito bites, especially in malaria-endemic areas.
                4. Wear long-sleeved shirts and long pants to reduce exposure to mosquito bites, particularly during dawn and dusk when mosquitoes are most active.
                5. Stay indoors in well-screened or air-conditioned areas, especially during peak mosquito activity times.
                6. Drain standing water around your home to eliminate mosquito breeding sites.
                7. Inform your healthcare provider if you have traveled to or live in areas where malaria is endemic, as preventive measures may be necessary.
                """)   
    
prediction_page()
