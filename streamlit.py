import streamlit as st
import pandas as pd
import boto3
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up the SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Define the endpoint name
ENDPOINT_NAME = 'sagemaker-scikit-learn-2024-09-18-03-06-17-234'  # Replace if different



# Define the features and their types
features = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Guam', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virgin Islands', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'],
    'Sex': ['Female', 'Male'],
    'GeneralHealth': ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'],
    'LastCheckupTime': ['5 or more years ago', 'Within past 2 years (1 year but less than 2 years ago)', 'Within past 5 years (2 years but less than 5 years ago)', 'Within past year (anytime less than 12 months ago)'],
    'PhysicalActivities': ['No', 'Yes'],
    'RemovedTeeth': ['1 to 5', '6 or more, but not all', 'All', 'None of them'],
    'HadAngina': ['No', 'Yes'],
    'HadStroke': ['No', 'Yes'],
    'HadAsthma': ['No', 'Yes'],
    'HadSkinCancer': ['No', 'Yes'],
    'HadCOPD': ['No', 'Yes'],
    'HadDepressiveDisorder': ['No', 'Yes'],
    'HadKidneyDisease': ['No', 'Yes'],
    'HadArthritis': ['No', 'Yes'],
    'HadDiabetes': ['No', 'No, pre-diabetes or borderline diabetes', 'Yes', 'Yes, but only during pregnancy (female)'],
    'DeafOrHardOfHearing': ['No', 'Yes'],
    'BlindOrVisionDifficulty': ['No', 'Yes'],
    'DifficultyConcentrating': ['No', 'Yes'],
    'DifficultyWalking': ['No', 'Yes'],
    'DifficultyDressingBathing': ['No', 'Yes'],
    'DifficultyErrands': ['No', 'Yes'],
    'SmokerStatus': ['Current smoker - now smokes every day', 'Current smoker - now smokes some days', 'Former smoker', 'Never smoked'],
    'ECigaretteUsage': ['Never used e-cigarettes in my entire life', 'Not at all (right now)', 'Use them every day', 'Use them some days'],
    'ChestScan': ['No', 'Yes'],
    'RaceEthnicityCategory': ['Black only, Non-Hispanic', 'Hispanic', 'Multiracial, Non-Hispanic', 'Other race only, Non-Hispanic', 'White only, Non-Hispanic'],
    'AgeCategory': ['Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 'Age 80 or older'],
    'AlcoholDrinkers': ['No', 'Yes'],
    'HIVTesting': ['No', 'Yes'],
    'FluVaxLast12': ['No', 'Yes'],
    'PneumoVaxEver': ['No', 'Yes'],
    'TetanusLast10Tdap': ['No, did not receive any tetanus shot in the past 10 years', 'Yes, received Tdap', 'Yes, received tetanus shot but not sure what type', 'Yes, received tetanus shot, but not Tdap'],
    'HighRiskLastYear': ['No', 'Yes'],
    'CovidPos': ['No', 'Tested positive using home test without a health professional', 'Yes']
}

# Define the order of features
FEATURE_ORDER = [
    'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms',
    'State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities', 'RemovedTeeth',
    'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
    'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
    'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands',
    'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory',
    'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap',
    'HighRiskLastYear', 'CovidPos'
]

def predict(input_data):
    logger.info("Starting prediction process")
    try:
        # Convert input data to a DataFrame
        df = pd.DataFrame([input_data])
        
        # Calculate BMI
        df['BMI'] = df['WeightInKilograms'] / (df['HeightInMeters'] ** 2)
        
        # Debug: Log input data
        logger.info("Input data:")
        for feature, value in input_data.items():
            logger.info(f"{feature}: {value}")
        
        # Define the encoding for each categorical column
        encoding_dict = {k: v for k, v in features.items() if isinstance(v, list)}

        # Perform one-hot encoding
        encoded_features = []
        for col, categories in encoding_dict.items():
            for i, category in enumerate(categories):
                col_name = f"{col}_encoded_{i}"
                encoded_features.append(col_name)
                if col in ['HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis']:
                    # For health conditions, 'Yes' should be 1, 'No' should be 0
                    df[col_name] = (df[col] == category).astype(int)
                else:
                    df[col_name] = (df[col] == category).astype(int)

        # Identify numeric columns
        numeric_cols = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']

        # Combine numeric columns and encoded features in the correct order
        final_cols = numeric_cols + encoded_features
        final_df = df[final_cols]

        # Ensure all columns are present (fill missing with 0)
        for col in final_cols:
            if col not in final_df.columns:
                final_df[col] = 0

        # Debug: Log encoded data
        logger.info("Encoded data:")
        for col in final_df.columns:
            logger.info(f"{col}: {final_df[col].values[0]}")

        # Convert to numpy array and then to list for JSON serialization
        payload = json.dumps(final_df.values.tolist())
        
        # Log the payload
        logger.info(f"Payload being sent to SageMaker endpoint: {payload}")
        logger.info(f"Shape of final dataframe: {final_df.shape}")
        
        # Make a prediction using the SageMaker endpoint
        logger.info("Sending request to SageMaker endpoint")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        logger.info(f"Received prediction: {result}")
        return result[0]
    except Exception as e:
        logger.error(f"An error occurred during prediction: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Indicadores de Ataque Cardíaco", page_icon="❤️", layout="wide")
    
    st.title("Indicadores de Ataque Cardíaco ❤️")
    st.write("Esta aplicação prevê o risco de um ataque cardíaco com base em diversos fatores de saúde e estilo de vida.")

    input_data = {feature: None for feature in FEATURE_ORDER}

    tabs = st.tabs(["Informações Pessoais", "Estilo de Vida", "Histórico de Saúde", "Procedimentos Médicos", "Fatores Adicionais"])

    with tabs[0]:
        st.header("Informações Pessoais")
        col1, col2 = st.columns(2)
        with col1:
            input_data['State'] = st.selectbox('Estado:', features['State'])
            sex_map = {'Feminino': 'Female', 'Masculino': 'Male'}
            input_data['Sex'] = sex_map[st.radio('Sexo:', ['Feminino', 'Masculino'])]
            age_map = {
                '18 a 24 anos': 'Age 18 to 24', '25 a 29 anos': 'Age 25 to 29', '30 a 34 anos': 'Age 30 to 34',
                '35 a 39 anos': 'Age 35 to 39', '40 a 44 anos': 'Age 40 to 44', '45 a 49 anos': 'Age 45 to 49',
                '50 a 54 anos': 'Age 50 to 54', '55 a 59 anos': 'Age 55 to 59', '60 a 64 anos': 'Age 60 to 64',
                '65 a 69 anos': 'Age 65 to 69', '70 a 74 anos': 'Age 70 to 74', '75 a 79 anos': 'Age 75 to 79',
                '80 anos ou mais': 'Age 80 or older'
            }
            input_data['AgeCategory'] = age_map[st.selectbox('Faixa Etária:', list(age_map.keys()))]
        with col2:
            race_map = {
                'Apenas Negro, Não-Hispânico': 'Black only, Non-Hispanic',
                'Hispânico': 'Hispanic',
                'Multirracial, Não-Hispânico': 'Multiracial, Non-Hispanic',
                'Apenas outra raça, Não-Hispânico': 'Other race only, Non-Hispanic',
                'Apenas Branco, Não-Hispânico': 'White only, Non-Hispanic'
            }
            input_data['RaceEthnicityCategory'] = race_map[st.selectbox('Raça/Etnia:', list(race_map.keys()))]
            input_data['HeightInMeters'] = st.number_input('Altura (em metros):', min_value=0.0, max_value=3.0, value=1.7, step=0.01)
            input_data['WeightInKilograms'] = st.number_input('Peso (em quilogramas):', min_value=0.0, max_value=500.0, value=70.0, step=0.1)

    with tabs[1]:
        st.header("Estilo de Vida e Saúde Geral")
        col1, col2 = st.columns(2)
        with col1:
            health_map = {'Ruim': 'Poor', 'Regular': 'Fair', 'Boa': 'Good', 'Muito boa': 'Very good', 'Excelente': 'Excellent'}
            input_data['GeneralHealth'] = health_map[st.select_slider('Saúde Geral:', options=list(health_map.keys()))]
            input_data['PhysicalHealthDays'] = st.slider('Dias de atividade física no último mês:', 0, 30, 0)
            input_data['MentalHealthDays'] = st.slider('Dias dedicados à saúde mental no último mês:', 0, 30, 0)
            input_data['SleepHours'] = st.slider('Média de horas de sono por dia:', 0, 24, 7)
        with col2:
            input_data['PhysicalActivities'] = 'Yes' if st.radio('Você pratica atividades físicas?', ['Não', 'Sim']) == 'Sim' else 'No'
            smoker_map = {
                'Fumante atual - fuma todos os dias': 'Current smoker - now smokes every day',
                'Fumante atual - fuma alguns dias': 'Current smoker - now smokes some days',
                'Ex-fumante': 'Former smoker',
                'Nunca fumou': 'Never smoked'
            }
            input_data['SmokerStatus'] = smoker_map[st.selectbox('Status de fumante:', list(smoker_map.keys()))]
            ecigarette_map = {
                'Nunca usou cigarros eletrônicos': 'Never used e-cigarettes in my entire life',
                'Não usa atualmente': 'Not at all (right now)',
                'Usa todos os dias': 'Use them every day',
                'Usa alguns dias': 'Use them some days'
            }
            input_data['ECigaretteUsage'] = ecigarette_map[st.selectbox('Uso de cigarros eletrônicos:', list(ecigarette_map.keys()))]
            input_data['AlcoholDrinkers'] = 'Yes' if st.radio('Você consome bebidas alcoólicas?', ['Não', 'Sim']) == 'Sim' else 'No'

    with tabs[2]:
        st.header("Histórico de Saúde")
        st.subheader("Você já teve alguma das seguintes condições?")
        st.write("Marque a caixa se você foi diagnosticado ou experimentou alguma dessas condições de saúde:")
        
        col1, col2, col3 = st.columns(3)
        conditions = [
            ('Angina', 'Angina'),
            ('Stroke', 'Derrame'),
            ('Asthma', 'Asma'),
            ('SkinCancer', 'Câncer de Pele'),
            ('COPD', 'DPOC'),
            ('DepressiveDisorder', 'Transtorno Depressivo'),
            ('KidneyDisease', 'Doença Renal'),
            ('Arthritis', 'Artrite')
        ]
        
        for i, (eng, ptbr) in enumerate(conditions):
            with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                input_data[f'Had{eng}'] = 'Yes' if st.checkbox(ptbr) else 'No'

        st.subheader("Outras Informações de Saúde")
        col1, col2 = st.columns(2)
        with col1:
            diabetes_map = {
                'Não': 'No',
                'Não, pré-diabetes ou diabetes limítrofe': 'No, pre-diabetes or borderline diabetes',
                'Sim': 'Yes',
                'Sim, mas apenas durante a gravidez (feminino)': 'Yes, but only during pregnancy (female)'
            }
            input_data['HadDiabetes'] = diabetes_map[st.selectbox("Status de diabetes:", list(diabetes_map.keys()))]
            teeth_map = {
                '1 a 5': '1 to 5',
                '6 ou mais, mas não todos': '6 or more, but not all',
                'Todos': 'All',
                'Nenhum': 'None of them'
            }
            input_data['RemovedTeeth'] = teeth_map[st.selectbox('Quantos dentes foram removidos?', list(teeth_map.keys()))]
        with col2:
            input_data['DeafOrHardOfHearing'] = 'Yes' if st.radio('Você é surdo ou tem séria dificuldade para ouvir?', ['Não', 'Sim']) == 'Sim' else 'No'
            input_data['BlindOrVisionDifficulty'] = 'Yes' if st.radio('Você é cego ou tem séria dificuldade para enxergar?', ['Não', 'Sim']) == 'Sim' else 'No'
        
        st.subheader("Dificuldade com Atividades Diárias")
        col1, col2 = st.columns(2)
        with col1:
            input_data['DifficultyConcentrating'] = 'Yes' if st.radio('Você tem séria dificuldade de concentração, memória ou tomada de decisões?', ['Não', 'Sim']) == 'Sim' else 'No'
            input_data['DifficultyWalking'] = 'Yes' if st.radio('Você tem séria dificuldade para andar ou subir escadas?', ['Não', 'Sim']) == 'Sim' else 'No'
        with col2:
            input_data['DifficultyDressingBathing'] = 'Yes' if st.radio('Você tem dificuldade para se vestir ou tomar banho?', ['Não', 'Sim']) == 'Sim' else 'No'
            input_data['DifficultyErrands'] = 'Yes' if st.radio('Você tem dificuldade para fazer tarefas sozinho?', ['Não', 'Sim']) == 'Sim' else 'No'

    with tabs[3]:
        st.header("Procedimentos Médicos e Exames")
        col1, col2 = st.columns(2)
        with col1:
            checkup_map = {
                '5 ou mais anos atrás': '5 or more years ago',
                'Nos últimos 2 anos (1 ano, mas menos de 2 anos atrás)': 'Within past 2 years (1 year but less than 2 years ago)',
                'Nos últimos 5 anos (2 anos, mas menos de 5 anos atrás)': 'Within past 5 years (2 years but less than 5 years ago)',
                'No último ano (menos de 12 meses atrás)': 'Within past year (anytime less than 12 months ago)'
            }
            input_data['LastCheckupTime'] = checkup_map[st.selectbox('Tempo desde o último check-up:', list(checkup_map.keys()))]
            input_data['ChestScan'] = 'Yes' if st.radio('Você já fez uma tomografia de tórax para câncer de pulmão?', ['Não', 'Sim']) == 'Sim' else 'No'
            input_data['HIVTesting'] = 'Yes' if st.radio('Você já fez o teste de HIV?', ['Não', 'Sim']) == 'Sim' else 'No'
        with col2:
            input_data['FluVaxLast12'] = 'Yes' if st.radio('Você tomou vacina contra gripe nos últimos 12 meses?', ['Não', 'Sim']) == 'Sim' else 'No'
            input_data['PneumoVaxEver'] = 'Yes' if st.radio('Você já tomou vacina contra pneumonia?', ['Não', 'Sim']) == 'Sim' else 'No'
            tetanus_map = {
                'Não recebeu nenhuma vacina contra tétano nos últimos 10 anos': 'No, did not receive any tetanus shot in the past 10 years',
                'Sim, recebeu Tdap': 'Yes, received Tdap',
                'Sim, recebeu vacina contra tétano, mas não tem certeza do tipo': 'Yes, received tetanus shot but not sure what type',
                'Sim, recebeu vacina contra tétano, mas não Tdap': 'Yes, received tetanus shot, but not Tdap'
            }
            input_data['TetanusLast10Tdap'] = tetanus_map[st.selectbox('Vacina contra tétano/Tdap nos últimos 10 anos:', list(tetanus_map.keys()))]

    with tabs[4]:
        st.header("Fatores de Risco Adicionais")
        col1, col2 = st.columns(2)
        with col1:
            input_data['HighRiskLastYear'] = 'Yes' if st.radio('Você esteve em alto risco para COVID-19 no último ano?', ['Não', 'Sim']) == 'Sim' else 'No'
        with col2:
            covid_map = {
                'Não': 'No',
                'Testou positivo usando teste caseiro sem um profissional de saúde': 'Tested positive using home test without a health professional',
                'Sim': 'Yes'
            }
            input_data['CovidPos'] = covid_map[st.selectbox('Você testou positivo para COVID-19?', list(covid_map.keys()))]

    st.header("Prever Risco de Ataque Cardíaco")
    if st.button('Prever', type='primary'):
        with st.spinner('Calculando risco...'):
            # Ensure input_data is in the correct order before calling predict
            ordered_input = {feature: input_data[feature] for feature in FEATURE_ORDER}
            prediction = predict(ordered_input)
        
        if prediction == 'Yes':
            st.error("⚠️ Alto Risco de Ataque Cardíaco")
            st.write("Com base nas informações fornecidas, você pode estar em maior risco de um ataque cardíaco. Por favor, consulte um profissional de saúde para uma avaliação completa e aconselhamento personalizado.")
        elif prediction == 'No':
            st.success("✅ Baixo Risco de Ataque Cardíaco")
            st.write("Com base nas informações fornecidas, você parece estar em menor risco de um ataque cardíaco. No entanto, é sempre uma boa ideia manter um estilo de vida saudável e consultar profissionais de saúde regularmente.")
        else:
            st.warning("⚠️ Não foi possível determinar o risco")
            st.write("Houve um problema ao calcular seu risco. Por favor, verifique suas entradas e tente novamente.")

    # Adicionando informações sobre os criadores
    with st.sidebar:
        st.markdown("## Criadores")
        st.markdown("""
        
        Curso: **Deep Learning com TensorFlow [24E3_3] - Infnet**
        
        **GRUPO 1:**
        - Anderson Oliveira Vaz
        - Elisandro Murliky
        - Lucas Maia Moreira
        - Rafael Leal Zimmer
        - Valdeci Gomes
        """)

if __name__ == "__main__":
    main()