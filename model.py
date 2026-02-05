from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def get_medical_model():
    # Define the structure: Disease -> Symptoms
    model = DiscreteBayesianNetwork([
        ('Disease', 'Fever'),
        ('Disease', 'Cough'),
        ('Disease', 'Fatigue'),
        ('Disease', 'Sneeze'),
        ('Disease', 'LossOfTaste')
    ])

    # Define CPDs
    # Disease states: 0: COVID-19, 1: Flu, 2: Allergy, 3: Cold, 4: Healthy
    cpd_disease = TabularCPD(
        variable='Disease', 
        variable_card=5, 
        values=[[0.02], [0.05], [0.10], [0.15], [0.68]]
    )

    # Symptom states: 0: Yes, 1: No
    # Order of Disease: COVID, Flu, Allergy, Cold, Healthy
    
    # Fever: High in COVID and Flu
    cpd_fever = TabularCPD(
        variable='Fever', variable_card=2,
        values=[[0.80, 0.70, 0.05, 0.10, 0.01],  # Yes
                [0.20, 0.30, 0.95, 0.90, 0.99]], # No
        evidence=['Disease'], evidence_card=[5]
    )

    # Cough: High in COVID, Flu, Cold
    cpd_cough = TabularCPD(
        variable='Cough', variable_card=2,
        values=[[0.70, 0.60, 0.10, 0.50, 0.02],
                [0.30, 0.40, 0.90, 0.50, 0.98]],
        evidence=['Disease'], evidence_card=[5]
    )

    # Fatigue: High in COVID and Flu
    cpd_fatigue = TabularCPD(
        variable='Fatigue', variable_card=2,
        values=[[0.90, 0.80, 0.10, 0.20, 0.05],
                [0.10, 0.20, 0.90, 0.80, 0.95]],
        evidence=['Disease'], evidence_card=[5]
    )

    # Sneeze: High in Allergy and Cold
    cpd_sneeze = TabularCPD(
        variable='Sneeze', variable_card=2,
        values=[[0.05, 0.10, 0.90, 0.70, 0.05],
                [0.95, 0.90, 0.10, 0.30, 0.95]],
        evidence=['Disease'], evidence_card=[5]
    )

    # Loss of Taste: Very specific to COVID-19
    cpd_loss_of_taste = TabularCPD(
        variable='LossOfTaste', variable_card=2,
        values=[[0.60, 0.01, 0.01, 0.01, 0.001],
                [0.40, 0.99, 0.99, 0.99, 0.999]],
        evidence=['Disease'], evidence_card=[5]
    )

    # Add CPDs to model
    model.add_cpds(cpd_disease, cpd_fever, cpd_cough, cpd_fatigue, cpd_sneeze, cpd_loss_of_taste)
    
    # Validate the model
    assert model.check_model()
    
    return model

def perform_inference(evidence):
    model = get_medical_model()
    inference = VariableElimination(model)
    
    # query results for Disease given symptoms
    result = inference.query(variables=['Disease'], evidence=evidence)
    
    disease_names = ['COVID-19', 'Flu', 'Allergy', 'Common Cold', 'Healthy']
    probabilities = {name: float(prob) for name, prob in zip(disease_names, result.values)}
    
    return probabilities
