from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def get_medical_model():
    # Diseases (States 0-8):
    # 0: COVID-19, 1: Flu, 2: Cold, 3: Allergy, 4: Pneumonia, 
    # 5: Bronchitis, 6: Strep Throat, 7: Sinusitis, 8: Healthy
    
    symptoms = [
        "Fever", "Fatigue", "Chills", "BodyAches", "LossOfAppetite", "NightSweats", "SwollenLymphNodes",
        "Cough", "ShortnessOfBreath", "SoreThroat", "Hoarseness", "Wheezing", "ChestPain",
        "Sneeze", "RunnyNose", "NasalCongestion", "SinusPressure", "EarAche",
        "Nausea", "Vomiting", "Diarrhea",
        "Headache", "Dizziness", "Confusion", "LossOfTaste", "LossOfSmell", "JointPain", "MuscleStiffness",
        "ItchyEyes", "WateryEyes", "Rashes"
    ]

    edges = [('Disease', s) for s in symptoms]
    model = DiscreteBayesianNetwork(edges)

    # Disease Prior Probabilities (Ailments)
    # Shape must be (card, 1) if no evidence: [[p1], [p2], ...]
    disease_prior = [[0.02], [0.05], [0.15], [0.10], [0.01], [0.02], [0.03], [0.05], [0.57]]
    cpd_disease = TabularCPD(variable='Disease', variable_card=9, values=disease_prior)

    # Symptom Conditional Probabilities (Prob of "Yes" given Disease)
    # Format: {Symptom: [P(S=Yes|D=0), P(S=Yes|D=1), ..., P(S=Yes|D=8)]}
    symptom_probs = {
        # Systemic
        "Fever":            [0.80, 0.85, 0.10, 0.01, 0.90, 0.20, 0.70, 0.10, 0.01],
        "Fatigue":          [0.85, 0.90, 0.30, 0.10, 0.80, 0.50, 0.40, 0.20, 0.05],
        "Chills":           [0.40, 0.60, 0.05, 0.01, 0.70, 0.10, 0.30, 0.05, 0.01],
        "BodyAches":        [0.60, 0.80, 0.20, 0.05, 0.50, 0.20, 0.20, 0.10, 0.02],
        "LossOfAppetite":   [0.40, 0.50, 0.10, 0.05, 0.60, 0.20, 0.30, 0.10, 0.02],
        "NightSweats":      [0.20, 0.10, 0.01, 0.01, 0.40, 0.05, 0.10, 0.02, 0.005],
        "SwollenLymphNodes":[0.10, 0.05, 0.05, 0.02, 0.10, 0.05, 0.80, 0.20, 0.01],
        
        # Respiratory
        "Cough":            [0.70, 0.75, 0.60, 0.10, 0.85, 0.90, 0.10, 0.20, 0.02],
        "ShortnessOfBreath":[0.50, 0.20, 0.02, 0.01, 0.80, 0.30, 0.01, 0.01, 0.001],
        "SoreThroat":       [0.20, 0.30, 0.70, 0.10, 0.10, 0.40, 0.90, 0.30, 0.02],
        "Hoarseness":       [0.05, 0.10, 0.30, 0.05, 0.05, 0.60, 0.40, 0.10, 0.01],
        "Wheezing":         [0.10, 0.15, 0.05, 0.20, 0.40, 0.70, 0.02, 0.05, 0.01],
        "ChestPain":        [0.30, 0.15, 0.05, 0.10, 0.70, 0.40, 0.05, 0.05, 0.01],
        
        # Nasal/Sinus
        "Sneeze":           [0.05, 0.10, 0.80, 0.90, 0.01, 0.05, 0.02, 0.20, 0.05],
        "RunnyNose":        [0.10, 0.20, 0.85, 0.80, 0.02, 0.10, 0.05, 0.75, 0.03],
        "NasalCongestion":  [0.10, 0.20, 0.80, 0.85, 0.02, 0.10, 0.05, 0.90, 0.03],
        "SinusPressure":    [0.05, 0.10, 0.30, 0.40, 0.02, 0.05, 0.02, 0.95, 0.01],
        "EarAche":          [0.01, 0.05, 0.20, 0.10, 0.02, 0.05, 0.10, 0.40, 0.005],
        
        # Digestive
        "Nausea":           [0.25, 0.30, 0.05, 0.01, 0.20, 0.05, 0.10, 0.05, 0.01],
        "Vomiting":         [0.10, 0.20, 0.02, 0.01, 0.10, 0.02, 0.05, 0.02, 0.005],
        "Diarrhea":         [0.15, 0.20, 0.02, 0.01, 0.05, 0.02, 0.02, 0.01, 0.01],
        
        # Neuro/Sensory
        "Headache":         [0.50, 0.70, 0.30, 0.20, 0.40, 0.20, 0.20, 0.80, 0.05],
        "Dizziness":        [0.20, 0.25, 0.05, 0.05, 0.20, 0.10, 0.05, 0.10, 0.01],
        "Confusion":        [0.15, 0.10, 0.01, 0.01, 0.30, 0.05, 0.01, 0.01, 0.001],
        "LossOfTaste":      [0.60, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10, 0.001],
        "LossOfSmell":      [0.65, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.15, 0.001],
        "JointPain":        [0.20, 0.30, 0.10, 0.20, 0.15, 0.10, 0.05, 0.05, 0.02],
        "MuscleStiffness":  [0.20, 0.35, 0.10, 0.05, 0.15, 0.10, 0.05, 0.05, 0.02],
        
        # Ocular/Other
        "ItchyEyes":        [0.01, 0.01, 0.05, 0.85, 0.01, 0.01, 0.01, 0.05, 0.005],
        "WateryEyes":       [0.01, 0.01, 0.10, 0.75, 0.01, 0.01, 0.01, 0.10, 0.005],
        "Rashes":           [0.05, 0.05, 0.01, 0.01, 0.10, 0.02, 0.30, 0.01, 0.01]
    }

    cpds = [cpd_disease]
    for sym, probs in symptom_probs.items():
        # Values for "Yes" (row 0) and "No" (row 1)
        # TabularCPD expects a list of lists: [[Prob(Yes|D0), Prob(Yes|D1)...], [Prob(No|D0), Prob(No|D1)...]]
        yes_probs = probs
        no_probs = [1.0 - p for p in probs]
        
        cpd = TabularCPD(
            variable=sym, 
            variable_card=2,
            values=[yes_probs, no_probs],
            evidence=['Disease'], 
            evidence_card=[9]
        )
        cpds.append(cpd)

    model.add_cpds(*cpds)
    assert model.check_model()
    return model

def perform_inference(evidence):
    model = get_medical_model()
    inference = VariableElimination(model)
    
    # Map raw evidence (True/False) to 0/1 for the model
    # Model expects 0 for 'Yes', 1 for 'No'
    processed_evidence = {k: (0 if v else 1) for k, v in evidence.items() if k in model.nodes()}
    
    result = inference.query(variables=['Disease'], evidence=processed_evidence)
    
    ailment_names = [
        'COVID-19', 'Influenza (Flu)', 'Common Cold', 'Allergic Rhinitis', 
        'Bacterial Pneumonia', 'Viral Bronchitis', 'Strep Throat', 'Acute Sinusitis', 'Healthy'
    ]
    
    probabilities = {name: float(prob) for name, prob in zip(ailment_names, result.values)}
    return probabilities
