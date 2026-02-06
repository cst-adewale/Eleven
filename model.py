from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def get_medical_model():
    # Diseases (States 0-14):
    # 0: COVID-19, 1: Flu, 2: Cold, 3: Allergy, 4: Pneumonia, 5: Bronchitis, 6: Strep Throat, 7: Sinusitis,
    # 8: Infectious Mono, 9: Pertussis, 10: Tuberculosis, 11: Gastroenteritis, 12: Migraine, 13: Ear Infection, 14: Healthy
    
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

    # Disease Prior Probabilities (15 states)
    # COVID, Flu, Cold, Allergy, Pneu, Bronch, Strep, Sinus, Mono, Pert, TB, Gastro, Migraine, Ear, Healthy
    disease_prior = [[0.02], [0.04], [0.12], [0.10], [0.01], [0.02], [0.03], [0.05], 
                    [0.01], [0.005], [0.005], [0.03], [0.01], [0.02], [0.53]]
    cpd_disease = TabularCPD(variable='Disease', variable_card=15, values=disease_prior)

    # Symptom Conditional Probabilities (Prob of "Yes" given Disease)
    # Each list has 15 values corresponding to the 15 disease states above.
    symptom_probs = {
        # Systemic
        "Fever":            [0.85, 0.90, 0.10, 0.01, 0.95, 0.20, 0.75, 0.10, 0.85, 0.10, 0.80, 0.65, 0.01, 0.45, 0.01],
        "Fatigue":          [0.85, 0.90, 0.30, 0.10, 0.85, 0.50, 0.40, 0.20, 0.98, 0.40, 0.85, 0.65, 0.35, 0.25, 0.05],
        "Chills":           [0.45, 0.65, 0.05, 0.01, 0.75, 0.10, 0.35, 0.05, 0.45, 0.05, 0.55, 0.45, 0.01, 0.15, 0.01],
        "BodyAches":        [0.65, 0.85, 0.20, 0.05, 0.55, 0.20, 0.25, 0.10, 0.65, 0.10, 0.45, 0.35, 0.20, 0.05, 0.01],
        "LossOfAppetite":   [0.45, 0.55, 0.10, 0.05, 0.65, 0.20, 0.35, 0.10, 0.75, 0.20, 0.65, 0.95, 0.10, 0.15, 0.01],
        "NightSweats":      [0.25, 0.15, 0.01, 0.01, 0.45, 0.05, 0.15, 0.02, 0.55, 0.05, 0.95, 0.05, 0.005, 0.02, 0.001],
        "SwollenLymphNodes":[0.10, 0.05, 0.05, 0.02, 0.15, 0.05, 0.85, 0.25, 0.98, 0.02, 0.35, 0.05, 0.01, 0.25, 0.005],
        
        # Respiratory
        "Cough":            [0.80, 0.85, 0.65, 0.10, 0.90, 0.95, 0.15, 0.25, 0.25, 0.98, 0.95, 0.05, 0.01, 0.02, 0.01],
        "ShortnessOfBreath":[0.60, 0.25, 0.02, 0.01, 0.85, 0.35, 0.01, 0.01, 0.05, 0.45, 0.70, 0.01, 0.001, 0.001, 0.001],
        "SoreThroat":       [0.25, 0.35, 0.75, 0.10, 0.15, 0.45, 0.95, 0.35, 0.85, 0.25, 0.35, 0.05, 0.01, 0.30, 0.01],
        "Hoarseness":       [0.05, 0.15, 0.35, 0.05, 0.10, 0.65, 0.45, 0.10, 0.25, 0.15, 0.35, 0.01, 0.01, 0.05, 0.005],
        "Wheezing":         [0.10, 0.20, 0.05, 0.25, 0.45, 0.75, 0.05, 0.05, 0.05, 0.85, 0.35, 0.01, 0.01, 0.02, 0.005],
        "ChestPain":        [0.40, 0.20, 0.05, 0.10, 0.75, 0.45, 0.05, 0.05, 0.25, 0.45, 0.65, 0.05, 0.01, 0.05, 0.005],
        
        # Nasal/Sinus
        "Sneeze":           [0.05, 0.10, 0.85, 0.95, 0.01, 0.05, 0.05, 0.25, 0.05, 0.05, 0.05, 0.01, 0.02, 0.05, 0.02],
        "RunnyNose":        [0.15, 0.25, 0.90, 0.85, 0.05, 0.15, 0.10, 0.80, 0.15, 0.10, 0.20, 0.05, 0.01, 0.25, 0.02],
        "NasalCongestion":  [0.15, 0.25, 0.85, 0.90, 0.05, 0.15, 0.10, 0.95, 0.15, 0.10, 0.20, 0.02, 0.01, 0.30, 0.02],
        "SinusPressure":    [0.05, 0.10, 0.35, 0.45, 0.02, 0.05, 0.05, 0.98, 0.05, 0.05, 0.05, 0.01, 0.20, 0.10, 0.005],
        "EarAche":          [0.01, 0.05, 0.25, 0.15, 0.05, 0.05, 0.15, 0.45, 0.15, 0.02, 0.05, 0.01, 0.15, 0.98, 0.001],
        
        # Digestive
        "Nausea":           [0.30, 0.35, 0.05, 0.01, 0.25, 0.05, 0.15, 0.05, 0.25, 0.05, 0.25, 0.95, 0.45, 0.15, 0.005],
        "Vomiting":         [0.15, 0.25, 0.02, 0.01, 0.15, 0.02, 0.05, 0.02, 0.05, 0.02, 0.15, 0.90, 0.35, 0.10, 0.001],
        "Diarrhea":         [0.20, 0.25, 0.02, 0.01, 0.05, 0.02, 0.02, 0.01, 0.05, 0.01, 0.05, 0.85, 0.01, 0.02, 0.001],
        
        # Neuro/Sensory
        "Headache":         [0.55, 0.75, 0.35, 0.25, 0.45, 0.25, 0.25, 0.85, 0.65, 0.25, 0.45, 0.25, 0.99, 0.35, 0.02],
        "Dizziness":        [0.25, 0.30, 0.05, 0.05, 0.25, 0.15, 0.05, 0.15, 0.35, 0.15, 0.25, 0.35, 0.65, 0.45, 0.005],
        "Confusion":        [0.20, 0.15, 0.01, 0.01, 0.35, 0.05, 0.01, 0.01, 0.15, 0.05, 0.45, 0.05, 0.20, 0.01, 0.001],
        "LossOfTaste":      [0.75, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10, 0.05, 0.01, 0.01, 0.15, 0.02, 0.01, 0.001],
        "LossOfSmell":      [0.80, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.15, 0.05, 0.01, 0.01, 0.05, 0.02, 0.01, 0.001],
        "JointPain":        [0.25, 0.35, 0.15, 0.25, 0.20, 0.15, 0.05, 0.05, 0.45, 0.15, 0.35, 0.05, 0.05, 0.01, 0.01],
        "MuscleStiffness":  [0.25, 0.40, 0.15, 0.05, 0.20, 0.15, 0.05, 0.05, 0.35, 0.15, 0.35, 0.05, 0.45, 0.01, 0.01],
        
        # Ocular/Other
        "ItchyEyes":        [0.01, 0.01, 0.05, 0.95, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.15, 0.05, 0.002],
        "WateryEyes":       [0.01, 0.01, 0.15, 0.85, 0.01, 0.01, 0.01, 0.10, 0.05, 0.01, 0.05, 0.01, 0.20, 0.05, 0.002],
        "Rashes":           [0.05, 0.05, 0.01, 0.01, 0.15, 0.05, 0.35, 0.01, 0.50, 0.01, 0.25, 0.02, 0.01, 0.15, 0.005]
    }

    cpds = [cpd_disease]
    for sym, probs in symptom_probs.items():
        yes_probs = probs
        no_probs = [1.0 - p for p in probs]
        
        cpd = TabularCPD(
            variable=sym, 
            variable_card=2,
            values=[yes_probs, no_probs],
            evidence=['Disease'], 
            evidence_card=[15]
        )
        cpds.append(cpd)

    model.add_cpds(*cpds)
    assert model.check_model()
    return model

def perform_inference(evidence):
    model = get_medical_model()
    inference = VariableElimination(model)
    
    processed_evidence = {k: (0 if v else 1) for k, v in evidence.items() if k in model.nodes()}
    result = inference.query(variables=['Disease'], evidence=processed_evidence)
    
    ailment_names = [
        'COVID-19', 'Influenza (Flu)', 'Common Cold', 'Allergic Rhinitis', 
        'Bacterial Pneumonia', 'Viral Bronchitis', 'Strep Throat', 'Acute Sinusitis',
        'Infectious Mononucleosis', 'Pertussis (Whooping Cough)', 'Tuberculosis', 
        'Gastroenteritis (Stomach Flu)', 'Migraine', 'Otitis Media (Ear Infection)', 'Healthy'
    ]
    
    probabilities = {name: float(prob) for name, prob in zip(ailment_names, result.values)}
    return probabilities
