from model import perform_inference

def test_covid_symptoms():
    print("Testing COVID-19 symptoms: Fever + Cough + Loss of Taste")
    evidence = {
        'Fever': 0,
        'Cough': 0,
        'LossOfTaste': 0,
        'Fatigue': 0,
        'Sneeze': 1
    }
    probs = perform_inference(evidence)
    print(f"Probabilities: {probs}")
    assert probs['COVID-19'] > probs['Flu']
    assert probs['COVID-19'] > probs['Allergy']
    print("COVID test passed!")

def test_allergy_symptoms():
    print("\nTesting Allergy symptoms: Sneeze (No Fever, No Cough)")
    evidence = {
        'Fever': 1,
        'Cough': 1,
        'LossOfTaste': 1,
        'Fatigue': 1,
        'Sneeze': 0
    }
    probs = perform_inference(evidence)
    print(f"Probabilities: {probs}")
    assert probs['Allergy'] > probs['COVID-19']
    assert probs['Allergy'] > probs['Flu']
    print("Allergy test passed!")

if __name__ == "__main__":
    try:
        test_covid_symptoms()
        test_allergy_symptoms()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
