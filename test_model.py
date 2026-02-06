from model import perform_inference

def test_covid_symptoms():
    print("Testing COVID-19 profile: Fever + Cough + Loss of Taste")
    evidence = {
        'Fever': True,
        'Cough': True,
        'LossOfTaste': True
    }
    probs = perform_inference(evidence)
    print(f"Top 3: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
    assert probs['COVID-19'] > 0.4
    print("COVID test passed!")

def test_strep_throat():
    print("\nTesting Strep Throat profile: Fever + Sore Throat + Swollen Lymph Nodes")
    evidence = {
        'Fever': True,
        'SoreThroat': True,
        'SwollenLymphNodes': True
    }
    probs = perform_inference(evidence)
    print(f"Top 3: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
    assert probs['Strep Throat'] > probs['Common Cold']
    print("Strep Throat test passed!")

def test_pneumonia():
    print("\nTesting Pneumonia profile: High Fever + Shortness of Breath + Chest Pain + Cough")
    evidence = {
        'Fever': True,
        'ShortnessOfBreath': True,
        'ChestPain': True,
        'Cough': True
    }
    probs = perform_inference(evidence)
    print(f"Top 3: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
    assert probs['Bacterial Pneumonia'] > probs['Common Cold']
    print("Pneumonia test passed!")

if __name__ == "__main__":
    try:
        test_covid_symptoms()
        test_strep_throat()
        test_pneumonia()
        print("\nAll expanded model tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
