from model import perform_inference

def test_covid_symptoms():
    print("Testing COVID-19 profile: Fever + Cough + Loss of Taste")
    evidence = {
        'Fever': True,
        'Cough': True,
        'LossOfTaste': True
    }
    probs = perform_inference(evidence)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted Top 5 Results: {sorted_probs[:5]}")
    assert sorted_probs[0][0] == 'COVID-19'
    print("COVID test passed!")

def test_gastroenteritis():
    print("\nTesting Gastroenteritis: Nausea + Vomiting + Diarrhea")
    evidence = {
        'Nausea': True,
        'Vomiting': True,
        'Diarrhea': True
    }
    probs = perform_inference(evidence)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted Top 5 Results: {sorted_probs[:5]}")
    assert sorted_probs[0][0] == 'Gastroenteritis (Stomach Flu)'
    print("Gastroenteritis test passed!")

def test_mono():
    print("\nTesting Infectious Mono: Extreme Fatigue + Fever + Swollen Nodes")
    evidence = {
        'Fatigue': True,
        'Fever': True,
        'SwollenLymphNodes': True
    }
    probs = perform_inference(evidence)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted Top 5 Results: {sorted_probs[:5]}")
    assert sorted_probs[0][0] == 'Infectious Mononucleosis'
    print("Mono test passed!")

def test_migraine():
    print("\nTesting Migraine: Headache + Nausea + Dizziness")
    evidence = {
        'Headache': True,
        'Nausea': True,
        'Dizziness': True
    }
    probs = perform_inference(evidence)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted Top 5 Results: {sorted_probs[:5]}")
    assert sorted_probs[0][0] == 'Migraine'
    print("Migraine test passed!")

def test_pertussis():
    print("\nTesting Pertussis: Severe Cough + Wheezing + Shortness of Breath")
    evidence = {
        'Cough': True,
        'Wheezing': True,
        'ShortnessOfBreath': True
    }
    probs = perform_inference(evidence)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted Top 5 Results: {sorted_probs[:5]}")
    assert sorted_probs[0][0] == 'Pertussis (Whooping Cough)'
    print("Pertussis test passed!")

if __name__ == "__main__":
    with open("test_results.txt", "w") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        try:
            log("--- Starting 15-Ailment Model Tests ---")
            
            # Test cases
            for test_func in [test_covid_symptoms, test_gastroenteritis, test_mono, test_migraine, test_pertussis]:
                log(f"\nRunning {test_func.__name__}...")
                test_func()
                log(f"{test_func.__name__} passed!")

            log("\n--- All tests passed successfully! ---")
        except Exception as e:
            log(f"\nTests failed: {e}")
            import traceback
            traceback.print_exc(file=f)
            traceback.print_exc()
