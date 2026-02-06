from model import perform_inference

print("Testing COVID-19 profile: Fever + Cough + Loss of Taste")
evidence = {
    'Fever': True,
    'Cough': True,
    'LossOfTaste': True
}
probs = perform_inference(evidence)
sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
for name, p in sorted_probs[:5]:
    print(f"{name}: {p:.4f}")
