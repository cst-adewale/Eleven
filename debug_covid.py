from model import perform_inference

print("Testing COVID-19 profile: Fever + Cough + Loss of Taste")
evidence = {
    'Fever': True,
    'Cough': True,
    'LossOfTaste': True
}
probs = perform_inference(evidence)
sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

with open("debug_log.txt", "w") as f:
    f.write("Testing COVID-19 profile: Fever + Cough + Loss of Taste\n")
    for name, p in sorted_probs:
        f.write(f"{name}: {p:.4f}\n")
    f.write("\nDone.")
print("Results written to debug_log.txt")
