from model import perform_inference

print("Testing Migraine profile: Headache + Nausea + Dizziness")
evidence = {
    'Headache': True,
    'Nausea': True,
    'Dizziness': True
}
probs = perform_inference(evidence)
sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

with open("debug_migraine_log.txt", "w") as f:
    f.write("Testing Migraine profile: Headache + Nausea + Dizziness\n")
    for name, p in sorted_probs:
        f.write(f"{name}: {p:.4f}\n")
    f.write("\nDone.")
print("Results written to debug_migraine_log.txt")
