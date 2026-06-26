import json

# Load the full STIX bundle.
with open("enterprise-attack.json", "r") as f:
    data = json.load(f)

techniques = []

for obj in data["objects"]:
    # We only want techniques. In STIX, a technique is an "attack-pattern".
    if obj.get("type") != "attack-pattern":
        continue
    # Skip techniques MITRE has retired — we don't want outdated data.
    if obj.get("revoked") or obj.get("x_mitre_deprecated"):
        continue

    # The ATT&CK ID (e.g. T1547.001) lives in external_references.
    attack_id = None
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            attack_id = ref.get("external_id")
            break

    name = obj.get("name", "")
    description = obj.get("description", "")
    detection = obj.get("x_mitre_detection", "")

    # Build one clean text document per technique.
    text = f"{attack_id} — {name}\n\n{description}"
    if detection:
        text += f"\n\nDetection: {detection}"

    techniques.append({
        "id": attack_id,
        "name": name,
        "text": text,
    })

# Save the parsed techniques for the next stage (embedding).
with open("techniques.json", "w") as f:
    json.dump(techniques, f)

print(f"Parsed {len(techniques)} techniques.")
print("\n--- Example document ---\n")
print(techniques[0]["text"][:500])