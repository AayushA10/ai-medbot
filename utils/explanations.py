# utils/explanations.py

EXPLANATIONS = {
    "Eczema": "Eczema causes itchy, inflamed skin. Moisturize frequently and avoid irritants. Hydrocortisone cream may help.",
    "Psoriasis": "Psoriasis leads to red, scaly skin patches. It's autoimmune. Topical steroids and light therapy are often prescribed.",
    "Melanoma": "Melanoma is a dangerous skin cancer. Urgent medical consultation is required for biopsy and treatment.",
    "Ringworm": "Ringworm is a fungal infection with ring-shaped rash. Apply antifungal cream like clotrimazole twice daily."
}

def get_explanation(disease):
    return EXPLANATIONS.get(disease, "No explanation available for this condition.")
