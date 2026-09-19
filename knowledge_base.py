"""Curated advisory content. This information is separate from model output."""

CROP_KNOWLEDGE_BASE = {
    "Apple___Healthy": {
        "status": "Healthy crop",
        "symptoms": "No disease symptoms identified by this prototype.",
        "preventive_actions": [
            "Continue regular crop scouting.",
            "Maintain field sanitation and balanced irrigation.",
        ],
        "treatment_guidance": "No disease treatment is indicated; keep monitoring the crop.",
        "safety_note": "This is general guidance. Consult a local agricultural extension officer before taking action.",
    },
    "Apple___Black_rot": {
        "status": "Possible black rot",
        "symptoms": "Dark leaf spots and fruit lesions can be associated with black rot.",
        "preventive_actions": [
            "Remove and safely dispose of visibly infected plant material.",
            "Prune to improve airflow and avoid prolonged leaf wetness.",
        ],
        "treatment_guidance": "Seek locally approved disease-management advice before applying any crop-protection product.",
        "safety_note": "Product choice, dosage, and waiting period must be confirmed by a qualified local expert.",
    },
    "Tomato___Late_blight": {
        "status": "Possible late blight",
        "symptoms": "Water-soaked or dark lesions may be associated with late blight.",
        "preventive_actions": [
            "Isolate affected foliage and avoid working in wet plants.",
            "Improve ventilation and remove crop debris responsibly.",
        ],
        "treatment_guidance": "Contact the local agriculture office promptly for an approved integrated disease-management plan.",
        "safety_note": "Do not self-prescribe pesticides. Follow local label, safety, and expert guidance.",
    },
    "Corn___Common_rust": {
        "status": "Possible common rust",
        "symptoms": "Small rust-coloured pustules may occur on leaves.",
        "preventive_actions": [
            "Inspect nearby plants to estimate spread.",
            "Maintain crop nutrition and follow local scouting advice.",
        ],
        "treatment_guidance": "Discuss threshold-based management with a local agricultural expert before treatment.",
        "safety_note": "Use only locally registered products under qualified guidance and follow all label directions.",
    },
}
