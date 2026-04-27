from predictor import Predictor
from llm_advice import generate_advice

predictor = Predictor()

def analyze_skin(image_path):
    disease, confidence = predictor.predict(image_path)

    print(f"\n🔍 Detected Disease: {disease}")
    print(f"Confidence: {confidence:.2f}")
    
    advice = generate_advice(disease)

    if not advice or advice.strip() == "":
        advice = "⚠️ Failed to generate advice. Check LLM configuration."

    print("\n💡 AI Medical Advice:\n")
    print(advice)


if __name__ == "__main__":
    analyze_skin("images/5.jpg")