from src.logic.MovieReviewClassifier import MovieReviewClassifier
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

class Chatbot_Interface:
    def __init__(self,model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 classifier_model_path="./multiNB_model.pkl"):

        self.system_prompt="Please extract the review from this paragraph. Reply with ONLY the review text."

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.classifier = MovieReviewClassifier(classifier_model_path)
        
    def encode_promp(self,prompt):
        return self.tokenizer(prompt,return_tensors="pt")
    

    def decode_promp(self,reply_ids):
        return self.tokenizer.decode(reply_ids,skip_special_tokens=True)
        

    def _generate_llm(self, prompt, max_new_tokens=128, temperature=0.7):
        inputs = self.encode_promp(prompt)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        gen_only = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.decode_promp(gen_only).strip()

    def generate_reply(self, user_message: str) -> str:
        """
        Opción 2:
        - Paso 1: Usar LLM para EXTRAER solo la reseña del mensaje del usuario.
        - Paso 2: Clasificar la reseña con MovieReviewClassifier.
        - Paso 3: Usar LLM para REDACTAR la respuesta final (con reseña + etiqueta).
        """

        
        extraction_prompt = (
            f"{self.system_prompt}\n{user_message}\n\n"  
        )
        extracted_review = self._generate_llm(extraction_prompt)

        extracted_review_vect = self.classifier.vectorizer.transform([extracted_review])
        proba_row = self.classifier.model.predict_proba(extracted_review_vect)[0]
        classes = self.classifier.model.classes_
        print("DEBUG → Classes:", classes)
        print("DEBUG → Probabilities:", proba_row)
        confidence = round((float(proba_row.max())),2)
        sentiment_label = self.classifier.classify_review(extracted_review)
       
        final_prompt = (
            "You are a helpful assistant. "
            "Write a concise, friendly reply to the user about their movie review, "
            "using the extracted review text and the classifier sentiment. "
            "Keep it under 3 sentences and be clear.\n\n"
            f"extracted review text: \"{extracted_review}\"\n\n"
            f"classifier sentiment:  {sentiment_label}\n\n"
            f"confidence: {confidence}\n"
            #f"{user_message}\n"
            
        )
        final_reply = self._generate_llm(final_prompt, max_new_tokens=160, temperature=0.7)
        return final_reply