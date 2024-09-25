# Replace this with your actual model code
import time
import json
from inference_api import generate_counternarrative, generate_hs_explanations
from semantic_similarity import find_most_similar_fact
from inference_custom import CustomLLMInference
model = CustomLLMInference('t5-small')

def analyze_statement(statement, return_dummy_values=False):
    time.sleep(1)
    if return_dummy_values:
     return {
        "nature": "dummy value for nature",
        "group": "dummy value for group",
        "bias": "dummy value for bias"
    }

    explanations = generate_hs_explanations(statement)
    explanations = json.loads(explanations) if isinstance(explanations, str) else explanations
    
    return {
        "nature": explanations['A1'],
        "group": explanations['A2'],
        "bias": explanations['A3']
    }

# Dummy function to generate counterspeech for the given statement
def generate_counterspeech(statement, inference_type="api", return_dummy_values=False):
    if return_dummy_values:
       return 'dummy counterspeech', 'dummy fact'

    if inference_type == "api":
        fact, similarity_score = find_most_similar_fact(statement)
        fact = fact['fact']
        response  = generate_counternarrative(hatespeech=statement, counter_fact=fact)
        return response, f"{fact}\n(Retrieval Confidence: {similarity_score*100}%)"
    elif inference_type=="custom":
       response = model.predict(hatespeech=statement, counter_fact=fact)
    
    return f"The statement is inappropriate and perpetuates harmful gender stereotypes. We should respect individuals regardless of their gender and value their contributions equally."
