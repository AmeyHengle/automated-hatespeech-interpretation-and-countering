import os
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
TEMPERATURE_SCALE = 0
model_name = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

system_description_base_cs = """
Given a hate speech statement, write an 'Informative' counter-narrative using the given counter fact. Don't use strong language, stick to facts.
Output must only contain the counter-narrative, and should be less that 150 words.
""".strip()

system_description_base_exp = """
Given a statement, answer each of the following questions:
Q1: Is the statement offensive? Answer in a single word only. 
Q2: Is there any social or demographic group targeted by the statement? Answer in two words only. 
Q3: What is the implied meaning, bias, or stereotype expressed by the statement? 

Please output strictly in json format: 
{
A1: ''
A2: ''
A3:''
}
""".strip()

prompt_template_cs = lambda hatespeech, counter_fact: f"""
statement: {hatespeech}
counter-fact: {counter_fact}
counter-narrative:
""".strip()

prompt_template_exp = lambda hatespeech: f"""
statement: {hatespeech}
""".strip()

def extract_content_from_response(response):
    try:
        response = response.choices[0].message.content
        response = response.strip().replace('\n','')
        return response
    except:
        print(response)
        return ''

def generate_counternarrative(hatespeech, counter_fact, system_description=system_description_base_cs):
    input_prompt = prompt_template_cs(hatespeech, counter_fact)
    print(f"Input: {input_prompt}")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_description},
            {"role": "user", "content": input_prompt},
        ],
        temperature=TEMPERATURE_SCALE,
        max_tokens=80,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        logprobs=False,
        n=1,
    )

    return extract_content_from_response(response)


def generate_hs_explanations(hatespeech, system_description=system_description_base_exp):
    input_prompt = prompt_template_exp(hatespeech)
    print(f"Input: {input_prompt}")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_description},
            {"role": "user", "content": input_prompt},
        ],
        temperature=TEMPERATURE_SCALE,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        logprobs=False,
        n=1,
    )

    return extract_content_from_response(response)