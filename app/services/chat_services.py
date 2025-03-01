import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def chat(domanda, contesto):
    try:
        risposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sei un assistente utile. Rispondi in base al contesto fornito."},
                {"role": "user", "content": f"Contesto: {contesto}"},
                {"role": "user", "content": f"Domanda: {domanda}"}
            ]
        )
        
        testo_risposta = risposta.choices[0].message.content
        return testo_risposta
    
    except Exception as e:
        return 500