from flask import jsonify
from google.cloud import storage
from pydantic import BaseModel
from typing import List
from transformers import T5Tokenizer, T5ForConditionalGeneration

#modelspredict new
class InputItem(BaseModel):
    id: str
    soal: str
    jawaban: str

class PredictionItem(BaseModel):
    id: str
    label: str

def predictmodel(save_dir, input_batch:List[InputItem]):
    try:
        tokenizer = T5Tokenizer.from_pretrained(save_dir, legacy=False)
        finetuned_model = T5ForConditionalGeneration.from_pretrained(save_dir)

        predictions = []
        for input_item in input_batch:
            input_text = f"{input_item.dimensi}; {input_item.jawaban}"

            # Tokenisasi teks input
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

            # inference model
            output = finetuned_model.generate(**inputs, max_length=100)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

            predictions.append({'id': input_item.id, 'label': prediction})

        return predictions

    except Exception as e:
        return jsonify({'error': str(e)})