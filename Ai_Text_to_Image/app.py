from flask import Flask, render_template, request
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None
    url = None

    if request.method == 'POST':
        url = request.form['url']
        image = Image.open(requests.get(url, stream=True).raw)

        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

    return render_template('index.html', url=url, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)