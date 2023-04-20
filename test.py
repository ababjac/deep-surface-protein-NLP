from transformers import TFAutoModelForSequenceClassification
#from torchinfo import summary
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
print(model.summary())
