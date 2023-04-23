from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
for param in model.parameters():
    print(param.requires_grad)
