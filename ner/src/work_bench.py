from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def predict_with_pipeline(text: str):
    """
    Eğitilmiş bir modelle, verilen metin üzerinde tahmin yapar.

    Args:
        model_path (str): Eğitilen modelin ve tokenizer'ın yolu (klasör).
        text (str): Girdi metni (uzun paragraflar olabilir).

    Returns:
        list: Tahmin edilen etiketler ve ilgili bilgiler.
    """

    # Yolları belirt.
    tokenizer_path = '../model/distilbert-base_conf67_noe9_weighted_nadam_all'
    model_path = '../tokenizer/distibert-base'

    # Tokenizer ve model yükle.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Pipeline oluştur.
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Tahminleri yap.
    predictions = nlp(text)

    return predictions
