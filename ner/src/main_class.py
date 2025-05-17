from argparse import ArgumentParser
from re import findall
from torch import argmax, long, no_grad, tensor
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from prediction_data_class import PredictionData


class Main:

    @staticmethod
    def predict(text: str):
        # model_path = '../model/distilbert-base_conf67_noe9_weighted_nadam_all'
        # tokenizer_path = '../tokenizer/distilbert-base'
        model_path = '../model'
        tokenizer_path = '../tokenizer'

        model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        # print(model.config)
        # print(tokenizer.tokenize(text))

        # inputs = tokenizer(text, return_tensors='pt')
        # tokens = inputs.tokens()
        # print(tokens)
        # outputs = model(**inputs)
        # logits = outputs.logits
        # predictions = argmax(logits, dim=2)
        # id2label = model.config.id2label
        # labels = [id2label[i.item()] for i in predictions[0]]
        # print(labels)

        # ner_pipeline = pipeline('ner',
        #                         model=model,
        #                         tokenizer=tokenizer,
        #                         ignore_labels=[],
        #                         aggregation_strategy='none'
        #                         )
        #
        # results = ner_pipeline(text)
        # # print(results)
        # for result in results:
        #     print(f'{result['word']}\t\t{result['entity']}')

        # Token'ların oluşturulması
        tokens = findall(r"[A-Za-z0-9]+|[^\w\s]", text)
        # Token'ların id'lere dönüştürülmesi
        input_ids_lists = []
        attention_mask_lists = []
        for token in tokens:
            ids = tokenizer.convert_tokens_to_ids([token])
            input_ids_lists.append(ids)
            attention_mask_lists.append([1] * len(ids))

        # Tensor'lere dönüştürme işlemi
        input_ids_tensor = tensor(input_ids_lists, dtype=long)
        attention_mask_tensor = tensor(attention_mask_lists, dtype=long)

        # Model tahminlerinin oluşturulması
        with no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        predicted_label_ids = argmax(outputs.logits, dim=2).tolist()
        flat_predicted_label_ids = [item[0] for item in predicted_label_ids]

        # Tahmin id'lerinin etiketlere dönüştürülmesi
        predicted_labels = []
        for id in zip(flat_predicted_label_ids):
            predicted_labels.append(model.config.id2label[id[0]])

        # Tahmin verisi nesnesinin oluşturulması
        prediction_data = PredictionData()

        # Subtoken'sız tahmin verisinin TXT dosyasına yazılması
        prediction_data.create_prediction(tokens, predicted_labels)

        # TXT dosyasına yazılmış subtoken'sız tahmin verisinin HTML dosyasına yazılması
        prediction_data.visualize()

        # Sonucun döndürülmesi
        return prediction_data.take()


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('text', type=str)
    parser.add_argument('--text', type=str, default='Named Entity Recognition')
    args = parser.parse_args()
    # Main.predict("i know that this type of problem has been reported before, but the answers weren't helpful to "
    #              "me. this is the code situation: start time: 01d9e3ffc17b8e54 i haven't been able to find "
    #              "similar questions on so, and since it i am not getting an out of memory error i think my "
    #              "suspicions might be off. any help and advice would be hugely appreciated here, happy to provide "
    #              "more info if i have left anything out that would be useful! it worked well on system version "
    #              "before ios 17. after the ios system version was upgraded to 17, a large number of crashes "
    #              "occurred. what happened?")
    Main.predict(args.text)
