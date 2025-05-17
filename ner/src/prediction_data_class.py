from data_class import Data


class PredictionData(Data):
    def __init__(self):
        super().__init__()
        self.path = f'../result/prediction'

    def create_prediction(self, tokens, predicted_labels):
        # Kelime kelime etiket tahminlerinin dosyaya yazdırılması
        with open(f'{self.path}.txt', 'w', encoding='utf-8') as f:
            for token, label in zip(tokens, predicted_labels):
                if token == '\n':
                    f.write(token)
                else:
                    f.write(token + '\t' + label + '\n')

        print(f'Modelin yaptığı tahminler TXT dosyasına CoNLL tipinde yazıldı.')
