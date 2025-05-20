from abc import ABC, abstractmethod
from spacy import blank, displacy
from spacy.tokens import Doc, Span


class Data(ABC):
    @abstractmethod
    def __init__(self):
        self.path = None

    # CoNLL tipindeki veriyi görselleştirerek veriyle aynı isimde ve konumda bir HTML dosyası üretir.
    def visualize(self):

        # Görselleştirilecek verinin yüklenmesi
        with open(self.path + '.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Verideki her satırın işleme sokularak bütün halindeki kelimelerle gerçek etiketlerin çıkarılması
        tokens = []  # Bütün halindeki kelimeleri tutmak için liste oluştur.
        labels = []  # Bütün halindeki kelimelere denk gelen etiketleri tutmak için liste oluştur.
        # spaces = []  # Satır boşluklarını ayrı bir listede tut.
        for line in lines:
            if not line.strip():
                tokens.append('\n')
                labels.append('O')
                # spaces.append(False)
            else:
                token, label = line.strip().split()  # Her satırı boşluğa göre ayır.
                tokens.append(token)
                labels.append(label)
                # spaces.append(True)

        # Boş spacy modeli oluşturulması
        nlp = blank('en')
        # doc = Doc(nlp.vocab, words=tokens, spaces=spaces)
        doc = Doc(nlp.vocab, words=tokens)

        # Etiketlere göre doc nesnesine entity bilgilerinin verilmesi
        spans = []
        for i, label in enumerate(labels):
            if label != 'O':  # 'O' etiketli kelimeleri göz ardı et.
                start = i
                end = i + 1
                span = Span(doc, start, end, label=label)
                spans.append(span)
        doc.ents = spans

        # Özelleştirilmiş renklerin belirlenmesi
        # colors = {'B-SOFT': '#16A5A5', 'I-SOFT': '#68CCCA', 'B-VER': '#7B64FF', 'I-VER': '#AEA1FF'}
        colors = {'B-SOFT': '#16A5A5', 'I-SOFT': '#68CCCA'}
        options = {'ents': labels, 'colors': colors}

        # HTML dosyasının oluşturulması
        html = displacy.render(doc, style='ent', page=True, options=options)
        with open(self.path + '.html', 'w', encoding='utf-8') as file:
            file.write(html)

        print('TXT verisi görselleştirilerek HTML dosyasına yazıldı.')

    def take(self):

        with open(self.path + '.html', 'r', encoding='utf-8') as file:
            html = file.read()

        return html
