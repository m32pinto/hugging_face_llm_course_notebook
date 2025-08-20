# Hugging face 🤗 llm course 📚 / curso de llm 📚

https://pytorch.org/get-started/locally/#supported-linux-distributions 

Nesse link deve-se decidir se irá usar a CPU ou GPU. Selecionar o SO utilizado e instalar as depedencias essenciais para rodar os elementos a seguir.

## Preparação do ambiente para windows 💻:

>Terminal: 



    pip install torch 

ℹ️Por padrão instala uma versão otimizada para cpu da biblioteca

    python -m venv venv_(nome_da_sua_pasta)
ℹ️ (Verificar)

    venv\Scripts\activate

ℹ️ Ativação do ambiente virtual

    deactivate 

ℹ️ Desativar o ambiente virtual sempre que terminar os estudos 


## Preparação do ambiente para linux 💻:

>Terminal:  

    sudo apt install python3 python3-pip python3-venv ; 

ℹ️Instalação do python e do ambiente virtual

    python3 -m venv venv_(nome_da_pasta) 

ℹ️ 

    source venv_(nome_da_pasta)/bin/activate  

ℹ️ativar o ambiente isolado para trabalho; O nome da pasta ficará entre parênteses. 





    deactivate
ℹ️Para desativar o ambiente virtual


ℹ️Nota: caso tenha dificuldade para achar o activate (linux) usar:

     "find . -name "activate" 

## Preparação do ambiente para utilização dos Transformers 🧠 :

    pip install transformers / datasets / evaluate / sentencepiec

ℹ️O código acima pode ser instalado por partes por questões de organização e estudo ex: "pip install transformers" apenas.

ℹ️**Nota: o modelo padrão é - distilbert/distilbert-base-uncased-finetuned-sst-2-englishand revision 714eb0f** 

ℹ️**Nota: sempre receber o resultado (result=) do objeto para depois imprimir (print(result)).**

## 1º Transformer sentiment analysis ➕ou ➖

--------------------------------

**Teste 1**

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier("I've been waiting for a HuggingFace course my whole life.")

    print(result)

Saída relevante📝: [{'label': 'POSITIVE', 'score': 0.9598046541213989}]

ℹ️Nota: aprovado.

--------------------------------

**Teste 2**

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier("I love instrumental music")

    print(result)

Saída relevante📝: [{'label': 'POSITIVE', 'score': 0.9998270869255066}]

ℹ️Nota: aprovado.

--------------------------------

**Teste 3**

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier("I admire those who still have the habit of reading")

    print(result)

Saída relevante📝:[{'label': 'POSITIVE', 'score': 0.9996665716171265}]

ℹ️Nota: aprovado.

--------------------------------
**Teste 4**

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier("I hate apple pie")

    print(result)

Saída relevante📝: [{'label': 'NEGATIVE', 'score': 0.9986454844474792}]

ℹ️Nota: aprovado.

--------------------------------

**Teste 5** (Teste em portuguêS)

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier("eu tenho ódio de uva passas no natal")

    print(result)

Saída relevante📝: [{'label': 'NEGATIVE', 'score': 0.9876565933227539}]

ℹ️Nota: aprovado.


--------------------------------
**Teste 6**

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier("I have a complaint to make")

    print(result)

Saída relevante📝: [{'label': 'POSITIVE', 'score': 0.9828368425369263}]

ℹ️Nota: reprovado

--------------------------------
**Teste 7**

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier(["I have a complaint to make",
                        "Eu gostei muito do cholate",
                        "I'm honestly disappointed",
                        "Gostaria de conversar a sós com você",
                        "I'll meet you at HR"])

    print(result)

Saída relevante📝:

[{'label': 'NEGATIVE', 'score': 0.9967466592788696}, 

ℹ️Nota:aprovado.

{'label': 'NEGATIVE', 'score': 0.9786876440048218}, 

ℹ️Nota:reprovado.

{'label': 'NEGATIVE', 'score': 0.9996434450149536}, 

ℹ️Nota:aprovado.
 

{'label': 'NEGATIVE', 'score': 0.7726762294769287}, 

ℹ️Nota:aprovado.
  
{'label': 'POSITIVE', 'score': 0.9994791150093079}] 

ℹ️Nota:reprovado.

--------------------------------
**Teste 8** 

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier(["EU amo você",
                        "Eu acho meu cunhado estranho",])

    print(result)

Saída relevante📝: [{'label': 'POSITIVE', 'score': 0.9777462482452393}, 

ℹ️Nota:aprovado.


Saída relevante📝: {'label': 'POSITIVE', 'score': 0.9578021168708801}] 

ℹ️Nota:reprovado.

--------------------------------
**Teste 9** 

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier(["Mãe, pai eu amo vocês",
                        "Eu odeio aquele cara que estudou comigo no fundamental",])

    print(result)

Saída relevante📝: [{'label': 'NEGATIVE', 'score': 0.8256018161773682}, 

ℹ️Nota:reprovado.

Saída relevante📝: {'label': 'NEGATIVE', 'score': 0.9777287840843201}]

ℹ️Nota:reprovado.

--------------------------------
**Teste 10** 

from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier(["mom, dad I love you",
                        "I hate that guy I went to elementary school with.",])

    print(result)

Saída relevante📝:[{'label': 'POSITIVE', 'score': 0.9998345375061035}, 

ℹ️Nota:aprovado

Saída relevante📝:{'label': 'NEGATIVE', 'score': 0.9987708926200867}] 

ℹ️Nota:aprovado.


--------------------------------
**Teste extra**

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier(["Before I felt bad about eating 3 slices of pizza every weekend,"
                        " but I think that if I hardly eat sugar during the week and I rarely drink soda then there is "
                        "no problem in eating 3 slices of pizza on the weekend, especially if it is homemade and with my wife."])

    print(result)

Saída relevante📝: [{'label': 'POSITIVE', 'score': 0.9828368425369263}] 

ℹ️Nota:aprovado.

--------------------------------


## 2º Transformer zero shot classification 📊🔫

✨✨Exercício hugging face: ✏️ Try it out! Play around with your own sequences and labels and see how the model behaves.
✨✨Experimente! Experimente com suas próprias sequências e rótulos e veja como o modelo se comporta.

✨✨Resposta do exercício nos testes:

Base:

    from transformers import pipeline

    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )

    print(result)

--------------------------------
**Teste 1**

from transformers import pipeline

    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "I would like to know if there is kanikama",
        candidate_labels=["Order", "Menu", "Information"],
    )

    print(result)

Saída relevante📝: {'sequence': 'I would like to know if there is kanikama', 'labels': ['Information', 'Order', 'Menu'], 'scores': [0.6081867814064026, 0.21226967871189117, 0.17954352498054504]}

ℹ️Nota: nesse caso era para retornar Menu com maior score. Logo seria reprovado o teste

--------------------------------
***Teste 2** 

    from transformers import pipeline

    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "I would like to know if there is kanikama",
        candidate_labels=["Order", "product availability"],
    )

    print(result)

Saída relevante📝: {'sequence': 'I would like to know if there is kanikama', 'labels': ['product availability', 'Order'], 'scores': [0.5289784669876099, 0.471021443605423]}

ℹ️Nota:aprovado.

--------------------------------
**Teste 3** 

    from transformers import pipeline

    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "I would like to know if you deliver to the Tucuna neighborhood",
        candidate_labels=["Order", "product availability","delivery information"],
    )

    print(result)

Saída relevante📝: {'sequence': 'I would like to know if you deliver to the Tucuna neighborhood', 'labels': ['delivery information', 'product availability', 'Order'], 'scores': [0.7124969363212585, 0.19233402609825134, 0.09516900032758713]}

ℹ️Nota: aprovado, porém deve se adicionar rótulos bem específicos (escolher bem as palavras), foram feitos teste para chegar nesse resultado.

--------------------------------
**Teste 4** 

    from transformers import pipeline

    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "I would like to order salmon, kani and tea",
        candidate_labels=["Order", "product availability","delivery information"],
    )

    print(result)

Saída relevante📝: {'sequence': 'I would like to order salmon, kani and tea', 'labels': ['Order', 'product availability', 'delivery information'], 'scores': [0.5683993697166443, 0.2863132059574127, 0.145287424325943]}

ℹ️Nota:aprovado, porém caso colocado o texte em português trouxe o rótulo delivey information com o maior score.

--------------------------------

**Teste 5** 

    from transformers import pipeline

    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "I would like to know if there are other branches of the store ?",
        candidate_labels=["Order", "product availability","delivery information","store information"],
    )

    print(result)

Saída relevante📝: 'labels': ['store information', 'product availability', 'Order', 'delivery information'], 'scores': [0.7502809762954712, 0.11712954193353653, 0.08225210756063461, 0.05033739656209946]

ℹ️Nota:aprovado com um rótulo a mais.

--------------------------------

## 3º transformer text generation. 💬💭✨

Base:

    from transformers import pipeline

    generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
    generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )

ℹ️Nota: A base gera um texto bem mal formatado se for executado de forma crua no editor.

✨✨✨Exercício hugging face: ✏️ Try it out! Use the num_return_sequences and max_length arguments to generate two sentences of 15 words each.

✨✨Experimente! Use os num_return_sequencesargumentos max_lengthe para gerar duas frases de 15 palavras cada.

✨Resposta:

    from transformers import pipeline

    generator = pipeline("text-generation") # Isso usará o modelo padrão que você viu

    result = generator(
        "The clouds are sereny and bright,",
        max_new_tokens=15, # Gerar no MÁXIMO 15 NOVOS tokens (aproximadamente 15 palavras)
        num_return_sequences=2, # Comece com 1 para facilitar a depuração da saída
        do_sample=True, # Ajuda na criatividade e reduz repetição
        temperature=0.7, # Controla a aleatoriedade (entre 0.5 e 1.0 é um bom range)
        pad_token_id=generator.tokenizer.eos_token_id # Importante para que o modelo saiba parar
    )

    print(result)

Saída relevante📝: [{'generated_text': 'The clouds are sereny and bright, and the air is so fresh that people can breathe freely without looking down.'}, 


Saída relevante📝: {'generated_text': 'The clouds are sereny and bright, and I can see the stars and the stars and the stars. And I'}]

ℹ️Nota: o segundo texto é muito aleatório e pode vir até mesmo incoerente e incompleto diferente do primeiro que tende a ser mais coerente e completo, o porém nesse teste a saída é mais organizada.

✨✨✨Exercícico hugging face: ✏️ Try it out! Use the filters to find a text generation model for another language. Feel free to play with the widget and use it in a pipeline!/
✨✨Experimente! Use os filtros para encontrar um modelo de geração de texto para outro idioma. Sinta-se à vontade para experimentar o widget e usá-lo em um pipeline!

✨Resposta:

    from transformers import pipeline

    generator = pipeline("text-generation", model="goldfish-models/por_latn_1000mb")
    result=generator(
        "Nuvens são serenas e",
        max_length=30,
        num_return_sequences=1,
    )

    print(result)

Saída relevante📝: [{'generated_text': 'Nuvens são serenas e Para se ter uma ideia da sua complexidade, o número de participantes que irá decorrer nas diferentes etapas do concurso de poesia vai ser cada vez menor.'}]

ℹ️Nota: português bom, porém com um pouco menos de coerência.



## 4º Transformer fill mask 👹 👺 🎭

Base: 

    from transformers import pipeline

    unmasker = pipeline("fill-mask")
    unmasker("This course will teach you all about <mask> models.", top_k=2)

✨✨✨Exercício hugging face: ✏️ Try it out! Search for the bert-base-cased model on the Hub and identify its mask word in the Inference API widget. What does this model predict for the sentence in our pipeline example above?

✨✨Experimente! Procure o modelo bert-base-cased no Hub e identifique sua palavra-máscara no widget da API de Inferência. O que esse modelo prevê para a frase em nosso exemplo de pipeline acima?

✨R:

    from transformers import pipeline

    unmasker = pipeline("fill-mask", model="neuralmind/bert-base-portuguese-cased")
    result=unmasker("This course will teach you all about [MASK] models", top_k=2)

    print(result)

Saída relevante📝 [{'score': 0.6997271180152893, 'token': 1621, 'token_str': 'the', 'sequence': 'This course will teach you all about the models'}, 


Saída relevante📝: {'score': 0.04927678406238556, 'token': 123, 'token_str': 'a', 'sequence': 'This course will teach you all about a models'}]

## 5º Transformer Named entity recognition 🏭 💁🗼

Base:

    from transformers import pipeline

    ner = pipeline("ner", grouped_entities=True)
    ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

✨✨✨Execícico hugging face: ✏️ Try it out! Search the Model Hub for a model able to do part-of-speech tagging (usually abbreviated as POS) in English. What does this model predict for the sentence in the example above? 

✨✨ ✏️ Experimente! Procure no Model Hub por um modelo capaz de fazer marcação de classes gramaticais (geralmente abreviado como POS) em inglês. O que esse modelo prevê para a frase do exemplo acima?

✨R:

    from transformers import pipeline

    ner = pipeline("ner", grouped_entities=True,model="dslim/bert-base-NER")
    result=ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

    print(result)

Saída relevante📝
[
  {'entity_group': 'PER', 'score': 0.9981525, 'word': 'Sylvain', 'start': 11, 'end': 18},
  {'entity_group': 'ORG', 'score': 0.93690395, 'word': 'Hugging Face', 'start': 33, 'end': 45},
  {'entity_group': 'LOC', 'score': 0.9971419, 'word': 'Brooklyn', 'start': 49, 'end': 57}
]

ℹ️Nota: esse conseguiu classificar como pessoa, organização e local.


**Teste 1:** 

    from transformers import pipeline

    ner = pipeline("ner", grouped_entities=True,model="vblagoje/bert-english-uncased-finetuned-pos")
    result=ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

    print(result)

Saída relevante📝:
[
    {'entity_group': 'PRON', 'score': np.float32(0.9994592), 'word': 'my', 'start': 0, 'end': 2},
    {'entity_group': 'NOUN', 'score': np.float32(0.99601364), 'word': 'name', 'start': 3, 'end': 7},
    {'entity_group': 'AUX', 'score': np.float32(0.9953696), 'word': 'is', 'start': 8, 'end': 10},
    {'entity_group': 'PROPN', 'score': np.float32(0.9981525), 'word': 'sylvain', 'start': 11, 'end': 18},
    {'entity_group': 'CCONJ', 'score': np.float32(0.99918765), 'word': 'and', 'start': 19, 'end': 22},
    {'entity_group': 'PRON', 'score': np.float32(0.9994679), 'word': 'i', 'start': 23, 'end': 24},
    {'entity_group': 'VERB', 'score': np.float32(0.99923587), 'word': 'work', 'start': 25, 'end': 29},
    {'entity_group': 'ADP', 'score': np.float32(0.90630955), 'word': 'at', 'start': 30, 'end': 32},
    {'entity_group': 'PROPN', 'score': np.float32(0.719051), 'word': 'hugging face', 'start': 33, 'end': 45},
    {'entity_group': 'ADP', 'score': np.float32(0.9993789), 'word': 'in', 'start': 46, 'end': 48},
    {'entity_group': 'PROPN', 'score': np.float32(0.9989513), 'word': 'brooklyn', 'start': 49, 'end': 57},
    {'entity_group': 'PUNCT', 'score': np.float32(0.99963903), 'word': '.', 'start': 57, 'end': 58}
]

ℹ️Nota: esse modelo, classificou a sintaxe do texto. (Resposta do exrcício)

## 6º Transformer Question answering 🎤💬💭

    Base:

    from transformers import pipeline

    question_answerer = pipeline("question-answering")
    question_answerer(
        question="Where do I work?",
        result=context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )

    print(result)

**Teste 1:**

    from transformers import pipeline

    question_answerer = pipeline("question-answering")
    result = question_answerer(
        question="You do delivery ?",
        context="menu,itens: fresh fish,frozen fish, kani and seaweed.," \
        "Order: If you want to place an order, please leave your name, your order,Localization: " \
        "your address and payment method "
        "and we will soon check the availability of the products and more information about the order."\
        "Addresses:Here are the of the stores and their opening hours..."\
        "Delivery informations: We deliver to neighborhoods x, y, x from time x to y, from x to y ..."

    )

    print(result)

ℹ️Nota: Deve-se fazer a pergunta de forma mais direta possível para obter o melhor retorno.

## 7º Transformer summarization. 📒📝📖

Base:

    from transformers import pipeline

    summarizer = pipeline("summarization")
    result=summarizer(
        """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.

        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
    """
    )

    print(result)

{'summary_text': ' America has changed dramatically during recent years . The number of engineering graduates in the U.S. has declined in traditional engineering disciplines such as mechanical, civil,    electrical, chemical, and aeronautical engineering . Rapidly developing economies such as China and India continue to encourage and advance the teaching of engineering .'}

## 8º Transformer Translation. Tradutor. 🇺🇸 > 🇫🇷 > 🇰🇷

ℹ️Nota: para esse transformer é usado o sentence piece, caso seja solicitado usar pip install sentecepiece

    from transformers import pipeline

    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    result=translator("Ce cours est produit par Hugging Face.")

    print(result)

Saída relevante📝: [{'translation_text': 'This course is produced by Hugging Face.'}]

✨✨✨Exercício hugging face: ✏️ Try it out! Search for translation models in other languages and try to translate the previous sentence into a few different languages.


✨✨ ✏️ Experimente! Pesquise modelos de tradução em outros idiomas e tente traduzir a frase anterior para vários idiomas diferentes.


✨R:

**(De pt para en)**

    from transformers import pipeline

    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-pt")
    result=translator("Hi, I would like to make a request")

    print(result)

Saída relevante📝: [{'translation_text': 'Olá, gostaria de fazer um pedido'}]

## 9º Transformer image classification 🎦📷💭

**Nota: esse transformer necessita de um biblioteca chamada pillow, para instalar**
     
     pip install pillow


Base:

    from transformers import pipeline

    image_classifier = pipeline(
        task="image-classification", model="google/vit-base-patch16-224"
    )
    result = image_classifier(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    )
    print(result)

Saída relevante📝:

Device set to use cpu
[{'label': 'lynx, catamount', 'score': 0.43349984288215637}, 
{'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'score': 0.03479618579149246},
{'label': 'snow leopard, ounce, Panthera uncia', 'score': 0.03240193799138069},
{'label': 'Egyptian cat', 'score': 0.02394479140639305},
{'label': 'tiger cat', 'score': 0.022889239713549614}]

**Teste 1:**

    from transformers import pipeline

    image_classifier = pipeline(
        task="image-classification", model="google/vit-base-patch16-224"
    )
    result = image_classifier("imagem local com uma piscina e um mergulhador")
    print(result)

Saída relevante📝:

[{'label': 'bathing cap, swimming cap', 'score': 0.5953492522239685}, 
{'label': 'swimming trunks, bathing trunks', 'score': 0.1526799201965332}, 
{'label': 'snorkel', 'score': 0.09047021716833115}, 
{'label': 'maillot, tank suit', 'score': 0.04795584827661514}, 
{'label': 'maillot', 'score': 0.02403387613594532}]

**10º Transformer Automatic speech recognition (Atenção esse não foi possivel rodar) 🔈📣📑**

Nota é necessário: ffmpeg e pip install soundfile librosa

Base:

    from transformers import pipeline

    transcriber = pipeline(
        task="automatic-speech-recognition", model="openai/whisper-large-v3"
    )
    result = transcriber(
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    )
    print(result)

ℹ️Nota: o modelo base utilizado é muito grande logo pode ser substituido por um menor como : openai/whisper-small

**Teste 1:**

    from transformers import pipeline

    transcriber = pipeline( task="automatic-speech-recognition", model="openai/whisper-small" ) 
    result = transcriber( "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac" ) 

    print(result)

Saída relevante📝: {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}

ℹ️Nota: teste aprovado.

## ☁️ ☁️  Pegadas de carbono: 

🔧🔧 Ferramentas: https://mlco2.github.io/impact/#compute ; https://codecarbon.io/ ; 

base:

    from codecarbon import EmissionsTracker

    tracker = EmissionsTracker()
    tracker.start()
    GPU Intensive code goes here
    tracker.stop()

Tokenizadores:

base:

    from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
result = tokenizer.tokenize("I want do a order")

print(result)

Saída relevante📝: 

['i', 'want', 'do', 'a', 'order']

## Usando transformadores. 📄📚

## Modulo 2 por trás da função pipeline.

## Funcionamento do analisador de sentimentos. 😊 ou 😡

ℹ️ Nota todo os códigos abaixo podem ser copiado juntos para o editor.

## Fase 0: O resultado

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier(
        [
            "I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!",
        ]
    )

    print(result)

Saída relevante da fase 0 📝:

    [{'label': 'POSITIVE', 'score': 0.9598046541213989}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]


## Fase 1: Pré-processamento com um tokenizador

    from transformers import AutoTokenizer


    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)


    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

Saída relevante da fase 1 📝:

    {'input_ids': tensor
    
    ([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],

    [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
    0,     0,     0,     0,     0,  0]]),

     'attention_mask': tensor
     
     ([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
     
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}

ℹ️ Nota: A saída em si é um dicionário contendo duas chaves, input_ids e attention_mask. input_ids contém duas linhas de inteiros (uma para cada frase) que são os identificadores exclusivos dos tokens em cada frase

ℹ️ Nota: os ## no código existem pois todo os códigos dessa parte foram unificados, logo eles podem ser utilizado juntos no editor, porém se esse em específico for utilizado só deve ser removido os ##.

## Fase 2: Passando pelo modelo.

    from transformers import AutoModel

    ##checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    ##checkpoint ja foi carregado mais acima caso o código seja usado só, deve-se usar o chackpoint acima

    model = AutoModel.from_pretrained(checkpoint)

    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

Saída relevante da fase 2 📝:


    torch.Size([2, 16, 768])




ℹ️Nota: A saída sera três números que tem o significado abaixo:

- Tamanho do lote: O número de sequências processadas por vez (2 em nosso exemplo).

- Comprimento da sequência: O comprimento da representação numérica da sequência (16 no nosso exemplo).

- Tamanho oculto: A dimensão vetorial de cada entrada do modelo (768 em nosso exemplo).

## Fase 2.1 Cabeçote do modelo: Dando sentido aos números

    from transformers import AutoModelForSequenceClassification

    ## Modelo com um cabeçalho de classificação de sequência (para poder classificar as sentenças como positivas ou negativas).

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    outputs = model(**inputs)

    print(outputs.logits.shape)

Saída relevante da fase 2 📝:

    torch.Size([2, 2])

ℹ️Nota:  a cabeça do modelo toma como entrada os vetores de alta dimensão que vimos antes e produz vetores contendo dois valores (um por rótulo)

ℹ️Nota: Como temos apenas duas frases e dois rótulos, o resultado que obtemos do nosso modelo tem o formato 2 x 2.


## Fase 3: Pós-processamento da saída

    import torch

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

    print(outputs.logits) ##Lógits

    print(model.config.id2label) ##Probabilidades

    model.config.id2label ##Resultado

Saída relevante da fase 3📝:

    ##logits

    tensor([[-1.5607,  1.6123],
            [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)

    ##Probabilidades

    tensor([[4.0195e-02, 9.5980e-01],
            [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)

    ##Resultado
            {0: 'NEGATIVE', 1: 'POSITIVE'}

ℹ️Nota: A saída são logits, pontuações brutas e não normalizadas emitidas pela última camada do modelo, precisam passar por uma camda SoftMax para serem convertidos em probabilidades


✨✨Exercício hugging face: ✏️ Experimente! Escolha dois (ou mais) textos próprios e execute-os no sentiment-analysis pipeline. Em seguida, replique você mesmo os passos que viu aqui e verifique se obtém os mesmos resultados!

    ## Resposta 

    ## Fase 0

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier(
        [
            "The food smels bad!, give me another dish",
            "So, the fish that i order is so good, can i repeat de order ?",
            "I love my work, this years are the greatest os my life",
            "Man you play for four hours, your brain are a beautiful",
            "I feel so sick, please buy me a remedy",
            
        ]
    )

    print(result)


    ## Fase 1

    from transformers import AutoTokenizer


    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)


    raw_inputs = [
            "So, the fish that i order is so good, can i repeat de order ?",
            "I love my work, this years are the greatest os my life",
            "Man you play for four hours, your brain are a beautiful",
            "I feel so sick, please buy me a remedy",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

    ## Fase 2

    from transformers import AutoModel

    ##checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    ##checkpoint ja foi carregado mais acima 

    model = AutoModel.from_pretrained(checkpoint)

    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

    ## A saída sera três números que tem o significado abaixo:
    ## Tamanho do lote: O número de sequências processadas por vez (2 em nosso exemplo).
    ## Comprimento da sequência: O comprimento da representação numérica da sequência (16 no nosso exemplo).
    ## Tamanho oculto: A dimensão vetorial de cada entrada do modelo.

    ## Fase 2.1

    from transformers import AutoModelForSequenceClassification

    ## Modelo com um cabeçalho de classificação de sequência (para poder classificar as sentenças como positivas ou negativas).
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    print(outputs.logits.shape)
    ## A cabeça do modelo toma como entrada os vetores de alta dimensão que vimos antes e produz vetores contendo dois valores (um por rótulo)
    ## Como temos apenas duas frases e dois rótulos, o resultado que obtemos do nosso modelo tem o formato 2 x 2.

    ## Fase 3

    print(outputs.logits)

    ## Serão logits, as pontuações brutas e não normalizadas emitidas pela última camada do modelo.
    ##precisam passar por uma camda SoftMax para serem convertidos em probabilidades

    import torch

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

    print(model.config.id2label)

## Modulo 3:  Modelos 📄📚

## Criando um transformer 🧠.

Utilizamos

    from transformers import AutoModel

    model = AutoModel.from_pretrained("bert-base-cased")

ℹ️Nota: O método baixará e armazenará em cache os dados do modelo do Hugging Face Hub.

ℹ️Nota: O nome do ponto de verificação corresponde a uma arquitetura e pesos de modelo específicos, neste caso um modelo BERT com uma arquitetura básica (12 camadas, 768 tamanhos ocultos, 12 cabeças de atenção)

ℹ️Nota: O modelo possui entradas de caixa (Distinção entra maiúsculas e minúsculas)

ℹ️Nota: O AutoModel class e seus associados são, na verdade, wrappers (empacotadores) simples projetados para buscar a arquitetura de modelo apropriada para um determinado ponto de verificação.

ℹ️Nota: É uma classe automática que adivinha a arquitetura de modelo apropriada para você e instanciará a classe de modelo correta.

**Porém caso saibamos a qual modelo usar podemos optar por:**

    from transformers import BertModel

    model = BertModel.from_pretrained("bert-base-cased")

## Carregando e salvando

Adicionamos no código:

    model.save_pretrained("nome_do_local_de_salvamento")

ℹ️Nota: No diretorio foi criado dois documentos de configurações: config.json e pytorch_model.bin (model safetensors)

ℹ️Nota: No config.json temos os atributos necessários para construir a arquitetura do modelo, metadados, versão do transformer do ponto de verificação que foi salvo.

ℹ️Nota: No pytorch_model.bin (model safetensors) temos o dicionário de estados, aqui temos todos os pesos do modelo (parâmetros do modelo)

ℹ️Nota: Os dois arquivos ficam juntos. 

ℹ️Nota: Para reutilizar um modelos salvo podemos usar:

    from transformers import AutoModel

    model = AutoModel.from_pretrained("directory_on_my_computer")

## Compartilhando modelos no hugging face

Utilizamos (Um por vez):

    huggingface-cli login

    model.push_to_hub("my-awesome-model")

ℹ️Explicação: Logo isso fará upload dos arquivos do modelo para o Hub, em um repositório sob seu namespace chamado my-awesome-model. Então, qualquer um pode carregar seu modelo com o from_pretrained() método!

**Para importar os modelos atualizados utilizamos**

    from transformers import AutoModel

    model = AutoModel.from_pretrained("your-username/my-awesome-model")

## Codificado o texto

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    encoded_input = tokenizer("Hello, I'm a single sentence!")
    print(encoded_input)


    texto_decodificado = tokenizer.decode(encoded_input["input_ids"][4])

    print(texto_decodificado)

Saída relevante📝:

    {'input_ids': [101, 8667, 117, 146, 112, 182, 170, 1423, 5650, 106, 102],
    
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    
    [CLS] Hello, I ' m a single sentence! [SEP]




ℹ️Nota: A saída desse código será Um dicionário com os seguintes campos:

input_ids: representações numéricas dos seus tokens

token_type_ids: eles informam ao modelo qual parte da entrada é a frase A e qual é a frase 

attention_mask: indica quais tokens devem ser atendidos e quais não devem

ℹ️Nota: Podemos decodificar os IDs de entrada para recuperar o texto original usando:

    texto_decodificado = tokenizer.decode(encoded_input["input_ids"][4])

    print(texto_decodificado)

ℹ️Nota:Atenção, entre colchetes no decodificador está a posição que será decodificada pode ser deixado vazio e será decodificado tudo


ℹ️Nota: [CLS] e [SEP] são tokens especiais exigidos pelo modelo, sendo que nem todos precisam.


**Com varias frases**

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    encoded_input = tokenizer(["How are you?", "I'm fine, thank you"])
    print(encoded_input)

Saída relevante📝:

    {'input_ids': 
    
    [[101, 1731, 1132, 1128, 136, 102], 
    
    [101, 146, 112, 182, 2503, 117, 6243, 1128, 102]],
    
    'token_type_ids': 
     
    [[0, 0, 0, 0, 0, 0], 
     
    [0, 0, 0, 0, 0, 0, 0, 0, 0]],
     
    'attention_mask': 
    
    [[1, 1, 1, 1, 1, 1], 
    
    [1, 1, 1, 1, 1, 1, 1, 1, 1]]}

**Extra: podemos pedir ao tokenizador para retornar tensores diretamente do PyTorch **

    encoded_input = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
    print(encoded_input)

Saída relevante📝:

    {'input_ids': 
    
    tensor
    
    ([[  101,  1731,  1132,  1128,   136,   102],

    [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]),

    'token_type_ids': 
    
    tensor
    
    ([[0, 0, 0, 0, 0, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 

    'attention_mask': 
    
    tensor
    
    ([[1, 1, 1, 1, 1, 1],


    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

ℹ️Nota: as duas listas não têm o mesmo comprimento! Matrizes e tensores precisam ser retangulares, então não podemos simplesmente converter essas listas em um tensor PyTorch (ou matriz NumPy). O tokenizador oferece uma opção para isso: preenchimento.

**Logo**:

    encoded_input = tokenizer(
        ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
    )
    print(encoded_input)

    Saída relevante📝:

    {'input_ids': 
    
    tensor
    
    ([[101,  1731,  1132,  1128,   136,   102,     0,     0,     0,     0],
         
         
    [101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]), 

    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 

    'attention_mask': 
    
    tensor
    
    ([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],


    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}


## Entradas de preenchimento (padding)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    encoded_input = tokenizer(["How are you?", "I'm fine, thank you"], padding = True, return_tensors = "pt")
    print(encoded_input)

Saída relevante📝:

    {'input_ids': 
    
    tensor
    
    ([[ 101, 1731, 1132, 1128,  136,  102,    0,    0,    0],

    [ 101,  146,  112,  182, 2503,  117, 6243, 1128,  102]]),
    
    'token_type_ids': 
    
    tensor
    
    ([[0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0]]),

    'attention_mask': tensor
    
    ([[1, 1, 1, 1, 1, 1, 0, 0, 0],

    [1, 1, 1, 1, 1, 1, 1, 1, 1]])}

**Logo: Agora temos tensores retangulares! os tokens de preenchimento foram codificados em IDs de entrada com ID 0 e também têm um valor de máscara de atenção de 0. Isso ocorre porque esses tokens de preenchimento não devem ser analisados pelo modelo: eles não fazem parte da frase real.**

## Truncando entradas 

**BERT só foi pré-treinado com sequências de até 512 token**

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    encoded_input = tokenizer(
        "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
        truncation=True,
    )
    print(encoded_input["input_ids"])


Saída relevante📝:

    [101, 1188, 1110, 170, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1304, 1263, 5650, 119, 102]


## Combinar os argumentos de preenchimento e truncamento, você pode garantir que seus tensores tenham o tamanho exato necessário:

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    encoded_input = tokenizer(
        ["How are you?", "I'm fine, thank you!"],
        padding=True,
        truncation=True,
        max_length=5,
        return_tensors="pt",
    )
    print(encoded_input)

Saída relevante📝:

    {'input_ids': tensor
    
    ([[ 101, 1731, 1132, 1128,  102],

    [ 101,  146,  112,  182,  102]]), 

    'token_type_ids': tensor
    
    ([[0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0]]), 
    
    'attention_mask': 
    
    tensor
    
    ([[1, 1, 1, 1, 1],

    [1, 1, 1, 1, 1]])}

## Sobre os tokens especiais :
**São usados quando o modelo é treinado com eles**

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    encoded_input = tokenizer("How are you?")
    print(encoded_input["input_ids"])
    decoded_input = tokenizer.decode(encoded_input["input_ids"])
    print(decoded_input)

Saída relevante📝:

    [101, 1731, 1132, 1128, 136, 102]
    [CLS] How are you? [SEP]

## Extras

    from transformers import BertConfig, BertModel

    # Building the config
    config = BertConfig()

    # Building the model from the config
    model = BertModel(config)   

    print(config)

Saída relevante📝:

    'BertConfig {

    "attention_probs_dropout_prob": 0.1,

    "classifier_dropout": null,

    "hidden_act": "gelu",

    "hidden_dropout_prob": 0.1,

    "hidden_size": 768,

    "initializer_range": 0.02,

    "intermediate_size": 3072,

    "layer_norm_eps": 1e-12,

    "max_position_embeddings": 512,

    "model_type": "bert",

    "num_attention_heads": 12,

    "num_hidden_layers": 12,

    "pad_token_id": 0,

    "position_embedding_type": "absolute",

    "transformers_version": "4.55.2",

    "type_vocab_size": 2,

    "use_cache": true,
    
    "vocab_size": 30522
    }'


## Fazendo uso de tensores como entrada para o modelo

    from transformers import BertModel, BertTokenizer
    import torch

    # Carregar o Tokenizador e o Modelo

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")

    # Opcional: Salvando o modelo para uso futuro

    model.save_pretrained("local_no_computador")

    # As entradas em formato de texto (strings)

    raw_inputs = ["Hello!", "Cool.", "Nice!"]

    # Tokenizar as entradas UMA VEZ
    # A variável 'model_inputs' agora vai conter o dicionário com os tensores de IDs

    model_inputs = tokenizer(raw_inputs, padding=True, return_tensors="pt")

    # Passar os inputs para o modelo
    # Os '**' (unpacking) passam o dicionário 'model_inputs' como argumentos nomeados
    # para o modelo, que espera 'input_ids', 'attention_mask', etc.

    output = model(**model_inputs)

    # O output do modelo agora é válido

    print(output.last_hidden_state.shape)

ℹ️Nota: **Embora o modelo aceite muitos argumentos diferentes, apenas os IDs de entrada são necessários.***

Saída relevante📝:

    torch.Size([3, 4, 768])

## Modulo 4: tokenizadores, pontos mais importantes.

ℹ️Nota: será explorado o que acontece no pipeline de tokenização.

Propósito: traduzir texto em dados (números) que podem ser processados pelo modelo.

ℹ️Nota: Traduzir texto para números é conhecido como codificação., é feita em um processo de duas etapas: a tokenização, seguida pela conversão em IDs de entrada.

O primeiro passo é dividir o texto em palavras (tokens), O segundo passo é converter esses tokens em números...

...Para fazer isso, o tokenizador possui um vocabulário, que é a parte que baixamos quando a instanciamos com o from_pretrained() método

**Tokenizador baseado em palavras**

Exemplos:

let's do tokenization!

let/'s/do/tokenization/!

    ## Tokenizadores

    from transformers import BertModel, BertTokenizer

    tokenized_text = "Jim henson was a puppeter".split()
    print(tokenized_text)

Saída relevante📝:

    ['Jim', 'henson', 'was', 'a', 'puppeter']

ℹ️Nota: Com esse tipo de tokenizador, podemos acabar com alguns “vocabulários” bem grandes, onde um vocabulário é definido pelo número total de tokens independentes que temos em nosso corpus.


ℹ️Nota: Cada palavra recebe um ID, começando em 0 e indo até o tamanho do vocabulário. O modelo usa esses IDs para identificar cada palavra.

ℹ️Nota: Palavras como “cão” são representadas de forma diferente de palavras como “cães”, e o modelo inicialmente não terá como saber que “cão” e “cães” são semelhantes: ele identificará as duas palavras como não relacionadas. O mesmo se aplica a outras palavras semelhantes, como “run” e “running”, que o modelo não verá como semelhantes inicialmente.

Precisamos de um token personalizado para representar palavras que não estão em nosso vocabulário. Isso é conhecido como token “desconhecido”, geralmente representado como ”[UNK]” ou ”<unk>”.

Caso seja visto muito unk é um mal sinal.

**Tokenizador baseado em caracteres**

ℹ️Nota: O vocabulário é muito menor.
Há muito menos tokens fora do vocabulário (desconhecidos), já que cada palavra pode ser construída a partir de caracteres.

Exemplos:

let's do tokenization!

l/e/t/'/s/d/o/t/o/k/e/n/i/z/a/t/i/o/n/!

ℹ️Nota: Acabaremos com uma quantidade muito grande de tokens a serem processados pelo nosso modelo

Logo usamos uma  técnica que combina as duas abordagens: tokenização de subpalavras.

**Tokenizador baseado em subpalavras**

ℹ️Nota: Aaseiam-se no princípio de que palavras usadas com frequência não devem ser divididas em subpalavras menores, mas palavras raras devem ser decompostas em subpalavras significativas.

Exemplos:

let's do tokenization!

let's/do/token/ization/!

**Carregando e salvando tokenizadores**

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    tokenizer.save_pretrained("local_no_computador")

    encoded_input = tokenizer(
        "Using a Transformer network is simple"
    )
    print(encoded_input)

Saída relevante📝:

    {'input_ids': 
    
    [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102],
    
    'token_type_ids': 
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    'attention_mask': 
    
    [1, 1, 1, 1, 1, 1, 1, 1, 1]}

**Codificação**

**Tokenização baseada em subpalavras**

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    sequence = "Using a Transformer network is simple"
    tokens = tokenizer.tokenize(sequence)

    print(tokens)

    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(ids)

ℹ️Nota: A conversão para IDs de entrada é feita pelo convert_tokens_to_ids() método tokenizador, Essas saídas, uma vez convertidas no tensor de estrutura apropriado, podem então ser usadas como entradas para um modelo.

Saída relevante📝:

    ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']

    [7993, 170, 13809, 23763, 2443, 1110, 3014]'

ℹ️Nota: Usaremos esses id's para decodificação. 

Hugging face exercise:

✏️ Try it out! Replicate the two last steps (tokenization and conversion to input IDs) on the input sentences we used in section 2 (“I’ve been waiting for a HuggingFace course my whole life.” and “I hate this so much!”). Check that you get the same input IDs we got earlier!

✏️ Experimente! Replique as duas últimas etapas (tokenização e conversão para IDs de entrada) nas frases de entrada que usamos na seção 2 (“Eu estive esperando por um curso HuggingFace toda a minha vida” e “Eu odeio tanto isso!”). Verifique se você obtém os mesmos IDs de entrada que obtivemos anteriormente!

Resposta abaixo:


    from transformers import AutoTokenizer

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)


Saída relevante📝:

    {'input_ids': 
    
    tensor
    
    ([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,2607,  2026,  2878,  2166,  1012,   102],

    [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,      0,     0,     0,     0,     0,     0]]), 
    
    'attention_mask': 
    
    tensor
    
    ([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}

**Decodificação**

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    sequence = "Using a Transformer network is simple"
    tokens = tokenizer.tokenize(sequence)

    decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
    print(decoded_string)

Saída relevante📝:

    ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']

    Using a transformer network is simple
    
ℹ️Nota:  método não apenas converte os índices de volta em tokens, mas também agrupa os tokens que faziam parte das mesmas palavras para produzir uma frase legível.

## Modulo 5: manipulando múltiplas sequências.

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence = "I've been waiting for a HuggingFace course my whole life."

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(ids)

    # Essa linha vai falhar
    model(input_ids)

Saída relevante📝:

    IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)

ℹ️Nota:O problema é que enviamos uma única sequência para o modelo, enquanto os modelos 🤗 Transformers esperam múltiplas sentenças por padrão. O tokenizador não apenas converteu a lista de IDs de entrada em um tensor, mas também adicionou uma dimensão a ele:

**Correção**

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence = "I've been waiting for a HuggingFace course my whole life."

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(ids)

    # Correção
    tokenized_inputs = tokenizer(sequence, return_tensors="pt")
    print(tokenized_inputs["input_ids"])

Saída relevante📝:

    tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102]])

    
**Adicionando novas dimensões**

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence = "I've been waiting for a HuggingFace course my whole life."

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = torch.tensor([ids])
    print("Input IDs:", input_ids)

    output = model(input_ids)
    print("Logits:", output.logits)

Saída relevante📝:

    Input IDs: tensor([[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
    2026,  2878,  2166,  1012]])
    Logits: tensor([[-2.7276,  2.8789]], grad_fn=<AddmmBackward0>)

**Batching(loteamento):  o ato de enviar várias frases através do modelo, todas de uma só vez. Se você tiver apenas uma frase, poderá simplesmente criar um lote com uma única sequência:**

    batched_ids = [ids, ids]

Este é um lote de duas sequências idênticas!

Exercício hugging face

✏️ Experimente! Converta isso batched_ids liste em um tensor e passe-o pelo seu modelo. Verifique se você obtém os mesmos logits de antes (mas duas vezes)!

**Preenchendo as entradas**

    batched_ids = [
        [200, 200, 200],
        [200, 200]
    ]

ℹ️Nota: A lista não esta retangular para contornar isso, usaremos enchimento (padding_id) para fazer com que nossos tensores tenham formato retangular. 

ℹ️Nota: Por exemplo, se você tiver 10 frases com 10 palavras e 1 frase com 20 palavras, o preenchimento garantirá que todas as frases tenham 20 palavras. No nosso exemplo, o tensor resultante é assim:

    padding_id = 100

    batched_ids = [
        [200, 200, 200],
        [200, 200, padding_id],
    ]

ℹ️Nota: O ID do token de preenchimento pode ser encontrado em tokenizer.pad_token_id. Vamos usá-lo e enviar nossas duas frases através do modelo individualmente e agrupadas:

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence1_ids = [[200, 200, 200]]
    sequence2_ids = [[200, 200]]
    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id],
    ]

    print(model(torch.tensor(sequence1_ids)).logits)
    print(model(torch.tensor(sequence2_ids)).logits)
    print(model(torch.tensor(batched_ids)).logits)

Saída relevante📝:

    tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
    tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)

    We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.

    tensor([[ 1.5694, -1.3895],
            [ 1.3374, -1.2163]], grad_fn=<AddmmBackward0>)



ℹ️Nota: Há algo errado com os logits em nossas previsões em lote: a segunda linha deve ser igual aos logits da segunda frase, mas temos valores completamente diferentes!

Isso ocorre porque a principal característica dos modelos Transformer são as camadas de atenção que contextualizar cada token. Eles levarão em consideração os tokens de preenchimento, pois atendem a todos os tokens de uma sequência. Para obter o mesmo resultado ao passar frases individuais de diferentes comprimentos pelo modelo ou ao passar um lote com as mesmas frases e preenchimento aplicados, precisamos dizer a essas camadas de atenção para ignorarem os tokens de preenchimento. Isso é feito usando uma máscara de atenção.

**Máscaras de atenção**

São tensores,  1s indicam que os tokens correspondentes devem ser atendidos, e 0s indicam que os tokens correspondentes não devem ser atendidos

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence1_ids = [[200, 200, 200]]
    sequence2_ids = [[200, 200]]
    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id],
    ]

    attention_mask = [
        [1, 1, 1],
        [1, 1, 0],
    ]


    outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
    print(outputs.logits)

Saída relevante📝:

    tensor([[ 1.5694, -1.3895],
            [ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)

Exercício hugging face: ✏️ Experimente! Aplique a tokenização manualmente nas duas frases usadas na seção 2 (“Estive esperando por um curso HuggingFace toda a minha vida.” e “odeio tanto isso!”). Passe-os pelo modelo e verifique se você obtém os mesmos logits da seção 2. Agora agrupe-os usando o token de preenchimento e crie a máscara de atenção adequada. Verifique se você obtém os mesmos resultados ao passar pelo modelo!

**Sequências mais longas**

A maioria dos modelos manipula sequências de até 512 ou 1024 tokens e trava quando solicitados a processar sequências mais longas

Para solucionar:

Use um modelo com um comprimento de sequência suportado maior.

Trunque suas sequências.
    






















