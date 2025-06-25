Hugging face llm course / curso de llm

Preparação do ambiente. para windows:

Terminal: pip install torch (instala um versão otimizada para cpu) ; python -m venv venv_(nome_da_pasta) ; venv\Scripts\activate (ativar o ambiente virtual) ;

  Para desativar utilizar: deactivate .

Preparação do ambiente. para linux:

Terminal:  sudo apt install python3 python3-pip python3-venv ; python3 -m venv venv_(nome_da_pasta) ; source venv_(nome_da_pasta)/bin/activate (para ativar o ambiente isolado para trabalho) ; https://pytorch.org/get-started/locally/#supported-linux-distributions nesse link decidir se irá usar a CPU ou GPU e selecionar o SO utilizado e instalar as depedencias de acordo com o escolhido ; 

O nome da pasta ficará entre parênteses.

  Para desativar utilizar: deactivate .

Preparação do ambiente para Transformers :

pip install transformers / datasets / evaluate / sentencepiece (pode ser instalado por partes por questões de organização e estudo ex: pip install transformers) ;  

Nota: o modelo padrão é - distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f

1º transformer sentiment analysis

--------------------------------

teste 1

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")

print(result)

[{'label': 'POSITIVE', 'score': 0.9598046541213989}]

Nota: aprovado.

--------------------------------

teste 2

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love instrumental music")

print(result)

[{'label': 'POSITIVE', 'score': 0.9998270869255066}]

Nota:aprovado.

--------------------------------

teste 3

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I admire those who still have the habit of reading")

print(result)

[{'label': 'POSITIVE', 'score': 0.9996665716171265}]

Nota: aprovado.

--------------------------------
teste 4

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I hate apple pie")

print(result)

[{'label': 'NEGATIVE', 'score': 0.9986454844474792}]

Nota:aprovado.

--------------------------------

teste 5 (teste em portuguêS)

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("eu tenho ódio de uva passas no natal")

print(result)

[{'label': 'NEGATIVE', 'score': 0.9876565933227539}]

Nota:aprovado


--------------------------------
teste 6

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I have a complaint to make")

print(result)

[{'label': 'POSITIVE', 'score': 0.9828368425369263}]

Nota: reprovado

--------------------------------
teste 7

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier(["I have a complaint to make",
                    "Eu gostei muito do cholate",
                    "I'm honestly disappointed",
                    "Gostaria de conversar a sós com você",
                    "I'll meet you at HR"])

print(result)

[{'label': 'NEGATIVE', 'score': 0.9967466592788696}, Nota:aprovado.
{'label': 'NEGATIVE', 'score': 0.9786876440048218},  Nota:reprovado.
{'label': 'NEGATIVE', 'score': 0.9996434450149536}, Nota:aprovado.
 {'label': 'NEGATIVE', 'score': 0.7726762294769287}, Nota:aprovado.
  {'label': 'POSITIVE', 'score': 0.9994791150093079}] Nota:reprovado.

--------------------------------
teste 8 

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier(["EU amo você",
                    "Eu acho meu cunhado estranho",])

print(result)

[{'label': 'POSITIVE', 'score': 0.9777462482452393}, Nota:aprovado.
{'label': 'POSITIVE', 'score': 0.9578021168708801}] Nota:reprovado.

--------------------------------
teste 9 

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier(["Mãe, pai eu amo vocês",
                    "Eu odeio aquele cara que estudou comigo no fundamental",])

print(result)

[{'label': 'NEGATIVE', 'score': 0.8256018161773682}, Nota:reprovado.
 {'label': 'NEGATIVE', 'score': 0.9777287840843201}]Nota:reprovado.

--------------------------------
teste 10 

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier(["mom, dad I love you",
                    "I hate that guy I went to elementary school with.",])

print(result)

[{'label': 'POSITIVE', 'score': 0.9998345375061035}, Nota:aprovado
{'label': 'NEGATIVE', 'score': 0.9987708926200867}] Nota:aprovado.


--------------------------------
teste extra

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier(["Before I felt bad about eating 3 slices of pizza every weekend,"
                    " but I think that if I hardly eat sugar during the week and I rarely drink soda then there is "
                    "no problem in eating 3 slices of pizza on the weekend, especially if it is homemade and with my wife."])

print(result)

[{'label': 'POSITIVE', 'score': 0.9828368425369263}] Nota:aprovado.

--------------------------------

obs: sempre receber o resultado do objeto para depois imprimir.


2º transformer zero shot classification

Exercício hugging face: ✏️ Try it out! Play around with your own sequences and labels and see how the model behaves.
Experimente! Experimente com suas próprias sequências e rótulos e veja como o modelo se comporta.

Base:

from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(result)

--------------------------------
teste 1

from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "I would like to know if there is kanikama",
    candidate_labels=["Order", "Menu", "Information"],
)

print(result)

{'sequence': 'I would like to know if there is kanikama', 'labels': ['Information', 'Order', 'Menu'], 'scores': [0.6081867814064026, 0.21226967871189117, 0.17954352498054504]}

Nota: nesse caso era para retornar Menu. Logo seria reprovado o teste

--------------------------------
teste 2 

from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "I would like to know if there is kanikama",
    candidate_labels=["Order", "product availability"],
)

print(result)

{'sequence': 'I would like to know if there is kanikama', 'labels': ['product availability', 'Order'], 'scores': [0.5289784669876099, 0.471021443605423]}

Nota:aprovado.

--------------------------------
teste 3 

from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "I would like to know if you deliver to the Tucuna neighborhood",
    candidate_labels=["Order", "product availability","delivery information"],
)

print(result)

{'sequence': 'I would like to know if you deliver to the Tucuna neighborhood', 'labels': ['delivery information', 'product availability', 'Order'], 'scores': [0.7124969363212585, 0.19233402609825134, 0.09516900032758713]}

Nota:aprovado, porém deve se adicionar rótulos bem específicos (escolher bem as palavras), foram feitos teste para chegar nesse resultado.

--------------------------------
teste 4 

from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "I would like to order salmon, kani and tea",
    candidate_labels=["Order", "product availability","delivery information"],
)

print(result)

{'sequence': 'I would like to order salmon, kani and tea', 'labels': ['Order', 'product availability', 'delivery information'], 'scores': [0.5683993697166443, 0.2863132059574127, 0.145287424325943]}

Nota:aprovado, porém caso colocado o texte em português trouxe o rótulo delivey information com o maior score.

--------------------------------

teste 5 

from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "I would like to know if there are other branches of the store ?",
    candidate_labels=["Order", "product availability","delivery information","store information"],
)

print(result)

'labels': ['store information', 'product availability', 'Order', 'delivery information'], 'scores': [0.7502809762954712, 0.11712954193353653, 0.08225210756063461, 0.05033739656209946]}

Nota:aprovado com um rótulo a mais.

--------------------------------

3º transformer geração de texto.

Exercício hugging face: ✏️ Try it out! Use the num_return_sequences and max_length arguments to generate two sentences of 15 words each./Experimente! Use os num_return_sequencesargumentos max_lengthe para gerar duas frases de 15 palavras cada.

R:

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

[{'generated_text': 'The clouds are sereny and bright, and the air is so fresh that people can breathe freely without looking down.'}, 
{'generated_text': 'The clouds are sereny and bright, and I can see the stars and the stars and the stars. And I'}]

Nota: o segundo texto é muito aleatório e pode vir até mesmo incoerente e incompleto diferente do primeiro que tende a ser mais coerente e completo.

Exercícico hugging face: ✏️ Try it out! Use the filters to find a text generation model for another language. Feel free to play with the widget and use it in a pipeline!/
Experimente! Use os filtros para encontrar um modelo de geração de texto para outro idioma. Sinta-se à vontade para experimentar o widget e usá-lo em um pipeline!

R:

from transformers import pipeline

generator = pipeline("text-generation", model="goldfish-models/por_latn_1000mb")
result=generator(
    "Nuvens são serenas e",
    max_length=30,
    num_return_sequences=1,
)

print(result)

[{'generated_text': 'Nuvens são serenas e Para se ter uma ideia da sua complexidade, o número de participantes que irá decorrer nas diferentes etapas do concurso de poesia vai ser cada vez menor.'}]

Nota: português bom, porém com um pouco menos de coerência.

Base:

from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

Nota: A base gera um texto bem mal formatado se for executado de forma crua no terminal.

4º transformer fill mask

Base: from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)

Exercício hugging face: ✏️ Try it out! Search for the bert-base-cased model on the Hub and identify its mask word in the Inference API widget. What does this model predict for the sentence in our pipeline example above? / Experimente! Procure o modelo bert-base-cased no Hub e identifique sua palavra-máscara no widget da API de Inferência. O que esse modelo prevê para a frase em nosso exemplo de pipeline acima?

R:

from transformers import pipeline

unmasker = pipeline("fill-mask", model="neuralmind/bert-base-portuguese-cased")
result=unmasker("This course will teach you all about [MASK] models", top_k=2)

print(result)

[{'score': 0.6997271180152893, 'token': 1621, 'token_str': 'the', 'sequence': 'This course will teach you all about the models'}, 
{'score': 0.04927678406238556, 'token': 123, 'token_str': 'a', 'sequence': 'This course will teach you all about a models'}]

5º transformer Named entity recognition

Base:

from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

Execícico hugging face: ✏️ Try it out! Search the Model Hub for a model able to do part-of-speech tagging (usually abbreviated as POS) in English. What does this model predict for the sentence in the example above? / ✏️ Experimente! Procure no Model Hub por um modelo capaz de fazer marcação de classes gramaticais (geralmente abreviado como POS) em inglês. O que esse modelo prevê para a frase do exemplo acima?

R:


