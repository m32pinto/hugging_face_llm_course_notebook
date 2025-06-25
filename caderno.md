Hugging face llm course / curso de llm

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
