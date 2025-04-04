# IA de Detecção de Tuberculose  

Este projeto utiliza um modelo de Deep Learning para classificar imagens de raios-X entre "Tuberculose" e "Normal". O modelo foi treinado com imagens radiológicas e pode ser testado utilizando novas imagens.  

## 📁 Estrutura do Projeto  

```
IA_Tuberculosis_Detector/
│── assets/                # Pasta contendo imagens de teste e atlas radiológico  
│── venv_tf/               # Ambiente virtual com dependências do TensorFlow e Keras  
│── .gitignore             # Arquivo para ignorar arquivos desnecessários no Git  
│── image_detector.ipynb   # Notebook principal para inferência das imagens  
│── README.md              # Documentação do projeto  
```

## 🚀 Como Rodar o Código  

### 1️⃣ Configurar o ambiente virtual  

Caso ainda não tenha o ambiente virtual configurado, crie e ative-o:  

```bash
python -m venv venv_tf  
source venv_tf/bin/activate  # MacOS/Linux  
venv_tf\Scripts\activate  # Windows  
```

Instale as dependências necessárias:  

```bash
pip install -r requirements.txt  
```

### 2️⃣ Testar uma imagem  

Para testar uma nova imagem, defina o caminho da imagem de teste no notebook `image_detector.ipynb` e execute o código abaixo:  

```python
import keras
import numpy as np

# Carregar a imagem e pré-processá-la
img = keras.utils.load_img(test_image_path, target_size=(225, 225))
img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalização

# Fazer a predição com o modelo treinado
predictions = model.predict(img_array)

# Classificação baseada na predição
if predictions[0] > 0.5:
    print(f'A imagem é classificada como **Tuberculose** ({predictions[0][0]:.2f})')
else:
    print(f'A imagem é classificada como **Normal** ({1 - predictions[0][0]:.2f})')
```

## 📌 Ajustando o limiar de decisão  

O código acima classifica uma imagem como "Tuberculose" se a predição for maior que `0.5`. Caso seja necessário um nível de confiança maior, o limiar pode ser ajustado para `0.8`, por exemplo:  

```python
limiar = 0.8
if predictions[0] > limiar:
    print(f'Classificado como TUBERCULOSE com {predictions[0][0]*100:.2f}% de confiança')
else:
    print(f'Classificado como NORMAL com {(1 - predictions[0][0])*100:.2f}% de confiança')
```

## 📷 Testando múltiplas imagens  

Se desejar testar todas as imagens dentro de uma pasta (`test_images/`), basta modificar o código para percorrer a pasta e classificar cada imagem:  

```python
import os

test_images_dir = "/assets/test_images/"
for image_name in os.listdir(test_images_dir):
    test_image_path = os.path.join(test_images_dir, image_name)
    img = keras.utils.load_img(test_image_path, target_size=(225, 225))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)

    classification = "Tuberculose" if predictions[0] > 0.5 else "Normal"
    print(f'{image_name}: {classification} ({predictions[0][0]:.2f})')
```

## 📌 Conclusão  

Este modelo permite classificar imagens de raios-X e pode ser melhorado com mais dados e ajustes na rede neural. O mesmo foi desenvolvido para fins acadêmicos e não deve ser usado para diagnósticos médicos.

Caso tenha dúvidas ou sugestões, entre em contato comigo antes de qualquer modificação no código. Estou disponível para discutir melhorias e ajustes! 📩
