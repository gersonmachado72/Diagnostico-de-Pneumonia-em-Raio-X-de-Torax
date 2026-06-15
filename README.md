# 🩺 Diagnóstico de Pneumonia em Raio-X de Tórax

Aplicativo de apoio ao diagnóstico médico que utiliza um modelo de deep learning (DenseNet-121 fine-tuned) para classificar radiografias de tórax como **Normal** ou **Pneumonia**, com explicação visual via Grad-CAM e geração automática de relatório clínico.

## 🚀 Funcionalidades

- 🔍 Classificação de raio-X de tórax com alta acurácia (>99% em validação)
- 🧠 Grad-CAM para destacar regiões importantes na imagem
- 📄 Relatório clínico detalhado com recomendações e links úteis (CID-10, Protocolo SUS)
- ⚡ Interface simples e intuitiva com Streamlit
- 💾 Download do relatório em formato .txt

## 🧪 Tecnologias

- **Python** 3.12
- **PyTorch** / **TorchVision**
- **DenseNet-121** fine-tuned
- **Grad-CAM** implementação manual
- **Streamlit** para interface

## 📦 Instalação local

```bash
# Clone o repositório
git clone https://github.com/gersonmachado72/Diagnostico-de-Pneumonia-em-Raio-X-de-Torax.git
cd diagnostico-pneumonia

# Crie ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou .\venv\Scripts\activate (Windows)

# Instale as dependências
pip install -r requirements.txt

# Faça o download do modelo treinado (link fornecido nas releases) e coloque na pasta raiz
# O arquivo deve se chamar 'pneumonia_weights.pth'

# Execute o app
streamlit run app.py
☁️ Deploy no Streamlit Cloud
Faça o fork deste repositório ou crie um novo.

Acesse share.streamlit.io e faça login com GitHub.

Selecione o repositório, branch (main) e arquivo app.py.

Adicione o arquivo do modelo pneumonia_weights.pth via Secrets ou hospede em nuvem e configure download automático.

Clique em Deploy.

Nota: O modelo pneumonia_weights.pth não está incluído neste repositório devido ao tamanho. Você pode obtê-lo em [link para download] ou recriá-lo com o notebook de fine-tuning incluso.

📊 Resultados esperados
Acurácia no teste: >99%

Tempo de inferência: <0,5s por imagem (CPU) / <0,1s (GPU)

Mapa de calor (Grad-CAM): evidencia as áreas pulmonares relevantes para a decisão

📋 Exemplo de relatório gerado
text
RELATÓRIO DE DIAGNÓSTICO POR IMAGEM
Data e hora: 2026-06-15 16:30:11

EXAME: Raio-X de Tórax
DIAGNÓSTICO: Pneumonia (confiança: 100,0%)

SINTOMAS RELATADOS: Tosse, febre alta e falta de ar.

INTERPRETAÇÃO CLÍNICA: Padrão radiológico compatível com processo inflamatório/infeccioso agudo (pneumonia). Correlacionar obrigatoriamente com quadro clínico.

RECOMENDAÇÕES:
- Avaliação presencial (ausculta, oximetria, exames laboratoriais)
- Considerar tratamento empírico para pneumonia comunitária
- Repetir radiografia em 48-72h se não houver melhora
- Em caso de sinais de gravidade (taquipneia, hipóxia), internação hospitalar
⚠️ Aviso
Este é um sistema de apoio diagnóstico. A decisão final é sempre do médico responsável. O modelo foi treinado com dataset pediátrico (Kaggle Chest X-Ray Pneumonia) e pode não generalizar para todas as populações.

📄 Licença
MIT

👤 Autor
[Gerson Machado] – [gerson72m@gmail.com]

Contribuições são bem-vindas! Abra uma issue ou pull request.
