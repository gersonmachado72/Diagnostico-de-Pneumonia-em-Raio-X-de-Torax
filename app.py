import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

# ------------------------------------------------------------
# Grad-CAM manual
# ------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, target_class]
        loss.backward(retain_graph=True)
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ------------------------------------------------------------
# Carregar modelo fine-tuned (Raio-X)
# ------------------------------------------------------------
@st.cache_resource
def load_pneumonia_model(weights_path='pneumonia_weights.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        st.success(f"✅ Modelo de Raio-X carregado de {weights_path}")
    except FileNotFoundError:
        st.error(f"Arquivo {weights_path} não encontrado. Faça o upload.")
        uploaded_file = st.file_uploader("Envie o modelo fine-tuned (.pth)", type=["pth"])
        if uploaded_file is not None:
            with open(weights_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            st.success("✅ Modelo de Raio-X carregado com sucesso!")
        else:
            st.stop()
    model.eval()
    model = model.to(device)
    return model, device

# ------------------------------------------------------------
# Pré-processamento para Raio-X
# ------------------------------------------------------------
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def overlay_heatmap(image_pil, cam, alpha=0.5):
    image_np = np.array(image_pil.convert('RGB'))
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

# ------------------------------------------------------------
# Interpretação clínica e geração de relatório (RX)
# ------------------------------------------------------------
def gerar_relatorio_rx(pred_class, confidence, symptoms):
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if pred_class == 0:
        diagnostico = "Normal"
        texto_clinico = "Sem evidências radiológicas de pneumonia."
        recomendacoes = "- Se houver sintomas, considerar causas extrapulmonares.\n- Acompanhamento clínico."
    else:
        diagnostico = "Pneumonia" if confidence > 0.8 else "Sugestivo de pneumonia"
        texto_clinico = "Padrão radiológico compatível com processo inflamatório/infeccioso agudo (pneumonia). Correlacionar obrigatoriamente com quadro clínico."
        recomendacoes = """- Avaliação presencial (ausculta, oximetria, exames laboratoriais).
- Considerar tratamento empírico para pneumonia comunitária.
- Repetir radiografia em 48-72h se não houver melhora.
- Em caso de sinais de gravidade (taquipneia, hipóxia), internação hospitalar."""

    relatorio = f"""
RELATÓRIO DE DIAGNÓSTICO POR IMAGEM
Data e hora: {data_hora}

EXAME: Raio-X de Tórax
DIAGNÓSTICO: {diagnostico} (confiança: {confidence:.1%})

SINTOMAS RELATADOS: {symptoms if symptoms else "Nenhum"}

INTERPRETAÇÃO CLÍNICA:
{texto_clinico}

RECOMENDAÇÕES:
{recomendacoes}

LINKS ÚTEIS:
- CID-10 (Pneumonia): https://icd.who.int/browse10/2019/en#/J12-J18
- Protocolo SUS: https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/p/pneumonia
- Diretrizes SBPT: https://www.sbpt.org.br/

⚠️ Este é um sistema de apoio diagnóstico. A decisão final é sempre do médico.
"""
    return relatorio

# ------------------------------------------------------------
# Interface Streamlit (apenas Raio-X)
# ------------------------------------------------------------
st.set_page_config(page_title="Diagnóstico de Pneumonia em Raio-X de Tórax", layout="centered")
st.title("🩺 Diagnóstico de Pneumonia em Raio-X de Tórax")
st.markdown("Modelo DenseNet-121 fine-tuned para raio-X + Grad-CAM.")

uploaded_file = st.file_uploader("Carregue uma radiografia (PNG, JPG)", type=["png", "jpg", "jpeg"])
threshold = st.slider("Limiar para alerta de pneumonia", 0.50, 0.95, 0.70, 0.01,
                      help="Acima deste valor, o diagnóstico é considerado positivo.")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Raio-X Carregado", width=400)
    symptoms = st.text_area("📝 Sintomas do paciente (opcional)", placeholder="Ex.: tosse, febre, falta de ar...")

    if st.button("Analisar e Gerar Diagnóstico"):
        with st.spinner("Processando..."):
            try:
                model, device = load_pneumonia_model()
                img_tensor = preprocess_image(image).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = F.softmax(outputs, dim=1)
                    confidence, pred = torch.max(probs, dim=1)
                    confidence = confidence.item()
                    pred_class = pred.item()

                # Exibe resultado
                if pred_class == 1 and confidence >= threshold:
                    st.error(f"⚠️ **Diagnóstico:** Pneumonia (confiança {confidence:.1%})")
                elif pred_class == 1 and confidence < threshold:
                    st.warning(f"⚠️ **Sugestivo de pneumonia** (confiança {confidence:.1%}) – correlacionar clinicamente.")
                else:
                    st.success(f"✅ **Diagnóstico:** Normal (confiança {confidence:.1%})")

                # Grad-CAM
                target_layer = model.features[-1]
                grad_cam = GradCAM(model, target_layer)
                cam = grad_cam(img_tensor, target_class=1)
                overlay_img = overlay_heatmap(image, cam)
                st.subheader("🔍 Mapa de Atenção (Grad-CAM)")
                st.image(overlay_img, caption="Regiões mais relevantes para a classe 'Pneumonia'", use_container_width=True)
                st.caption("Áreas em vermelho/amarelo indicam maior influência na decisão do modelo.")

                # Interpretação clínica
                st.subheader("📋 Interpretação Clínica e Recomendações")
                relatorio = gerar_relatorio_rx(pred_class, confidence, symptoms)
                st.markdown(relatorio)

                # Botão de download do relatório
                st.download_button(
                    label="📄 Gerar Relatório (RX) - Download .txt",
                    data=relatorio,
                    file_name=f"relatorio_rx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                st.caption("⚠️ Sistema de apoio diagnóstico – a decisão final é sempre do médico.")

            except Exception as e:
                st.error(f"Erro na análise: {str(e)}")
                st.exception(e)
