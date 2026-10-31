import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Détection d'Anomalies Routières",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
DEVICE = torch.device('cpu')
MODEL_PATH = 'modele_final.pth'
INPUT_SIZE = (256, 256)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    """Charge le modèle U-Net pré-entraîné"""
    try:
        # Création du modèle U-Net avec ResNet34 encoder
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # Chargement des poids
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        return model
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# Fonction de prétraitement
def preprocess_image(image):
    """Prétraite l'image pour l'inférence"""
    # Conversion PIL -> numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Conversion RGB si nécessaire
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Redimensionnement
    image_resized = cv2.resize(image, INPUT_SIZE)
    
    # Normalisation [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Transposition (H, W, C) -> (C, H, W)
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    
    # Conversion en tensor et ajout de la dimension batch
    tensor = torch.from_numpy(image_transposed).unsqueeze(0).to(DEVICE)
    
    return tensor, image_resized

# Fonction de post-traitement
def postprocess_mask(mask_tensor, original_size):
    """Convertit le masque de prédiction en image visualisable"""
    # Conversion tensor -> numpy
    mask = mask_tensor.squeeze().cpu().numpy()
    
    # Application sigmoïde pour convertir en probabilités
    mask = 1 / (1 + np.exp(-mask))
    
    # Seuillage à 0.5
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    
    # Redimensionnement à la taille originale
    mask_resized = cv2.resize(mask_binary, original_size, interpolation=cv2.INTER_NEAREST)
    
    return mask_resized, mask

# Fonction de visualisation
def create_visualization(original_image, mask, heatmap):
    """Crée une visualisation combinée"""
    # Conversion du masque binaire en BGR
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:, :, 1] = 0  # Suppression du canal vert
    mask_colored[:, :, 2] = 0  # Rouge pour les anomalies
    
    # Heatmap (probabilité)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay (superposition)
    overlay = cv2.addWeighted(original_image, 0.6, mask_colored, 0.4, 0)
    
    return mask_colored, heatmap_colored, overlay

# Fonction principale de prédiction
def predict(input_image, model):
    """Effectue la prédiction sur une image"""
    if model is None:
        st.error("❌ Le modèle n'est pas chargé correctement.")
        return None
    
    try:
        # Prétraitement
        tensor, resized_image = preprocess_image(input_image)
        
        # Inférence
        with torch.no_grad():
            output = model(tensor)
        
        # Post-traitement
        original_size = (input_image.shape[1], input_image.shape[0])
        mask_binary, mask_prob = postprocess_mask(output, original_size)
        
        # Redimensionnement de l'image originale pour la visualisation
        resized_for_viz = cv2.resize(np.array(input_image), original_size)
        
        # Visualisation
        mask_colored, heatmap, overlay = create_visualization(
            resized_for_viz, mask_binary, mask_prob
        )
        
        # Calcul des statistiques
        anomaly_percentage = (mask_binary > 0).sum() / mask_binary.size * 100
        
        return {
            'mask': mask_colored,
            'heatmap': heatmap,
            'overlay': overlay,
            'percentage': anomaly_percentage
        }
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
        return None

# Interface principale
def main():
    # Header
    st.markdown('<div class="main-header">🛣️ Détection d\'Anomalies Routières</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Détection automatique avec Deep Learning (U-Net + ResNet34)</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=100)
        st.title("📋 Instructions")
        st.markdown("""
        1. **Uploadez** une image de route
        2. Le modèle analysera automatiquement
        3. Consultez les résultats :
           - 🎯 Masque binaire
           - 🌡️ Heatmap de probabilité
           - 🔍 Overlay (superposition)
        
        ### 🎨 Interprétation
        - 🟢 **< 1%** : Route normale
        - 🟡 **1-5%** : Anomalie mineure
        - 🔴 **> 5%** : Anomalie critique
        
        ### 🔬 Modèle
        - **Architecture** : U-Net
        - **Encoder** : ResNet34
        - **Loss** : Dice + BCE
        - **Résolution** : 256×256
        """)
        
        st.markdown("---")
        st.info("💡 **Astuce** : Pour de meilleurs résultats, utilisez des images bien éclairées et cadrées.")
    
    # Chargement du modèle
    with st.spinner("🔄 Chargement du modèle..."):
        model = load_model()
    
    if model is not None:
        st.success("✅ Modèle chargé avec succès!")
    else:
        st.error("❌ Impossible de charger le modèle. Vérifiez que 'modele_final.pth' existe.")
        st.stop()
    
    # Upload d'image
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📸 Choisissez une image de route",
        type=['jpg', 'jpeg', 'png'],
        help="Formats supportés: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Affichage de l'image uploadée
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Image Originale")
            st.image(image, use_container_width=True)
        
        # Prédiction
        with st.spinner("🔍 Analyse en cours..."):
            results = predict(np.array(image), model)
        
        if results is not None:
            # Affichage du pourcentage
            with col2:
                st.subheader("📊 Résultat d'Analyse")
                
                percentage = results['percentage']
                
                # Détermination du statut
                if percentage > 5:
                    status = "🔴 ANOMALIE CRITIQUE"
                    color = "red"
                    message = "Intervention nécessaire !"
                elif percentage > 1:
                    status = "🟡 ANOMALIE MINEURE"
                    color = "orange"
                    message = "Surveillance recommandée"
                else:
                    status = "🟢 ROUTE NORMALE"
                    color = "green"
                    message = "Aucune action requise"
                
                st.markdown(f"### {status}")
                st.metric(
                    label="Pourcentage d'anomalie détecté",
                    value=f"{percentage:.2f}%",
                    delta=message
                )
            
            # Affichage des visualisations
            st.markdown("---")
            st.subheader("🎨 Visualisations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🎯 Masque Binaire**")
                st.image(
                    cv2.cvtColor(results['mask'], cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption="Zones d'anomalies en rouge"
                )
            
            with col2:
                st.markdown("**🌡️ Heatmap de Probabilité**")
                st.image(
                    cv2.cvtColor(results['heatmap'], cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption="Bleu=faible, Rouge=élevé"
                )
            
            with col3:
                st.markdown("**🔍 Overlay**")
                st.image(
                    cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption="Superposition sur l'original"
                )
            
            # Informations supplémentaires
            st.markdown("---")
            with st.expander("ℹ️ Détails Techniques"):
                st.markdown(f"""
                **Statistiques de Détection:**
                - Pourcentage d'anomalie: **{percentage:.4f}%**
                - Taille de l'image: **{image.size[0]} × {image.size[1]} pixels**
                - Résolution d'analyse: **256 × 256 pixels**
                - Device: **{DEVICE}**
                
                **Paramètres du Modèle:**
                - Architecture: U-Net avec ResNet34 encoder
                - Fonction de perte: Dice Loss + Binary Cross Entropy
                - Seuil de détection: 0.5
                - Activation: Sigmoid
                """)
    
    else:
        # Message d'invitation
        st.info("👆 Uploadez une image de route pour commencer l'analyse")
        
        # Exemples (si vous avez des images)
        st.markdown("---")
        st.subheader("💡 Exemples de Détections")
        st.markdown("""
        Cette application détecte automatiquement :
        - 🕳️ Nids-de-poule
        - 🔨 Fissures et crevasses
        - 🚧 Dégradations de la chaussée
        - 💥 Zones endommagées
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>Développé avec ❤️ pour la détection d'anomalies routières</p>
        <p><strong>Frameworks:</strong> PyTorch • Streamlit • OpenCV • segmentation_models_pytorch</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
