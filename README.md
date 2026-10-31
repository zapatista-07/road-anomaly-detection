# 🛣️ Détection d'Anomalies Routières - Streamlit

Application de détection automatique d'anomalies sur les routes utilisant Deep Learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://votre-app.streamlit.app)

## 📋 Description

Ce projet utilise un modèle **U-Net** avec un encodeur **ResNet34** pour détecter automatiquement les anomalies sur les images de routes (nids-de-poule, fissures, dégradations, etc.).

## ✨ Démo en Ligne

🌐 **Application déployée** : [Voir l'app](https://votre-app.streamlit.app)

## 🎯 Fonctionnalités

- ✅ **Détection automatique** d'anomalies routières
- 🎨 **Visualisation triple** : masque binaire, heatmap de probabilité, overlay
- 📊 **Analyse statistique** : pourcentage d'anomalie détecté avec code couleur
- 🖼️ **Interface moderne** avec Streamlit
- ⚡ **Inférence rapide** sur CPU
- 📱 **Responsive** : fonctionne sur mobile et desktop

## 🏗️ Architecture

- **Modèle** : U-Net avec ResNet34 encoder
- **Framework** : PyTorch 2.0.1
- **Bibliothèque** : segmentation_models_pytorch 0.6.0
- **Interface** : Streamlit 1.28.0
- **Loss function** : Dice Loss + Binary Cross Entropy
- **Résolution** : 256×256 pixels

## 🚀 Déploiement

### Option 1 : Streamlit Cloud (Recommandé - Gratuit)

1. **Créez un repository GitHub** avec ces fichiers :
   - `streamlit_app.py`
   - `modele_final.pth`
   - `requirements.txt`
   - `README.md`

2. **Déployez sur Streamlit Cloud** :
   - Allez sur https://streamlit.io/cloud
   - Connectez-vous avec GitHub
   - Cliquez sur "New app"
   - Sélectionnez votre repository
   - Main file : `streamlit_app.py`
   - Cliquez sur "Deploy!"

3. **C'est tout !** 🎉

📖 **Guide détaillé** : Consultez `DEPLOIEMENT_STREAMLIT.md`

### Option 2 : Local

```bash
# Cloner le repository
git clone https://github.com/votre-nom/road-anomaly-detection.git
cd road-anomaly-detection

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run streamlit_app.py
```

L'interface sera accessible sur `http://localhost:8501`

## 📦 Fichiers du Projet

```
road-anomaly-detection/
├── streamlit_app.py           # Application Streamlit principale
├── modele_final.pth          # Modèle U-Net pré-entraîné (93 MB)
├── requirements.txt          # Dépendances Python
├── README.md                 # Documentation
└── DEPLOIEMENT_STREAMLIT.md  # Guide de déploiement détaillé
```

## 🎨 Interface Utilisateur

### Sections Principales

1. **Upload d'image** : Glissez-déposez ou sélectionnez une image
2. **Résultat d'analyse** : Statut et pourcentage d'anomalie
3. **Visualisations** :
   - 🎯 Masque binaire (zones rouges = anomalies)
   - 🌡️ Heatmap de probabilité (gradient de couleur)
   - 🔍 Overlay (superposition sur l'original)
4. **Détails techniques** : Statistiques et paramètres du modèle

### Codes Couleur

| Pourcentage | Statut | Action |
|-------------|--------|--------|
| < 1% | 🟢 Route normale | Aucune action |
| 1-5% | 🟡 Anomalie mineure | Surveillance recommandée |
| > 5% | 🔴 Anomalie critique | Intervention nécessaire |

## 🔬 Détails Techniques

### Pipeline de Prédiction

1. **Prétraitement** :
   - Redimensionnement à 256×256
   - Normalisation [0, 1]
   - Conversion RGB
   - Tensorisation

2. **Inférence** :
   - Passage dans le U-Net
   - Activation sigmoïde
   - Seuillage à 0.5

3. **Post-traitement** :
   - Génération du masque binaire
   - Création de la heatmap
   - Overlay coloré sur l'original

### Performance

- ⏱️ **Chargement initial** : 10-20 secondes
- ⚡ **Inférence** : 2-4 secondes/image sur CPU
- 💾 **RAM utilisée** : ~800 MB
- 🔄 **Utilisateurs simultanés** : Illimité (Streamlit Cloud)

## 🛠️ Technologies Utilisées

| Technologie | Version | Usage |
|-------------|---------|-------|
| **PyTorch** | 2.0.1 | Framework Deep Learning |
| **segmentation_models_pytorch** | 0.6.0 | Architecture U-Net |
| **Streamlit** | 1.28.0 | Interface web |
| **OpenCV** | 4.8.1.78 | Traitement d'image |
| **NumPy** | 1.24.3 | Calculs numériques |
| **Pillow** | 10.0.1 | Manipulation d'images |

## 📊 Exemples de Détection

### Types d'Anomalies Détectées

- 🕳️ **Nids-de-poule** : Trous profonds dans la chaussée
- 🔨 **Fissures** : Crevasses longitudinales et transversales
- 💥 **Dégradations** : Zones endommagées de la surface
- 🚧 **Déformations** : Affaissements et bosses

## 🔐 Sécurité et Confidentialité

- ✅ Aucune donnée n'est stockée sur les serveurs
- ✅ Images traitées en mémoire uniquement
- ✅ Aucun tracking ou analytics invasif
- ✅ Code open-source et auditable

## 🆚 Comparaison avec Gradio

| Fonctionnalité | Streamlit | Gradio |
|----------------|-----------|--------|
| **Interface** | Plus personnalisable | Plus simple |
| **Layout** | Colonnes, expanders, tabs | Moins flexible |
| **Styling** | CSS personnalisé | Limité |
| **Quota gratuit** | Illimité ✅ | Limité ⚠️ |
| **Déploiement** | Via GitHub | Direct upload |

## 🐛 Résolution de Problèmes

### L'app ne charge pas

**Solution** :
1. Vérifiez les logs Streamlit Cloud
2. Assurez-vous que tous les fichiers sont présents
3. Vérifiez `requirements.txt`

### Erreur "Module not found"

**Solution** :
```bash
pip install -r requirements.txt
```

### Modèle ne charge pas

**Solution** :
1. Vérifiez que `modele_final.pth` existe
2. Vérifiez la taille du fichier (93 MB)
3. Réuploadez si nécessaire

### Prédictions lentes

**Solution** :
- Normal sur CPU gratuit (2-4 secondes)
- Considérez Streamlit Pro pour plus de ressources

## 📈 Roadmap / Améliorations Futures

- [ ] Support des vidéos (frame par frame)
- [ ] Batch processing (plusieurs images)
- [ ] Export des résultats en PDF
- [ ] Comparaison avant/après
- [ ] Statistiques agrégées
- [ ] API REST
- [ ] Application mobile (PWA)
- [ ] Multi-langue (FR/EN/AR)

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Commitez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👤 Auteur

**Votre Nom**

- GitHub: [@votre-username](https://github.com/votre-username)
- LinkedIn: [Votre Profil](https://linkedin.com/in/votre-profil)

## 🙏 Remerciements

- [PyTorch](https://pytorch.org/) pour le framework Deep Learning
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) pour l'architecture U-Net
- [Streamlit](https://streamlit.io/) pour l'excellent framework d'interface
- [OpenCV](https://opencv.org/) pour le traitement d'image

## 📞 Support

Pour toute question ou problème :

- 🐛 Ouvrez une [Issue](https://github.com/votre-nom/road-anomaly-detection/issues)
- 💬 Rejoignez la [Discussion](https://github.com/votre-nom/road-anomaly-detection/discussions)
- 📧 Email: votre.email@example.com

## ⭐ Star le Projet

Si ce projet vous aide, n'oubliez pas de mettre une ⭐ sur GitHub !

---

<div align="center">

**Développé avec ❤️ pour la sécurité routière**

[🌐 Demo](https://votre-app.streamlit.app) • [📖 Documentation](DEPLOIEMENT_STREAMLIT.md) • [🐛 Issues](https://github.com/votre-nom/road-anomaly-detection/issues)

</div>
