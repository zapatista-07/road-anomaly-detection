# Détection d'Anomalie — Projet Deep Learning

Plateforme d'intelligence artificielle développée par une équipe de 6 étudiants.

---

## 🚀 Installation rapide

### 1. Cloner le repo
```bash
git clone https://github.com/zapatista-07/road-anomaly-detection.git
cd road-anomaly-detection
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Télécharger les fichiers modèles

> ⚠️ Les fichiers `.pth` sont trop lourds pour GitHub. Chaque membre partage ses modèles séparément.

#### 📁 Modèles de Louay (Classification + Segmentation)
Télécharge les deux fichiers et place-les dans `models/objectif_louay/` :

| Fichier | Description | Lien |
|---------|-------------|------|
| `resnet18_types.pth` | Classification 3 classes (ResNet18) | *(lien à ajouter)* |
| `unet_best.pth` | Segmentation binaire (U-Net) | *(lien à ajouter)* |

```
models/
└── objectif_louay/
    ├── resnet18_types.pth   ← mettre ici
    └── unet_best.pth        ← mettre ici
```

### 4. Lancer le serveur Flask
```bash
cd app
python app.py
```

Ouvrir **http://127.0.0.1:5000** dans le navigateur.

---

## 📁 Structure du projet

```
road-anomaly-detection/
├── app/
│   ├── app.py              # Serveur Flask (routes + API)
│   ├── models.py           # Architectures ResNet18 + U-Net + inférence
│   └── templates/
│       ├── home.html       # Page d'accueil équipe
│       └── index.html      # Interface démo
├── models/
│   ├── objectif_louay/     # Modèles Louay (Classification + Segmentation)
│   ├── objectif_samar/     # Modèles Samar
│   ├── objectif_azza/      # Modèles Azza
│   ├── objectif_roua_lz/   # Modèles Roua LZ
│   ├── objectif_roua_toui/ # Modèles Roua Toui
│   └── objectif_ghofrane/  # Modèles Ghofrane
├── requirements.txt
└── README.md
```

---

## 👥 Membres de l'équipe

| Membre | Objectif | Statut |
|--------|----------|--------|
| Louay | Déploiement Flask — ResNet18 + U-Net | ✅ Terminé |
| Samar | - | 🔄 En cours |
| Azza | - | 🔄 En cours |
| Roua LZ | - | 🔄 En cours |
| Roua Toui | - | 🔄 En cours |
| Ghofrane | - | 🔄 En cours |

---

## 🌐 URLs

| Page | URL |
|------|-----|
| Accueil équipe | http://127.0.0.1:5000/ |
| Démo modèles | http://127.0.0.1:5000/demo |
| Santé API | http://127.0.0.1:5000/health |
