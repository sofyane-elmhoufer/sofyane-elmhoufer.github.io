# ML Quest

Application web installable (PWA) pour apprendre le machine learning.

Le contenu (cours, jeux, cartes, QCM, code) est chiffré dans `content.enc.json` (AES-256-GCM, clé dérivée du mot de passe avec PBKDF2). `index.html` ne contient que l'écran de verrouillage : le déchiffrement se fait dans le navigateur, le mot de passe n'est jamais envoyé ni stocké dans le dépôt.
