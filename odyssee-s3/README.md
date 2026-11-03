# Odyssée S3

Application d'apprentissage gamifiée du semestre 3 (ISDIA, ENSA Fès) : Machine Learning, Big Data, Développement Web et Réalité Virtuelle.

**Accès protégé par mot de passe.** Tout le contenu de l'application (cours, jeux, code) est chiffré dans `index.html` :

- AES-256-GCM, clé dérivée du mot de passe par PBKDF2-SHA256 (600 000 itérations, sel aléatoire) ;
- le déchiffrement se fait uniquement dans le navigateur, après saisie du mot de passe ;
- la progression (XP, cartes, badges) est elle aussi enregistrée chiffrée dans le navigateur, avec un nouvel IV à chaque sauvegarde.

Le mot de passe n'est stocké nulle part dans ce dépôt.
