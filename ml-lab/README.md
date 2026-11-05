# ML Lab

Application web installable (PWA) pour apprendre le workflow de préparation des données en machine learning, en pratiquant : règle → code → résultat affiché sous la cellule, comme dans Jupyter.

- 9 leçons, 32 exercices, 110 cellules à compléter, plus un bac à sable libre
- Python s'exécute **dans le navigateur** (Pyodide : pandas, matplotlib, scikit-learn, seaborn), rien à installer
- Bouton « Discuter avec Claude » : copie le contexte de l'onglet (leçon, ton code, résultat, ta question) et ouvre Claude
- Fonctionne hors-ligne après le premier chargement

## Sécurité

Les leçons, solutions et jeux de données sont chiffrés (AES-256-GCM, clé dérivée du mot de passe par PBKDF2-SHA256, 600 000 itérations) dans `content.enc`. Le mot de passe n'est écrit dans aucun fichier : il sert uniquement, dans le navigateur, à déchiffrer le contenu. Le code de l'application (`app.js`, etc.) est public, seul le contenu est protégé.

Un dépôt public permet à n'importe qui de télécharger `content.enc` et de tester des mots de passe hors-ligne : choisis un mot de passe long et unique.

## Déploiement (GitHub Pages)

1. Pousse ce dossier dans un dépôt GitHub.
2. Dépôt → **Settings → Pages** → Source : *Deploy from a branch* → branche `main`, dossier `/ (root)`.
3. Ouvre `https://<utilisateur>.github.io/<dépôt>/` sur ton téléphone.
4. Installe : Android (Chrome) menu ⋮ → *Installer l'application* ; iPhone (Safari) Partager → *Sur l'écran d'accueil*.

## Modifier le contenu ou le mot de passe

Les sources (leçons en clair, script de chiffrement) ne sont **pas** dans ce dépôt, volontairement. Avec le dossier source : `ML_LAB_PASSWORD='…' python build.py` régénère `content.enc`.
