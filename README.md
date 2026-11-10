# AI-TestGuard

Stratégie de test et de qualification d'un modèle de détection de fraude bancaire
(régression logistique, jeu de données Kaggle Credit Card Fraud).

## Installation
pip install -r requirements.txt

## Lancer les tests
pytest --cov=app

## Contenu
- app/ : code du pipeline (prétraitement, entraînement, prédiction, évaluation)
- tests/ : tests unitaires, intégration, robustesse, sécurité, non-régression
- reports/ : métriques, graphiques, baseline
- .github/workflows/tests.yml : intégration continue