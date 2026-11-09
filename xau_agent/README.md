# xau_agent — XAUUSD sur challenge FTMO 2-step (10 000 €)

3 agents déterministes, chacun avec un rôle unique :

| Agent | Fichier | Rôle |
|---|---|---|
| Stratégie | `strategy.py` | Signal M15 : tendance EMA200 + cassure Donchian 20, SL 1.5×ATR, TP 3×ATR (R:R 2) |
| Risque | `risk.py` | Taille de position + coupe-circuits. **A le dernier mot.** |
| Exécution | `live.py` | MT5, ordres avec SL/TP serveur, journal `journal.log`, DRY-RUN par défaut |

## Garde-fous (config.py)
- Risque 0,5 % / trade (50 €), max 2 trades/jour, stop après 2 pertes d'affilée.
- Stop journalier interne **2 %** (FTMO : 5 %). Stop définitif interne **6 %** (FTMO : 10 %).
- Arrêt automatique à l'objectif (Phase 1 : +10 %, Phase 2 : +5 %).
- Pas d'entrée si spread > 0,50 $. Flat le vendredi 19h UTC. Session 7h–19h UTC.

## Procédure
1. Données : exporter l'historique XAUUSD M15 depuis MT5 (≥ 3 ans) en CSV `time,open,high,low,close`.
2. `python backtest.py XAUUSD_M15.csv` → valider sur des périodes **différentes** (pas d'optimisation sur toute la donnée).
3. `python live.py` (dry-run) sur compte démo FTMO ≥ 2 semaines, comparer au backtest.
4. `python live.py --live` sur le challenge. Phase 2 : `--target 5`.

Tests : `python -m unittest discover -s tests`
