# Résultats du banc d'essai (XAUUSD 2015-2025, données HistData M1 → M15)

Protocole : sélection sur 2018-2021 (IS), verdict sur 2022-2025 (OOS). Frais = 1 bp + 0,10 $/trade.
IA = gradient boosting walk-forward (entraîné uniquement sur le passé) qui filtre les trades d'un agent de base.

## M15 (intraday) : tous perdants
Espérance par trade entre −0,03 R et −0,23 R en OOS. Avantage avant frais ≈ 0 : les frais suffisent à rendre l'espérance négative.
Le filtre IA réduit un peu les pertes (trend : −0,067 → −0,057 R en IS) mais ne crée pas d'avantage, et dégrade le pullback en OOS (−0,23 R).

## H4 / D1 (suivi de tendance) : petit avantage positif
| Agent | IS espR | OOS espR |
|---|---|---|
| H4 Donchian 20 | +0,045 | +0,127 |
| H4 Donchian 55 | +0,118 | +0,075 |
| D1 Donchian 20/55 | +0,12 / +0,41 | +0,72 / +0,75 (trop peu de trades, biaisé par la hausse de l'or 2022-25) |

## Probabilité de réussir les 2 phases (H4, départ chaque lundi 2016-2025, stop interne −6 %)
| Risque/trade | H4 dc20 | H4 dc55 | Durée médiane |
|---|---|---|---|
| 0,5 % | 43 % | 8 % | 6 à 9 ans |
| 1 % | 27 % | 31 % | 11 à 20 mois |
| 2 % | 25 % | 25 % | 3 à 4 mois |

Reproduire : `python prep_data.py <dossier_m1> xau_m15.csv && python lab.py xau_m15.csv`
