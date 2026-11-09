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

## M1 (à la minute) : pire que M15
| Agent | espR brut (sans frais) IS / OOS | Frais en R | espR net OOS |
|---|---|---|---|
| Trend Donchian 20/55 | −0,03 à −0,05 | 0,35–0,37 | −0,36 |
| Pullback | +0,02 à +0,05 | 0,28–0,38 | −0,23 à −0,31 |
| ORB Londres | −0,01 / +0,03 | 0,04 | −0,02 |
| IA + pullback | — | — | −0,27 |

Sans frais, l'avantage est quasi nul (±0,05 R) ; à la minute, les frais coûtent environ 0,3 R par trade. Réussite FTMO : 0 %.

## Autres marchés + stratégies horaires de bourse (`lab2.py`, OANDA 5 min, 2015 → oct. 2026)
Marchés : S&P 500, Nasdaq 100, Dow Jones, DAX, or, EUR/USD, pétrole WTI. Swap facturé chaque nuit.
Stratégies : cassure du range des 30 premières min (2 variantes), momentum dernière 1/2 h, comblement de gap,
overnight acheteur, effet fin de mois, tendance H4.

**0 stratégie retenue sur 40** (critère IS : espérance > 0 et t-stat ≥ 2).
- Meilleures en IS : effet fin de mois sur indices (+0,12 à +0,17 R, t ≈ 1,5–1,9) → nul ou négatif en OOS.
- Or tendance H4 : +0,05 R (IS) / +0,09 R (OOS), positif mais non significatif (t ≈ 1).
- Momentum dernière 1/2 h : négatif partout (t jusqu'à −4,8) : l'effet publié en 2018 a disparu.
- Cassures d'ouverture, gaps, overnight : autour de 0, l'avantage avant frais est quasi nul.

Détail : `lab2_results.csv`.

## Suivi de tendance diversifié, 27 marchés en journalier (`lab3.py`, OANDA, 2006 → nov. 2026)
Règles Turtle publiques, non optimisées (entrée 55 j / sortie 20 j, stop 2 ATR), 0,5 % de risque par position,
risque ouvert ≤ 4 %, max 3 positions par secteur. Variante 20/10 j pour la robustesse.

| Système 55/20 | Espérance nette | Total sur 20 ans | Pire perte (max DD) | Réussite FTMO 2016-2026 |
|---|---|---|---|---|
| Swap estimé (2 bp/nuit indices, prudent) | −0,08 R | −51 % | 93 % | 7 à 10 % |
| Swap ÷ 2 | +0,02 R | +15 % | 45 % | 13 % |
| Sans swap (cas futures) | +0,12 R | +80 % | 39 % | 21 % |

- Avantage brut réel : +0,13 R/trade, 28 % de gagnants (profil classique).
- Gains concentrés sur métaux, énergie, agricoles ; indices et devises perdants.
- Rendement très irrégulier : 2008 +29 %, 2020 +23 %, beaucoup d'années à −10/−15 %.
- Le swap (frais de financement des CFD) détruit l'avantage ; les fonds tradent des futures, sans cette marge.
- Les creux de 20 à 40 % sont normaux pour ce style, incompatibles avec la perte max FTMO de 10 %.

### Avec les swaps réels FTMO (`ftmo_swaps.py` → `ftmo_swaps.csv`, relevés le 09/11/2026)
Exemples (coût annuel si position tenue toute l'année) : indices à l'achat −7 à −10 %/an, or à l'achat −5 %/an,
pétrole Brent à la vente −85 %/an, gaz naturel à l'achat −253 %/an, maïs à l'achat −32 %/an.

| Système 55/20, swaps FTMO | Espérance nette | Total 20 ans | Réussite FTMO 2016-2026 |
|---|---|---|---|
| 27 marchés | −0,19 R | −118 % | 7 à 8 % |
| Métaux seuls (choisis après coup) | +0,04 R | +7 % | 17 à 21 % |

Limite : swaps actuels appliqués à tout l'historique (ils varient avec les taux et la courbe des futures).
Conclusion : sur des CFD FTMO, le suivi de tendance n'est pas rentable.

## Forex seul : 28 paires FTMO, suivi de tendance journalier (`lab3.py --fx`)
Swaps FTMO forex : de −7 %/an à +1,5 %/an selon le sens (les plus faibles de FTMO). Le portage (carry) est quasi nul.

| Système | Espérance brute | Nette (swaps FTMO) | Total 20 ans | Réussite FTMO 2016-2026 |
|---|---|---|---|---|
| 55/20 j | −0,08 R | −0,18 R | −120 % | 3 à 8 % |
| 20/10 j | −0,09 R | −0,17 R | −186 % | 3 à 5 % |

Négatif avant même les frais : la tendance sur le forex ne fonctionne plus depuis 2009 (années positives : 6/21).
Le forex en intraday (EUR/USD, `lab2.py`) était lui aussi sans avantage.

## Retour à la moyenne sur le forex, 28 paires FTMO (`lab4.py`, journalier + horaire, 2006 → oct. 2026)
4 stratégies classiques, paramètres fixés avant le test, swaps FTMO réels.

| Stratégie | Trades/an | Espérance brute IS / OOS | Nette OOS (t) | Paires positives OOS |
|---|---|---|---|---|
| RSI(2) journalier | ~850 | +0,004 / +0,003 R | −0,010 R (−2,3) | 8/28 |
| Bollinger journalier | ~330 | 0,000 / −0,006 R | −0,045 R (−2,8) | 7/28 |
| Bollinger horaire, session asiatique | ~1 600 | −0,006 / −0,006 R | −0,074 R (−10,3) | 5/28 |
| RSI(14) horaire | ~2 100 | −0,021 / +0,003 R | −0,055 R (−7,3) | 2/28 |

Avantage brut nul partout ; après frais, toutes perdantes, sur IS comme sur OOS. Réussite FTMO : 0 à 15 % (peu de challenges terminés).
