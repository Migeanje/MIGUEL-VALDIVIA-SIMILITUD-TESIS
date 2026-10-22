# Labeled-set draw of snapshot 20261002T224412Z

The pilot and the labeled set of T20a (D34), drawn once at commit `2eb324944c4cb95cb73a28c6c6770eb7b29cc386` with seed 20261021 from every pair of the 744 included theses. Which theses form each pair is in the gitignored `pairs.parquet` only; the tracked records hold counts and scores.

## Universe

| Stratum | sistemas | industrial | electronica | mecanica | minas | between | All |
|---|---|---|---|---|---|---|---|
| Pairs | 2,080 | 57,291 | 1,035 | 18,145 | 5,253 | 192,592 | 276,396 |

## Top 1 % of each representation

Each representation's top is its 2,764 highest-ranked pairs, by rank (D28 C10); the cut is the lowest score in it.

| Representation | Cut | Highest | sistemas | industrial | electronica | mecanica | minas | between |
|---|---|---|---|---|---|---|---|---|
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 0.7350 | 0.9898 | 13 | 1,909 | 8 | 198 | 272 | 364 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.7668 | 0.9905 | 10 | 1,765 | 3 | 271 | 259 | 456 |
| tfidf | 0.1512 | 0.9826 | 63 | 1,403 | 34 | 364 | 349 | 551 |

## Mid-range

Mean percentile rank over the representations at least 0.9 and below 0.99: 17,866 candidates.

| Stratum | sistemas | industrial | electronica | mecanica | minas | between |
|---|---|---|---|---|---|---|
| Candidates | 166 | 11,150 | 108 | 1,441 | 1,181 | 3,820 |

## Pilot and labeled set by pool and stratum

Each cell is the pairs drawn over the pool's candidates left in the stratum when the pool was drawn. The labeled set's high and mid-range pools are stratified (a stratum's quota is its drawn count); the pilot and the random pool are not.

| Phase | Pool | Model | Pairs | sistemas | industrial | electronica | mecanica | minas | between |
|---|---|---|---|---|---|---|---|---|---|
| pilot | sbert_high | paraphrase-multilingual-MiniLM-L12-v2 | 2 | 0 / 13 | 0 / 1,909 | 0 / 8 | 0 / 198 | 1 / 272 | 1 / 364 |
| pilot | sbert_high | paraphrase-multilingual-mpnet-base-v2 | 1 | 0 / 10 | 0 / 1,765 | 0 / 3 | 0 / 271 | 1 / 258 | 0 / 455 |
| pilot | tfidf_high |  | 2 | 0 / 63 | 2 / 1,403 | 0 / 34 | 0 / 364 | 0 / 347 | 0 / 550 |
| pilot | mid_range |  | 3 | 0 / 166 | 2 / 11,148 | 0 / 108 | 0 / 1,441 | 0 / 1,181 | 1 / 3,820 |
| pilot | random |  | 2 | 0 / 2,080 | 0 / 57,287 | 0 / 1,035 | 0 / 18,145 | 0 / 5,251 | 2 / 192,590 |
| labeled | sbert_high | paraphrase-multilingual-MiniLM-L12-v2 | 20 | 2 / 13 | 7 / 1,908 | 2 / 8 | 3 / 198 | 3 / 270 | 3 / 363 |
| labeled | sbert_high | paraphrase-multilingual-mpnet-base-v2 | 20 | 2 / 9 | 7 / 1,758 | 2 / 3 | 3 / 268 | 3 / 254 | 3 / 453 |
| labeled | tfidf_high |  | 20 | 2 / 62 | 6 / 1,400 | 2 / 33 | 3 / 359 | 3 / 342 | 4 / 547 |
| labeled | mid_range |  | 30 | 2 / 164 | 13 / 11,133 | 2 / 104 | 4 / 1,438 | 3 / 1,180 | 6 / 3,814 |
| labeled | random |  | 30 | 0 / 2,072 | 5 / 57,254 | 0 / 1,027 | 1 / 18,132 | 1 / 5,239 | 23 / 192,572 |

## Score ranges per pool

Each cell is the range of the pool's scores, then of its percentile ranks.

| Phase | Pool | Model | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | tfidf | Mean rank |
|---|---|---|---|---|---|---|
| pilot | sbert_high | paraphrase-multilingual-MiniLM-L12-v2 | 0.7927 to 0.7971 (0.9978 to 0.9980) | 0.8070 to 0.8256 (0.9970 to 0.9982) | 0.2164 to 0.2417 (0.9960 to 0.9972) | 0.9973 to 0.9974 |
| pilot | sbert_high | paraphrase-multilingual-mpnet-base-v2 | 0.8105 to 0.8105 (0.9987 to 0.9987) | 0.8749 to 0.8749 (0.9997 to 0.9997) | 0.4018 to 0.4018 (0.9997 to 0.9997) | 0.9994 to 0.9994 |
| pilot | tfidf_high |  | 0.7310 to 0.7365 (0.9890 to 0.9903) | 0.7267 to 0.7333 (0.9745 to 0.9776) | 0.1717 to 0.1803 (0.9926 to 0.9934) | 0.9854 to 0.9871 |
| pilot | mid_range |  | 0.6127 to 0.6820 (0.9126 to 0.9701) | 0.5958 to 0.6549 (0.8130 to 0.9101) | 0.0611 to 0.0999 (0.9301 to 0.9751) | 0.9002 to 0.9438 |
| pilot | random |  | 0.4237 to 0.4550 (0.5206 to 0.6042) | 0.4116 to 0.5353 (0.3307 to 0.6721) | 0.0062 to 0.0198 (0.1501 to 0.5872) | 0.4476 to 0.5074 |
| labeled | sbert_high | paraphrase-multilingual-MiniLM-L12-v2 | 0.7390 to 0.9005 (0.9910 to 0.9999) | 0.7224 to 0.9424 (0.9721 to 1.0000) | 0.0633 to 0.7423 (0.9348 to 1.0000) | 0.9759 to 1.0000 |
| labeled | sbert_high | paraphrase-multilingual-mpnet-base-v2 | 0.6675 to 0.7957 (0.9617 to 0.9980) | 0.7675 to 0.8330 (0.9902 to 0.9985) | 0.0455 to 0.3743 (0.8755 to 0.9995) | 0.9489 to 0.9984 |
| labeled | tfidf_high |  | 0.5091 to 0.8035 (0.7363 to 0.9984) | 0.5196 to 0.8730 (0.6304 to 0.9997) | 0.1547 to 0.4933 (0.9906 to 0.9999) | 0.7982 to 0.9993 |
| labeled | mid_range |  | 0.5341 to 0.8164 (0.7891 to 0.9989) | 0.6228 to 0.8130 (0.8621 to 0.9974) | 0.0341 to 0.1683 (0.7972 to 0.9922) | 0.9002 to 0.9895 |
| labeled | random |  | 0.2250 to 0.6861 (0.0876 to 0.9723) | 0.2645 to 0.7118 (0.0634 to 0.9654) | 0.0027 to 0.1270 (0.0374 to 0.9850) | 0.0727 to 0.9180 |

## Known duplicates (O09)

1 listed, 1 in the universe, 0 drawn; a drawn one is flagged in `pairs.parquet`.

Declared values: seed 2.0261e+07, sbert_high_per_model 20, tfidf_high 20, mid_range 30, random 30, top_share 0.01, mid_low 0.9, mid_high 0.99, stratum_floor 2, pilot_size 10, pilot_floor 2.
