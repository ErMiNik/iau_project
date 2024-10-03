Zadanie (The QUEST)
Každá dvojica bude pracovať s pridelenou dátovou sadou od 2. týždňa. Vašou úlohou je predikovať závislé hodnoty premennej “mwra” (predikovaná premenna) pomocou metód strojového učenia. Budete sa musieť pritom vysporiadať s viacerými problémami, ktoré sa v dátach nachádzajú ako formáty dát, chýbajúce, vychýlené hodnoty a mnohé ďalšie. 

Očakavaným výstupom projektu  je:
najlepší model strojového učenia; 
data pipeline pre jeho vybudovanie na základe vstupných dát.

Fáza 1 - Prieskumná analýza: 15% = 15 bodov
1.1 Základný opis dát spolu s ich charakteristikami (5b)
EDA s vizualizáciou
(A-1b) Analýza štruktúr dát ako súbory (štruktúry a vzťahy, počet, typy, …), záznamy (štruktúry, počet záznamov, počet atribútov, typy, …)
(B-1b) Analýza jednotlivých atribútov: pre zvolené významné atribúty (min 10) analyzujte ich distribúcie a základné deskriptívne štatistiky. 
(C-1b) Párová analýza dát: Identifikujte vzťahy a závislostí medzi dvojicami atribútov.
(D-1b) Párová analýza dát: Identifikujte závislosti medzi predikovanou premennou a ostatnými premennými (potenciálnymi prediktormi).
(E-1b) Dokumentujte Vaše prvotné zamyslenie k riešeniu zadania projektu, napr. sú niektoré atribúty medzi sebou závislé? od ktorých atribútov závisí predikovaná premenná? či je potrebné kombinovať záznamy z viacerých súborov? 
1.2 Identifikácia problémov, integrácia a čistenie dát (5b)
(A-2b) Identifikujte aj prvotne riešte problémy v dátach napr.: nevhodná štruktúra dát, duplicitné záznamy (riadky, stlpce), nejednotné formáty, chýbajúce hodnoty, vychýlené hodnoty. V dátach sa môžu nachádzať aj iné, tu nevymenované problémy. 
(B-2b) Chýbajúce hodnoty (missing values): vyskúšajte riešiť problém min. 2 technikami
odstránenie pozorovaní s chýbajúcimi údajmi
nahradenie chýbajúcej hodnoty napr. mediánom, priemerom, pomerom, interpoláciou, alebo kNN
(C-1b) Vychýlené hodnoty (outlier detection), vyskúšajte riešiť problém min. 2 technikami
odstránenie vychýlených alebo odľahlých pozorovaní
nahradenie vychýlenej hodnoty hraničnými hodnotami rozdelenia (napr. 5%, 95%)
1.3 Formulácia a štatistické overenie hypotéz o dátach (5b)
(A-4b) Sformulujte dve hypotézy o dátach v kontexte zadanej predikčnej úlohy. Formulované hypotézy overte vhodne zvolenými štatistickými testami.
Príklad formulovania: 
android.defcontainer má v priemere vyššiu váhu v stave malware-related-activity ako v normálnom stave
(B-1b) Overte či Vaše štatistické testy majú dostatok podpory z dát, teda či majú dostatočne silnú štatistickú silu.

V odovzdanej správe (Jupyter notebook) by ste tak mali odpovedať na otázky:
Majú dáta vhodný formát pre ďalšie spracovanie? Aké problémy sa v nich vyskytujú? Nadobúdajú niektoré atribúty nekonzistentné hodnoty? Ako riešíte tieto Vami identifikované problémy?

Správa sa odovzdáva v 5. týždni semestra. Dvojica svojmu cvičiacemu odprezentuje vykonanú fázu v Jupyter Notebooku podľa potreby na cvičení. V notebooku uveďte percentuálny podiel práce členov dvojice. Následne správu elektronicky odovzdá jeden člen z dvojice do systému AIS do nedele 20.10.2024 23:59.
