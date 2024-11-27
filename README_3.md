# Fáza 3 - Strojové učenie: 20 bodov
Pri dátovej analýze nemusí byť naším cieľom získať len znalosti obsiahnuté v aktuálnych dátach, ale aj natrénovať model, ktorý bude schopný robiť rozumné **predikcie** pre nové pozorovania pomocou techniky **strojového učenia**. 
## 3.1 Jednoduchý klasifikátor na základe závislosti v dátach  (5b)
- (A-3b) Naimplementujte jednoduchý **ID3** klasifikátor s hĺbkou min 2 (vrátane root/koreň). 
- (B-1b) Vyhodnoťte Váš ID3 klasifikátor pomocou metrík accuracy, precision a recall.
- (C-1b) Zístite či Váš ID3 klasifikátor má overfit.
## 3.2 Trénovanie a vyhodnotenie klasifikátorov strojového učenia (5b)
- (A-1b) Na trénovanie využite jeden **stromový algoritmus** v scikit-learn.
- (B-1b) Porovnajte s jedným iným **nestromovým algoritmom** v scikit-learn.
- (C-1b) Porovnajte výsledky s ID3 z prvého kroku.
- (D-1b) Vizualizujte natrénované pravidlá **minimálne** pre jeden Vami vybraný algoritmus
- (E-1b) Vyhodnoťte natrénované modely pomocou metrík accuracy, precision a recall
## 3.3 Optimalizácia alias hyperparameter tuning (5b)
- (A-1b) Vyskúšajte rôzne nastavenie hyperparametrov (tuning) pre zvolený algoritmus tak, aby ste optimalizovali výkonnosť (bez underfitingu).
- (B-1b) Vyskúšajte kombinácie modelov (ensemble) pre zvolený algoritmus tak, aby ste optimalizovali výkonnosť (bez **underfitingu**) . 
- (C-1b) Využite krížovú validáciu (**cross validation**) na trénovacej množine.
- (D-2b) Dokážte že Váš nastavený najlepší model je bez **overfitingu**.
## 3.4 Vyhodnotenie vplyvu zvolenej stratégie riešenia na klasifikáciu (5b) 
Vyhodnoťte Vami zvolené stratégie riešenia projektu z hľadiska classification accuracy, či sú učinné pre Váš dataset: 
- (A-1b) Stratégie riešenia chýbajúcich hodnôt a outlierov
- (B-1b) Dátová transformácia (scaling, transformer, …)
- (C-1b) Výber atribútov, výber algoritmov, hyperparameter tuning, ensemble learning
- (D-1b) Ktorý model je Váš **najlepší model** pre nasadenie (deployment)? 
- (E-1b) Aký je **data pipeline** pre jeho vybudovanie na základe Vášho datasetu **v produkcii**?

**Všetky hodnotenia podložte dôkazmi**. Najlepší model má byť stabilný, bez overfitu a bez underfitu. Jeho data pipeline má byť dodaný s metadátami, ak tie metadáta sú potrebné a vyrobené v developmente. 

Správa sa odovzdáva v 10. týždni semestra. Dvojica svojmu cvičiacemu odprezentuje vykonanú fázu v Jupyter Notebooku podľa potreby na cvičení. V notebooku uveďte percentuálny podiel práce členov dvojice. Následne správu elektronicky odovzdá jeden člen z dvojice do systému AIS do nedele 24.11.2024 23:59.​​
