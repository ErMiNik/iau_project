# Postup na rozdelenie dát pre analýzu pomocou pandas a iných knižníc v Python-e

## 1. Import potrebných knižníc
Začnite s importovaním základných knižníc potrebných pre spracovanie dát.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
```

## 2. Načítanie dát zo súborov
Načítajte jednotlivé datasety zo súborov .csv pomocou pandas. Predpokladajme, že vaše dátové súbory sú vo formáte CSV.

```py
connections_df = pd.read_csv('connections.csv')
devices_df = pd.read_csv('devices.csv')
processes_df = pd.read_csv('processes.csv')
profiles_df = pd.read_csv('profiles.csv')
```

## 3. Rozdelenie dát na tréningovú a testovaciu množinu
Použite train_test_split z sklearn.model_selection na rozdelenie dát každého datasetu. Nastavte pomer, napríklad 80:20, pričom random_state zaistí opakovateľnosť.

```py
connections_train, connections_test = train_test_split(connections_df, test_size=0.2, random_state=42)
devices_train, devices_test = train_test_split(devices_df, test_size=0.2, random_state=42)
processes_train, processes_test = train_test_split(processes_df, test_size=0.2, random_state=42)
profiles_train, profiles_test = train_test_split(profiles_df, test_size=0.2, random_state=42)
```

## 4. Export tréningovej a testovacej množiny (voliteľné)
Ak chcete uložiť tréningové a testovacie dáta pre ďalšie použitie, môžete ich exportovať ako nové .csv súbory.

```py
connections_train.to_csv('connections_train.csv', index=False)
connections_test.to_csv('connections_test.csv', index=False)
devices_train.to_csv('devices_train.csv', index=False)
devices_test.to_csv('devices_test.csv', index=False)
processes_train.to_csv('processes_train.csv', index=False)
processes_test.to_csv('processes_test.csv', index=False)
profiles_train.to_csv('profiles_train.csv', index=False)
profiles_test.to_csv('profiles_test.csv', index=False)
```

## 5. Ďalšia analýza iba s tréningovou množinou
Po rozdelení pracujte ďalej len s tréningovými dátami, napríklad s connections_train. Tu je príklad, ako zobraziť základné informácie a vykonať prvotnú analýzu dát.


```py
# Zobrazenie základných informácií o tréningovej množine
connections_train.info()

# Popis základných štatistík pre numerické stĺpce
connections_train.describe()
```

Tento postup vám umožní bezpečne rozdeliť dáta a pracovať ďalej len s tréningovou množinou pre analytické účely.



# Postup na transformáciu dát pre modelovanie strojového učenia

## 1. Identifikácia chýbajúcich hodnôt a ich spracovanie
Začnite s identifikáciou chýbajúcich hodnôt a zvoľte stratégiu ich spracovania (napríklad doplnenie priemerom, medianom alebo odstránenie riadkov).

```python
# Skontrolovanie počtu chýbajúcich hodnôt v jednotlivých stĺpcoch
connections_train.isnull().sum()

# Príklad doplnenia chýbajúcich hodnôt medianom pre číselné stĺpce
connections_train.fillna(connections_train.median(), inplace=True)
```
## 2. Detekcia a ošetrenie outlierov
Vykonajte detekciu outlierov, najmä pre číselné atribúty, napríklad pomocou IQR (interquartile range) alebo Z-score.

```py
from scipy.stats import zscore

# Identifikácia outlierov podľa Z-score
connections_train['c.android.youtube_zscore'] = zscore(connections_train['c.android.youtube'])
outliers = connections_train[connections_train['c.android.youtube_zscore'] > 3]

# Odstránenie outlierov (voliteľné)
connections_train = connections_train[connections_train['c.android.youtube_zscore'] <= 3]
```

## 3. Kódovanie kategórií do numerického formátu
Použite LabelEncoder pre atribúty s malým počtom kategórií alebo OneHotEncoder pre viacero kategórií. Ak je nutné, preveďte aj atribúty dátumu na numerický formát (napr. rok, mesiac, deň).

```py
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Príklad s LabelEncoder pre stĺpec 'store_name' v devices_train
le = LabelEncoder()
devices_train['store_name_encoded'] = le.fit_transform(devices_train['store_name'])

# Príklad s OneHotEncoder pre 'location' v profiles_train
profiles_train = pd.get_dummies(profiles_train, columns=['location'], drop_first=True)
```

## 4. Konverzia dátumov na numerické atribúty
Pre atribúty obsahujúce časové dáta, rozdeľte na rok, mesiac, deň alebo ďalšie užitočné časové zložky.

```py
# Prevod stĺpca 'ts' na časové komponenty v connections_train
connections_train['ts'] = pd.to_datetime(connections_train['ts'])
connections_train['year'] = connections_train['ts'].dt.year
connections_train['month'] = connections_train['ts'].dt.month
connections_train['day'] = connections_train['ts'].dt.day
connections_train['hour'] = connections_train['ts'].dt.hour
```

## 5. Zlúčenie dát z rôznych tabuliek (join)
Ak je to potrebné, skombinujte rôzne datasety na základe spoločného kľúča, napríklad imei, aby ste mali kompletný pohľad na jedno pozorovanie v jednom riadku.

```py
# Spojenie connections_train a devices_train na základe stĺpca 'imei'
merged_data = pd.merge(connections_train, devices_train, on='imei', how='left')
```

## 6. Finalizácia dátového rámca pre ML modelovanie
Vyberte len numerické atribúty a uistite sa, že každý riadok obsahuje jedno pozorovanie s konzistentnými stĺpcami. Ak je potrebné odstrániť nepotrebné stĺpce, použite príkaz drop().

```py 
# Výber číselných atribútov
final_data = merged_data.select_dtypes(include=[int, float])

# Odstránenie pomocných stĺpcov
final_data.drop(['imei', 'ts'], axis=1, inplace=True)
```

Tento postup vám umožní pripraviť dáta na modelovanie, pričom každé pozorovanie je opísané jedným riadkom s čisto numerickými atribútmi.


# Postup na transformáciu atribútov dát pre strojové učenie (Scaling, Transformers a Pipeline)

## 1. Import potrebných knižníc
Najprv importujte knižnice potrebné na škálovanie a transformáciu dát.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
```

## 2. Scaling (škálovanie)
Aplikujte dve rôzne techniky škálovania na numerické atribúty, aby ste zabezpečili ich porovnateľnosť. Príklady obsahujú StandardScaler a MinMaxScaler.

### 2.1 StandardScaler
StandardScaler štandardizuje atribúty tak, že ich prevedie na distribúciu s priemerom 0 a štandardnou odchýlkou 1.

```py
scaler_standard = StandardScaler()
scaled_data_standard = scaler_standard.fit_transform(final_data)
```

### 2.2 MinMaxScaler
MinMaxScaler škáluje atribúty do zvoleného rozsahu, napríklad 0 až 1.

```py
scaler_minmax = MinMaxScaler()
scaled_data_minmax = scaler_minmax.fit_transform(final_data)
```

## 3. Transformers
Aplikujte dve techniky transformácie dát, ako napríklad PowerTransformer a QuantileTransformer, aby ste zlepšili normalitu a stabilitu rozdelenia atribútov.

### 3.1 PowerTransformer
PowerTransformer využíva Box-Cox alebo Yeo-Johnson transformáciu na stabilizáciu rozdelenia.

```py
power_transformer = PowerTransformer(method='yeo-johnson')
transformed_data_power = power_transformer.fit_transform(final_data)
```

### 3.2 QuantileTransformer
QuantileTransformer prevedie distribúciu dát na približne normálnu distribúciu pomocou kvantilov.

```py
quantile_transformer = QuantileTransformer(output_distribution='normal')
transformed_data_quantile = quantile_transformer.fit_transform(final_data)
```

## 4. Kombinovanie v dátovom pipeline
Vytvorte pipeline na kombináciu vybraných krokov predspracovania dát. Skúste rôzne kombinácie škálovania a transformácií, aby ste zistili ich efektivitu pre model.

```py
from sklearn.compose import ColumnTransformer

# Definovanie jednotlivých krokov v pipeline
pipeline = Pipeline([
    ('scaling', StandardScaler()),  # použite MinMaxScaler pre alternatívne testovanie
    ('transform', PowerTransformer())  # použite QuantileTransformer pre alternatívne testovanie
])

# Transformácia dát pomocou pipeline
processed_data = pipeline.fit_transform(final_data)
```

## 5. Testovanie rôznych kombinácií
Skúste rôzne kombinácie (StandardScaler + PowerTransformer, MinMaxScaler + QuantileTransformer) a porovnajte efekty na pripravených dátach pre modelovanie.

Každú transformáciu otestujte a analyzujte pomocou základných štatistických ukazovateľov alebo vizualizácií (napr. histogramy, boxploty), aby ste určili najlepšiu kombináciu pre konkrétne dáta a modely.


# Výber Atribútov pre Strojové Učenie (Feature Selection)

Aby sme identifikovali najinformatívnejšie atribúty pre modelovanie, aplikujeme minimálne tri rôzne techniky výberu atribútov. Tieto techniky nám pomôžu posúdiť, ktoré atribúty sú najsilnejšie pre predikciu cieľovej premennej. Po identifikovaní najlepších atribútov ich porovnáme, aby sme zistili, ktoré z nich sú konzistentne najinformatívnejšie naprieč rôznymi metódami.

## 1. Korelačná Matica
Korelačná matica identifikuje silu vzťahu medzi každým atribútom a cieľovou premennou. Pre číselné atribúty sa často využíva Pearsonova korelácia; vysoká hodnota (kladná alebo záporná) naznačuje informatívnosť atribútu pre predikciu.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Vytvorenie korelačnej matice
correlation_matrix = final_data.corr()

# Vizualizácia korelačnej matice pre zistenie vzťahu k cieľovej premennej
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.show()
```

- Výhoda: Korelačná matica je jednoduchá metóda, ktorá dokáže rýchlo identifikovať silné vzťahy medzi atribútmi a cieľovou premennou.

- Obmedzenie: Funguje dobre len s lineárnymi vzťahmi a s číselnými atribútmi.

## 2. Feature Importance pomocou Random Forest
Použitím modelu Random Forest môžeme zistiť význam atribútov podľa toho, ako často a s akým dopadom sa daný atribút používa pri delení uzlov v rozhodovacích stromoch.

```py
from sklearn.ensemble import RandomForestClassifier

# Predpokladajme, že cieľová premenná je uložená ako 'target'
X = final_data.drop('target', axis=1)
y = final_data['target']

# Tréning Random Forest modelu
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

# Zobrazenie významu atribútov
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

# Vizualizácia významu atribútov
feature_importances.plot(kind='barh', x='feature', y='importance', legend=False)
plt.show()
```

- Výhoda: Random Forest dokáže efektívne identifikovať informatívne atribúty a je robustný voči šumu a outlierom.
- Obmedzenie: Môže byť menej presný pri dátach s vysokou dimenziou a pri kategorizovaných dátach.

## 3. Selektívna Metóda (SelectKBest s ANOVA alebo Chi-Square)
SelectKBest vyberá atribúty podľa štatistických testov. Pre klasifikačné úlohy sa často používa ANOVA alebo Chi-Square (Chi2) test. Táto metóda umožňuje identifikovať atribúty s najvyššou variabilitou voči cieľovej premennej.

```py
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# Použitie ANOVA pre numerické atribúty (alternatívne Chi2 pre kategórie)
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

# Získanie skóre a zoradenie atribútov podľa informatívnosti
scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values(by='score', ascending=False)

# Vizualizácia najlepších atribútov podľa SelectKBest
scores.plot(kind='barh', x='feature', y='score', legend=False)
plt.show()
```

- Výhoda: SelectKBest je vhodný pre rýchlu selekciu atribútov a pracuje dobre s numerickými aj kategorizovanými dátami.
- Obmedzenie: Selektuje atribúty iba na základe ich nezávislého vzťahu k cieľovej premennej, ignorujúc ich spoločné vzťahy.

## 4. Porovnanie výsledkov jednotlivých metód
- Cieľom je identifikovať konzistentné atribúty, ktoré sú vysoko hodnotené viacerými metódami. Ak sa určité atribúty objavia v popredí vo viacerých technikách, možno ich považovať za kľúčové.

- V rámci prípravy dát je možné tieto identifikované atribúty následne integrovať do pipeline na modelovanie.
Tento kombinovaný prístup výberu atribútov nám pomôže zaistiť, že najdôležitejšie informácie budú zahrnuté v modeli, čím zlepšíme jeho presnosť a efektivitu.


# Zoradenie Atribútov podľa Dôležitosti

Po aplikácii troch metód na určenie informatívnosti atribútov zoradíme vybrané atribúty podľa dôležitosti. Aby sme zabezpečili robustnosť nášho poradia, použijeme vážený prístup, ktorý zohľadňuje skóre z každej techniky.

## 1. Spojenie výsledkov z metód (Korelačná Matica, Random Forest, SelectKBest)
Najprv zhromaždíme dôležitostné skóre alebo hodnotenia z jednotlivých metód. Skombinujeme výsledky do jednej tabuľky, aby sme získali prehľad, ktoré atribúty sú konzistentne vysoko hodnotené.

```python
# Príklad: spájanie výsledkov do jedného DataFrame
importance_df = pd.DataFrame({
    'feature': X.columns,
    'correlation_score': correlation_matrix['target'].abs().drop('target').values,  # Korelácia s cieľovou premennou
    'random_forest_importance': rf_model.feature_importances_,  # Významnosť podľa Random Forest
    'selectkbest_score': selector.scores_  # Významnosť podľa SelectKBest
})

# Normalizácia skóre na jednotnú mierku
importance_df['correlation_score'] = importance_df['correlation_score'] / importance_df['correlation_score'].max()
importance_df['random_forest_importance'] = importance_df['random_forest_importance'] / importance_df['random_forest_importance'].max()
importance_df['selectkbest_score'] = importance_df['selectkbest_score'] / importance_df['selectkbest_score'].max()
```

## 2. Výpočet Celkovej Dôležitosti
Na základe normalizovaných skóre z každej metódy vypočítame priemerné skóre pre každý atribút. Toto skóre bude určovať celkovú dôležitosť atribútov.

```py
# Priemerné skóre ako vážená dôležitosť atribútu
importance_df['average_importance'] = importance_df[['correlation_score', 'random_forest_importance', 'selectkbest_score']].mean(axis=1)
```

## 3. Zoradenie atribútov podľa dôležitosti
Zoradíme atribúty podľa vypočítanej priemernej dôležitosti od najdôležitejších po najmenej dôležité. Tieto atribúty predstavujú finálny výber najinformatívnejších premenných.

```py
# Zoradenie podľa priemernej dôležitosti
importance_df = importance_df.sort_values(by='average_importance', ascending=False)

# Zobrazenie výsledkov
print(importance_df[['feature', 'average_importance']])
```

## 4. Interpretácia Výsledkov
Týmto spôsobom sme zoradili atribúty podľa ich dôležitosti pre cieľovú premennú. Atribúty s najvyšším skóre by mali byť zahrnuté v ďalšom modelovaní, pričom menej dôležité atribúty môžeme odstrániť alebo im prideliť nižšiu prioritu v ďalších analýzach.


# Zdokumentovanie a Zdôvodnenie Rozhodnutí pri Výbere a Zoradení Atribútov

Pri výbere a zoradení atribútov sme použili rôzne metódy, aby sme zabezpečili robustnosť výsledkov a optimalizovali model pre predikciu. Nižšie sú uvedené hlavné dôvody výberu použitých techník a prístupu k zoradeniu atribútov.

## 1. Kombinácia Viacerých Techník pre Výber Atribútov
- **Dôvod**: Každá metóda výberu atribútov má svoje špecifiká a klady, a preto kombinácia viacerých techník znižuje riziko nesprávneho výberu a zvýrazňuje atribúty, ktoré sú konzistentne dôležité naprieč rôznymi metódami.
- **Prístup**: Použili sme korelačnú maticu, ktorá poskytuje intuitívny a rýchly prehľad o vzťahoch medzi atribútmi a cieľovou premennou. Random Forest a SelectKBest sme pridali, pretože poskytujú pokročilejšie metódy hodnotenia dôležitosti, pričom využívajú silné stránky štatistických testov a stromových algoritmov.

## 2. Normalizácia a Výpočet Priemernej Dôležitosti
- **Dôvod**: Každá metóda používa rôzne mierky dôležitosti, preto sme výsledky zjednotili cez normalizáciu, aby boli porovnateľné. Výpočet priemernej dôležitosti umožňuje kombinovať výsledky z rôznych metód do jedného koherentného skóre, čo nám dáva spoľahlivý základ na zoradenie atribútov.
- **Prístup**: Normalizované skóre každého atribútu z jednotlivých metód sme použili na výpočet priemernej dôležitosti. Táto metóda zaručuje, že naše konečné poradie nie je ovplyvnené extrémami v jednej z metód, a súčasne zohľadňuje viaceré aspekty dôležitosti.

## 3. Zoradenie Atribútov na základe Priemernej Dôležitosti
- **Dôvod**: Zoradenie podľa priemernej dôležitosti umožňuje zamerať sa na najvýznamnejšie atribúty pri vytváraní modelu, čo vedie k efektívnejšiemu a presnejšiemu modelu.
- **Výhoda pre model**: Zahrnutím najdôležitejších atribútov zvyšujeme presnosť modelu a znižujeme šum v dátach. Na základe tohto poradia môžeme rozhodnúť o odstránení menej dôležitých atribútov, čím znížime riziko nadmerného fitovania a zvýšime generalizovateľnosť modelu.

## Celkové Zdôvodnenie
Tieto rozhodnutia boli navrhnuté s cieľom optimalizovať efektívnosť a interpretabilitu modelu. Použitím viacerých techník a váženého prístupu sme sa snažili zaistiť, že výber atribútov nie je založený na jednej metóde, ale zohľadňuje rozmanité prístupy. Týmto spôsobom sme schopní presnejšie identifikovať najinformatívnejšie atribúty, čo je kľúčové pre dosiahnutie kvalitných výsledkov v ďalšej fáze modelovania.


# Postup na Úpravu Kódu pre Predspracovanie Testovacej Množiny

Aby sme zabezpečili konzistentné a automatizované predspracovanie testovacej množiny, použijeme pipeline, ktorá bude aplikovať rovnaké kroky predspracovania ako pri tréningovej množine. Tento prístup umožní, aby všetky transformácie, ktoré sme použili na tréningovej množine, boli identicky aplikované aj na testovaciu množinu bez manuálnych úprav kódu.

## 1. Vytvorte Pipeline pre Predspracovanie
Vytvorte pipeline, ktorá obsahuje všetky kroky predspracovania, ako sú dopĺňanie chýbajúcich hodnôt, škálovanie a transformácie. Každý krok pipeline by mal zodpovedať jednej fáze predspracovania dát, ako je napríklad dopĺňanie medianom, normalizácia alebo selekcia atribútov.

- **Príklad**: Definujte pipeline s nasledujúcimi krokmi:
  - Dopĺňanie chýbajúcich hodnôt (napr. pomocou medianu alebo inej vhodnej stratégie).
  - Škálovanie pomocou `StandardScaler` alebo `MinMaxScaler`.
  - Transformácia dát podľa vybraných metód, ako `PowerTransformer` alebo `QuantileTransformer`.
  - Selekcia atribútov podľa dôležitosti.

## 2. Použite fit na Tréningovej Množine a Transform na Testovacej Množine
Pipeline nastavte tak, aby bola natrénovaná len na tréningovej množine, a následne použite metódu `transform()` na testovaciu množinu. Týmto sa zabezpečí, že parametre (napríklad priemer, štandardná odchýlka pri štandardizácii alebo hranice pri MinMax škálovaní) sú založené na tréningových dátach a použité identicky na testovacích dátach.

- **Dôležité**: Vyhnite sa použitiu `fit_transform` na testovacej množine, pretože by to spôsobilo, že pipeline natrénuje nové parametre priamo na testovacích dátach, čím by sa zmenila distribúcia testovacích dát a mohlo by dôjsť k úniku dát (data leakage).

## 3. Validácia Použitia Pipeline
Na overenie správnosti aplikácie pipeline použite `cross-validation` na celom procese predspracovania a trénovania. Cross-validation zabezpečí, že model bude validovaný na dátach, ktoré neboli použité pri natrénovaní pipeline, a tým poskytne lepšiu predstavu o generalizácii modelu na nových dátach.

## 4. Automatické Použitie na Nových Dátach
Pipeline umožňuje aplikáciu rovnakého predspracovania na nových, nevidených dátach. Tento prístup je mimoriadne užitočný, ak je potrebné aplikovať model na nové alebo reálne dáta (napr. pri prediktívnej analýze po nasadení modelu).

Nasadenie pipeline s rovnakými transformačnými krokmi pre všetky dátové množiny garantuje konzistentnosť výsledkov a zabraňuje nesúladu medzi tréningovými a testovacími dátami.
