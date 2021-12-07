# Gépi tanulás órai anyagok

## 1. labor. 2021.09.06. hétfő és 2021.09.08. szerda 

A gépi tanulás egy Python ökoszisztémájának ismertetése, melyet a laboron használni fogunk. Azért az egy, mert nagyon sokféle módon és sokféle eszközzel lehet gépi tanulási feladatokat megoldani. Jelen kurzusban az Anaconda disztribúciót fogjuk használni az automatikusan települő Numpy, Scipy, Matplotlib, Pandas és Scikit-learn könyvtárakkal. A kurzusban nem lesz szó olyan csomagokról mint a TensorFlow vagy a Keras.

Figyelmeztetés. A kurzus feltételezi a Python minimális ismeretét. Ezek az ismeretek könnyen elsajátíthatóak az alábbi linkekről.

1. Scipy Lecture Notes: egy gyorstalpaló Python alapismeretekről

http://www.scipy-lectures.org/

2. Python adattudományi ökoszisztéma ismertetése: 

http://www.scipy-lectures.org/intro/intro.html

3. Néhány olyan Python disztribúció amely minden szükséges csomagot tartalmaz (ajánlott és támogatott az Anaconda):

Anaconda: https://www.anaconda.com/download/

EPD:  https://store.enthought.com/downloads/

WinPython: https://winpython.github.io/

4. Amiben dolgozni fogunk: Spyder Python IDE,  https://pythonhosted.org/spyder/

5. Numpy: Numerical Python, egy hatékony, numerikus tömbök kezelésére és numerikus számításokra szolgáló könyvtár, http://www.numpy.org/

Reference Guide: https://docs.scipy.org/doc/numpy/reference/index.html

6. Scipy: Scientific Python, magaszintű matematikai módszereket, pl. optimalizáció, regresszió, interpoláció stb., támogató könyvtár, http://www.scipy.org/

Reference Guide: https://docs.scipy.org/doc/scipy/reference/

7. Matplotlib: 2D grafikus könyvtár, https://matplotlib.org/

Pyplot, egy Matlab-szerű modul: https://matplotlib.org/api/pyplot_summary.html

8. Pandas: adatelemző könyvtár (input-output, alapstatisztikák és grafikák), http://pandas.pydata.org/

9. Seaborn: statisztikai adatvizualizációs könyvtár (matplotlib alapú), http://seaborn.pydata.org/

10.  Scikit-learn: gépi tanulásra szolgáló könyvtár, http://scikit-learn.org/stable/

Manual: http://scikit-learn.org/stable/user_guide.html

Manual in pdf: http://scikit-learn.org/0.20/_downloads/scikit-learn-docs.pdf

Tutorial: http://www.scipy-lectures.org/packages/scikit-learn/index.html#scikit-learn-chapter


## 2. labor. 2021.09.13. hétfő és 2021.09.15. szerda

A regresszió az egyik alapvető felügyelt tanítási feladat. Felügyelt tanítás: van egy kitüntetett célváltozó, melyet a többi ún. input változó segítségével szeretnénk előrejelezni. Regressziónál a célváltozó folytonos, azaz az értékek tetszőleges valós számok lehetnek. Az input változókra nincs megkötés, lehetnek diszkrétek, folytonosak, akár vegyesen is.

A lineáris regressziónál lineáris kapcsolatot, melyet egy lineáris egyenlettel lehet leírni, tételezünk fel az input változók és a célváltozó között.

Példa. Egyváltozós lineáris regresszióÁllomány
A példában egy egyváltozós lineáris regressziót hajtunk végre szimulált adatokon szemléltetve a LinearRegression sklearn osztály alkalmazását. Az osztály legfontosabb metódusai az alábbiak:

LinearRegression(): az osztály példányosítása,
- fit(X,y): a modell illesztése az adatokra, ahol X az input, y a célváltozó,
- score(X,y): a model jóságát mérő érték, általában [0,1]-ben, jelen esetben az R-négyzet mutató,
- predict(X): a célérték előrejelzése az illesztett modell alapján az X inputra.

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linear%20regression#sklearn.linear_model.LinearRegression

## 3. labor. 2021.09.20. hétfő és 2021.09.22. szerda

Az osztályozás a másik alapvető felügyelt tanítási feladat. Felügyelt tanítás: van egy kitüntetett célváltozó, melyet a többi ún. input változó segítségével szeretnénk előrejelezni. Osztályozásnál a célváltozó diszkrét, azaz az értékek egy véges halmazból kerülhetnek ki. A különböző értékeket osztályoknak nevezzük. Bináris osztályozásról beszélünk ha két lehetséges osztály van, melyekre pozitív és negatív osztályként szokás hivatkozni.
A lineáris regresszió módszerének osztályozási feladatra való alkalmazását a logisztikus regresszió teszi lehetővé. Az input és a célváltozó közé egy látens folytonos változót (z) és egy látens valószínűséget (p) teszünk. A z az inputtól lineáris (regressziós) kapcsolaton keresztül függ. Az output osztály egy érmedobás eredménye, ahol a pozitív osztály (fej) valószínűsége p . A z és a p közötti kapcsolatot az ún. link függvény biztosítja, mely most a logisztikus függvény melynek inverze a logit függvény. A modell speciális esete az ún. általánosított lineáris modellnek (GLM - generalized linear model).
Példa. Egyváltozós logisztikus regresszióÁllomány
A példában egy egyváltozós logisztikus regressziót hajtunk végre szimulált adatokon szemléltetve a LogisticRegression sklearn osztály alkalmazását. Az osztály legfontosabb metódusai az alábbiak:

LogisticRegression(): az osztály példányosítása,
- fit(X,y): a modell illesztése az adatokra, ahol X az input, y a célváltozó,
- score(X,y): a model pontosságát mérő érték a [0,1]-ben,
- predict(X): a célérték előrejelzése az illesztett modell alapján az X inputra,
- predict_proba(X): a posterior eloszlás becslése az illesztett modell alapján az X inputra,
- decision_function(X): a folytonos látens változó, döntési határ, előrejelzése az X inputra.

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.decision_function

## 4. labor. 2021.09.27. hétfő és 2021.09.29. szerda

Gépi tanulási feladatok megoldásánál egy vagy több algoritmust alkalmazunk egy vagy több adatállományon melynek eredménye:

- regresszió esetén a becsült célérték,
- osztályozás esetén a becsült osztálycímke,
- klaszterezés esetén pedig a becsült klaszter.
Az algoritmusok építhetnek modellt, melyet paraméterekkel jellemezhetünk és általuk tárolhatunk el, vagy megoldhatják a feladatot modell építés nélkül közvetlen a tanító állományt használva. Ez utóbbiakat lusta tanítóknak (lazy learner) hívjuk.
Fontos azonban azt látni hogy minden ilyen algoritmus eredményét véletlen mennyiségnek kell tekinteni. A véletlen sok forrásból eredhet:

- Már maga az input adatállomány is hibával terhelt lehet, amelyről még akár információval is rendelkezhetünk. Gyakran előfordul az is, hogy hiányzó adatok vannak az állományban melyeket pótolnunk kell (imputáció) és a pótlás módja véletlen generáláson alapszik.
- Az algoritmusok, még ha determinisztikusak is, gyakran függnek a kiinduló állapottól: a kezdeti paraméter értékektől vagy akár attól, hogy milyen sorrendben vannak a rekordok.
- Végül vannak sztochasztikus algoritmusok ahol a véletlen, véletlen szám generálás formájában, közvetlenül megjelenik az algoritmus egyes lépéseiben. Erre példa a sztochasztikus gradiens módszer vagy a véletlen alapú jellemző szelekció nagyméretű feladatoknál.
Így ezen algoritmusok eredményeit, úm. becsült célérték vagy klaszter-tagság illetve az illesztett modell paraméterei, vizsgálhatjuk úgy, mint véletlen mennyiségeket. Ki számíthatunk rá különböző statisztikákat, pl átlag és szórás, illetve megvizsgálhatjuk az eloszlásukat grafikus eszközökkel, pl. hisztogram vagy doboz ábra. Ennek alapján egy gépi tanulási probléma megoldása végén, ahhoz hogy az eredményeket hasznosítsuk, választ kell adni többek között az alábbi kérdésekre is:
- Milyen az eloszlása a becsült regressziós célértéknek, mi az átlaga és mekkora a szórása?
- Mekkora a valószínűsége a kapott osztálycímkének vagy klaszter-tagságnak?
- Mekkora a kapott eredmények hibája és mit várhatunk a tanított modelltől alkalmazás közben?
Az első esetre példaként, nyilvánvaló,  hogy ha egy cukorbetegségre való hajlamosságot jelző változó becsült értékének szórása kicsi egy betegnél, akkor ott ez az érték megbízhatónak tekinthető, míg ha nagy egy másik betegnél, akkor csak fenntartásokkal lehet kezelni az algoritmus által kapott értéket. Hasonlóan, egy bináris osztályozási feladatnál, ha egy emailt spam-nek osztályoztunk és ennek valószínűsége is nagyon közel van az 1-hez, akkor erre az előrejelzésre biztosan támaszkodhatunk. Ezzel szemben, ha ez a valószínűség csak kicsit nagyobb mint 0.5, pl. 0.6, akkor már fenntartásokkal kell élnünk. 
Ezen kérdések megválaszolására a gépi tanulás számos módszert dolgozott ki.
- A legegyszerűbb módszer a gépi tanulási algoritmusok hibájának megbízható mérésére ha két adatállományt használunk. A tanító állományon illesztjük a modellt és egy tőle független tesztelőn mérjük a modell jóságát vagy hibáját. A két adatállományt (véletlen) particionálással kaphatjuk meg. Gyakran használnak egy harmadik, ún. validációs adatállományt is annak meghatározására, hogy mikor álljunk le a tanítással. Ez a módszer segít abban, hogy reálisabb képet kapjunk a modellünk jóságáról, viszont az eredmények eloszlásáról már nem mond semmit.
- Részletesebb képet kapunk a keresztellenőrzés (CV - crossvalidation) módszerével. Ekkor K egyenlő részre osztjuk az adatállományt, amelyből mindig egyet választunk teszt állománynak és az összes többit tanítónak. Így minden az algoritmus által kapott eredményből pontosan K darabot kapunk, melynek eloszlását már vizsgálhatjuk.
- Még részletesebb képet kaphatunk a replikáció módszerével. Ekkor már az adat előállítás folyamatában is véletlen generálást alkamazunk. Egy nagyméretű adattárházból több (a pontos számot a replikáció paramétere mondja meg) adatállományt válogatunk le, majd ezekre egyenként alkalmazzuk a teljes gépi tanulási eljárást. Az alkalmazott algoritmus által kapott eredményekre annyi érték lesz amennyi a replikáció paramétere. Így világos képet kaphatunk a modell pontosságáról vagy az egyes paraméterértékek viselkedéséről. A módszer nagyon jól párhuzamosítható, hiszen az egyes adatállományok egymástól teljesen függetlenül kezelhetőek.
Példa. Replikáció analizis az egyváltozós lineáris regresszióraÁllomány
A példában a korábbi egyváltozós lineáris regressziós példát fejlesztjük tovább. Először mintavételezéssel több tanító adatállományt hozunk létre, majd mindegyikre modellt illesztve vizsgáljuk a paraméterek és az R2 score eloszlásának viselkedését. Ezután egy tanító adatállományon modellt illesztve több teszt állományon vizsgáljuk az R2 eloszlását. Végül tanító-teszt felosztásos és keresztellenőrzéses vizsgálatokat végzünk.

https://scikit-learn.org/stable/modules/generated/sklearn.utils.random.sample_without_replacement.html?highlight=sample_without_replacement#sklearn.utils.random.sample_without_replacement

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train%20test%20split#sklearn.model_selection.train_test_split

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html?highlight=cross_validation

## 5. labor. 2021.10.04. hétfő és 2021.10.06. szerda

Az alábbi két példában valós adatállományokon mutatjuk be a két alapvető model, a lineáris regresszió (regressziós feladatra) és a logisztikus regresszió (osztályozási feladatra) használatát.

- Példa. Többváltozós lineáris regresszió a Diabetes adatállományra
Az sklearn Diabetes adatállományán építünk lineáris regressziós modellt a betegség előrehaladottságát (disease progression) mérő folytonos célváltozó előrejelzésére. Az adatállományt tanító és teszt részre bontjuk majd összehasonlítjuk az R-négyzet értékeket. Az előrejelzés pontosságát grafikusan is szemléltetjük.

- Példa. Többváltozós logisztikus regresszió a Breast_cancer adatállományon
Az sklearn Breast_cancer adatállományán építünk logisztikus regressziós modellt a daganat rossz vagy jóindulatúságát jellemző bináris célváltozó előrejelzésére. A modellt illesztjük a teljes adatállományra és kétféle módon számolunk előrejelzést ugyanarra az eredményre jutva. Ezután tanító és tesz állományokra bontva végezzük el a modell illesztést. Végül replikáció elemzéssel vizsgáljuk meg a pontosság ingadozását.

## 6. labor. 2021.10.11. hétfő és 2021.10.13. szerda

Sok gépi tanulási feladat esetén problémát okoz a nagyszámú attribútum jelenléte, pl. jelentősen megnöveli a futás idejét vagy megnehezíti az adatok grafikus ábrázolását. Gyakran a sok attribútum elrejti a valós struktúrát, amely sokkal alacsonyabb dimenziójú térre koncentrálódik. Ezen a problémán dimenziócsökkentéssel segíthetünk, amely az attribútumok számának csökkentését jelenti. A legfontosabb megközelítések az alábbiak:

- az attribútumok egy részének (legfontosabbaknak) a kiválogatása
- az attribútumok transzformálása új attribútumokká.
A mindkét megközelítésre többféle módszer ismert. Az elsőre példa amikor egy fontossági (korrelációs) mérőszámot határozzunk meg egyenként az input attribútumokra és az első K legnagyobb értékkel rendelkezőt tartjuk meg. A másodikra a legelterjedtebb módszer a főkomponens analízis (PCA-Principal Component Analysis).

- Példa. Többféle dimenziócsökkentés az Iris adatállományra
A Fisher-féle Iris adatállomány ábrázolása a 2D térben. 1. módszer: két tetszőleges attribútum megadása, 2. módszer: a 2 legfontosabb attribútum kiválasztása a SelectKBest módszerrel, 3. módszer: pontdiagram-mátrix készítése seaborn-nel, 4. módszer:  főkomponens analízissel.

A SelectKBest osztály kiválasztja az első K legnagyobb fontossági súllyal rendelkezó input atrribútumot. Legfontosabb jellemzői az alábbiak:

SelectKBest(): az osztály példányosítása, legfontosabb paraméter a megtartandó attribútumok K száma,
- fit(X,y): a modell illesztése és az attribútumok súlyainak meghatározása,
- transform(X): a kiválasztott attribútumok előállítása,
- get_support(): a kiválasztott attribútumok maszkja.
- Példa. A Digits adatállomány főkomponens analízise
A 10-féle számjegy kézzel írt képeit tartalmazó adatállomány főkomponens analízise, a fökomponensek varianciáinak és kumulatív varianciáinak ábrázolása. Az első két dimenzió ábrázolása a tanító és teszt állományra az osztályok színezésével. Hasonlítsuk össze a két ábrát!

A PCA osztály végrehajtja a főkomponens analízist, legfontosabb jellemzői az alábbiak:

- PCA(): az osztály példányosítása, legfontosabb paramétere a komponensek száma (n_components),
- fit(X): a modell illesztése az adatokra,
- transform(X): az X transzformálása a főkomponensek terébe.

https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA

## 7. labor. 2021.10.18. hétfő és 2021.10.20. szerda

Az osztályozási feladat (felügyelt tanítás diszkrét célváltozóval) megoldásának az egyik legszemléletesebb módszere  döntési fa osztályozó építése. Döntési fa alatt egy olyan fa-gráfot értünk, melynek gyökerében és belső csúcsaiban ún. vágási kifejezések vannak, a terminális vagy levél csúcsokban pedig osztálycímkék. A vágási feltétel általában egy input attribútumon alapuló logikai kifejezés, pl. az életkor>20. Ha a feltétel igaz, akkor az egyik, ha hamis akkor a másik ágon megyünk tovább. Egy rekord abba az osztályba sorolódik amelyhez tartozó levélbe a döntések végén jutunk. A döntési fa illesztése olyan fa építését jelenti, amelynek a lehető leghomogénebbek a levelei. A homogenitást többféle módon mérhetjük, az sklearn implementációjában a Gini és entrópia alapú mérőszámok szerepelnek.

Döntési fát a DecisionTreeClassifier osztállyal tudunk illeszteni, az osztály legfontosabb paraméterei és metódusai az alábbiak:

- DecisionTreeClassifier(): az osztály példányosítása, legfontosabb paraméterek a fa építésének kritériuma (criterion) és a fa mélysége (max_depth),

- fit(X,y): a modell illesztése az adatokra, ahol X az input, y az osztályozó változó,

- score(X,y): a model pontosságát mérő érték [0,1]-ben,

- predict(X): az osztály előrejelzése az illesztett modell alapján az X input rekordjaira,

- predict_proba(X): a posterior eloszlás becslése az illesztett modell alapján az X input rekordjaira

- decision_path(X): a döntések sorozata az X input rekordjaira.

- Példa. Döntési fák illesztése a spambase adatállományra
A példában kétféle döntési fát (Gini és entrópia alapút) illesztünk a Spambase adatállományra. Bináris osztályozási feladat: a cél a spam versus not spam előrejelzése. A program kirajzolja a fákat, meghatározza a pontosságot a tanító és teszt állományra, és prediktál a teszt állományon.

Az adatállomány az oktató honlapjáról töltődik be automatikusan.

- Példa. Döntési fák illesztése az Iris adatállományra
A példában kétféle döntési fát (Gini és entrópia alapút) illesztünk az Iris adatállományon. Nem bináris osztályozási feladat: a cél a 3 osztály előrejelzése az input rekordokon.  A program kirajzolja és el is menti a döntési fákat, valamint meghatározza a pontosságot a teljes adatállományra.

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontree#sklearn.tree.DecisionTreeClassifier

https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html?highlight=plot_tree#sklearn.tree.plot_tree

## 8. labor. 2021.10.25. és 2021.10.27. szerda

Néhány további alapvető osztályozási módszer ismertetése és összehasonlítása a logisztikus regresszióval.

A naív Bayes módszer a valószínűségszámítás Bayes-formuláján alapszik. Feltevése szerint mind az input mind pedig a cél attribútumok véletlen változók. A célváltozó feltételes valószínűségét kell kiszámolni az input változókra nézve és maximalizálni az osztályokra (maximum aposzteriori becslés). A Bayes formula alapján ez az input változók együttes feltételes valószínűségére vezet az osztályra nézve. Függetlenséget feltételezve, ezért naív, ez az input változók egyenkénti feltételes valószínűségére vezet az osztályra nézve. Ha az input változó folytonos akkor ezt a feltételes eloszlást normális eloszlással érdemes közelíteni. Így a Gauss-féle naív Bayes módszert kapjuk.

A legközelebbi társ módszer egy ún. lusta tanító. Nem épít modellt hanem mindig a tanító halmaz segítségével végzi el az előrejelzést. Legfontosabb paramétere a K, az ún. szomszédok száma. Egy új osztályozandó rekordnak meghatározzuk a K legközelebbi szomszédját a tanító állományban egy távolság-definíciót (pl. euklideszi) használva, majd ezek osztálycímkéi alapján a többségi szavazás elvével meghatározzuk a leggyakoribb osztálycímkét. Ebbe az osztályba fog majd az új rekord kerülni.

A neurális hálók napjaink egyik legelterjedtebb gépi tanítási módszere. Elemei számolási egységek, ún. neuronok, egy gráfba szervezett hálózata. Legfontosabb esete a ún. többrétegű perceptron (MLP-multilayer perceptron). Ez a hálózat egymás utáni rétegekbe szervezi a neuronjait, az első az input réteg az input attribútumokkal, az utolsó az output réteg a célváltozóval. Bármennyi közbenső, ún. rejtett réteget tartalmazhat. A modell paraméterei az egymás utáni rétegekbe eső neuronokat összekötő élek súlyai valamint a rejtett és az ouput réteg neuronjainak torzításai. További paraméter még az ún. aktivációs függvény, amely azt szabályozza hogy a bejövő súlyozott összegből a neuron mennyit ad tovább a következő réteg neuronjainak.

Az illesztett osztályozók a pontosságuk, amely a helyesen osztályozott és az összes rekord hányadosa, alapján hasonlíthatóak össze.

- Példa. Osztályozók illesztése a Spambase adatállományra
A példában a Spambase adatállományt olvassuk be az oktató URL-jéről és 4-féle osztályozót illesztünk az adatokra. A feladat bináris osztályozás. Meghatározzuk az osztályozók pontosságát (score) a tanító és a teszt állományokon.

- fit(X,y): a modell illesztése az adatokra, ahol X az input, y a célváltozó,
- score(X,y): a model pontosságát mérő érték a [0,1]-ben,
- predict(X): a célérték előrejelzése az illesztett modell alapján az X inputra,
- predict_proba(X): a posterior eloszlás becslése az illesztett modell alapján az X inputra.

- Példa. Osztályozók illesztése a Digits adatállományra
A kézzel írt számjegyek tanítása 4-féle osztályozóval. Többosztályos feladat: 0..9 (10 osztály).

https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html?highlight=gaussiannb#sklearn.naive_bayes.GaussianNB

https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?highlight=mlpclassifier#sklearn.neural_network.MLPClassifier

## 9. labor. 2021.11.08. hétfő és 2021.11.10. szerda

Ahhoz, hogy dönteni tudjunk a különböző gépi tanulási modellek közül melyeket ugyanarra az adatállományra illesztettünk, olyan mérőszámokra van szükségünk melyek jellemzik az adott modell jóságát. A modellek nagyon eltérő természete miatt ezek a mérőszámok csak a prediktív (előrejelző) képességen alapulhatnak. Itt csak az osztályozási feladatot tárgyaljuk. Legegyszerűbb az ún. pontosság, amely a helyesen osztályozott és az összes rekord aránya. Ettől kicsit részletesebb képet mutat az ún. tévesztési mátrix (confusion matrix). Ennél még részletesebb képet mutató eszközök a ROC-görbe melynek minél közelebb kell esnie az y tengely és y=1 egyeneshez és az az alatti terület, az ún. AUC (area under the curve) érték.

- Példa. Osztályozók összehasonlítása a Spambase adatállományra
A Spambase adatállomány bináris osztályozási feladatára két osztályozó (logisztikus regresszió és naív Bayes) kiértékelése. A bemutatott eszközök: pontosság, tévesztési mátrix, ROC görbe és AUC. A tévesztési mátrixot a confusion_matrix függvénnyel tudjuk kiszámolni, majd a plot_confusion_matrix függvénnyel megjeleníteni. A ROC görbe pontjait a roc_curve függvény számolja ki, majd a Pyplot megfelelő eszközeivel tudjuk megjeleníteni akár több görbét is egy koordináta-rendszerben. Egy másik, közvetlen grafikus eszköz a plot_roc_curve függvény de ez csak egy osztályozó ROC görbéjének megjelenítésére alkalmas.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html?highlight=confusion_matrix#sklearn.metrics.confusion_matrix

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html?highlight=plot#sklearn.metrics.plot_confusion_matrix

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html?highlight=roc_curve#sklearn.metrics.roc_curve

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html?highlight=plot#sklearn.metrics.plot_roc_curve

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html?highlight=auc#sklearn.metrics.auc

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=auc#sklearn.metrics.roc_auc_score

## 10. labor. 2021.11.15. hétfő és 2021.11.17. szerda

A nem-felügyelt tanítás egyik legfontosabb módszere a klaszterezés vagy klaszteranalízis. Nem-felügyelt tanításról akkor beszélünk amikor nincs kitüntetett célváltozó, minden változó inputként szerepel. A klaszterezés feladata olyan csoportok kialakítása az input rekordokból, melyekre az igaz, hogy az azonos csoportokba tartozó rekordok hasonlóan míg a különbözőekbe tartozók különbözően viselkednek. Ezt pontosan megfogalmazni a hasonlóság vagy távolság fogalmainak bevezetése révén lehet. Távolságra a legismertebb példa az euklideszi távolság (ld. Pitagorasz tétel), hasonlóságra pedig a koszinusz-hasonlóság.

A legelterjedtebb klaszterező algoritmus az ún. K-közép módszer. A módszer algoritmusa egy kétlépéses iteráció két ismeretlenben, a klaszter-középpontokban és a klaszter-tagságban:

- ha ismerjük a klaszter-középpontokat akkor egy rekordot abba a klaszterbe sorolunk amely középpontjához a legközelebb van,
- ha ismerjük a klaszter-tagságokat, akkor az új klaszter-középpontok a klaszterbeli rekordok átlagaként adódnak.
A módszer gyors és az iteráció viszonylag korán konvergál.
Klaszterezések jóságának mérésére többféle módszer is ismert, az egyik legelterjedtebb a Davies-Bouldin index. Minél kisebb az értéke annál jobb a klaszterezés. Minden címkével ellátott, csoportosított adatmezőre használható.
- Példa. Az Iris adatállomány klaszterezése a K-közép módszerrel
A példában az Iris adatállományt klaszterezzük a K-közép módszerrel szemléltetve a Kmeans sklearn osztályt. Az osztály legfontosabb attribútumai és metódusai az alábbiak:

- Kmeans() : az osztály példányosítása, legfontosabb parameter a klaszterek n_clusters száma,
- fit(X): a modell illesztése az adatokra a klaszter-középpontok meghatározásával, ahol X az input,
- score(X): -1-szerese a klaszteren belüli négyzetösszegnek, amit az inertia_ attribútum is előállít,
- predict(X): a klasztercímke előrejelzése az X rekordjaira, ami lehet más is mint az input, az inputra ugyanezt a labels_ attribútum is megadja,
- fit_predict(X): a modell illesztése és előrejelzés együtt az X inputra,
- transform(X): az X transzformálása a klaszter-középpontoktól való távolságok terébe,
- fit_transform(X): a modell illesztése és transformálása a távolságtérbe az X inputra.
A K-közép modell illesztése mellett a példában klaszteren belüli négyzetösszeget (SSE) és Davies–Bouldin indexet is számolunk.


- Példa. Az Aggregation adatállomány K-közép klaszterezése
Az Aggregation adatállomány egy mesterségesen generált példa, melyet az oktató honlapjáról tölt be a program. Konzolról megadható a K klaszterszám valamint diagnosztikai ábrákat csinál az SSE és DB klaszterezettségi mutatókra. A klasztereket pontdiagrammal ábrázolja a 2D térben.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

## 11. labor. 2021.11.22. hétfő és 2021.11.24. szerda

A középpont alapú K-közép módszer mellett más elveken működő klaszterezési módszerek is léteznek, melyek közül a legfontosabbak a hierarchikus és sűrűség alapú módszerek.

A hierarchikus módszereknél klaszterek egy hierarchiája épül fel, melyet egy ún. dendrogrammal (fa) lehet ábrázolni. Két fajtája van az összevonó (agglomeratív) vagy felosztó (divizív). Az első esetben kezdetben minden adatpont (rekord) egy külön klasztert alkot és egy távolság vagy hasonlóság függvény alapján az egymáshoz legközelebbi két klasztert összevonjuk amíg minden adatpont egy klaszterbe nem kerül. A második esetben pontosan fordítva járunk el. Az AgglomerativeClustering osztály az összevonó hierarchikus klaszterezést implementálja. A paraméterezésben 4-féle összevonási mód közül lehet választani: egyszerű, átlagos és teljes kapcsolás valamint Ward módszer.

A sűrűség alapú klaszterezésnél az adatpontok körüli más adatpontok sűrűsége a meghatározó. A DBSCAN függvény a DBSCAN módszert implementálja, amely belső, külső és határpontokra bontja a rekordjainkat. Belső pont az lesz, amelyik egy adott sugarú környezetében egy minimálisnál több pont van. Belső pont határán lévő pont határpont, míg a fennmaradó pontok külső vagy zajos pontok lesznek. A belső pontok összefüggő lánca alkot egy klasztert.

Egy klaszterezés eredményét össze lehet venni előre megadott osztálycímkékkel, ami referenciaként szolgálhat. Erre a kontingencia mátrix szolgál, melynek sorai az osztálycímkék, oszlopai a klasztercímkék. Egy cella értéke az abba a kategóriába eső rekordok száma.

- Péda. Az Aggregation adatállomány haladó klaszterezése
A példában az Aggregation mesterséges példa adatállományt (letölthető az oktató honlapjáról, a kód automatikusan beolvassa) klaszterezzük haladó módszerekkel (hierarchikus és DBSCAN) szemléltetve az AgglomerativeClustering és DBSCAN  sklearn osztályokat. Az AgglomerativeClustering osztály legfontosabb attribútumai és metódusai az alábbiak:

- AgglomerativeClustering() : az osztály példányosítása, legfontosabb parameter a klaszterezés módszere (linkage),
- fit(): a hierarchikus klaszter modell illesztése az adatokra vagy a távolságmátrixra,
- fit_predict(): a modell illesztése és a klasztercímkék előrejelzése.
- Az DBSCAN osztály legfontosabb attribútumai és metódusai az alábbiak::
- DBSCAN(): az osztály példányosítása, két legfontosabb paramétere a környezet sugara (eps) és a minimális pontok száma (min_samples),
- labels_: a klasztercímkék attribútuma,
- fit(): a modell illesztése az adatokra vagy a távolságmátrixra,
- fit_predict(): a modell illesztése és a klasztercímkék előrejelzése.

Meg kell jegyezni, hogy a DBSCAN osztálynak létezik egy dbscan függvény változata is.
A modellek illesztése mellett a példában Davies–Bouldin indexet és kontingencia mátrixot is számolunk.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html?highlight=agglomerativeclustering#sklearn.cluster.AgglomerativeClustering

## 12. labor. 2021.11.29. hétfő és 2021.12.01. szerda

Gépi tanulási algoritmusok sikeres futtatásához és az eredmények helyes értelmezéséhez számos további Python eszköztárat használhatunk, akár az sklearn könyvtáron belül, akár más könyvtárakból.

Az sklearn könyvtáron belül az alábbi segédeszközök érhetőek el:

- jellemző kinyerés (feature_selection)
- hiányzó értékek pótlása (impute)
- előfeldolgozás, úm. változó transzformáció, skálázás, diszkretizálás (preprocessing)
- csővezetékek (pipeline)
- modellek kiértékelése (metrics)
- utilitik, úm. véletlen mintavételezés, keverés (utils)
- kivételkezelés (exceptions)
Fontos segédeszközök az olyan könyvtárak, melyekkel gyorsan és egyszerűen készíthetünk alapstatisztikákat (pandas) és statisztikai grafikonokat (seaborn). 

A pandas eszközei (dataframe és leíró statisztikák).

Adatvizualizációs eszközök a MatPlotLib-ben és a Pandasban
