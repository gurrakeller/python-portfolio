import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#vi använder pandas för att läsa in datasetet och tar bort ; med sep då alla instanser är separerade med semikolon när vi läser in den.
#vi benämner även variabeln som innehåller all denna data "data"

data = pd.read_csv("student-mat.csv", sep=";")

#här separerar vi den data vi vill använda i datasetet för att träna vår modell. i detta fall annvänder vi alltså 5 stycken attributer, då vi kommer att separera G3, vilket är den datan vi kommer försöka lista ut.

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#här ger vi våran "list", den variabeln vi vill försöka lista ut namnet "predict"

predict = "G3"

#här gör vi två arrays, i x använder vi data.drop för att ta bort predict(G3), så att vi tränar modellen utan det då vi annars hade memoriserat det.
#i y gör vi istället tvärt om och lagrar svars datan för att sedan kunna jämnföra och få fram enn accuracy på vår modell

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#här delar vi upp datan, test_size används för att ange mängden data vi vill använda för att träna och testa. i detta fall har vi valt 0.1 som då motsvarar 10% av datan.
#vi delar upp det i 4 olika variablar och använder en träningsmodell från sklearn.
#vad vi gjort nu är att förbereda datan för att kunna användas och tränas/testas på.

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#här använder vi nu statistiska modellen linearregression, vi har valt att gå för linearregression då den data vi använder är rätt så lokaliserad,
# hade datan vi använt istället varit mer spontan och inte så beroende av sigsjälv, hade det varit bättre att använda en annan modell då det skulle bli svårt att dra ett snitt sträck.
#linearregression använder formeln y=mx+b (i en två dimesionel linje, inte i vårat fall då vi använder fler attributer) detta kan visualiseras som en linjär ekvation.

linear = linear_model.LinearRegression()

#här säger vi till vilken data vi vill att den drar vårat sträck i, i detta fall vill vi att den använder x_train, y_train

linear.fit(x_train, y_train)

#här ger vi vårat resutat benämningen acc så när vi kallar på den behöver vi enbart göra print(acc)

acc = linear.score(x_test, y_test)
print(acc)

#här printar vi ut våra "coefficients" vilket är alla våra attributer, det vill säga g1, g2, studytime o.s.v i en array. /n används helt enkelt för att printa på en ny rad.
#intercept använda för att printa ut vart vi skär Y axeln i vår ekvation. i detta fall -1.5

print(('coefficient: \n', linear.coef_))
print(('intercept: \n', linear.intercept_))

#vi använder nu linear.predict för att helt enkelt säga till vilken data det är vi vill att den testar sin nu tränade modell på.
#vi ger även denna variabeln namnet predictions.

predictions = linear.predict(x_test)

# Här går loopen igenom varje instans i 'predictions' och jämför förutsägelsen med de verkliga värdena från 'y_test'.
# Vi skriver ut varje förutsagt värde, de ingående funktionerna (features) och det verkliga värdet för att visuellt kunna jämföra modellens prestanda.

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])