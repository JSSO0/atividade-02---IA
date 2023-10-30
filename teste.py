from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregue os dados
from ucimlrepo import fetch_ucirepo

wine = fetch_ucirepo(id=109)#carregando a base de dados
X = wine.data.features #recursos armazenados
y = wine.data.targets#os rotulos armazenados aqui


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#divide o conjunto de dados em um conjunto de treinamento o Xtrain e Ytrain e  o conjunto de teste Ytest e Xtest
#a divisão ocorre pelo split do train test
#test_size define o tamanho do teste, pra saber a % vc subtrai o tamanho por 1 = 1 - 0,3 = 70%
#e ramdom_state é pra garantir a reprodutibilidade dos resultados


knn = KNeighborsClassifier(n_neighbors=2)
#definimos a quantidade de vizinhos.


knn.fit(X_train, y_train)
#treinamos o nosso modelo.


y_pred = knn.predict(X_test)
#depois do modelo treinado, fazemos a previsão e guardamos em y-pred

accuracy = accuracy_score(y_test, y_pred)
#calculamos a acuracia, usando o teste com a previsão
print(f'Acurácia do KNN: {accuracy * 100:.2f}%')
#print dela.
