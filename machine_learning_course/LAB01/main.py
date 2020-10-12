import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split


def todo2_todo3():
    # Korzystajac z biblioteki scikit-learn załadować dataset digits
    digits = datasets.load_digits()

    # Jakie dane przechowuje ta baza? Jaki jest ich format?
    print("Dataset keys: {}".format(digits.keys()))
    data = digits['data']
    images = digits['images']
    targets = digits['target']

    # Jaka jest różnica pomiędzy danymi w data a images?
    # - Roznica pomiedzy data a images jest taka, że w tablicy dane te 1797 zdjec zapisane jest w jednowymiarowym
    # wektorze o długosci 64, podczas gdy w macierzy images te 1797 zdjęć zapisane jest jako macierz 8x8. Przechowywanie
    # danych w jednowymiarowym wektorze poprawia szybkość dostepu do danych i jest łatwiejsze dla komputera

    # Wyswietlic jedno ze zdjec jako macierz numpy korzystajac z biblioteki matplotlib
    plt.imshow(images[0], cmap='gray')
    plt.show()

    # Stworz klaysifkator SVC i wytrenuj (metoda fit) go na jednym zdjęciu z bazy danych
    svc_clf = svm.SVC()
    svc_clf.fit(data[0:2], targets[0:2])

    # Przetestuj klasyfikator na tym samym zdjeciu (metoda predict), wyswietl wynik
    print("Wynik predykcji {}".format(svc_clf.predict([data[0]])[0]))
    print("Rzeczywista wartosc {}".format(targets[0]))

    # Wykorzystaj funkcje do podzialu danych dla zbioru digits:
    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.8, shuffle=True)
    svc_clf_normal = svm.SVC()
    svc_clf_normal.fit(x_train, y_train)
    score = svc_clf_normal.score(x_test, y_test)
    print("Wynik trenowania algorytmu: ", score)
    prediction = svc_clf_normal.predict([x_test[-1]])
    print("Wynik predykcji: {}. Rzeczywista wartość: {}".format(prediction[0], y_test[-1]))


def todo4():
    # Korzystając z biblioteki scikit-learn załadować dataset Olivetti faces
    olivetti_faces = datasets.fetch_olivetti_faces()
    data = olivetti_faces['data']
    targets = olivetti_faces['target']
    images = olivetti_faces['images']

    # Podzielić zbiór zdjęć wraz z przypisanymi im klasami na zbiory treningowy (80% wszystkich zdjęć)
    # oraz testowy (20% wszystkich zdjęć).
    number_of_classes = 40
    number_of_samples_per_class = 10
    number_of_pixels_per_feature = data.shape[1]
    number_of_pixels_per_feature_dimension = images.shape[2]

    target_reshaped = np.reshape(targets, (number_of_classes, number_of_samples_per_class))
    data_reshaped = np.reshape(data, (number_of_classes, number_of_samples_per_class, number_of_pixels_per_feature))
    images_reshaped = np.reshape(images, (number_of_classes, number_of_samples_per_class,
                                          number_of_pixels_per_feature_dimension,
                                          number_of_pixels_per_feature_dimension))
    x_train, x_test, y_train, y_test = train_test_split(images_reshaped, target_reshaped, test_size=0.2, shuffle=True)

    # Wyświetlić zdjęcia osób ze zbioru testowego wraz z etykietami.
    for i in range((x_test.shape[0])):
        plt.figure()
        plt.imshow(x_test[i][0], cmap='gray')
        plt.title(y_test[i][0])
        plt.show()


def todo5():
    # Korzystając z biblioteki scikit-learn załadować wybraną przez siebie, ale inną niż wykorzystywane do tej pory,
    # bazę danych
    iris = datasets.load_iris()
    feature_names = iris['feature_names']  # nazwy cech, dlugosc lodygi, szerokosc lodyki, dlugosc kwiatostanu, szerokosc
    data = iris['data']  # 150 roznych kwiatow, kazdy ma 4 wymienione wyzej pola
    target = iris['target']  # odmiany irysow
    target_names = iris['target_names']  # nazwy odmian

    # Jakie dane przechowuje ta baza? Jaki jest ich format? Jakiego typu jest to problem?
    # Czy niezbędne jest dodatkowe przetwarzania danych?
    # Jest to problem klasyfikacji

    # Wyświetlić / wypisać przykładowe dane.
    print("Rodzaje kwiatow: {}".format(target_names))
    print("Cechy kwiatow: {}".format(feature_names))

    # Podzielić dane na podzbiór do uczenia oraz testowania.
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True)

    # Spróbować wytrenować model.
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("Wynik uczenia: {}".format(score))
    prediction = clf.predict([x_test[7]])
    print("Przykladowa predykcja: {}. Rzeczywista wartosc: {}".format(target_names[prediction[0]],
                                                                      target_names[y_test[7]]))


def todo6():
    # Korzystając z funkcji make_classification wygenerować nowy zbiór danych
    x, y = datasets.make_classification()

    # Jakie dane przechowuje ta baza? Jaki jest ich format? Jakiego typu jest to problem?
    # Czy niezbędne jest dodatkowe przetwarzania danych?
    # Jest to problem klasyfikacji. Ta baza przechowuje 100 próbek, po 20 cech na każdą. Przetwarzanie danych może
    # polegać na pozbyciu się niektórych cech

    # Model bedzie bral pod uwage tylko 4 zhardkodowane cechy
    x_reduced = x[:, (0, 5, 8, 9)]
    x_train, x_test, y_train, y_test = train_test_split(x_reduced, y, test_size=0.2, shuffle=True)

    # Spróbować wytrenować model.
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("Wynik uczenia: {}".format(score))
    prediction = clf.predict([x_test[7]])
    print("Przykladowa predykcja: {}. Rzeczywista wartosc: {}".format(prediction[0],
                                                                      y_test[7]))


def todo7():
    # Korzystając z platformy https://www.openml.org/ oraz funkcji fetch_openml
    # (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html) pobierz wybrany przez
    # siebie zbiór danych (dostępnych jest ponad 3000 zestawów danych, można filtrować po liczbie "lajków",
    # liczbie klas, brakujących danych itp.).

    # This database encodes the complete set of possible board configurations at the end of tic-tac-toe games,
    # where "x" is assumed to have played first. The target concept is "win for x" (i.e., true when "x" has one of
    # 8 possible ways to create a "three-in-a-row").
    tic_tac_toe = datasets.fetch_openml(name='tic-tac-toe')
    target = tic_tac_toe['target']
    data = tic_tac_toe['data']
    categories = tic_tac_toe['categories']
    list_of_moves = categories['top-left-square']

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True)

    clf = svm.SVC()
    x_train_diagonal = x_train[:, (0, 4, 8)]
    x_test_diagonal = x_test[:, (0, 4, 8)]
    clf.fit(x_train_diagonal, y_train)
    score = clf.score(x_test_diagonal, y_test)
    print("Wynik modelu: {}".format(score))
    prediction = clf.predict([x_test_diagonal[-1]])
    print("Wynik positive ma miejsce wtedy kiedy wygral x")
    print("Przykladowa predykcja: {}. Rzeczywisty wynik: {}.".format(prediction[0], y_test[-1]))
    rozklad = x_test[-1].reshape(3, 3)
    rozklad_string = ""
    for i in range(0, 3):
        for j in range(0, 3):
            rozklad_string = rozklad_string + list_of_moves[int(rozklad[i][j])]
        rozklad_string = rozklad_string + "\n"

    print("Rozklad na planszy: \n", rozklad_string)
