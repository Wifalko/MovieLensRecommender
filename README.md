# MovieLensRecommender
Systemy rekomendacji oparte na sieciach neuronowych stanowią jedno z najbardziej zaawansowanych rozwiązań w dziedzinie filtracji treści, umożliwiając modelowanie złożonych preferencji użytkowników i dokładniejsze przewidywanie ich zainteresowań. W oparciu o zbiór danych MovieLens, który zawiera obszerny zestaw ocen filmów oraz szczegóły dotyczące interakcji użytkowników, można stworzyć model rekomendacyjny, który wykorzystuje sieci neuronowe do przechwycenia wzorców i zależności w danych.

Przy budowie takiego systemu rekomendacji można wykorzystać zarówno głębokie sieci neuronowe (DNN), jak i bardziej wyspecjalizowane architektury, takie jak sieci konwolucyjne (CNN) czy rekurencyjne sieci neuronowe (RNN), które znajdują zastosowanie w analizie sekwencji preferencji użytkowników. Przykładowo, sieci oparte na architekturze Collaborative Filtering z wykorzystaniem uczenia reprezentacji użytkowników i filmów pozwalają na efektywne modelowanie wektorów latentnych, odzwierciedlających cechy wpływające na podobieństwo preferencji i gustów.

Użycie sieci neuronowych do rekomendacji na danych MovieLens ma na celu nie tylko personalizację treści, ale także optymalizację procesu uczenia się preferencji użytkownika w czasie. Wprowadzenie warstw głębokiego uczenia pozwala na przechwytywanie bardziej abstrakcyjnych cech, które mogą być niewidoczne w klasycznych podejściach opartych na macierzach współczynników korelacji, dzięki czemu rekomendacje są precyzyjniejsze i bardziej spójne z rzeczywistymi zainteresowaniami użytkowników.


# Metoda KNN (k-Nearest Neighbors)
Metoda k-Nearest Neighbors (KNN) jest klasyczną techniką uczenia maszynowego używaną do wyszukiwania podobieństw między użytkownikami lub elementami w systemach rekomendacyjnych. W kontekście rekomendacji na danych MovieLens, KNN można zastosować do identyfikacji użytkowników o podobnych preferencjach (User-Based Collaborative Filtering) lub filmów podobnych do tych już ocenionych przez użytkownika (Item-Based Collaborative Filtering).

Działanie metody KNN w rekomendacjach:
Reprezentacja danych:

Użytkownicy i filmy są reprezentowani w przestrzeni wielowymiarowej. W przypadku filtracji użytkowników, każdy wymiar może odpowiadać ocenie użytkownika dla konkretnego filmu, a brakujące oceny są zastępowane wartością domyślną lub pomijane.
Metryka podobieństwa:

Do określenia podobieństwa pomiędzy użytkownikami lub filmami stosuje się różne metryki, takie jak:
Cosine Similarity: Miara kąta między wektorami w przestrzeni wielowymiarowej.
Pearson Correlation: Miara liniowej zależności między ocenami.
Jaccard Index: Miara zgodności między zestawami danych binarnych.
Wyszukiwanie najbliższych sąsiadów:

Dla danego użytkownika (lub filmu) identyfikowanych jest 𝑘 najbliższych sąsiadów, czyli tych użytkowników (lub filmów), które mają najwyższe wartości podobieństwa.

Zalety KNN:
Prosta implementacja i intuicyjne działanie.
Brak potrzeby długiego uczenia modelu, ponieważ algorytm jest obliczeniowy.

Wady KNN:
Skalowalność: Wyszukiwanie sąsiadów w dużych zbiorach danych może być kosztowne obliczeniowo.
Nie przechwytuje złożonych wzorców w danych, ograniczając jego skuteczność w bardziej skomplikowanych scenariuszach.


# Metoda NCF (Neural Collaborative Filtering)
Neural Collaborative Filtering (NCF) to nowoczesne podejście do systemów rekomendacji, które łączy zalety tradycyjnego Collaborative Filtering (CF) z możliwościami uczenia głębokiego. Metoda ta wykorzystuje sieci neuronowe do modelowania nieliniowych zależności między użytkownikami a elementami, co pozwala na dokładniejsze odwzorowanie złożonych preferencji.

Działanie metody NCF:
Reprezentacja użytkowników i elementów:

Każdy użytkownik i element (np. film) jest reprezentowany przez wektor osadzony (embedding), który jest wyuczony w trakcie trenowania modelu.
Warstwy interakcji:

Wektory użytkowników i elementów są łączone za pomocą różnych operacji:
Operacje liniowe: Takie jak iloczyn skalarny (dot product) lub konkatenacja.
Operacje nieliniowe: Sieć neuronowa przekształca połączone wektory, aby uchwycić nieliniowe zależności.
Architektura modelu:

Architektura NCF zazwyczaj zawiera następujące komponenty:
Input Layer: Przyjmuje wektory osadzeń dla użytkowników i elementów.
Hidden Layers: Głębokie warstwy neuronowe przekształcające dane wejściowe w sposób nieliniowy, umożliwiając przechwytywanie złożonych wzorców.
Output Layer: Przewiduje ocenę filmu przez użytkownika lub prawdopodobieństwo, że użytkownik polubi dany film.
Funkcja straty:
Model jest trenowany za pomocą funkcji straty, np.:
- Mean Squared Error (MSE): W przypadku przewidywania ocen.
- Binary Cross-Entropy: W przypadku przewidywania klikalności lub prawdopodobieństwa interakcji.

Proces uczenia:
Wagi w sieci neuronowej są aktualizowane za pomocą algorytmu optymalizacji (np. Adam, SGD) na podstawie danych treningowych.

Zalety NCF:
Umożliwia modelowanie złożonych nieliniowych zależności, których nie wychwytuje tradycyjny CF.
Skalowalność i elastyczność dzięki zastosowaniu sieci neuronowych.
Może być łatwo rozbudowany o dodatkowe dane wejściowe, takie jak metadane filmów lub demografia użytkowników.

Wady NCF:
Wymaga dużej liczby danych i mocy obliczeniowej.
Może być podatny na nadmierne dopasowanie (overfitting), jeśli nie zastosuje się odpowiednich technik regularyzacji.
