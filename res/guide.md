🔧 MONITOR STANU NARZĘDZI - INSTRUKCJA UŻYTKOWNIKA
═══════════════════════════════════════════════════════════

📋 PRZEGLĄD
Aplikacja umożliwia trenowanie modeli uczenia maszynowego do klasyfikacji stanu technicznego narzędzi oraz testowanie ich na nowych danych. Obsługuje algorytmy Random Forest i SVM do wykrywania:
• Stanu normalnego
• Niewyważenia
• Niewspółosiowości
• Uszkodzeń łożysk

═══════════════════════════════════════════════════════════

🎓 TRYB UCZENIA – Trenowanie nowych modeli

1 WYBÓR TYPU MODELU
   • Random Forest: szybki trening, dobra wydajność
   • SVM: dokładniejszy, ale wolniejszy

2 PRZYGOTOWANIE DANYCH
   Dane powinny być zorganizowane tak:
   
   res/
   └data/
      ├── misalignment/
      ├── normal/
      ├── unbalance/
      └── wearing/

   Każdy plik CSV: kolumny t, x, y, z (czas + przyspieszenia w danych osiach)

3 WYBÓR FOLDERU
   Kliknij "Browse", wybierz folder danych treningowych

4 NAZWA MODELU
   • Pusta: nazwa automatyczna np. random_forest_2024_01_15_14_30
   • Własna: np. "silnik_model_v1"

5 START
   • Kliknij "Start Training"
   • Obserwuj pasek postępu i zakładkę "Results"

═══════════════════════════════════════════════════════════

🔍 TRYB TESTOWANIA – Wykorzystanie wytrenowanego modelu

1 WYBÓR MODELU
   • Wybierz z listy
   • Kliknij "Refresh", by odświeżyć

2 WCZYTAJ MODEL
   • Kliknij "Load Model"
   • Wybierz plik pkl
   • Poczekaj na potwierdzenie

3 WYBÓR PLIKU TESTOWEGO
   • Kliknij "Browse", wskaż plik CSV (kolumny t,x,y,z jak w przypadku sygnałów treningowych)

4 TEST
   • Kliknij "Run Test"
   • Wyniki pojawią się w zakładce "Results"

═══════════════════════════════════════════════════════════

📊 WIZUALIZACJA DANYCH TRENINGOWYCH

1 WYBÓR PLIKU
   • Wskaż plik CSV dla danych treningowych

2 WIZUALIZUJ
   • Wykresy parametrów dla wybranej osi

💡 TYPY Łożysk:
   • Normal – normalne / poprawne 
   • Wearing – zespute / wystepujące szumy
   • Unbalance – niewyważone łożysko
   • Misalignment – niewspółosiowe łożysko

═══════════════════════════════════════════════════════════

📈 ZAKŁADKA WYNIKÓW

Zawiera:
• Graficzne przedstawienie zużycia łożyska
• Graficzne przedstawienie parametrów sygnału
• Postęp treningu
• Wyniki modelu
• Szczegóły testów
• Dziennik aktywności

═══════════════════════════════════════════════════════════

🛠️ ROZWIĄZYWANIE PROBLEMÓW

❌ "Training data folder does not exist" – sprawdź ścieżkę  
❌ "No data processed" – sprawdź strukturę folderów i CSV  
❌ "Model file not found" – kliknij "Refresh"  
❌ Pasek postępu nie działa – daj czas, sprawdź dziennik  
❌ Niska dokładność – zwiększ ilość i jakość danych

═══════════════════════════════════════════════════════════

📁 LOKALIZACJA PLIKÓW

Modele:
• res/model/random_forest/
• res/model/svm

Dodatkowo:
• Dziennik błędów: zakładka "Results"

═══════════════════════════════════════════════════════════
