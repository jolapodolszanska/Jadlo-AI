# JadłoAI - Rozpoznawanie jedzenia ze zdjęcia

nteligentna aplikacja do planowania posiłków, generowania przepisów i bilansowania diety z pomocą modeli AI. „Jadło AI” łączy preferencje użytkownika, cele żywieniowe i zawartość lodówki, 
aby automatycznie tworzyć zdrowe, zbilansowane jadłospisy oraz listy zakupów.

**Aplikacja jest w trakcie rozbudowy**

<img src="https://github.com/jolapodolszanska/Jadlo-AI/blob/main/screen-apki.png"></img>

## Najważniejsze funkcje

- Wgrywanie obrazu metodą „kliknij lub przeciągnij i upuść” (JPG/PNG/WEBP, limit 10 MB) i podgląd zdjęcia. 
- Uruchomienie modelu ONNX w przeglądarce przez onnxruntime-web (CDN ort.min.js). 
- Top-3 predykcje z prawdopodobieństwami (softmax), pasek postępu oparty o pewność. 
- Karta potrawy: makra (białko, węglowodany, tłuszcze), kalorie i wybrane mikro (błonnik, cukier, sód, wapń, żelazo, wit. C, potas). 
- Klasyfikacja indeksu glikemicznego (niski/średni/wysoki) oraz wyliczana heurystycznie „ocena zdrowotna” 1–10 z listą czynników na plus/minus. 
- Sprawdzanie zgodności z dietą wybraną z listy: standard, keto, wegańska, low-carb, śródziemnomorska, wraz z komunikatem i piktogramem zgodności. 
- Wskazanie potencjalnych alergenów i lista „zdrowszych alternatyw” zależna od kategorii potrawy i profilu diety. 

## Wymagania

- Dowolny prosty serwer statyczny (potrzebny, aby przeglądarka mogła pobrać *.onnx/*.json/*.txt przez fetch). 
- Nowoczesna przeglądarka z włączonym JavaScriptem.

## Szybki start

1. Umieść w jednym katalogu: app.html, food101.onnx, labels.txt, nutrition.json. 
2. Uruchom lokalny serwer, np.:

```pyton
# Python
python -m http.server 8080
# albo node (serve)
npx serve .
```

3. Wejdź na http://localhost:8080/app.html.

## Struktura plików

```
.
├── app.html            # aplikacja 
├── food101.onnx        # model ONNX (klasy Food-101)
├── labels.txt          # etykiety (jedna klasa na linię)
└── nutrition.json      # dane żywieniowe i metadane potraw
```

## Format nutrition.json

Plik nutrition.json to słownik, w którym kluczem jest nazwa potrawy (zgodna z labels.txt lub możliwa do rozpoznania przez dopasowanie fragmentu), a wartością obiekt z polami żywieniowymi i metadanymi. Minimalny przykład:

```
{
  "pizza": {
    "calories": 266,
    "protein": 11,
    "carbs": 33,
    "fat": 10,
    "fiber": 2.3,
    "sugar": 3.6,
    "sodium": 640,
    "calcium": 188,
    "iron": 2.6,
    "vitaminC": 2,
    "potassium": 172,
    "saturatedFat": 4.5,
    "glycemicIndex": 80,
    "category": "fast_food",
    "allergens": ["mleko", "gluten"],
    "highRiskAllergens": ["gluten"]
  }
}
```

Pola używane w UI: makra/mikro (jak wyżej), glycemicIndex, category, allergens, highRiskAllergens. Brak wpisu oznacza skromniejszą kartę potrawy („Brak szczegółowych danych żywieniowych”). 

## Pomysły na rozwój

- Dodanie estymacji porcji/gramatury i przeliczeń „na porcję”.
- Obsługa WebGPU/WebGL EP w ONNX Runtime Web dla przyspieszenia inferencji.
- Lokalizacja nazw potraw i mapowanie etykiet → nazwy przyjazne dla użytkownika.
- Zapisywanie historii analiz w localStorage.
- Rozszerzenie schematu nutrition.json (np. witaminy A/E/K, profil kwasów tłuszczowych).
- Eksport karty potrawy do PDF.

  ## Licencja

  MIT License

Copyright (c) [YEAR] [Full name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
