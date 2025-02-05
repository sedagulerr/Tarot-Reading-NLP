# Tarot-Reading-NLP
Doğal dil işleme (NLP) modeli kullanılarak kullanıcı girdilerine göre tarot kartları öneriliyor ve kartların anlamları kullanıcıya sunuluyor.

#This code was made via "Colab".
#The dataset was found on "kaggle". https://www.kaggle.com/datasets/lsind18/tarot-json

#Using the NLP Model: The code utilizes the SentenceTransformer model, which is used for text-based similarity calculations.
#Processing User Inputs: The user's input is analyzed to determine their intent, and a suitable tarot card is selected.
#Displaying Images: Tarot card images are loaded and displayed.
#Console Colors: ANSI escape sequences are used to create colorful and stylized console outputs.
#Q&A and Recommendation Mechanism: A system is built to provide suggestions based on user input and offer a second tarot card recommendation.

# Doğal dil işleme (NLP) modeli kullanılarak kullanıcı girdilerine göre tarot kartları öneriliyor ve kartların anlamları kullanıcıya sunuluyor.
# NLP Modeli Kullanımı: Kodu yazarken SentenceTransformer modeli kullanılıyor. Bu model, metin tabanlı benzerlik hesaplamaları için kullanılıyor.
# Kullanıcı Girdilerinin İşlenmesi: Kullanıcının girdiği metin analiz edilerek niyet belirleniyor ve uygun tarot kartı seçiliyor.
# Görsellerin Gösterimi: Kartların görselleri yüklenip gösteriliyor.
# Konsol Renkleri: Konsolda renkli ve stilize edilmiş çıktılar oluşturmak için ANSI kaçış dizileri kullanılıyor.
# Soru-Cevap ve Tavsiye Mekanizması: Kullanıcı girdilerine göre önerilerde bulunan ve kullanıcıya ikinci bir kart önerisi sunan bir yapı oluşturuluyor.


import numpy as np
import pandas as pd
import random
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import json
import seaborn
import pandas as pd
import os
import cv2
import spacy
import ipywidgets as widgets
from IPython.display import display, clear_output
from pandas import json_normalize
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import SentenceTransformer
from scipy.ndimage import rotate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# JSON dosyasından tarot kartlarını okuma
data_dir = '/content/'  # Görsellerin ve JSON'un olduğu dizin
df = pd.read_json(data_dir + 'tarot-images.json', orient='records')
df_cards = pd.json_normalize(df['cards'])
df_cards

# Konsol renkleri
class ConsoleColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    ORANGE = "\033[38;5;214m"

# Bugünün nasıl geçecek görmek ister misin?
# JSON dosyasından tarot kartlarını okuma
df_cards = pd.json_normalize(df['cards'])

# Kullanıcıya sorma
response = input("Bugünün nasıl geçecek görmek ister misin? (evet/hayır): ").strip().lower()

if response in ["evet", "yes"]:
    # Rastgele bir kart seçme
    random_card_idx = random.randint(0, len(df_cards) - 1)
    random_card = df_cards.iloc[random_card_idx]

    # Light ve Shadow anlamlarını karşılaştırma
    light_meanings = random_card['meanings.light']
    shadow_meanings = random_card['meanings.shadow']
    light_count = len(light_meanings)
    shadow_count = len(shadow_meanings)

    # Light ve Shadow'a göre mesaj belirleme
    if light_count > shadow_count:
        dominance_message = f"{ConsoleColors.GREEN}Kartın genel yorumu: Olumlu 🌟{ConsoleColors.RESET}"
    elif shadow_count > light_count:
        dominance_message = f"{ConsoleColors.RED}Kartın genel yorumu: Olumsuz ⚡{ConsoleColors.RESET}"
    else:
        dominance_message = f"{ConsoleColors.BLUE}Kartın genel yorumu: Dengeli ⚖{ConsoleColors.RESET}"

    # Kart bilgilerini yazdırma
    print(f"\nSizin için seçilen kart: {random_card['name']}")
    print(f"Anlamı (Light): {light_meanings}")
    print(f"Anlamı (Shadow): {shadow_meanings}")
    print(dominance_message)

    # Kart görselini gösterme
    image_path = os.path.join(data_dir, random_card['img'])
    if os.path.exists(image_path):
        img = plt.imread(image_path)
        plt.figure(figsize=(5, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title(random_card['name'], fontsize=16, fontweight='bold')
        plt.show()
    else:
        print(f"{ConsoleColors.RED}Görsel bulunamadı: {image_path}{ConsoleColors.RESET}")
else:
    print(f"{ConsoleColors.BLUE}Tamam, bir dahaki sefere görüşürüz!{ConsoleColors.RESET}")

# Ne hisettigini yazinca bir kart onerme (ing olarak yazılan)
# Modeli yükleme
model = SentenceTransformer('all-MiniLM-L6-v2')

# JSON dosyasından tarot kartlarını okuma
df_cards = pd.json_normalize(df['cards'])

# Kart açıklamalarını hazırlama
card_descriptions = df_cards['meanings.light'].fillna('').tolist()
card_embeddings = model.encode(card_descriptions)

# Kullanıcı girdisi
user_input = input("Bir soru sorun ya da ne hissettiğinizi yazın: ") #Kullanıcı Girdisinin Analiz Edilmesi.
user_embedding = model.encode(user_input)

# Benzerlik hesaplama
# kart açıklamaları ile kullanıcının girdisinin benzerliği hesaplanıyor ve en yüksek
scores = util.cos_sim(user_embedding, card_embeddings)[0]
best_card_idx = scores.argmax().item()  # Tamsayıya dönüştürme

# En iyi eşleşmeyi yazdırma
best_card = df_cards.iloc[best_card_idx]

# Light ve Shadow anlamlarını kontrol etme
light_meanings = best_card['meanings.light']
shadow_meanings = best_card['meanings.shadow']
light_count = len(light_meanings)
shadow_count = len(shadow_meanings)

# Light ve Shadow'a göre mesaj belirleme
if light_count > shadow_count:
    dominance_message = f"{ConsoleColors.GREEN}Kartın genel yorumu: Olumlu 🌟{ConsoleColors.RESET}"
elif shadow_count > light_count:
    dominance_message = f"{ConsoleColors.RED}Kartın genel yorumu: Olumsuz ⚡{ConsoleColors.RESET}"
else:
    dominance_message = f"{ConsoleColors.BLUE}Kartın genel yorumu: Dengeli ⚖{ConsoleColors.RESET}"

# Kart bilgilerini yazdırma
print(f"\nSizin için seçilen kart: {best_card['name']}")
print(f"Anlamı (Light): {light_meanings}")
print(f"Anlamı (Shadow): {shadow_meanings}")
print(dominance_message)

# Kart görselini gösterme
image_path = os.path.join(data_dir, best_card['img'])
if os.path.exists(image_path):
    img = plt.imread(image_path)
    plt.figure(figsize=(5, 7))
    plt.imshow(img)
    plt.axis('off')
    plt.title(best_card['name'], fontsize=16, fontweight='bold')
    plt.show()
else:
    print(f"Görsel bulunamadı: {image_path}")


 # soru örnekleri   How will my career progress?    I'm feeling so stressed, what can I do?    What will my friendships be like?


# Niyet Analizi

# NLP modeli yükle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Tarot kartlarını yükle
data_dir = '/content/'  # Görsellerin olduğu klasör yolu
json_path = '/content/tarot-images.json'
df = pd.read_json(json_path, orient='records')
df_cards = pd.json_normalize(df['cards'])

# Suit açıklamaları
suit_explanations = {
    "Wands": "Ateş elementinin serisidir. Aslan, Koç, Yay burçlarını temsil eder. Bu kartlar hareketliliğe, hıza ve yaratıcılığa dair bilgileri bize sunar.",
    "Cups": "Su elementinin serisidir. Yengeç, Akrep, Balık burçlarını temsil eder. Bu kartlar duygulara, alışkanlıklara ve bilinçaltına dair bilgileri bize sunar.",
    "Swords": "Hava elementinin serisidir. İkizler, Kova, Terazi burçlarını temsil eder. Bu kartlar düşüncelere, inançlara ve zihinsel mücadelelere dair bilgileri sunar.",
    "Pentacles": "Toprak elementinin serisidir. Boğa, Başak, Oğlak burçlarını temsil eder. Bu kartlar dünya işleri, doğa ve fiziksel yapıya dair bilgileri verir.",
    "Major Arcana": "Büyük Arkana kartları yaşamın önemli temalarını ve dönüşümlerini temsil eder. Her biri farklı bir ders sunar."
}

# Kart görselini gösterme
def display_card_image(card):
    image_path = os.path.join(data_dir, card['img'])
    if os.path.exists(image_path):
        img = plt.imread(image_path)
        plt.figure(figsize=(5, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title(card['name'])
        plt.show()
    else:
        print(f"{ConsoleColors.RED}Görsel bulunamadı: {image_path}{ConsoleColors.RESET}")

# Light ve Shadow baskınlığına göre mesaj ekleme
def determine_dominance(light_meanings, shadow_meanings):
    light_count = len(light_meanings)
    shadow_count = len(shadow_meanings)

    if light_count > shadow_count:
        return f"{ConsoleColors.GREEN}Kartın genel yorumu: Olumlu 🌟{ConsoleColors.RESET}"
    elif shadow_count > light_count:
        return f"{ConsoleColors.RED}Kartın genel yorumu: Olumsuz ⚡{ConsoleColors.RESET}"
    else:
        return f"{ConsoleColors.BLUE}Kartın genel yorumu: Dengeli ⚖{ConsoleColors.RESET}"

# Niyet ve girişlere göre kart önerme
def recommend_card_by_intent(user_input):
    intent = "Genel"
    suit_selection = None

    if any(word in user_input for word in ["aşk", "ilişki", "sevgi", "arkadaş", "duygu", "romantizm"]):
        suit_selection = "Cups"
        intent = "Aşk ve İlişkiler"
    elif any(word in user_input for word in ["iş", "kariyer", "para", "zenginlik", "başarı", "maddi"]):
        suit_selection = "Pentacles"
        intent = "Kariyer ve Maddiyat"
    elif any(word in user_input for word in ["sağlık", "şifa", "hastalık", "iyileşme", "beden", "ruh"]):
        suit_selection = "Major Arcana"
        intent = "Sağlık ve Şifa"
    elif any(word in user_input for word in ["düşünce", "zihin", "mantık", "karar", "analiz", "fikri"]):
        suit_selection = "Swords"
        intent = "Zihin ve Karar"
    elif any(word in user_input for word in ["yaratıcı", "enerji", "hareket", "tutku", "hedef"]):
        suit_selection = "Wands"
        intent = "Yaratıcılık ve Hareket"

    # Seçilen suit'e göre kart filtreleme
    filtered_cards = df_cards[df_cards['suit'].str.contains(suit_selection, case=False, na=False)] if suit_selection else df_cards
    selected_card = filtered_cards.sample(1).iloc[0]

    # Kart bilgilerini yazdırma
    print(f"{ConsoleColors.ORANGE}{intent} kategorisinde sizin için bir kart seçtim!{ConsoleColors.RESET}")
    print(f"{ConsoleColors.BOLD}Kart İsmi:{ConsoleColors.RESET} {selected_card['name']}")
    print(f"{ConsoleColors.GREEN}Light Anlam:{ConsoleColors.RESET} {selected_card['meanings.light']}")
    print(f"{ConsoleColors.RED}Shadow Anlam:{ConsoleColors.RESET} {selected_card['meanings.shadow']}")

    # Suit açıklaması ekleme
    suit_description = suit_explanations.get(suit_selection, "Genel bir kategori açıklaması mevcut değil.")
    print(f"{ConsoleColors.BOLD}Suit Açıklaması:{ConsoleColors.RESET} {suit_description}")

    # Light ve Shadow anlamlarını değerlendir
    dominance_message = determine_dominance(selected_card['meanings.light'], selected_card['meanings.shadow'])
    print(dominance_message)

    display_card_image(selected_card)

    return selected_card


# İkinci kart önerme
def recommend_related_card(first_card):
    print(f"\n{ConsoleColors.YELLOW}İlk kartınıza göre ikinci kart öneriliyor...{ConsoleColors.RESET}")

    # İlk kartın light anlamını al
    first_card_meaning = first_card['meanings.light']

    # Diğer kartları filtrele (ilk kart hariç)
    other_cards = df_cards[df_cards['name'] != first_card['name']]

    # Tüm kartların light anlamlarını al ve benzerlik skoru hesapla
    card_meanings = other_cards['meanings.light'].tolist()
    embeddings = model.encode([first_card_meaning] + card_meanings)
    similarity_scores = util.cos_sim([embeddings[0]], embeddings[1:])

    # En yüksek benzerlik skoruna sahip olan kartı seç
    best_match_index = similarity_scores.numpy().argmax()  # NumPy yöntemi ile argmax işlemi
    second_card = other_cards.iloc[best_match_index]

    # İkinci kart bilgilerini yazdır
    print(f"{ConsoleColors.BOLD}İkinci Kart:{ConsoleColors.RESET} {second_card['name']}")
    print(f"{ConsoleColors.GREEN}Light Anlam:{ConsoleColors.RESET} {second_card['meanings.light']}")
    print(f"{ConsoleColors.RED}Shadow Anlam:{ConsoleColors.RESET} {second_card['meanings.shadow']}")

    # Suit açıklaması ekle
    suit_description = suit_explanations.get(second_card['suit'], "Genel bir kategori açıklaması mevcut değil.")
    print(f"{ConsoleColors.BOLD}Suit Açıklaması:{ConsoleColors.RESET} {suit_description}")

    # Görseli göster
    display_card_image(second_card)

# Tarot kart önerme uygulaması

def main():
    output = widgets.Output()
    display(output)

    while True:
        with output:
            clear_output(wait=True)
            user_input = input("\nBir soru sorun ya da ne hissettiğinizi yazın (Çıkmak için 'çıkış'): ").strip()

            if user_input.lower() in ["çıkış", "exit", "quit"]:
                print(f"{ConsoleColors.ORANGE}Görüşmek üzere, tekrar bekleriz! 🌟{ConsoleColors.RESET}")
                break

            # İlk kartı seç
            first_card = recommend_card_by_intent(user_input)

            while True:
                cont = input(f"\nİlk karta göre ikinci bir kart seçmek ister misiniz? (evet/hayır): ").strip().lower()
                if cont in ["evet", "yes"]:
                    recommend_related_card(first_card)
                    break  # İkinci kart seçildi, döngüden çık
                elif cont in ["hayır", "no"]:
                    print(f"{ConsoleColors.ORANGE}Harika bir gün geçirmenizi dilerim! 🌟{ConsoleColors.RESET}")
                    break  # Döngüden çık
                else:
                    print(f"{ConsoleColors.RED}Lütfen 'evet' veya 'hayır' yazın.{ConsoleColors.RESET}")

            # Kullanıcı "evet" veya "hayır" dedikten sonra dış döngüye dön
            if cont in ["hayır", "no"]:
                break  # Kullanıcı istemiyorsa tamamen çık

# Programı başlat
if __name__ == "__main__":
    main()


# EVET-HAYIR sorusu sorma

# NLP modeli yükle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Suit açıklamaları
suit_explanations = {
    "Wands": "Ateş elementinin serisidir. Aslan, Koç, Yay burçlarını temsil eder. Bu kartlar hareketliliğe, hıza ve yaratıcılığa dair bilgileri bize sunar.",
    "Cups": "Su elementinin serisidir. Yengeç, Akrep, Balık burçlarını temsil eder. Bu kartlar duygulara, alışkanlıklara ve bilinçaltına dair bilgileri bize sunar.",
    "Swords": "Hava elementinin serisidir. İkizler, Kova, Terazi burçlarını temsil eder. Bu kartlar düşüncelere, inançlara ve zihinsel mücadelelere dair bilgileri sunar.",
    "Pentacles": "Toprak elementinin serisidir. Boğa, Başak, Oğlak burçlarını temsil eder. Bu kartlar dünya işleri, doğa ve fiziksel yapıya dair bilgileri verir.",
    "Major Arcana": "Büyük Arkana kartları yaşamın önemli temalarını ve dönüşümlerini temsil eder. Her biri farklı bir ders sunar."
}

# Kart görselini gösterme
def display_card_image(card):
    image_path = os.path.join(data_dir, card['img'])
    if os.path.exists(image_path):
        img = plt.imread(image_path)
        plt.figure(figsize=(5, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title(card['name'])
        plt.show()
    else:
        print(f"{ConsoleColors.RED}Görsel bulunamadı: {image_path}{ConsoleColors.RESET}")

# Cevap değerlendirme fonksiyonu
def evaluate_card(card, user_input):
    # Kullanıcının sorusunu vektörleştir
    user_embedding = model.encode(user_input)

    # Kartın "light" ve "shadow" anlamları
    light_meaning = card['meanings.light']
    shadow_meaning = card['meanings.shadow']

    # Vektörleştirme ve benzerlik hesaplama
    light_embedding = model.encode(light_meaning)
    shadow_embedding = model.encode(shadow_meaning)

    light_score = util.cos_sim(user_embedding, light_embedding).mean().item()
    shadow_score = util.cos_sim(user_embedding, shadow_embedding).mean().item()

    # Evet veya Hayır cevabını belirleme
    response = "Evet" if light_score > shadow_score else "Hayır"

    # Sonuçları döndür
    return {
        "response": response,
        "light_score": light_score,
        "shadow_score": shadow_score,
    }

# Rastgele kart çekme ve analiz yapma
def answer_user_question(user_input):
    # Rastgele bir kart çek
    selected_card = df_cards.sample(1).iloc[0]

    # Kartı analiz et
    result = evaluate_card(selected_card, user_input)

    # Kart bilgilerini yazdır
    print(f"{ConsoleColors.ORANGE}\nSizin için bir kart çektim!{ConsoleColors.RESET}")
    print(f"{ConsoleColors.BOLD}Kart İsmi:{ConsoleColors.RESET} {selected_card['name']}")
    print(f"{ConsoleColors.GREEN}Light Anlam:{ConsoleColors.RESET} {selected_card['meanings.light']}")
    print(f"{ConsoleColors.RED}Shadow Anlam:{ConsoleColors.RESET} {selected_card['meanings.shadow']}")
    print(f"{ConsoleColors.BLUE}Çekiliş Tarihi:{ConsoleColors.RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Suit açıklaması ekle
    suit_description = suit_explanations.get(selected_card['suit'], "Genel bir kategori açıklaması mevcut değil.")
    print(f"{ConsoleColors.BOLD}Suit Açıklaması:{ConsoleColors.RESET} {suit_description}")

    # Kartın görselini göster
    display_card_image(selected_card)

    # "Evet" veya "Hayır" cevabı yazdır
    print(f"\n{ConsoleColors.BOLD}Cevap:{ConsoleColors.RESET} {result['response']}")
    print(f"{ConsoleColors.GREEN}Light Skor:{ConsoleColors.RESET} {result['light_score']:.2f}")
    print(f"{ConsoleColors.RED}Shadow Skor:{ConsoleColors.RESET} {result['shadow_score']:.2f}")

# Ana program
def main():
    print(f"{ConsoleColors.BOLD}Bir soru sorun ya da ne hissettiğinizi yazın (Çıkmak için 'çıkış'): {ConsoleColors.RESET}")
    user_input = input(">>> ").strip()

    if user_input.lower() in ["çıkış", "exit", "quit"]:
        print(f"\n{ConsoleColors.ORANGE}Görüşmek üzere, tekrar bekleriz! 🌟{ConsoleColors.RESET}")
        return

    # Kullanıcının sorusunu cevapla
    answer_user_question(user_input)

    print(f"\n{ConsoleColors.ORANGE}Harika bir gün geçirmenizi dilerim! 🌟{ConsoleColors.RESET}")

# Programı başlat
if __name__ == "__main__":
    main()


# Kullanıcıdan bir kart ID'si alma ve o kartin anlamini cikarma

# NLP modeli yükle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Tarot kartlarını yükle
data_dir = '/content/'  # Görsellerin olduğu klasör yolu
json_path = '/content/tarot-images.json'
df = pd.read_json(json_path, orient='records')
df_cards = pd.json_normalize(df['cards'])

# Suit açıklamaları
suit_explanations = {
    "Wands": "Ateş elementinin serisidir. Aslan, Koç, Yay burçlarını temsil eder. Bu kartlar hareketliliğe, hıza ve yaratıcılığa dair bilgileri bize sunar.",
    "Cups": "Su elementinin serisidir. Yengeç, Akrep, Balık burçlarını temsil eder. Bu kartlar duygulara, alışkanlıklara ve bilinçaltına dair bilgileri bize sunar.",
    "Swords": "Hava elementinin serisidir. İkizler, Kova, Terazi burçlarını temsil eder. Bu kartlar düşüncelere, inançlara ve zihinsel mücadelelere dair bilgileri sunar.",
    "Pentacles": "Toprak elementinin serisidir. Boğa, Başak, Oğlak burçlarını temsil eder. Bu kartlar dünya işleri, doğa ve fiziksel yapıya dair bilgileri verir.",
    "Major Arcana": "Büyük Arkana kartları yaşamın önemli temalarını ve dönüşümlerini temsil eder. Her biri farklı bir ders sunar."
}

# Kart görselini gösterme
def display_card_image(card):
    image_path = os.path.join(data_dir, card['img'])
    if os.path.exists(image_path):
        img = plt.imread(image_path)
        plt.figure(figsize=(5, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title(card['name'])
        plt.show()
    else:
        print(f"{ConsoleColors.RED}Görsel bulunamadı: {image_path}{ConsoleColors.RESET}")

# Kart bilgisini gösterme ve suit açıklaması ekleme
def show_card_by_name(card_name):
    print("\nKart bilgileri getiriliyor...\U0001F52E\n")

    # Kartı adından bul
    selected_card = df_cards[df_cards['img'].str.lower() == card_name.lower()]

    if not selected_card.empty:
        card_info = selected_card.iloc[0]

        # Kart bilgilerini yazdır
        print(f"Kart İsmi: {card_info['name']}")
        print(f"Numarası: {card_info['number']}")
        print(f"Arcana: {card_info['arcana']}")
        print(f"Sembol: {card_info['suit']}")
        print(f"Kehanet: {card_info['fortune_telling']}")
        print(f"Olumlu Anlam (Light): {card_info['meanings.light']}")
        print(f"Gölge Anlam (Shadow): {card_info['meanings.shadow']}")

        # Suit açıklaması ekleme
        suit_description = suit_explanations.get(card_info['suit'], "Genel bir kategori açıklaması mevcut değil.")
        print(f"{ConsoleColors.BOLD}Suit Açıklaması:{ConsoleColors.RESET} {suit_description}")

        # Light ve Shadow anlamlarını kontrol etme
        light_meanings = card_info['meanings.light']
        shadow_meanings = card_info['meanings.shadow']
        light_count = len(light_meanings)
        shadow_count = len(shadow_meanings)

        # Light ve Shadow'a göre mesaj belirleme
        if light_count > shadow_count:
            dominance_message = f"{ConsoleColors.GREEN}Kartın genel yorumu: Olumlu 🌟{ConsoleColors.RESET}"
        elif shadow_count > light_count:
            dominance_message = f"{ConsoleColors.RED}Kartın genel yorumu: Olumsuz ⚡{ConsoleColors.RESET}"
        else:
            dominance_message = f"{ConsoleColors.BLUE}Kartın genel yorumu: Dengeli ⚖{ConsoleColors.RESET}"

        print(dominance_message)

        # Görseli yükle ve göster
        image_path = os.path.join(data_dir, card_info['img'])
        if os.path.exists(image_path):
            img = plt.imread(image_path)

            # Görseli çiz
            fig, ax = plt.subplots(figsize=(5, 7))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(card_info['name'], fontsize=16, fontweight='bold')
            plt.show()
        else:
            print(f"{ConsoleColors.RED}Görsel bulunamadı: {image_path}{ConsoleColors.RESET}")
    else:
        print(f"{ConsoleColors.RED}'{card_name}' adında bir kart bulunamadı.{ConsoleColors.RESET}")

# Ana program
def main():
    print("\nKart kodunu girin (örneğin 'p13.jpg'): ")
    card_name = input(">>> ").strip()

    if card_name.lower() in ["çıkış", "exit", "quit"]:
        print(f"\n{ConsoleColors.CYAN}Görüşmek üzere, tekrar bekleriz! 🌟{ConsoleColors.RESET}")
        return

    show_card_by_name(card_name)

    print(f"\n{ConsoleColors.ORANGE}Harika bir gün geçirmenizi dilerim! 🌟{ConsoleColors.RESET}")

# Programı başlat
if __name__ == "__main__":
    main()



# 3 KART ACILIMI
 # Bu fonksiyon, tarot kartlarını rastgele çeker, seçilen kartların falını gösterir.

def playDefault(mode): # kartları çekme ve falı okuma fonksiyonunun başlangıcı

  print('\nFalınızı okuyorum... 🔮\n')

  modes = [['Geçmiş','Şimdi','Gelecek'], ['Durum','Aksiyon','Sonuç'], ['Sen','Partnerin','İlişki']] #tarot kartı okumasında kullanabileceğimiz üç farklı kart dizilimi

  cards3 = random.sample(range(78), k=3) # rastgele 3 kart seçme

  #seçilen kartların falını, adlarını ve görsellerini depolamak için boş listelerdir.
  fortunes = []
  names = []
  images = []

  # Kartların Okunması (for döngüsü)
  # Kartların Görselleştirilmesi
  for i in range(3):
    fortunes.append( df_cards.loc[cards3[i],'fortune_telling'][random.randint(0, len(df_cards.loc[cards3[i],'fortune_telling'])-1)] )
    names.append( df_cards.loc[cards3[i],'name'] )
    images.append( df_cards.loc[cards3[i],'img'] )

  fig, ax = plt.subplots(1, 3, figsize = (8,4))

  for i in range(3):
    img = plt.imread(data_dir + '/' + images[i])
    ax[i].imshow(img)
    ax[i].set_title(names[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])

#Falın Gösterilmesi
    print(f'Your \033[1m{modes[mode][i]}\033[0m is \033[1m{names[i]}.\033[0m {fortunes[i]}.')
  print()
def playLightDark(mode):

  print('\nFalınızı okuyorum...  🔮\n')

# Modlar ve Light-Dark Tanımlaması
  modes = [['Geçmiş','Şimdi','Gelecek'], ['Durum','Aksiyon','Sonuç'], ['Sen','Partnerin','İlişki']]
  lightDark = ['meanings.light', 'meanings.shadow']

#Kartları ve Durumu Seçme
  cards3 = random.sample(range(78), k=3)
  binary3 = [random.randint(0,1), random.randint(0,1), random.randint(0,1)]

#Kartların Bilgilerini Hazırlama
  fortunes = []
  names = []
  images = []

#Kartın Anlamını Seçme ve Görselleri Çekme
  for i in range(3):
    orientation = lightDark[binary3[i]]
    fortunes.append( df_cards.loc[cards3[i], orientation][random.randint(0, len(df_cards.loc[cards3[i], orientation])-1)] )
    names.append( df_cards.loc[cards3[i], 'name'] )
    images.append( df_cards.loc[cards3[i], 'img'] )

#Görsel Oluşturma (Matplotlib Kullanarak)
  fig, ax = plt.subplots(1, 3, figsize = (8,4))

#  Kartın Görselinin Yönlendirilmesi ve Başlıklandırılması
  for i in range(3):

    # Çekilen görsel ters ise çevir
    img = plt.imread(data_dir + '/' + images[i])
    if binary3[i] == 1:
      img = scipy.ndimage.rotate(img, 180)
    else:
      img = scipy.ndimage.rotate(img, 0)

    ax[i].imshow(img)

    # Kasrt ters çevrilmişse reversed yaz
    name = names[i]
    if binary3[i] == 1:
      name += ' (Reversed)'

    ax[i].set_title(names[i])

    ax[i].set_xticks([])
    ax[i].set_yticks([])

#Fal Mesajı Yazdırma
    print(f'Your \033[1m{modes[mode][i]}\033[0m is \033[1m{name}.\033[0m {fortunes[i]}.')
  print()
# Kullanıcı girişi

def validInt(min, max):
  while True:
    try:
      ans = int(input('Bir sayı giriniz: '))

      if ans < min:
        raise Exception
      elif ans > max:
        raise Exception
      else:
        return ans

    except:
      return 0
# Kullanıcıdan Ana Seçim Yapmasını İsteme

print('Nasıl bır fal bakmak istersiniz? \n[1] 3- Klasik Fal \n[2] 3 Light & Shadow Falı \n[*] Çıkış')

r = validInt(1,2)
if r == 1:
  print('\n[1] Geçmiş-Şimdi-Gelecek \n[2] Durum-Aksiyon-Sonuç \n[3] Sen-Partnerin-İlişki \n[*] Exit')

  r = validInt(1,3)
  if r != 0:
    playDefault(r-1)
  else:
    print('\nFalınızı okuyorum... 🔮')

elif r == 2:
  print('\n[1] Geçmiş-Şimdi-Gelecek \n[2] Durum-Aksiyon-Sonuç \n[3] Sen-Partnerin-İlişki \n[*] Exit')

  r = validInt(1,3)
  if r != 0:
    playLightDark(r-1)
  else:
    print('\nCFalınızı okuyorum... 🔮')

else:
  print('\nFalınızı okuyorum... 🔮')


  r = validInt(1,3)
  if r != 0:
    playDefault(r-1)
  else:
    print('\nFalınızı okuyorum... 🔮')

