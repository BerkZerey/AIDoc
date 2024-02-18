import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import sqlite3
from os.path import dirname, join
from com.chaquo.python import Python


def main(kullanicisemptomlari):

    trainingfile = join(dirname(__file__),"Training.csv")
    testingfile = join(dirname(__file__),"Testing.csv")

    egitim_veri = pd.read_csv(trainingfile)
    test_veri = pd.read_csv(testingfile)

    # Eğitim ve test dosyalarının yolunu tanımladım, Pandas kütüphanesini kullanarak verileri okumalarını sağladım.


    egitim_veri = egitim_veri.drop(['Unnamed: 133'], axis=1)

    # 133 adet boş sütun var onları sildim.


    def process_dataset(data):
        x = data.drop(['prognosis'], axis=1)
        y = data['prognosis']

        le = LabelEncoder()
        y = pd.DataFrame(le.fit_transform(y))

        return x, y

    x_train, y_train = process_dataset(egitim_veri)
    x_test, y_test = process_dataset(test_veri)

    # YSA'nın çalışacağı verisetlerini oluşturdum.

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.flatten(), dtype=torch.long)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values.flatten(), dtype=torch.long)

    # Veriyi PyTorch tensörüne dönüştürdüm.

    unique_labels = np.unique(y_train)

# Define the neural network model using PyTorch
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(x_train.shape[1], 200) # İlk katman 200 adet giriş düğümü içeriyor.
            self.fc2 = nn.Linear(200, 150) # İkinci katman 150 adet düğüm içeriyor.
            self.fc3 = nn.Linear(150, 100) # Üçüncü katman 100 adet düğüm içeriyor.
            self.fc4 = nn.Linear(100, 42) # Çıkış katmanı 42 adet düğüm içeriyor.

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = nn.functional.softmax(self.fc4(x), dim=1)
            # Çıkış katmanı softmax fonksiyonuyla çalışıyor, diğer katmanlar relu fonksiyonunu kullanıyor.
            return x

    model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    batch_size = 128

    for epoch in range(epochs):
        for i in range(0, len(x_train_tensor), batch_size):
            batch_x = x_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Eğitim burada yapılıyor, her epoch başına döngü tekrarlanıyor.

    with torch.no_grad():
        model.eval()
        test_outputs = model(x_test_tensor)
        _, predicted = torch.max(test_outputs, 1)

    # Burada eğitimin doğruluğu test ediliyor.

    correct = (predicted == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    accuracy = correct / total * 100
    #print("Dogruluk: ", accuracy, "%")

    # Burada eğitimin verimli çalışma yüzdesi hesaplanıyor.

    pragnoise_columns = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
                     'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
                     'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold',
                     'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
                     'Fungal infection', 'Gastroenteritis', 'GERD', 'Heart attack', 'Hepatitis A', 'Hepatitis B',
                     'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ', 'Hyperthyroidism',
                     'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
                     'Osteoarthristis', 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia',
                     'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins']

    turkce_hastaliklar = ['Paroksismal Pozisyonel Vertigo', 'AIDS', 'Akne', 'Alkolik Hepatit', 'Alerji', 'Artrit', 'Bronşiyal Astım',
                         'Servikal Spondiloz', 'Su Çiçeği', 'Kronik Kolestaz', 'Soğuk Algınlığı', 'Dang Humması', 'Diyabet ', 'Hemoroid (Basur)', 'İlaç Reaksiyonu',
                            'Mantar Enfeksiyonu', 'Gastroenterit', 'Reflü', 'Kalp Krizi', 'Hepatit A', 'Hepatit B', 'Hepatit C', 'Hepatit D', 'Hepatit E', 'Hipertansiyon',
                        'Hipertiroidizm', 'Hipoglisemi', 'Hipotiroidizm', 'Impetigo', 'Sarılık', 'Sıtma', 'Migren', 'Osteoartrit', 'Felç (Beyin kanaması)',
                         'Ülser', 'Zatürre', 'Sedef', 'Tüberküloz', 'Tifo', 'İdrar Yolu Enfeksiyonu', 'Varis']

    turkce_tedaviler = ['Zamanla kendi kendine geçebilir, ancak ağır durumlarda doktora başvurmalı ve vestibüler rehabilitasyon terapisi uygulanmalıdır.',
    'Belirli antiretroviral ilaçlarla tedavi edilebilir, ancak tam bir tedavi mümkün olmayabilir. Hasta, sağlıklı bir yaşam tarzı sürdürmeli ve düzenli olarak ilaçlarını almalıdır.',
    'Çeşitli topikal ve sistemik tedaviler mevcuttur. Doktor tavsiyesine göre cilt temizliği ve nemlendirici kullanımı önemlidir.',
    'Alkolü bırakmak ve sağlıklı bir diyet, belirli ilaçlarla birlikte, ilk adımdır.',
    'Alerjenlerden kaçınma ve antihistaminikler genellikle semptomları hafifletir.',
    'Anti-inflamatuar ilaçlar ve ağrı kesiciler genellikle semptomları hafifletir.',
    'Inhaler ve öksürük şurubu gibi ilaçlar genellikle semptomları hafifletir.',
    'Fiziksel terapi ve ağrı kesiciler genellikle semptomları hafifletir.',
    'Kendi kendine sınırlıdır ve genellikle belirli bir tedavi gerektirmez.',
    'Kendi kendine sınırlıdır, belirtilerin yönetilmesi genellikle gereklidir.',
    'Yeterli dinlenme, bol sıvı alımı ve ağrı kesici ilaçlar genelde semptomları hafifletir.',
    'Dehidrasyonu önlemek için bol su ve elektrolit içeceği alınmalıdır.',
    'Kan şekeri seviyelerini kontrol altında tutmak için sağlıklı bir yaşam tarzı ve ilaç gereklidir.',
    'Ağrıyı hafifletmek için anti-inflamatuarlar, ağrı kesiciler veya topikal kremler kullanılabilir.',
    'İlaç kesilmeli ve alerji belirtileri için antihistaminikler kullanılabilir.',
    'Antifungal ilaçlar genellikle semptomları hafifletir.',
    'Bol su veya elektrolit içecek almalı ve tuzlu yiyecekler tüketmelidir.',
    'Yiyeceklere ve içeceklere dikkat edilmesi ve antasid kullanımı genellikle yararlıdır.',
    'Acil tıbbi yardım gereklidir - kalp masajı, aspirin ve oksijen uygulanabilir.',
    'Belirli antiviral ilaçlar ile tedavi edilir.',
    'Belirli antiviral ilaçlar ile tedavi edilir.',
    'Belirli antiviral ilaçlar ile tedavi edilir.',
    'Belirli antiviral ilaçlar ile tedavi edilir.',
    'Belirli antiviral ilaçlar ile tedavi edilir.',
    'Diyet ve yaşam tarzı değişiklikleri, bazen ilaç kullanımı gereklidir.',
    'Tiroit aktivitesini düşürmek için ilaçlar gereklidir.',
    'Düşük kan şekeri seviyesini yükseltmek için hemen glikoz veya şekerli bir yiyecek tüketmelidir.',
    'Tiroid hormonları takviye edilmelidir.',
    'Antibiyotik kremler veya oral antibiyotikler genellikle semptomları hafifletir.',
    'Altta yatan nedene bağlı olarak tedavi, düzenli olarak taze meyve ve sebze tüketilmesini ve bol sıvı alınmasını içerebilir.',
    'Antimalaryal ilaçlar genellikle semptomları hafifletir.',
    'Ağrı kesiciler ve yaşam tarzı değişiklikleri genellikle semptomları hafifletir.',
    'Ağrı kesiciler ve yaşam tarzı değişiklikleri genellikle semptomları hafifletir.',
    'Acil tıbbi yardım gereklidir - Hasta hemen hastaneye götürülmelidir.',
    'Yiyecek ve içeceklere dikkat etmek ve antasid kullanmak genellikle yararlıdır.',
    'Antibiyotikler genellikle semptomları hafifletir.',
    'Topikal ve sistemik tedavilerle kontrol altına alınabilir.',
    'Uygun antibiyotiklerle tedavi edilmelidir.',
    'Antibiyotiklerle tedavi edilmelidir.',
    'Yeterli hidrasyon ve doktor reçeti üzerine antibiyotikler gereklidir.',
    'Yeterli su alımı ve doktor önerisi üzerine antibiyotikler gereklidir.']

# Kullanıcının girdiği semptomları buraya ekle
    kullanicisemptomlari = torch.tensor([kullanicisemptomlari], dtype=torch.float32)
# Modeli kullanarak tahminleri al
    user_prediction = model(kullanicisemptomlari)
    predicted_probabilities = nn.functional.softmax(user_prediction, dim=1)

# En yüksek olasılığa sahip sınıfın index'ini bul
    _, predicted_class_index = torch.max(predicted_probabilities, 1)

# Benzersiz hastalık türlerine uyacak şekilde güncellenmiş sınıf adını bul
    predicted_class = unique_labels[predicted_class_index.item()]

# Hastalığın ismini hastaliklar listesinden al, tedavisini de tedaviler listesinden al.
    hastalikadi = turkce_hastaliklar[predicted_class]
    hastaliktedavi = turkce_tedaviler[predicted_class]

    semptomornek = kullanicisemptomlari.tolist()[0]
    stringsemptom = str(semptomornek)
    listetemizle = stringsemptom.split(".0, ")

    ilkbolum = listetemizle[0:131]
    ikincibolum = listetemizle[131:132]

    sonkisim = str(ikincibolum[0])
    temizson = str(sonkisim.replace(".0]", ""))

    # ilk liste son eleman 130 son liste 0 yaz
    #print("GELGELGEL" + str(ilkbolum) + "SONUNCU" + temizson)

    files_dir = str(Python.getPlatform().getApplication().getFilesDir())
    filename = join(dirname(files_dir),"logs.db")
    connection = sqlite3.connect(filename)
    cursor = connection.cursor()

    sql_tablo_olustur = "CREATE TABLE IF NOT EXISTS Predictions (predict_id INTEGER PRIMARY KEY AUTOINCREMENT, itching INTEGER, skin_rash INTEGER, nodal_skin_eruptions INTEGER, continuous_sneezing INTEGER, shivering INTEGER, chills INTEGER, joint_pain INTEGER, stomach_pain INTEGER, acidity INTEGER, ulcers_on_tongue INTEGER, muscle_wasting INTEGER, vomiting INTEGER, burning_micturition INTEGER, spotting_urination INTEGER, fatigue INTEGER, weight_gain INTEGER, anxiety INTEGER, cold_hands_and_feets INTEGER, mood_swings INTEGER, weight_loss INTEGER, restlessness INTEGER, lethargy INTEGER, patches_in_throat INTEGER, irregular_sugar_level INTEGER, cough INTEGER, high_fever INTEGER, sunken_eyes INTEGER, breathlessness INTEGER, sweating INTEGER, dehydration INTEGER, indigestion INTEGER, headache INTEGER, yellowish_skin INTEGER, dark_urine INTEGER, nausea INTEGER, loss_of_appetite INTEGER, pain_behind_the_eyes INTEGER, back_pain INTEGER, constipation INTEGER, abdominal_pain INTEGER, diarrhoea INTEGER, mild_fever INTEGER, yellow_urine INTEGER, yellowing_of_eyes INTEGER, acute_liver_failure INTEGER, fluid_overload INTEGER, swelling_of_stomach INTEGER, swelled_lymph_nodes INTEGER, malaise INTEGER, blurred_and_distorted_vision INTEGER, phlegm INTEGER, throat_irritation INTEGER, redness_of_eyes INTEGER, sinus_pressure INTEGER, runny_nose INTEGER, congestion INTEGER, chest_pain INTEGER, weakness_in_limbs INTEGER, fast_heart_rate INTEGER, pain_during_bowel_movements INTEGER, pain_in_anal_region INTEGER, bloody_stool INTEGER, irritation_in_anus INTEGER, neck_pain INTEGER, dizziness INTEGER, cramps INTEGER, bruising INTEGER, obesity INTEGER, swollen_legs INTEGER, swollen_blood_vessels INTEGER, puffy_face_and_eyes INTEGER, enlarged_thyroid INTEGER, brittle_nails INTEGER, swollen_extremities INTEGER, excessive_hunger INTEGER, extra_marital_contacts INTEGER, drying_and_tingling_lips INTEGER, slurred_speech INTEGER, knee_pain INTEGER, hip_joint_pain INTEGER, muscle_weakness INTEGER, stiff_neck INTEGER, swelling_joints INTEGER, movement_stiffness INTEGER, spinning_movements INTEGER, loss_of_balance INTEGER, unsteadiness INTEGER, weakness_of_one_body_side INTEGER, loss_of_smell INTEGER, bladder_discomfort INTEGER, foul_smell_of_urine INTEGER, continuous_feel_of_urine INTEGER, passage_of_gases INTEGER, internal_itching INTEGER, toxic_look_typhos INTEGER, depression INTEGER, irritability INTEGER, muscle_pain INTEGER, altered_sensorium INTEGER, red_spots_over_body INTEGER, belly_pain INTEGER, abnormal_menstruation INTEGER, dischromic_patches INTEGER, watering_from_eyes INTEGER, increased_appetite INTEGER, polyuria INTEGER, family_history INTEGER, mucoid_sputum INTEGER, rusty_sputum INTEGER, lack_of_concentration INTEGER, visual_disturbances INTEGER, receiving_blood_transfusion INTEGER, receiving_unsterile_injections INTEGER, coma INTEGER, stomach_bleeding INTEGER, distention_of_abdomen INTEGER, history_of_alcohol_consumption INTEGER, fluid_overload2 INTEGER, blood_in_sputum INTEGER, prominent_veins_on_calf INTEGER, palpitations INTEGER, painful_walking INTEGER, pus_filled_pimples INTEGER, blackheads INTEGER, scurrying INTEGER, skin_peeling INTEGER, silver_like_dusting INTEGER, small_dents_in_nails INTEGER, inflammatory_nails INTEGER, blister INTEGER, red_sore_around_nose INTEGER, yellow_crust_ooze INTEGER, prediction VARCHAR(255))"
    cursor.execute(sql_tablo_olustur)
    connection.commit()

    sql_sorgu = f"""
    INSERT INTO Predictions (
        itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering, chills, joint_pain, stomach_pain, acidity, ulcers_on_tongue,
        muscle_wasting, vomiting, burning_micturition, spotting_urination, fatigue, weight_gain, anxiety, cold_hands_and_feets, mood_swings,
        weight_loss, restlessness, lethargy, patches_in_throat, irregular_sugar_level, cough, high_fever, sunken_eyes, breathlessness, sweating,
        dehydration, indigestion, headache, yellowish_skin, dark_urine, nausea, loss_of_appetite, pain_behind_the_eyes, back_pain, constipation,
        abdominal_pain, diarrhoea, mild_fever, yellow_urine, yellowing_of_eyes, acute_liver_failure, fluid_overload, swelling_of_stomach,
        swelled_lymph_nodes, malaise, blurred_and_distorted_vision, phlegm, throat_irritation, redness_of_eyes, sinus_pressure, runny_nose,
        congestion, chest_pain, weakness_in_limbs, fast_heart_rate, pain_during_bowel_movements, pain_in_anal_region, bloody_stool,
        irritation_in_anus, neck_pain, dizziness, cramps, bruising, obesity, swollen_legs, swollen_blood_vessels, puffy_face_and_eyes,
        enlarged_thyroid, brittle_nails, swollen_extremities, excessive_hunger, extra_marital_contacts, drying_and_tingling_lips, slurred_speech,
        knee_pain, hip_joint_pain, muscle_weakness, stiff_neck, swelling_joints, movement_stiffness, spinning_movements, loss_of_balance,
        unsteadiness, weakness_of_one_body_side, loss_of_smell, bladder_discomfort, foul_smell_of_urine, continuous_feel_of_urine, passage_of_gases,
        internal_itching, toxic_look_typhos, depression, irritability, muscle_pain, altered_sensorium, red_spots_over_body, belly_pain,
        abnormal_menstruation, dischromic_patches, watering_from_eyes, increased_appetite, polyuria, family_history, mucoid_sputum, rusty_sputum,
        lack_of_concentration, visual_disturbances, receiving_blood_transfusion, receiving_unsterile_injections, coma, stomach_bleeding,
        distention_of_abdomen, history_of_alcohol_consumption, fluid_overload2, blood_in_sputum, prominent_veins_on_calf, palpitations,
        painful_walking, pus_filled_pimples, blackheads, scurrying, skin_peeling, silver_like_dusting, small_dents_in_nails, inflammatory_nails,
        blister, red_sore_around_nose, yellow_crust_ooze , prediction
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{temizson}', '{hastalikadi}')"""


    cursor.execute(sql_sorgu, ilkbolum)

    connection.commit()
    connection.close()

    verikumesi  = f"{hastalikadi},{hastaliktedavi}"  #Burada ikisini tek veri haline getirme amacım Java tarafında çekerken sıkıntı çıkarmamak.
    return verikumesi
