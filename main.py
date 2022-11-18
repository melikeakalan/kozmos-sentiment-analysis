##################################################
# Amazon Yorumları için Duygu Analizi
##################################################

#### İş Problemi ####
# Amazon üzerinden satışlarını gerçekleştiren ev tesktili ve günlük giyim odaklı üretimler yapan Kozmos
# ürünlerine gelen yorumları analiz ederek ve aldığı şikayetlere göre özelliklerini geliştirerek satışlarını
# artırmayı hedeflemektedir. Bu hedef doğrultusunda yorumlara duygu analizi yapılarak etiketlenecek ve
# etiketlenen veri ile sınıflandırma modeli oluşturulacaktır.

#### Veri Seti Hikayesi ####
# Veri seti belirli bir ürün grubuna ait yapılan yorumları, yorum başlığını, yıldız sayısını ve yapılan yorumu
# kaç kişinin faydalı bulduğunu belirten değişkenlerden oluşmaktadır.

##### Değişkenler #####
# Star   : Ürüne verilen yıldız sayısı
# HelpFul: Yorumu faydalı bulan kişi sayısı
# Title  : Yorum içeriğine verilen başlık, kısa yorum
# Review : Ürüne yapılan yorum

#### Gerekli Kütüphaneler ####
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, learning_curve,  train_test_split
from sklearn.metrics import classification_report, roc_auc_score, plot_roc_curve, accuracy_score
from mlxtend.plotting import plot_learning_curves
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################
# Görev 1: Metin Ön İşleme
##################################################

# Adım 1: amazon.xlsx verisini okutunuz.
df = pd.read_excel("amazon.xlsx")
df.head()
df.info()

# Adım 2: Review değişkeni üzerinde ;
# a. Tüm harfleri küçük harfe çeviriniz.
df['Review'] = df['Review'].str.lower()

# b. Noktalama işaretlerini çıkarınız.
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# c. Yorumlarda bulunan sayısal ifadeleri çıkarınız.
df['reviewText'] = df['Review'].str.replace('\d', '')

# d. Bilgi içermeyen kelimeleri (stopwords) veriden çıkarınız.
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# e. 1000'den az geçen kelimeleri veriden çıkarınız.
drops = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# f. Lemmatization işlemini uygulayınız
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##################################################
# Görev 2: Metin Görselleştirme
##################################################

# Adım 1: Barplot görselleştirme işlemi için;
# a. "Review" değişkeninin içerdiği kelimeleri frekanslarını hesaplayınız, tf olarak kaydediniz
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

# b. tf dataframe'inin sütunlarını yeniden adlandırınız: "words", "tf" şeklinde
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

# c. "tf" değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirme işlemini
# tamamlayınız.
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# Adım 2: WordCloud görselleştirme işlemi için;
# a. "Review" değişkeninin içerdiği tüm kelimeleri "text" isminde string olarak kaydediniz.
text = " ".join(i for i in df.Review)

# b. WordCloud kullanarak şablon şeklinizi belirleyip kaydediniz.
wordcloud = WordCloud(background_color="white",
               max_words=1000,
               contour_width=3,
               contour_color="firebrick")

# c. Kaydettiğiniz wordcloud'u ilk adımda oluşturduğunuz string ile generate ediniz.
wordcloud = wordcloud.generate(text)

# d. Görselleştirme adımlarını tamamlayınız.
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

##################################################
# Görev 3: Duygu Analizi
##################################################

# Adım 1: Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturunuz.
sia = SentimentIntensityAnalyzer()

# Adım 2: SentimentIntensityAnalyzer nesnesi ile polarite puanlarının inceleyiniz;
# a. "Review" değişkeninin ilk 10 gözlemi için polarity_scores() hesaplayınız.
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

# b. İncelenen ilk 10 gözlem için compund skorlarına göre filtrelenerek tekrar gözlemleyiniz.
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# c. 10 gözlem için compound skorları 0'dan büyükse "pos" değilse "neg" şeklinde güncelleyiniz.
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# d. "Review" değişkenindeki tüm gözlemler için pos-neg atamasını yaparak yeni bir değişken olarak dataframe'e ekleyiniz.
# NOT: SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorum sınıflandırma makine öğrenmesi modeli için bağımlı değişken
# oluşturulmuş oldu.
df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["sentiment_label"].value_counts()
# neg    846
# pos    4765

########################
# Plottig
########################

sizes = [4765, 846]
labels = 'POSITIVE', 'NEGATIVE'
explode = (0, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=75)
ax1.axis('equal')
ax1.set_title("Sentiment Analysis Result")
ax1.legend(labels)
plt.show()

##################################################
# Görev 4: Makine Öğrenmesine Hazırlık
##################################################

# Adım 1: Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train test olarak ayırınız.
y = df["sentiment_label"]
y.value_counts()
X = df["Review"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# Adım 2: Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte;
# a. TfidfVectorizer kullanarak bir nesne oluşturunuz.
tf_idf_word_vectorizer = TfidfVectorizer()

# b. Daha önce ayırmış olduğumuz train datamızı kullanarak oluşturduğumuz nesneye fit ediniz.
tf_idf_word_vectorizer = TfidfVectorizer().fit(X_train)

# c. Oluşturmuş olduğumuz vektörü train ve test datalarına transform işlemini uygulayıp kaydediniz.
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(X_train)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(X_test)


##################################################
# Görev 5: Modelleme (Lojistik Regresyon)
##################################################

# Adım 1: Lojistik regresyon modelini kurarak train dataları ile fit ediniz.
log_model = LogisticRegression().fit(x_train_tf_idf_word, y_train)

# Adım 2: Kurmuş olduğunuz model ile tahmin işlemleri gerçekleştiriniz;
# a. Predict fonksiyonu ile test datasını tahmin ederek kaydediniz.
y_pred = log_model.predict(x_test_tf_idf_word)

acc_test = accuracy_score(y_test, y_pred)
#  0.884

# b. classification_report ile tahmin sonuçlarınızı raporlayıp gözlemleyiniz.
print(classification_report(y_test, y_pred))

# c. cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.
cross_val_score(log_model,
                x_test_tf_idf_word,
                y_test,
                scoring="accuracy",
                cv=5).mean()
# 0.84

# Adım 3: Veride bulunan yorumlardan ratgele seçerek modele sorulması;
# a. sample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçerek yeni bir değere atayınız
random_review = pd.Series(df["reviewText"].sample(1).values)

# b. Elde ettiğiniz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştiriniz.
# c. Vektörleştirdiğiniz örneklemi fit ve transform işlemlerini yaparak kaydediniz.
random_review_vec = CountVectorizer().fit(X_train).transform(random_review)

# d. Kurmuş olduğunuz modele örneklemi vererek tahmin sonucunu kaydediniz.
random_review_pred = log_model.predict(random_review_vec)


##################################################
# Görev 6: Modelleme (Random Forest)
##################################################

# Adım 1: Random Forest modeli ile tahmin sonuçlarının gözlenmesi;
# a. RandomForestClassifier modelini kurup fit ediniz.
rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, y_train)

# b. Cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız
cross_val_score(rf_model, x_test_tf_idf_word, y_test, cv=5, scoring="accuracy", n_jobs=-1).mean()

# c. Lojistik regresyon modeli ile sonuçları karşılaştırınız
# rf_model: 0.87
# log_model:  0.84

# review of overfitting
plot_learning_curves(x_train_tf_idf_word, y_train, x_test_tf_idf_word, y_test, rf_model)
plt.show()

plot_learning_curves(x_train_tf_idf_word, y_train, x_test_tf_idf_word, y_test, log_model)
plt.show()