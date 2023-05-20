import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/amazon_review.csv")
df.head(5)


#Adım 1: Ürünün ortalama puanını hesaplayalım.

df["overall"].mean()

#Sonuç: 4.587589013224822

#Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayalım.

df.info()

df["unixReviewTime"] = pd.to_datetime(df["unixReviewTime"])
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

df["unixReviewTime"].max() #veri setinin en son zamanını bulalım.
df["reviewTime"].max() #veri setinin en son zamanını bulalım.

current_date = pd.to_datetime("2014-12-08 00:00:00") #bugün tarihi olarak bir gün belirleyelim.

df["days"] = (current_date - df["reviewTime"]).dt.days #bugünden veri setinde ki tarihi çıkarıp gün formatına dönüştürelim.

df.loc[df["days"] <= 30, "overall"].mean() #örnek bir tarihe göre ortalama
#Sonuç: 4.742424242424242

#şimdi kendi isteğimize göre tarih aralıklarına ağırlık verelim.
df.loc[df["days"] <= 30, "overall"].mean() * 23/100 + \
    df.loc[df["days"] <= 60, "overall"].mean() * 24/100 + \
    df.loc[(df["days"] > 60) & (df["days"] <= 120), "overall"].mean() * 25/100 + \
    df.loc[df["days"] > 120, "overall"].mean() * 28/100
#Sonuç: 4.700268767591435

#Şimdi kodu fonksiyonel hale getirelim.

def time_based_weighted_average(df, w1 = 23, w2 = 24, w3 = 25, w4 = 28):
   return df.loc[df["days"] <= 30, "overall"].mean() * w1 / 100 + \
        df.loc[df["days"] <= 60, "overall"].mean() * w2 / 100 + \
        df.loc[(df["days"] > 60) & (df["days"] <= 120), "overall"].mean() * w3 / 100 + \
        df.loc[df["days"] > 120, "overall"].mean() * w4 / 100

time_based_weighted_average(df)
#Sonuç: 4.700268767591435

# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştıralım ve yorumlayalım.

df.loc[df["days"] <= 30, "overall"].mean()
df.loc[df["days"] <= 60, "overall"].mean()
df.loc[(df["days"] > 60) & (df["days"] <= 180), "overall"].mean()
df.loc[df["days"] > 180, "overall"].mean()

"""ilk başlarda değerlendirme puanı daha düşükken günümüze yaklaştıkça puanlar artmıştır."""

# Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyelim.

# Adım 1: helpful_no değişkenini üretiniz

df["helpful_yes"].head(10)
df["total_vote"].head(10)

df["helpful_no"] = (df["total_vote"] - df["helpful_yes"])

df.head(5)

# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyelim.

# score_pos_neg_diff
def score_pos_neg_diff(up, down):
    return up - down

df["helpful_yes"].max()
df["helpful_no"].max()

score_pos_neg_diff(1952,183)

df["score_pos_neg_diff"] = (df["helpful_yes"] - df["helpful_no"])

# score_average_rating

def score_average_rating(up,down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(1952,183)

df["score_average_rating"] = (df["helpful_yes"] / (df["helpful_yes"] + df["helpful_no"]))

# wilson_lower_bound

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(1952,183)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Adım 3: 20 Yorumu belirleyelim (lower bound'a göre).

df.sort_values("wilson_lower_bound", ascending=False).head(20)