import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 学習データの作成
data = pd.DataFrame([
    # 取引ID    | 時間帯 (hour): 取引が行われた時間 (0〜23)
    # 取引額 (amount): 取引の金額（円）
    # 取引エリア異常 (area_mismatch): 取引が通常のエリア外で行われた場合は1、正常なエリア内であれば0
    # 不正 (fraud): 取引が不正である場合は1、正常であれば0
    ["TXN00001", 2, 5000, 0, 0],   # 通常の取引
    ["TXN00002", 23, 120000, 1, 1], # 高額 + 取引エリア異常
    ["TXN00003", 14, 800, 0, 0],   # 通常の取引
    ["TXN00004", 3, 900000, 1, 1], # 深夜 + 高額 + 取引エリア異常
    ["TXN00005", 12, 1000, 0, 0],  # 通常の取引
    ["TXN00006", 20, 150000, 1, 1], # 高額 + 取引エリア異常
    ["TXN00007", 5, 10000, 0, 0],  # 通常の取引
    ["TXN00008", 22, 500000, 1, 1], # 高額 + 取引エリア異常
    ["TXN00009", 9, 3000, 0, 0],   # 通常の取引
    ["TXN00010", 16, 30000, 1, 1],  # 高額 + 取引エリア異常
    ["TXN00011", 1, 200000, 0, 1],  # 深夜 + 高額 
    ["TXN00012", 10, 20000, 0, 0],  # 通常の取引
    ["TXN00013", 14, 500000, 1, 1], # 高額 + 取引エリア異常
    ["TXN00014", 8, 1500, 0, 0],    # 通常の取引
    ["TXN00015", 21, 700000, 1, 1], # 深夜 + 高額 + 取引エリア異常
    ["TXN00016", 5, 30000, 0, 0]    # 通常の取引
], columns=["txn_id", "hour", "amount", "area_mismatch", "fraud"])


data.columns = ["txn_id", "hour", "amount", "area_mismatch", "fraud"]

# 特徴量とラベル（X：特徴量データ、Y：ラベルデータ（不正:1 or 正常:0））
X = data[["hour", "amount", "area_mismatch"]]
y = data["fraud"]

# データ分割（データを「学習用（X_train, y_train）80％」と「テスト用（X_test, y_test）」20％に分ける）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル学習（ランダムフォレストで10本の決定技を使う、決定木が増えると精度はあがるが計算コストが増える）
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度確認
print(f"予測精度: {accuracy_score(y_test, y_pred):.2f}")

# 新しい取引データ（TXN12345, TXN12346, TXN12347）の予測
new_data = pd.DataFrame([
    [2, 300000, 1],  # TXN12345: 深夜の取引、取引金額は高額、取引エリア異常あり
    [10, 5000, 0],   # TXN12346: 日中の取引、取引金額は小額、取引エリアは正常
    [22, 980000, 1]  # TXN12347: 深夜の取引、取引金額は非常に高額、取引エリア異常あり
], columns=["hour", "amount", "area_mismatch"])

predictions = model.predict(new_data)

print("予測結果:")
for i, pred in enumerate(predictions):
    status = "不正" if pred == 1 else "正常"
    print(f"TXN1234{i+5}: {status}")
