import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. データ読み込み
    df = pd.read_csv('fish_data.csv')  # CSVファイルを読み込む

    # 2. 入力 (content) と ラベル (label) に分割
    X = df['content']
    y = df['label']

    # 3. 学習データとテストデータに分割
    # 例：学習80%、テスト20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4. パイプラインの作成
    # TF-IDF で単語ベクトル化 → ロジスティック回帰モデルで分類
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1,2),    # 単語・2-gramも考慮
            max_df=0.9,          # 文書全体の90%以上に出現する単語を除外
            min_df=2             # 2文書未満でしか出現しない単語を除外
        )),
        ('clf', LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # クラス重みの設定
        ))
    ])

    # 5. モデルの学習
    text_clf.fit(X_train, y_train)

    # 6. テストデータで予測
    y_pred = text_clf.predict(X_test)

    # 7. 評価指標を表示 (classification_report)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # 8. 混同行列を表示
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 9. 新規テキストを含むCSVファイルを読み込んで予測
    new_text_df = pd.read_csv('new_fish_data.csv')  # 新規テキストのCSVファイルを読み込む
    new_texts = new_text_df['content']  # 'content'カラムに新規テキストが含まれていると仮定

    # 新規テキストでの予測
    new_predictions = text_clf.predict(new_texts)

    # 新規テキストと予測結果を表示
    for text, pred in zip(new_texts, new_predictions):
        print(f"Text: {text}\nPredict: {pred}\n")

if __name__ == "__main__":
    main()
