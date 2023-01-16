import pandas as pd
import utils
import pickle


if __name__ == "__main__":
    df = pd.read_csv('data_test.csv', index_col='Unnamed: 0')

    print("Обработка данных...")
    test = utils.prepare_test(df, 'features.csv/features.csv')

    with open('LGBMClassifier', 'rb') as f:
        lgbm = pickle.load(f)

    print("Пресказание вероятностей...")
    df['target_proba'] = lgbm.predict_proba(test)[:, 1]
    df.to_csv('answers_test.csv', index=False)

    print("Готово! Файл с предсказаниями вероятностей покупок пользователей находятся в answers_test.csv")
