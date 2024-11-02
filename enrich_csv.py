import pandas as pd
from openai import OpenAI

from env import OPENAI_API, CSV_FILENAME, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API)

df = pd.read_csv(CSV_FILENAME)

# столбец с текстом должен называться text
# внутри лямбды Функция вычисления эмбедингов,
# она отправляет chatGPT строки для ее токенизации
df['embedding'] = df.text.apply(
    lambda row: client.embeddings.create(input=[row], model=EMBEDDING_MODEL).data[0].embedding
)

# Сохранение результата
df.to_csv(CSV_FILENAME, index=False)

print(df.head())
