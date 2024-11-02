import ast
import logging

from openai import OpenAI
import pandas as pd
from scipy.spatial.distance import cosine  # вычисляет сходство векторов
from tiktoken import encoding_for_model

from env import OPENAI_API, CSV_FILENAME, EMBEDDING_MODEL, GPT_MODEL


class GptService:
    def __init__(self, openai_api, csv_filename, embedding_model, gpt_model):
        """
        Создаёт объект для работы с ChatGPT.

        Args:
            openai_api (str): API-ключ OpenAI
            csv_filename (str): имя файла с текстом и эмбеддингами
            embedding_model (str): название модели эмбеддингов
            gpt_model (str): название GPT-модели

        Attributes:
            openai (OpenAI): Экземпляр OpenAI
            df (pd.DataFrame): DataFrame со столбцами text и embedding (база знаний)
            top_n (int): 100, выбор лучших n-результатов
            token_budget (int): 4096 - 500, ограничение на число отсылаемых токенов в модель
            print_message (bool): False, нужно ли выводить сообщение перед отправкой
            query (str): текст вопроса
            response_message (str): ответ от GPT
        """

        self.openai = OpenAI(api_key=openai_api)
        self.df = self.read_df(csv_filename)
        self.embedding_model = embedding_model
        self.gpt_model = gpt_model
        self.top_n = 100
        self.token_budget = 4096 - 500,
        self.print_message = False,
        self.query = ''
        self.response_message = ''

    @staticmethod
    def relatedness_fn(x, y):
        """функция схожести, возвращает косинусное расстояние"""
        return 1 - cosine(x, y)

    def strings_ranked_by_relatedness(self) -> tuple[list[str], list[float]]:
        """
        Функция поиска.
        Возвращает строки и схожести,
        отсортированные от большего к меньшему

        Формат - кортеж двух списков,
        первый содержит строки,
        второй - числа с плавающей запятой
        """

        # Отправляем в OpenAI API пользовательский запрос для токенизации
        query_embedding_response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=self.query,
        )

        # Получен токенизированный пользовательский запрос
        query_embedding = query_embedding_response.data[0].embedding

        # Сравниваем пользовательский запрос
        # с каждой токенизированной строкой DataFrame
        strings_and_relatednesses = [
            (row["text"], self.relatedness_fn(query_embedding, row["embedding"]))
            for _, row in self.df.iterrows()
            # for i, row in self.df.iterrows()
        ]

        # Сортируем по убыванию схожести полученный список
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)

        # Преобразовываем наш список в кортеж из списков
        # strings, relatednesses = zip(*strings_and_relatednesses)
        strings, _ = zip(*strings_and_relatednesses)

        # Возвращаем n лучших результатов
        return strings[:self.top_n] #, relatednesses[:self.top_n]

    def num_tokens(self, text: str) -> int:
        """Возвращает число токенов в строке для заданной модели"""
        encoding = encoding_for_model(self.gpt_model)
        return len(encoding.encode(text))

    def query_message(self) -> str:
        """
        Функция формирования запроса к chatGPT
        по пользовательскому вопросу и базе знаний.

        Возвращает сообщение для GPT
        с соответствующими исходными текстами,
        извлеченными из фрейма данных (базы знаний).
        """
        # функция ранжирования базы знаний по пользовательскому запросу
        # strings, relatednesses = self.strings_ranked_by_relatedness()
        strings = self.strings_ranked_by_relatedness()

        # Шаблон инструкции для chatGPT
        message = (
            'Use the below articles on Linux Distributions '
            'to answer the subsequent question. '
            'If the answer cannot be found in the articles, '
            'write "I could not find an answer."'
        )
        # Шаблон для вопроса
        question = f"\n\nQuestion: {self.query}"

        # Добавляем к сообщению для chatGPT
        # релевантные строки из базы знаний,
        # пока не выйдем за допустимое число токенов
        for string in strings:
            next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
            if (self.num_tokens(message + next_article + question) > self.token_budget):
                break
            else:
                message += next_article
        return message + question

    def get_response(self):
        """
        Отвечает на вопрос, используя GPT и базу знаний.
        """

        # Формируем сообщение к chatGPT (функция выше)
        message = self.query_message()
        # Если параметр True, то выводим сообщение
        if self.print_message:
            logging.info(message)
        messages = [
            {"role": "system", "content": "You answer questions about Linux Distributions"},
            {"role": "user", "content": message},
        ]
        response = self.openai.chat.completions.create(
            model=self.gpt_model,
            messages=messages,
            temperature=0
            # гиперпараметр степени случайности
            # при генерации текста.
            # Влияет на то, как модель выбирает
            # следующее слово в последовательности.
        )
        self.response_message = response.choices[0].message.content

    @staticmethod
    def read_df(csv_filename):
        """
        Читает CSV и конвертирует эмбединги из строк в списки
        """
        df = pd.read_csv(csv_filename)
        logging.info('CSV is read')
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        logging.info('embedding is set')
        return df

    def ask_gpt(self, message):
        self.query = message
        self.get_response()
        return self.response_message
