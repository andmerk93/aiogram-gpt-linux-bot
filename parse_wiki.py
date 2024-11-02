import logging           # для сообщений об ошибках
import re                # для вырезания ссылок <ref> из статей Википедии

from requests import ConnectionError
import mwclient          # для работы с MediaWiki API (для загрузки)
import mwparserfromhell  # Парсер для MediaWiki
import pandas as pd      # для сохранения в csv базы знаний и токенов
import tiktoken          # для подсчета токенов
# import openai  # будем использовать для токинизации

from env import CSV_FILENAME, GPT_MODEL


# Задаем англоязычную версию Википедии для поиска
WIKI_SITE = "en.wikipedia.org"
# Задаем категорию
CATEGORY_TITLE = "Category:Linux distributions"
# Задаем секции, которые будут отброшены при парсинге статей
SECTIONS_TO_IGNORE = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
]
# only matters insofar as it selects which tokenizer to use
# GPT_MODEL = "gpt-3.5-turbo"
# Для сохранения в CSV
# CSV_FILENAME = 'linux_distribs.csv'


def titles_from_category(
    category: mwclient.listing.Category,
    max_depth: int = 1
) -> set[str]:
    """
    Возвращает набор заголовков страниц
    в данной категории Википедии и ее подкатегориях.

    Args:
        category (mwclient.listing.Category): Задает категории статей.
        max_depth (int): Определяет глубину вложения категорий.
    Returns:
        Набор заголовков страниц (set)
    """
    titles = set()
    # Используем множество для хранения заголовков статей
    for current_page in category.members():
        # Перебираем вложенные объекты категории
        if isinstance(current_page, mwclient.listing.Category) and max_depth > 0:
            # Если объект является категорией
            # и глубина вложения не достигла максимальной
            deeper_titles = titles_from_category(current_page, max_depth - 1)
            # вызываем рекурсивно функцию для подкатегории
            titles.update(deeper_titles)
            # добавление в множество элементов из другого множества
        elif type(current_page) is mwclient.page.Page:
            # Если объект является страницей
            titles.add(current_page.name)
            # в хранилище заголовков добавляем имя страницы
    return titles


def all_subsections_from_section(
    section: mwparserfromhell.wikicode.Wikicode,
    parent_titles: list[str],
    sections_to_ignore: set[str],
) -> list[tuple[list[str], str]]:
    """
    Из раздела Википедии возвращает список всех вложенных секций.
    Каждый подраздел представляет собой кортеж, где:
      - первый элемент представляет собой список родительских секций, 
      начиная с заголовка страницы
      - второй элемент представляет собой текст секции

    Args:
        section (mwparserfromhell.wikicode.Wikicode): текущая секция
        parent_titles (list[str]): Заголовки родителя
        sections_to_ignore (set[str]): Секции, которые необходимо проигнорировать

    Returns:
        список всех вложенных секций для заданной секции страницы
    """

    # Извлекаем заголовки текущей секции
    # headings = [str(h) for h in section.filter_headings()]
    headings = list(map(str, section.filter_headings()))
    title = headings[0]
    # Заголовки Википедии имеют вид: "== Heading =="

    if title.strip("= ") in sections_to_ignore:
        # Если заголовок секции в списке для игнора, то пропускаем его
        return []

    # Объединим заголовки и подзаголовки, чтобы сохранить контекст для chatGPT
    titles = parent_titles + [title]

    # Преобразуем wikicode секции в строку
    full_text = str(section)

    # Выделяем текст секции без заголовка
    section_text = full_text.split(title)[1]
    if len(headings) == 1:
        # Если один заголовок, то формируем результирующий список
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        # Формируем результирующий список из текста до первого подзаголовка
        results = [(titles, section_text)]
        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(
                # Вызываем функцию получения вложенных секций для заданной секции
                all_subsections_from_section(subsection, titles, sections_to_ignore)
                )  # Объединяем результирующие списки данной функции и вызываемой
        return results


def all_subsections_from_title(
    title: str,
    site: mwclient.client.Site,
    sections_to_ignore: set[str] = SECTIONS_TO_IGNORE,
) -> list[tuple[list[str], str]]:
    """
    Из заголовка страницы Википедии возвращает список всех вложенных секций.
    Каждый подраздел представляет собой кортеж, где:
      - первый элемент представляет собой список родительских секций,
      начиная с заголовка страницы
      - второй элемент представляет собой текст секции

    Args:
        title (str): Заголовок статьи Википедии, которую парсим
        site (mwclient.client.Site): Объект сайта (HTTP-сессия)
        sections_to_ignore (set[str]): Секции, которые игнорируем

    Returns:
        список всех секций страницы, за исключением тех, которые отбрасываем
    """
    # Запрашиваем страницу по заголовку
    page = site.pages[title]

    # Получаем текстовое представление страницы
    text = page.text()

    # Удобный парсер для MediaWiki
    parsed_text = mwparserfromhell.parse(text)
    # Извлекаем заголовки
    # headings = [str(h) for h in parsed_text.filter_headings()]
    headings = list(map(str, parsed_text.filter_headings()))
    if headings:
        # Если заголовки найдены
        # В качестве резюме берем текст до первого заголовка
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        # Если нет заголовков, то весь текст считаем резюме
        summary_text = str(parsed_text)
    results = [([title], summary_text)]
    # Добавляем резюме в результирующий список
    for subsection in parsed_text.get_sections(levels=[2]):
        # Извлекаем секции 2-го уровня
        results.extend(
            # Вызываем функцию получения вложенных секций для заданной секции
            # Объединяем результирующие списки данной функции и вызываемой
            all_subsections_from_section(subsection, [title], sections_to_ignore)
        )
    return results


def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    """
    Очистка текста секции от ссылок <ref>xyz</ref>,
    начальных и конечных пробелов
    """
    titles, text = section
    # Удаляем ссылки
    text = re.sub(r"<ref.*?</ref>", "", text)
    # Удаляем пробелы вначале и конце
    text = text.strip()
    return (titles, text)


def keep_section(
    section: tuple[list[str], str],
    string_length: int = 16
) -> bool:
    """
    Фильтрует короткие и пустые секции

    Возвращает значение True,
    если раздел должен быть сохранен,
    в противном случае значение False.

    Args:
        section (tuple[list[str], str]): Фильтруемый раздел
        string_length (int): Фильтр по длине, по умолчанию 16
    """
    titles, text = section
    if len(text) < string_length:
        return False
    return True


def run_parser():
    # Инициализация объекта MediaWiki
    # WIKI_SITE ссылается на англоязычную часть Википедии
    site = mwclient.Site(WIKI_SITE)

    # Загрузка раздела заданной категории
    category_page = site.pages[CATEGORY_TITLE]
    # Получение множества всех заголовков категории
    #  с вложенностью на 2 уровеня
    titles = titles_from_category(category_page, max_depth=2)

    # Разбивка статей на секции
    # придется немного подождать,
    # на парсинг 100 статей требуется около минуты
    wikipedia_sections = []
    count = 0
    for title in titles:
        try:
            wikipedia_sections.extend(all_subsections_from_title(title, site))
        except ConnectionError as exc:
            logging.error(f'Error with title: {title}')
            logging.error(exc)
            site = mwclient.Site(WIKI_SITE)
            # return []
        print(title)
        count += 1
        print(count)

    wikipedia_sections = [
        clean_section(ws)
        for ws in wikipedia_sections
        if keep_section(ws)
    ]
    return wikipedia_sections


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Возвращает число токенов в строке."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """
    Функция разделения строк
    Разделяет строку надвое с помощью разделителя (delimiter),
    балансирует токены с каждой стороны.
    """

    # Делим строку на части по разделителю, по умолчанию \n - перенос строки
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # разделитель не найден
    elif len(chunks) == 2:
        return chunks  # нет необходимости искать промежуточную точку
    else:
        # Считаем токены
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        # Предварительное разделение по середине числа токенов
        best_diff = halfway
        # В цикле ищем какой из разделителей, будет ближе всего к best_diff
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        # Возвращаем левую и правую часть
        # оптимально разделенной строки
        return [left, right]


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """
    Обрезка строки до максимально разрешенного числа токенов.

    Args:
        string (str): строка
        model (str): модель
        max_tokens (int): максимальное число разрешенных токенов
        print_warning (bool): True, флаг вывода предупреждения
    """
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    # Обрезаем строку и декодируем обратно
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(
            f"Предупреждение: Строка обрезана с {len(encoded_string)} токенов до {max_tokens} токенов."
        )
    # Усеченная строка
    return truncated_string


def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Разделяет секции на список из частей секций,
    в каждой части не более max_tokens.
    Каждая часть представляет собой кортеж
    родительских заголовков [H1, H2, ...] и текста (str).

    Args:
        subsection (tuple[list[str], str]): секции
        max_tokens (int): 1000, максимальное число токенов
        model (str): GPT_MODEL,  модель
        max_recursion (int): 5,  максимальное число рекурсий
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # Если длина соответствует допустимой, то вернет строку
    if num_tokens_in_string <= max_tokens:
        return [string]
    # если в результате рекурсии не удалось разделить строку,
    # то просто усечем ее по числу токенов
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # иначе разделим пополам и выполним рекурсию
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            # Пробуем использовать разделители
            # от большего к меньшему (разрыв, абзац, точка)
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # если какая-либо половина пуста, повторяем попытку
                # с более простым разделителем
                continue
            else:
                # применим рекурсию на каждой половине
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                        # уменьшаем максимальное число рекурсий
                    )
                    results.extend(half_strings)
                return results
    # иначе никакого разделения найдено не было,
    # поэтому просто обрезаем строку (должно быть очень редко)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]


def section_saver(strings: list[str], path: str):
    df = pd.DataFrame(dict(text=strings))
    # df['embedding']
    df.to_csv(path, index=False)
    return df


if __name__ == '__main__':
    wikipedia_sections = run_parser()

    # Делим секции на части
    MAX_TOKENS = 1600
    wikipedia_strings = []
    for section in wikipedia_sections:
        wikipedia_strings.extend(
            split_strings_from_subsection(section, max_tokens=MAX_TOKENS)
        )

    print(
        f"{len(wikipedia_sections)} секций Википедии "
        f"поделены на {len(wikipedia_strings)} строк."
    )
    df = section_saver(wikipedia_strings, CSV_FILENAME)
    print(df.head())
