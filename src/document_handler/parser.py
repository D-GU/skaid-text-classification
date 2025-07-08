import pymupdf as pm


class Parser:
    def __call__(self, path: str):
        document = self.__open(path)
        return self.__get_text(document)

    def __open(self, path: str):
        """
        Метод, открывающий PDF-файл. Если указанный путь к файлу неверный,
        то срабатывает исключение FileNotFound и пользователь уведомляется о том,
        что файла по указанному пути не существует.

        :return: PDF-файл
        """
        try:
            return pm.open(path)
        except pm.FileNotFoundError:
            raise Exception(f"Указанный файл: {path} не найден.")

    def __get_text(self, document):
        """
        Метод, позволяющий распарсить текст из указанного PDF-файла.
        Если по итогам парсинга не было найдено текста, то срабатывает исключение,
        уведомляющее пользователя о том, что текста не было найдено.

        :return: parsed_text: str - строка с найденным текстом.
        """
        parsed_text = ""

        for page in document:
            parsed_text += page.get_text()

        if not parsed_text or parsed_text.strip() == "":
            raise Exception(
                "В указанном файле нет текста."
            )

        return parsed_text
