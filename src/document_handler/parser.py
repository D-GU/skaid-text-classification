import pymupdf as pm


class Parser:
    def __init__(self, path):
        self.path = path
        self.__document = self.__open()

    def __open(self):
        """
        Метод, открывающий PDF-файл. Если указанный путь к файлу неверный,
        то срабатывает исключение FileNotFound и пользователь уведомляется о том,
        что файла по указанному пути не существует.

        :return: PDF-файл
        """

        try:
            return pm.open(self.path)
        except pm.FileNotFoundError:
            raise Exception(f"Указанный файл: {self.path} не найден.")

    def __get_text(self):
        """
        Метод, позволяющий распарсить текст из указанного PDF-файла.
        Если по итогам парсинга не было найдено текста, то срабатывает исключение,
        уведомляющее пользователя о том, что текста не было найдено.

        :return: parsed_text: str - строка с найденным текстом.
        """
        parsed_text = ""

        for page in self.__document:
            parsed_text += page.get_text()

        if not parsed_text or parsed_text.strip() == "":
            raise Exception(
                "В указанном файле нет текста."
            )

        return parsed_text

    def parse(self):
        return self.__get_text()


if __name__ == "__main__":
    parser = Parser("../../JPMartinez2004.pdf").parse()
