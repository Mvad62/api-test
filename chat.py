import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def load_settings() -> tuple[str, str, list[str]]:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_API_BASE", "").strip()
    models = list(
        dict.fromkeys(
            model
            for name in ("MODEL_NAME1", "MODEL_NAME2")
            if (model := os.getenv(name, "").strip())
        )
    )

    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_API_BASE", base_url),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"В .env не заданы обязательные настройки: {', '.join(missing)}")
    if not models:
        raise RuntimeError("В .env не задана ни одна модель (MODEL_NAME1 или MODEL_NAME2).")

    return api_key, base_url, models


def choose_model(models: list[str]) -> str:
    print("Выберите модель:")
    for index, model in enumerate(models, start=1):
        print(f"{index}. {model}")

    while True:
        selection = input(f"Номер модели [1-{len(models)}]: ").strip()
        if selection.isdigit() and 1 <= int(selection) <= len(models):
            return models[int(selection) - 1]
        print("Некорректный выбор. Введите номер модели из списка.")


class ChatSession:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model
        self.messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "Ты полезный ассистент."}
        ]

    def generate_text(self, user_text: str) -> str:
        """Отправляет сообщение с историей диалога и сохраняет ответ в контекст."""
        conversation = [*self.messages, {"role": "user", "content": user_text}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
        )
        answer = response.choices[0].message.content
        if answer is None:
            raise RuntimeError("API вернул ответ без текста.")

        self.messages = [
            *conversation,
            {"role": "assistant", "content": answer},
        ]
        return answer


def main() -> None:
    api_key, base_url, models = load_settings()
    model = choose_model(models)
    session = ChatSession(OpenAI(api_key=api_key, base_url=base_url), model)
    print(f"Используется модель: {model}")
    print("Чат запущен. Для выхода введите /exit.")

    while True:
        user_text = input("Вы: ").strip()
        if user_text.lower() == "/exit":
            break
        if not user_text:
            continue

        print(f"Ассистент: {session.generate_text(user_text)}")


if __name__ == "__main__":
    main()
