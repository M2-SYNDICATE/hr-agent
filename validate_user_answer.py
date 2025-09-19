import time

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal, Any
from enum import Enum
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from ollama import Client
import re

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY") #NVIDIA_API OPENROUTER_API_KEY
if not api_key:
    raise ValueError("Не найден API ключ OPENROUTER_API_KEY")

# ==============================================================================
# БЛОК 1: КЛАССЫ И ФУНКЦИИ ДЛЯ ПРОВЕДЕНИЯ ИНТЕРВЬЮ (без изменений)
# ==============================================================================

class InterviewState:
    def __init__(self, questions_data: Dict):
        self.questions_asked = 0
        self.collected_data_by_category = {}
        self.all_questions = []
        for category, questions_list in questions_data.items():
            for i, q in enumerate(questions_list):
                self.all_questions.append({
                    "id": f"{category}_{i}",
                    "category": category,
                    "question": q.get("question"),
                    "example_answer": q.get("expected_response")
                })

    def get_current_question(self):
        if self.questions_asked < len(self.all_questions):
            return self.all_questions[self.questions_asked]
        return None

    def collect_answer(self, category: str, question_text: str, user_answer: str):
        if category not in self.collected_data_by_category:
            self.collected_data_by_category[category] = []
        answers_for_category = self.collected_data_by_category[category]
        updated_answers = [ans for ans in answers_for_category if ans.get("question") != question_text]
        updated_answers.append({"question": question_text, "answer": user_answer})
        self.collected_data_by_category[category] = updated_answers

    def move_to_next_question(self):
        self.questions_asked += 1

    def move_to_previous_question(self):
        if self.questions_asked > 0:
            self.questions_asked -= 1

    def is_interview_complete(self):
        return self.questions_asked >= len(self.all_questions)


class AIHRPipeline:
    def __init__(self, questions_data: Dict, vacancy_name: str):
        self.state = InterviewState(questions_data)
        self.ollama = Client()  # локальный Ollama client
        self.model_name = "hf.co/RefalMachine/RuadaptQwen3-8B-Hybrid-GGUF:Q4_K_M"
        self.vacancy_name = vacancy_name
        self.interaction_tool = self._create_interaction_tool_definitions()

        # Pre-warm модели (снижает холодный старт)
        try:
            warm_messages = [
                {"role": "system", "content": self._create_system_prompt(self.state.get_current_question())},
                {"role": "user", "content": "Прогрев. Пожалуйста, верни 1 токен."}
            ]
            warm_resp = self.ollama.chat(
                model=self.model_name,
                messages=warm_messages,
                stream=False,
                options={"max_tokens": 1, "temperature": 0.0}
            )
        except Exception as e:
            print("[WARN] Prewarm failed:", e)

    def _create_interaction_tool_definitions(self):
        return [{"type": "function",
                 "function": {"name": "process_response", "description": "Классифицирует ответ кандидата.",
                              "parameters": {"type": "object", "properties": {"response_type": {"type": "string",
                                                                                                "enum": ["ANSWER",
                                                                                                         "UNCERTAIN",
                                                                                                         "REPEAT_REQUEST",
                                                                                                         "PREVIOUS_QUESTION_REQUEST"]},
                                                                              "message": {"type": "string",
                                                                                          "description": "Ответ для кандидата (на русском)."}},
                                             "required": ["response_type", "message"]}}}]

    def _create_system_prompt(self, current_question: Dict) -> str:
        return f"""Ты — AI HR-интервьюер на позицию '{self.vacancy_name}' Ты женского пола, все твои ответы в женском роде.
Текущий вопрос: {current_question['question']}

**Задачи:**
1. Классифицируй ответ кандидата:
   - ANSWER: кандидат отвечает по теме или явно просит перейти к следующему вопросу или говорит прямо, что не знает ответа.
   - UNCERTAIN: кандидат растерян, не знает, что сказать или с чего начать, просит уточнение по вопросу или его ответ не полный и краткий. Если ответ не полный, попроси его дополнить ответ. Если человек задается вопросом относительно вопроса, просит что-то уточнить, то ответь на его вопрос. Если вопрос пользователя не касается текущего вопроса, то попроси, чтобы человек говорил по делу
   - REPEAT_REQUEST: кандидат попросил повторить вопрос.
   - PREVIOUS_QUESTION_REQUEST: просьба вернуться к предыдущему вопросу.
2. Сформулируй краткий ответ кандидату на русском языке, исходя из типа ответа.
   - Если это ANSWER, то выведи краткий ответ о том, что принял ответ и всё, не озвучивай следующий вопрос.
   - Если REPEAT_REQUEST, то просто повтори вопрос. Текущий вопрос: {current_question['question']}
   - Если UNCERTAIN, то ответь на запрос пользователя кратко.
   - Если PREVIOUS_QUESTION_REQUEST, то просто скажи что то вроде "Хорошо, возвращаюсь к предыдущему вопросу" и всё, не пиши предыдущий вопрос.

**Важно:** Если кандидат написал что-то вроде "следующий вопрос", "давайте дальше", "пропустить", "не хочу отвечать", это считается `ANSWER`, а не `UNCERTAIN`.  
Ответ для кандидата помещай в поле `message` в tool_call.
Отвечай как человек, отвечай кратко и дружелюбно. Примеры:
"Поняла, Давайте перейдем к следуюшему вопросу"
"Да, конечно, давайте вернемся к предыдущему вопросу"
Не говори "Всё, переход" и т.д., соблюдай деловой стиль речи и говори как человек/no_think"""








    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        import json
        import time

        current_question = self.state.get_current_question()
        if not current_question:
            return {
                "message": "Собеседование завершено! Спасибо.",
                "interview_complete": True,
                "collected_data": self.state.collected_data_by_category
                }

        messages = [
            {"role": "system", "content": self._create_system_prompt(current_question)},
            {"role": "user", "content": f"Ответ кандидата для классификации: <<< {user_input} >>>"}
            ]

        tool_args = None
        try:
            print("\n[INFO] Отправка запроса модели...")
            start_total = time.time()
            start_ttft = time.time()

            stream = self.ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                tools=self.interaction_tool,
                options={"max_tokens": 128, "temperature": 0.1}
                )

            # Замеряем TTFT — время до первого tool_call
            for chunk in stream:
                msg = chunk.get("message", {})
                tool_calls = msg.get("tool_calls") or msg.get("tool_call") or []
                if tool_calls:
                    first_call = tool_calls[0]
                    tool_args = first_call.get("function", {}).get("arguments") or first_call.get("arguments")
                    print(f"[TTFT] Время до получения первого tool_call: {time.time() - start_ttft:.3f} сек")
                    break  # первый tool_call получен

            print(f"[GEN_TIME] Полная генерация заняла: {time.time() - start_total:.3f} сек")
            print(f"[TOOL_CALL]: {tool_args}")

            # Если tool_args — str, преобразуем в dict
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
            
            if not tool_args:
                return {"message": "Не удалось разобрать ответ модели (нет tool_call).", "interview_complete": False}

        except Exception as e:
            print(f"[ERROR] Ошибка при вызове локальной модели: {e}")
            return {"message": "Произошла внутренняя ошибка."}

        response_type = tool_args.get("response_type")
        action = None
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)

        raw_message = tool_args.get("message", "") or ""

        # --- удаляем содержимое между <think>...</think> (если есть) ---
        cleaned = re.sub(r"<think.*?>.*?</think>", "", raw_message, flags=re.S | re.I)

        # --- удаляем одиночные теги <think> или </think> на всякий случай ---
        cleaned = re.sub(r"</?think.*?>", "", cleaned, flags=re.I)

        # --- удаляем другие служебные маркеры типа [THINK] ... [/THINK] ---
        cleaned = re.sub(r"\[/?THINK\].*?\[/THINK\]", "", cleaned, flags=re.S | re.I)

        # --- тримим и используем как финальный текст ---
        cleaned = cleaned.strip()

        # если после очистки пусто — fallback: взять raw_message без тегов
        final_text_for_user = cleaned if cleaned else raw_message.strip()

        # записываем в финальный ответ
        final_response = {"message": final_text_for_user, "interview_complete": False}


        if response_type == "ANSWER":
            action = "next_question"
            self.state.collect_answer(current_question["category"], current_question["question"], user_input)
            self.state.move_to_next_question()
            next_q = self.state.get_current_question()
            if not next_q:
                final_response["interview_complete"] = True
                final_response["message"] += "\n\nЭто был последний вопрос."
                final_response["collected_data"] = self.state.collected_data_by_category
            else:
                final_response["next_question"] = next_q["question"]
        elif response_type in ["REPEAT_REQUEST", "UNCERTAIN"]:
            action = "stay_on_question"
        elif response_type == "PREVIOUS_QUESTION_REQUEST":
            index_before = self.state.questions_asked
            self.state.move_to_previous_question()
            if index_before == self.state.questions_asked:
                action = "stay_on_question"
                final_response["message"] = "Это самый первый вопрос, возвращаться некуда."
                final_response["next_question"] = current_question["question"]
            else:
                action = "previous_question"
                prev_q = self.state.get_current_question()
                if prev_q:
                    final_response["next_question"] = prev_q["question"]

        final_response["action"] = action
        return final_response


    def start_interview(self) -> str:
        first_question = self.state.get_current_question()
        return f"Добро пожаловать на собеседование на позицию '{self.vacancy_name}'! Давайте начнем.\n\n{first_question['question']}"

# ==============================================================================
# БЛОК 2: ФУНКЦИИ ДЛЯ АНАЛИЗА (с изменениями)
# ==============================================================================

def _create_evaluation_tool_definitions():
    # --- ИЗМЕНЕНИЕ 1: Уточняем, что фидбек должен быть коротким ---
    return [{"type": "function", "function": {"name": "evaluate_answer", "description": "Оценивает ответ кандидата.",
                                              "parameters": {"type": "object",
                                                             "properties": {"score": {"type": "number"},
                                                                            "passed": {"type": "boolean"},
                                                                            "feedback": {"type": "string",
                                                                                         "description": "Масимально краткий отзыв на русском."}},
                                                             "required": ["score", "passed", "feedback"]}}}]


def _create_evaluation_prompt(question: str, answer: str, expected_response: Optional[str], vacancy_name: str) -> str:
    expected_text = f"Критерии для идеального ответа: {expected_response}" if expected_response else "Четких критериев нет. Оцени ответ на основе логичности и полноты."
    # --- ИЗМЕНЕНИЕ 2: Добавляем требование к краткости фидбека ---
    return f"Ты — технический эксперт, оценивающий кандидата на позицию '{vacancy_name}'.\n{expected_text}\n\nПроанализируй связку вопрос-ответ и вызови `evaluate_answer`. **Твой отзыв (`feedback`) должен быть коротким и по существу (1-2 предложения).**\nВопрос: \"{question}\"\nОтвет: \"{answer}\""


# --- ИЗМЕНЕНИЕ 3: Новая функция для генерации итогового резюме ---
def _generate_final_summary(feedbacks: List[str], vacancy_name: str, client: OpenAI) -> str:
    if not feedbacks:
        return "Итоговое резюме не может быть составлено, так как не было получено ни одного отзыва."

    # Соединяем все фидбеки в один текст
    all_feedbacks_text = "\n- ".join(feedbacks)

    system_prompt = f"""Ты — опытный HR-менеджер. Проанализируй следующие краткие комментарии по ответам кандидата на позицию '{vacancy_name}'. 
На основе этих комментариев дай очень короткую итоговую выжимку, сплошным текстом.

Вот комментарии:
- {all_feedbacks_text}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.3
            )
        summary = response.choices[0].message.content
        return summary.strip()
    except Exception as e:
        print(f"[ERROR] Не удалось сгенерировать итоговое резюме: {e}")
        return "Ошибка при генерации итогового резюме."


def analyze_interview_data(collected_data: Dict, questions_data: Dict, vacancy_name: str) -> Dict:
    print("\n--- НАЧАЛО ПОСЛЕДОВАТЕЛЬНОГО АНАЛИЗА РЕЗУЛЬТАТОВ ---")
    api_key_2 = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key_2)
    evaluation_tool = _create_evaluation_tool_definitions()

    analysis_report = []
    total_score = 0
    category_scores = {}
    all_feedbacks = []  # Собираем все фидбеки для итогового резюме

    all_questions = []
    for category, questions_list in questions_data.items():
        category_scores.setdefault(category, 0)
        for q in questions_list:
            all_questions.append(
                {"category": category, "question": q.get("question"), "expected_response": q.get("expected_response")})

    for question_data in all_questions:
        category, question_text, expected_response = question_data["category"], question_data["question"], \
        question_data["expected_response"]
        candidate_answer = next(
            (ans["answer"] for ans in collected_data.get(category, []) if ans["question"] == question_text), None)

        evaluation = {}
        if candidate_answer is None:
            evaluation = {"score": 0, "passed": False, "feedback": "Ответ не был дан."}
        else:
            print(f"Анализирую ответ на вопрос: '{question_text[:40]}...'")
            system_prompt = _create_evaluation_prompt(question_text, candidate_answer, expected_response, vacancy_name)
            try:
                response = client.chat.completions.create(model="deepseek/deepseek-chat-v3.1:free",
                                                          messages=[{"role": "system", "content": system_prompt}],
                                                          tools=evaluation_tool, tool_choice={"type": "function",
                                                                                              "function": {
                                                                                                  "name": "evaluate_answer"}})
                tool_args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                evaluation = {"score": tool_args.get("score"), "passed": tool_args.get("passed"),
                              "feedback": tool_args.get("feedback")}
                time.sleep(1)
                # --- ИЗМЕНЕНИЕ 4: Сохраняем фидбек в список ---
                if evaluation.get("feedback"):
                    all_feedbacks.append(evaluation["feedback"])
            except Exception as e:
                print(f"  - ОШИБКА при оценке: {e}")
                evaluation = {"error": str(e)}

        score = evaluation.get("score", 0)
        total_score += score
        category_scores[category] += score
        analysis_report.append({"category": category, "question": question_text, "expected_response": expected_response,
                                "answer": candidate_answer or "Ответ не дан.", "evaluation": evaluation})

    # --- ИЗМЕНЕНИЕ 5: После цикла вызываем генерацию итогового резюме ---
    print("\nГенерирую итоговое резюме...")
    final_summary_text = _generate_final_summary(all_feedbacks, vacancy_name, client)

    print("\n--- АНАЛИЗ ЗАВЕРШЕН ---")

    score_summary = {"total_score": total_score, "scores_by_category": category_scores}

    return {
        "detailed_report": analysis_report,
        "score_summary": score_summary,
        "final_summary": final_summary_text  # Добавляем резюме в итоговый отчет
        }


# ==============================================================================
# БЛОК 3: ТОЧКА ВХОДА И ОРКЕСТРАЦИЯ (с изменениями)
# ==============================================================================

if __name__ == '__main__':
    vacancy = "Ведущий специалист по обслуживанию ЦОД"
    pipeline = AIHRPipeline(interview_questions2, vacancy_name=vacancy)
    print(pipeline.start_interview())
    raw_data = {}
    while True:
        try:
            user_input = input("\nКандидат: ")
            if user_input.lower() in ['quit', 'exit', 'завершить']:
                print("Интервью завершено по команде пользователя.")
                raw_data = pipeline.state.collected_data_by_category
                break
            result = pipeline.process_user_input(user_input)
            print(f"\nИнтервьюер: {result['message']}")
            if result.get('interview_complete'):
                print(f"\nИнтервью завершено.")
                raw_data = result.get('collected_data', {})
                break
            if result.get('action') == 'next_question' and 'next_question' in result:
                print(f"\nСледующий вопрос: {result['next_question']}")
            elif result.get('action') in ['repeat_question', 'stay_on_question',
                                          'previous_question'] and 'next_question' in result:
                print(f"\n{result['next_question']}")
        except Exception as e:
            print(f"\nПроизошла ошибка: {e}")
            print("Пожалуйста, попробуйте еще раз или завершите интервью.")

    if raw_data:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        raw_filename = f"interview_results_{timestamp}.json"
        try:
            with open(raw_filename, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=4)
            print(f"\n[INFO] 'Сырые' данные сохранены в: {raw_filename}")
        except Exception as e:
            print(f"\n[ERROR] Не удалось сохранить 'сырые' данные: {e}")

        # --- ИЗМЕНЕНИЕ 6: Обновляем вывод, чтобы показать все три части отчета ---
        final_report_data = analyze_interview_data(collected_data=raw_data, questions_data=interview_questions2,
                                                   vacancy_name=vacancy)

        print("\n" + "=" * 40)
        print("--- ИТОГОВЫЙ ОТЧЕТ ПО КАНДИДАТУ ---")
        print("=" * 40)

        print("\n--- ИТОГОВОЕ РЕЗЮМЕ ---")
        print(final_report_data.get("final_summary"))

        print("\n--- СВОДКА ПО БАЛЛАМ ---")
        print(json.dumps(final_report_data.get("score_summary"), indent=2, ensure_ascii=False))

        print("\n--- ДЕТАЛЬНЫЙ РАЗБОР ---")
        print(json.dumps(final_report_data.get("detailed_report"), indent=2, ensure_ascii=False))

        report_filename = f"final_report_{timestamp}.json"
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(final_report_data, f, ensure_ascii=False, indent=4)
            print(f"\n[INFO] Итоговый отчет сохранен в: {report_filename}")
        except Exception as e:
            print(f"\n[ERROR] Не удалось сохранить итоговый отчет: {e}")
    else:
        print("\nНет данных для анализа.")
