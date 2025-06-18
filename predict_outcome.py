import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx # Для визуализации, если нужно

# --- Константы ---
MODEL_FILE = 'trained_bn_model.pkl' # Файл для сохранения/загрузки обученной модели
CONFIG_FILE = 'bn_config.json'      # Файл конфигурации
EVIDENCE_FILE = 'incident_evidence.json'      # Входной файл со свидетельствами

# --- Функция Загрузки Обученной Модели ---
def load_model(filename=MODEL_FILE):
    """Загружает обученную модель из файла."""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Обученная модель успешно загружена из файла: {filename}")
        if not model.cpds:
             print("ПРЕДУПРЕЖДЕНИЕ: Модель загружена, но не содержит обученных CPT.")
             return None
        return model
    except FileNotFoundError:
        print(f"Ошибка: Файл с обученной моделью '{filename}' не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None

# --- Функция Загрузки Конфигурации ---
def load_config(filename=CONFIG_FILE):
    """Загружает конфигурацию из JSON."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Конфигурация JSON успешно загружена из файла: {filename}")
        return config
    except FileNotFoundError:
        print(f"Ошибка: Файл конфигурации '{filename}' не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке конфигурации: {e}")
        return None

# --- Функция Загрузки Свидетельств ---
def load_evidence(filename=EVIDENCE_FILE):
    """Загружает свидетельства из JSON."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            evidence_data = json.load(f)
        if 'evidence' not in evidence_data:
             print(f"Ошибка: В файле '{filename}' отсутствует ключ 'evidence'.")
             return None
        evidence_dict = evidence_data['evidence']
        print(f"Свидетельства успешно загружены из файла: {filename}")
        return evidence_dict
    except FileNotFoundError:
        print(f"Ошибка: Файл со свидетельствами '{filename}' не найден.")
        return None
    except json.JSONDecodeError:
         print(f"Ошибка: Файл '{filename}' не является корректным JSON.")
         return None
    except Exception as e:
        print(f"Ошибка при загрузке свидетельств: {e}")
        return None

# --- Функция Предсказания Исхода ---
def predict_outcome(model, evidence, config):
    """Выполняет логический вывод для предсказания исхода."""
    if not model or not model.cpds or not evidence or not config or \
       'outcome_node' not in config or 'outcome_states' not in config:
        print("Ошибка: Некорректные входные данные для predict_outcome.")
        return None, None # Возвращаем None для распределения и None для словаря

    outcome_node = config['outcome_node']
    outcome_states_config = config['outcome_states']

    print("\n--- Выполнение логического вывода ---")
    print("Заданные свидетельства (Evidence):")
    valid_evidence = {}
    for node, state in evidence.items():
        if node in model.nodes():
            valid_evidence[node] = state
            print(f"  - {node}: {state}")
        else:
            print(f"  - ПРЕДУПРЕЖДЕНИЕ: Узел '{node}' из файла свидетельств отсутствует в модели, игнорируется.")

    if not valid_evidence:
         print("Ошибка: Нет валидных свидетельств для выполнения запроса.")
         return None, None

    try:
        inference = VariableElimination(model)
        result_distribution = inference.query(variables=[outcome_node], evidence=valid_evidence)
        print(f"\nПредсказанное распределение вероятностей для '{outcome_node}':")
        print(result_distribution)

        print("\nВероятности конкретных исходов:")
        results_dict = {}
        for state_info in outcome_states_config:
            code = state_info['code']
            label = state_info.get('label', f'Состояние {code}')
            try:
                 state_dict = {outcome_node: code}
                 probability = result_distribution.get_value(**state_dict)
                 results_dict[label] = probability
                 print(f"  P({outcome_node}={code} - {label} | Свидетельства) = {probability:.4f}")
            except (KeyError, ValueError): # Добавляем ValueError для случаев, когда состояние может быть некорректным
                 print(f"  ПРЕДУПРЕЖДЕНИЕ: Состояние {code} для '{outcome_node}' отсутствует или некорректно в результате запроса (вероятность 0).")
                 results_dict[label] = 0.0
            except Exception as e_gv:
                 print(f"  Ошибка при получении значения для состояния {code}: {e_gv}")
                 results_dict[label] = None
        # Возвращаем и распределение, и словарь
        return result_distribution, results_dict

    except ValueError as e:
        print(f"\nОШИБКА ValueError во время выполнения запроса: {e}")
    except Exception as e:
        print(f"\nНеожиданная ОШИБКА во время выполнения запроса: {e}")
        return None, None

# --- Функция Оценки Влияния Факторов ---
def assess_factor_influence(model, base_evidence, config):
    """
    Оценивает влияние каждого фактора на вероятность неблагоприятных исходов
    (Авария или Катастрофа).
    Возвращает ранжированный список факторов.
    """
    if not model or not model.cpds or not config or \
       'outcome_node' not in config or 'outcome_states' not in config or 'factors' not in config:
        print("Ошибка: Некорректные входные данные для assess_factor_influence.")
        return None

    outcome_node = config['outcome_node']
    # Определяем коды неблагоприятных исходов (Авария=1, Катастрофа=2)
    adverse_outcome_codes = [s['code'] for s in config['outcome_states'] if s['code'] > 0]
    if not adverse_outcome_codes:
        print("Ошибка: Не найдены коды для неблагоприятных исходов (Авария/Катастрофа) в конфигурации.")
        return None

    print("\n--- Оценка влияния факторов на неблагоприятные исходы ---")
    print(f"(Неблагоприятные исходы: коды {adverse_outcome_codes})")

    try:
        inference = VariableElimination(model)

        # 1. Рассчитываем базовую вероятность неблагоприятного исхода с УЧЕТОМ свидетельств
        current_evidence = {}
        print("Используемые свидетельства для базовой вероятности:")
        for node, state in base_evidence.items():
             if node in model.nodes():
                 current_evidence[node] = state
                 print(f"  - {node}: {state}")
             else:
                 print(f"  - ПРЕДУПРЕЖДЕНИЕ (база): Узел '{node}' игнорируется.")

        if not current_evidence:
             print("Нет валидных свидетельств для расчета базовой вероятности.")
             # Можно расчитать априорную вероятность как базу
             print("Расчет базовой вероятности без свидетельств (априорная)...")
             base_prob_dist = inference.query(variables=[outcome_node])
        else:
             base_prob_dist = inference.query(variables=[outcome_node], evidence=current_evidence)

        base_adverse_prob = 0.0
        for code in adverse_outcome_codes:
            try:
                base_adverse_prob += base_prob_dist.get_value(**{outcome_node: code})
            except (KeyError, ValueError):
                 continue # Пропускаем, если состояния нет
        print(f"Базовая вероятность неблагоприятного исхода (P(Исход>0 | Св-ва)): {base_adverse_prob:.4f}")


        # 2. Итерируем по факторам, меняя их состояние на "проблемное"
        factor_influence = {}
        factors_config = config['factors']

        for factor_info in factors_config:
            factor_id = factor_info['id']
            if factor_id not in model.nodes():
                # print(f"Фактор '{factor_id}' отсутствует в модели, пропускаем.")
                continue # Пропускаем факторы не из модели

            # Находим первое "проблемное" состояние (код > 0)
            problem_states = [s['code'] for s in factor_info.get('states', []) if s['code'] > 0]
            if not problem_states:
                # print(f"Для фактора '{factor_id}' не определены проблемные состояния (>0), пропускаем.")
                continue # Пропускаем, если нет проблемных состояний

            # Используем первое проблемное состояние для оценки
            problem_state_code = min(problem_states)
            problem_state_label = next((s.get('label', str(problem_state_code)) for s in factor_info.get('states', []) if s['code'] == problem_state_code), str(problem_state_code))


            # Формируем свидетельства для этого фактора
            # Берем базовые свидетельства и добавляем/заменяем текущий фактор
            evidence_with_factor = current_evidence.copy()
            evidence_with_factor[factor_id] = problem_state_code

            # Рассчитываем вероятность неблагоприятного исхода при наличии этого фактора
            try:
                 prob_dist_with_factor = inference.query(variables=[outcome_node], evidence=evidence_with_factor)
                 adverse_prob_with_factor = 0.0
                 for code in adverse_outcome_codes:
                     try:
                          adverse_prob_with_factor += prob_dist_with_factor.get_value(**{outcome_node: code})
                     except (KeyError, ValueError):
                          continue
                 # Оцениваем ВЛИЯНИЕ как ОТНОШЕНИЕ или РАЗНИЦУ
                 # Используем отношение - во сколько раз увеличивается вероятность
                 if base_adverse_prob > 1e-9: # Избегаем деления на ноль
                      influence_ratio = adverse_prob_with_factor / base_adverse_prob
                 else:
                      influence_ratio = float('inf') if adverse_prob_with_factor > 1e-9 else 1.0 # Бесконечность или 1, если оба близки к нулю

                 # Можно использовать разницу: influence_diff = adverse_prob_with_factor - base_adverse_prob
                 factor_influence[f"{factor_id} (Сост. {problem_state_code} - {problem_state_label})"] = {
                     'prob_with_factor': adverse_prob_with_factor,
                     'influence_ratio': influence_ratio
                     # 'influence_diff': influence_diff
                 }

            except ValueError as e_inf:
                 print(f"  ПРЕДУПРЕЖДЕНИЕ: Ошибка инференса для фактора '{factor_id}'={problem_state_code}: {e_inf}. Пропускаем.")
            except Exception as e_inf_other:
                 print(f"  Неожиданная ошибка инференса для фактора '{factor_id}': {e_inf_other}. Пропускаем.")


        # 3. Ранжируем факторы по степени влияния (по убыванию отношения вероятностей)
        ranked_factors = sorted(factor_influence.items(), key=lambda item: item[1]['influence_ratio'], reverse=True)

        return ranked_factors

    except Exception as e:
        print(f"\nОшибка при оценке влияния факторов: {e}")
        return None


# --- Основной блок выполнения ---
if __name__ == "__main__":
    # 1. Загружаем конфигурацию
    config_data = load_config()
    if not config_data:
        exit()

    # 2. Загружаем обученную модель
    trained_model = load_model()
    # --- Сюда нужно вставить ваш код для переобучения модели, если файл не найден ---
    # --- (как в предыдущем ответе), чтобы trained_model был определен ---
    if not trained_model or not trained_model.cpds:
        print("Не удалось загрузить или обучить модель. Выход.")
        exit()

    # 3. Загружаем свидетельства
    evidence_dict = load_evidence()
    if not evidence_dict:
        exit()

    # 4. Делаем предсказание исхода
    predicted_distribution, predicted_probabilities = predict_outcome(trained_model, evidence_dict, config_data)

    # 5. Оцениваем и ранжируем влияние факторов
    if trained_model and config_data and evidence_dict: # Убедимся что все загружено
        ranked_influence = assess_factor_influence(trained_model, evidence_dict, config_data)

        if ranked_influence:
            print("\n--- Ранжированная таблица влияния факторов на неблагоприятный исход ---")
            print("(Факторы отсортированы по убыванию отношения P(Небл. исход | Фактор > 0, Св-ва) / P(Небл. исход | Св-ва))")
            print("-" * 80)
            print("{:<50} | {:<15} | {:<15}".format("Фактор (Проблемное состояние)", "P(Небл.|Факт.)", "Коэфф. влияния"))
            print("-" * 80)
            for factor_name, influence_data in ranked_influence:
                 prob_w_factor = influence_data['prob_with_factor']
                 ratio = influence_data['influence_ratio']
                 # Форматируем бесконечность для вывода
                 ratio_str = "inf" if ratio == float('inf') else f"{ratio:.2f}"
                 print("{:<50} | {:<15.4f} | {:<15}".format(factor_name, prob_w_factor, ratio_str))
            print("-" * 80)

    # 6. Выводим наиболее вероятный исход (из шага 4)
    if predicted_probabilities:
        most_likely_outcome_label = max(predicted_probabilities, key=predicted_probabilities.get)
        print(f"\nНаиболее вероятный исход при заданных свидетельствах: {most_likely_outcome_label} (Вероятность: {predicted_probabilities[most_likely_outcome_label]:.4f})")

    print("\n--- Выполнение скрипта завершено ---")