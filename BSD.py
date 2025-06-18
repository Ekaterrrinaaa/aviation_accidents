import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx # Для визуализации
import pickle

MODEL_FILE = 'trained_bn_model.pkl' # Имя файла должно совпадать с константой в predict_outcome.py

# --- 1. Загрузка данных и конфигурации ---
try:
    df = pd.read_csv('aviation_incidents_all.csv')
    with open('bn_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"Данные CSV ('aviation_incidents_refined.csv', {df.shape[0]} строк) и конфигурация JSON успешно загружены.")
except FileNotFoundError:
    print("Ошибка: Один из файлов ('aviation_incidents_refined.csv' или 'bn_config.json') не найден.")
    exit()
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# --- 2. Подготовка данных и валидация ---
factor_nodes = [f['id'] for f in config['factors']]
outcome_node = config['outcome_node']
all_nodes_in_config = factor_nodes + [outcome_node]

state_names = {}
nodes_in_data = []
issues_found_data_config = False

print("\n--- Проверка соответствия данных CSV и конфигурации JSON ---")
# Обрабатываем факторы
for factor in config['factors']:
    node_id = factor['id']
    if node_id in df.columns:
        if 'states' not in factor or not isinstance(factor['states'], list):
             print(f"ОШИБКА: Некорректная структура 'states' для фактора '{node_id}' в JSON.")
             issues_found_data_config = True
             continue
        try:
            allowed_states = sorted([s['code'] for s in factor['states']])
            if not allowed_states:
                 print(f"ОШИБКА: Пустой список состояний для фактора '{node_id}' в JSON.")
                 issues_found_data_config = True
                 continue
            state_names[node_id] = allowed_states
            nodes_in_data.append(node_id)
            numeric_series = pd.to_numeric(df[node_id], errors='coerce')
            unique_vals = numeric_series.dropna().unique()
            unique_int_vals = []
            for val in unique_vals:
                try:
                    int_val = int(val)
                    unique_int_vals.append(int_val)
                    if int_val < 0:
                         print(f"ОШИБКА: Найдено отрицательное значение {int_val} в исходном столбце '{node_id}'.")
                         issues_found_data_config = True
                except (ValueError, TypeError):
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось преобразовать значение '{val}' в столбце '{node_id}' к int для проверки.")
                    continue
            invalid_vals = [val for val in unique_int_vals if val not in allowed_states]
            if invalid_vals:
                print(f"ОШИБКА: В столбце '{node_id}' найдены значения {invalid_vals}, не соответствующие разрешенным состояниям {allowed_states} из JSON.")
                issues_found_data_config = True
        except KeyError as e:
            print(f"ОШИБКА: Отсутствует ключ '{e}' в описании состояний фактора '{node_id}' в JSON.")
            issues_found_data_config = True
        except Exception as e:
             print(f"ОШИБКА: Непредвиденная ошибка при обработке фактора '{node_id}': {e}")
             issues_found_data_config = True
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Фактор '{node_id}' из JSON отсутствует в CSV.")

# Обрабатываем исход
if outcome_node in df.columns:
    if 'outcome_states' not in config or not isinstance(config['outcome_states'], list):
         print(f"ОШИБКА: Некорректная структура 'outcome_states' в JSON.")
         issues_found_data_config = True
    else:
        try:
            allowed_states_outcome = sorted([s['code'] for s in config['outcome_states']])
            if not allowed_states_outcome:
                 print(f"ОШИБКА: Пустой список состояний для исхода '{outcome_node}' в JSON.")
                 issues_found_data_config = True
            else:
                state_names[outcome_node] = allowed_states_outcome
                nodes_in_data.append(outcome_node)
                numeric_series_outcome = pd.to_numeric(df[outcome_node], errors='coerce')
                unique_vals_outcome = numeric_series_outcome.dropna().unique()
                unique_int_vals_outcome = []
                for val in unique_vals_outcome:
                     try:
                         int_val = int(val)
                         unique_int_vals_outcome.append(int_val)
                         if int_val < 0:
                             print(f"ОШИБКА: Найдено отрицательное значение {int_val} в исходном столбце исхода '{outcome_node}'.")
                             issues_found_data_config = True
                     except (ValueError, TypeError):
                         print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось преобразовать значение '{val}' в столбце исхода '{outcome_node}' к int для проверки.")
                         continue
                invalid_vals_outcome = [val for val in unique_int_vals_outcome if val not in allowed_states_outcome]
                if invalid_vals_outcome:
                    print(f"ОШИБКА: В столбце исхода '{outcome_node}' найдены значения {invalid_vals_outcome}, не соответствующие разрешенным состояниям {allowed_states_outcome} из JSON.")
                    issues_found_data_config = True
        except KeyError as e:
             print(f"ОШИБКА: Отсутствует ключ '{e}' в описании состояний исхода в JSON.")
             issues_found_data_config = True
        except Exception as e:
             print(f"ОШИБКА: Непредвиденная ошибка при обработке исхода '{outcome_node}': {e}")
             issues_found_data_config = True
else:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Столбец исхода '{outcome_node}', указанный в JSON, отсутствует в CSV.")
    issues_found_data_config = True

# --- Остановка при наличии проблем валидации ---
if issues_found_data_config:
    print("\nОбнаружены несоответствия или ошибки. Исправьте CSV или JSON и запустите снова.")
    exit()
print("\nПроверка соответствия данных и конфигурации JSON завершена успешно.")

# --- Оставляем только нужные столбцы ---
df_prepared = df[nodes_in_data].copy()

# --- Очистка данных ---
print("\nНачало очистки данных...")
for col in nodes_in_data:
    df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')
df_prepared.replace([np.inf, -np.inf], np.nan, inplace=True)
print("ВНИМАНИЕ: Пропущенные значения (NaN) будут заполнены нулями.")
df_prepared.fillna(0, inplace=True)
try:
    df_prepared = df_prepared.astype(int)
    print("Данные успешно подготовлены и очищены.")
except Exception as e:
    print(f"\nОшибка при преобразовании данных в целые числа: {e}.")
    exit()

# --- 3. Определение структуры РЕАЛИСТИЧНОЙ сети ---
print("\n--- Определение структуры реалистичной сети ---")
edges = []

# === Связи: Организационные факторы (Org) ===
# Узлы Org_PlanningScheduling, Org_Oversight, Org_TrainingQuality, Org_MaintOrganization удалены
edges.extend([
    ('Org_ProceduresQuality', 'HF_Violation'),
    ('Org_ProceduresQuality', 'HF_DecisionError'),
    ('Org_SMS_Effectiveness', 'Org_SafetyCulture'),
    ('Org_Resources', 'Ctx_Workload'),
    ('Org_Pressure', 'HF_Stress'),
    ('Org_Pressure', 'HF_Violation'),
    ('Org_Pressure', 'HF_Maint_Pressure'),
    ('Org_SafetyCulture', 'HF_Violation'),
    # Добавлено влияние Госнадзора
    ('Org_RegulatoryOversight', 'Org_ProceduresQuality'),
    ('Org_RegulatoryOversight', 'Org_SMS_Effectiveness'),
    ('Org_RegulatoryOversight', 'Org_SafetyCulture'),
    # ('Org_RegulatoryOversight', 'Org_Oversight'), # Org_Oversight был удален
])

# === Связи: Внешние условия (Env) ===
# Узлы HF_ATC_Workload, HF_ATC_Separation, Env_VolcanicAsh, HF_Ground_Damage удалены
edges.extend([
    ('Env_Icing', 'Ctx_TaskComplexity'),
    ('Env_Turbulence', 'Ctx_TaskComplexity'),
    ('Env_WindShear', 'Ctx_TaskComplexity'),
    ('Env_Thunderstorm', 'Ctx_TaskComplexity'),
    ('Env_Visibility', 'Ctx_TaskComplexity'),
    ('Env_Ceiling', 'Ctx_TaskComplexity'),
    ('Env_RunwayCondition', 'Ctx_TaskComplexity'),
    ('Env_Terrain', 'Ctx_TaskComplexity'),
    ('Env_TrafficDensity', 'Ctx_Workload'),
    ('Env_AirportLights', 'HF_PerceptionError'),
    ('Env_NavAids', 'HF_PerceptionError'),
    ('Env_Visibility', 'HF_PerceptionError'),
    ('Env_Ceiling', 'HF_PerceptionError'),
    ('Env_Thunderstorm', 'HF_Stress'),
    ('Env_Turbulence', 'HF_Stress'),
    ('Env_WindShear', 'HF_Stress'),
    ('Env_WindShear', 'HF_SkillError'),
    ('Env_RunwayCondition', 'HF_SkillError'),
    ('Env_SurfaceWind', 'HF_SkillError'),
    ('Env_Icing', 'Tech_SystemFailure'),
    ('Env_BirdstrikeRisk', 'Tech_SystemFailure'),
    ('Env_Thunderstorm', 'Tech_SystemFailure'),
    # Добавлено влияние Температуры
    ('Env_Temperature', 'Tech_SystemFailure'),
    ('Env_Temperature', 'Ctx_TaskComplexity'),
    ('Env_Temperature', 'Env_RunwayCondition'), # Низкая T -> лед
    ('Env_Temperature', 'Env_Icing'),          # T около нуля -> обледенение
    # Добавлено влияние Препятствий
    ('Env_Obstacles', 'Ctx_TaskComplexity'),
    ('Env_Obstacles', 'HF_PerceptionError'),
    ('Env_Obstacles', 'HF_SkillError'),
    ('Env_Obstacles', outcome_node), # Прямое влияние на исход
    # Добавлено влияние Сложности ВП
    ('Env_AirspaceComplexity', 'Ctx_Workload'),
    ('Env_AirspaceComplexity', 'HF_DecisionError'),
    ('Env_AirspaceComplexity', 'HF_SkillError'),
    # ('Env_AirspaceComplexity', 'HF_ATC_InstructionError'), # Узел УВД удален
])

# === Связи: Контекст полета (Ctx) -> HF ===
# Узел HF_Fatigue удален
edges.extend([
    ('Ctx_TaskComplexity', 'Ctx_Workload'),
    ('Ctx_TaskComplexity', 'HF_Stress'),
    ('Ctx_Workload', 'HF_Stress'),
    ('Ctx_Workload', 'HF_Distraction'),
    ('Ctx_Workload', 'HF_SkillError'),
    ('Ctx_Workload', 'HF_DecisionError'),
    ('Ctx_UnexpectedEvents', 'HF_Stress'),
    ('Ctx_UnexpectedEvents', 'HF_DecisionError'),
    ('Ctx_FlightPhase', 'Ctx_TaskComplexity'),
    ('Ctx_FlightPhase', 'Ctx_Workload'),
])

# === Связи: Технические факторы (Tech) ===
# Узел Tech_ManufacturingDefect удален
edges.extend([
    ('Tech_SystemFailure', 'Ctx_UnexpectedEvents'),
    ('Tech_SystemFailure', 'Ctx_Workload'),
    ('Tech_Fire', 'Ctx_UnexpectedEvents'),
    ('Tech_Fire', 'Ctx_Workload'),
    ('Tech_Fire', 'HF_Stress'),
    ('Tech_SystemFailure', 'HF_DecisionError'),
    ('Tech_SystemFailure', 'HF_SkillError'),
    ('Tech_SystemFailure', outcome_node),
    ('Tech_Fire', outcome_node),
    ('Tech_DesignFlaw', 'Tech_SystemFailure'),
])

# === Связи: Человеческий фактор (HF) ===
# ... (удалены связи с устраненными узлами) ...
edges.extend([
    ('HF_Stress', 'HF_SkillError'),
    ('HF_Stress', 'HF_DecisionError'),
    ('HF_Stress', 'HF_PerceptionError'),
    ('HF_Distraction', 'HF_SkillError'),
    ('HF_Distraction', 'HF_DecisionError'),
    ('HF_KnowledgeSkill', 'HF_SkillError'),
    ('HF_KnowledgeSkill', 'HF_DecisionError'),
    ('HF_CRM', 'HF_DecisionError'),
    ('HF_CRM', 'HF_SkillError'),
    ('HF_Ground_FuelError', 'Tech_SystemFailure'),
    ('HF_Ground_Deicing', 'Env_Icing'),
    ('HF_Ground_FOD', 'Tech_SystemFailure'),
    ('HF_ATC_InstructionError', 'HF_DecisionError'),
    ('HF_SkillError', outcome_node),
    ('HF_DecisionError', outcome_node),
    ('HF_PerceptionError', outcome_node),
    ('HF_Violation', outcome_node),
])


# Уникальные ребра и создание модели
unique_edges = list(set(edges))
# Узлы, которые должны быть в модели (все из конфига, что есть в данных)
model_nodes_list = [node for node in all_nodes_in_config if node in df_prepared.columns]

model = DiscreteBayesianNetwork() # Создаем пустую модель
model.add_nodes_from(model_nodes_list) # Добавляем все нужные узлы
model.add_edges_from(unique_edges) # Добавляем связи

# Проверка на ацикличность и создание модели
try:
    if not nx.is_directed_acyclic_graph(model):
         raise ValueError("Структура сети содержит циклы!")
    print("Структура сети (реалистичная) определена и является ацикличной.")
    print(f"  Количество узлов: {len(model.nodes())}")
    print(f"  Количество ребер: {len(model.edges())}")
except ImportError:
     print("Не удалось проверить на циклы, т.к. NetworkX не установлен. Продолжаем...")
except Exception as e:
    print(f"\nОшибка при создании модели или проверке на циклы: {e}")
    exit()

# --- Проверка кардинальностей перед обучением ---
print("\n--- Проверка данных перед обучением ---")
data_to_fit = df_prepared[model_nodes_list].copy()
issues_found_cardinality = False
for node in model.nodes():
    if node not in state_names:
        print(f"ОШИБКА: Узел '{node}' есть в модели, но отсутствует в state_names.")
        issues_found_cardinality = True
        continue
    if node not in data_to_fit.columns:
         print(f"КРИТИЧЕСКАЯ ОШИБКА: Узел '{node}' из модели отсутствует в данных 'data_to_fit'.")
         issues_found_cardinality = True
         continue
    model_cardinality = len(state_names[node])
    expected_max_state = model_cardinality - 1
    try:
        actual_max_state = data_to_fit[node].max()
        actual_min_state = data_to_fit[node].min()
    except TypeError:
         print(f"  ОШИБКА: Не удалось получить мин/макс для столбца '{node}'.")
         issues_found_cardinality = True
         continue
    print(f"Узел '{node}': Кардинальность={model_cardinality}. Мин/Макс в данных={actual_min_state}/{actual_max_state}")
    if actual_max_state > expected_max_state:
        print(f"  ОШИБКА: Макс. значение ({actual_max_state}) > ожидаемого ({expected_max_state}).")
        issues_found_cardinality = True
    if actual_min_state < 0:
         print(f"  ОШИБКА: Отрицательные значения ({actual_min_state}) в '{node}'.")
         issues_found_cardinality = True
    if data_to_fit[node].nunique(dropna=False) <= 1 :
         print(f"  ПРЕДУПРЕЖДЕНИЕ: Столбец '{node}' содержит только одно уникальное значение или пуст.")

if issues_found_cardinality:
    print("\nОбнаружены несоответствия кардинальностей/значений. Исправьте данные или JSON.")
    exit()
else:
    print("\nПроверка соответствия данных и кардинальностей перед обучением успешно завершена.")

# --- Финальная проверка на отрицательные значения ---
print("\n--- Финальная проверка данных на отрицательные значения перед model.fit ---")
negative_found_final = False
for col in data_to_fit.columns:
    min_val = data_to_fit[col].min()
    if min_val < 0:
        print(f"  КРИТИЧЕСКАЯ ОШИБКА: Отрицательное значение ({min_val}) в '{col}'!")
        negative_found_final = True
if negative_found_final:
    print("Исправьте отрицательные значения.")
    exit()
else:
    print("Отрицательные значения не обнаружены.")

# --- 4. Обучение параметров (CPT) ---
try:
    print("\nНачало обучения параметров...")
    model.fit(data=data_to_fit, estimator=BayesianEstimator, state_names=state_names, equivalent_sample_size=10)
    print("Параметры сети (CPT) успешно обучены.")
except ValueError as e:
     print(f"\nОШИБКА ValueError во время model.fit: {e}")
     print("Проверьте соответствие данных и state_names, отсутствие отрицательных значений.")
     print("\nДиагностика:")
     for node in model.nodes():
         if node in data_to_fit.columns:
             unique_in_data = data_to_fit[node].unique()
             print(f"  Узел '{node}': Уникальные={unique_in_data}. Разрешенные={state_names.get(node)}")
             if node in state_names:
                 missing_in_states = [v for v in unique_in_data if v not in state_names[node]]
                 if missing_in_states:
                     print(f"    ОШИБКА! Значения {missing_in_states} есть в данных, но не в state_names[{node}]")
         else:
             print(f"  Узел '{node}': Отсутствует в data_to_fit.")
     exit()
except Exception as e:
    print(f"\nНеожиданная ОШИБКА во время model.fit: {e}")
    exit()


# --- 5. Анализ Вероятностей ---
print("\n--- Анализ Вероятностей ---")
total_incidents = len(data_to_fit)
print(f"\nАприорные вероятности Исхода ({outcome_node})")
outcome_states = config['outcome_states']
outcome_counts = data_to_fit[outcome_node].value_counts()
outcome_probs = data_to_fit[outcome_node].value_counts(normalize=True)
for state_info in outcome_states:
    code = state_info['code']
    label = state_info.get('label', f'Состояние {code}')
    prob = outcome_probs.get(code, 0)
    count = outcome_counts.get(code, 0)
    print(f"  P(Исход={code} - {label}) = {prob:.4f} ({count} из {total_incidents})")

# --- Визуализация априорных вероятностей Исхода ---
if 'outcome_probs' in locals() and not outcome_probs.empty:
    plt.figure(figsize=(8, 6))
    outcome_labels_for_plot = [s.get('label', f'Состояние {s["code"]}') for s in config['outcome_states']]
    # Убедимся, что порядок меток соответствует данным в outcome_probs.index
    # Это важно, если не все состояния есть в данных
    ordered_labels = []
    ordered_probs = []
    for code in outcome_probs.index: # Используем порядок из value_counts
        label = next((s.get('label', f'Состояние {s["code"]}') for s in config['outcome_states'] if s["code"] == code), f'Состояние {code}')
        ordered_labels.append(label)
        ordered_probs.append(outcome_probs[code])

    if ordered_probs: # Если есть данные для отображения
        plt.pie(ordered_probs, labels=ordered_labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'darkred'])
        plt.title(f'Априорное распределение Исхода ({outcome_node})')
        plt.axis('equal') # Делает круг круглым
        plt.savefig('apriori_outcome_distribution.png', dpi=100)
        print("График априорного распределения исхода сохранен в 'apriori_outcome_distribution.png'")
        # plt.show() # Раскомментировать для немедленного показа
    else:
        print("Недостаточно данных для построения графика априорного распределения исхода.")
else:
     print("Данные для 'outcome_probs' не были вычислены, график не будет построен.")
print(f"\nАприорные вероятности некоторых Факторов (P(Фактор>0))")
factors_to_show = ['HF_DecisionError', 'HF_Violation', 'Env_Visibility', 'Env_Ceiling',
                   'Tech_SystemFailure', 'Env_Icing', 'Env_RunwayCondition', 'HF_Ground_Deicing']
for factor_id in factors_to_show:
    if factor_id in data_to_fit.columns:
      try:
          factor_config = next((f for f in config['factors'] if f['id'] == factor_id), None)
          if factor_config is None or 'states' not in factor_config:
              print(f"  ПРЕДУПРЕЖДЕНИЕ: Конфигурация для '{factor_id}' не найдена. Пропускаем.")
              continue
          problem_codes = [s['code'] for s in factor_config['states'] if s['code'] > 0]
          if not problem_codes: continue
          total_prob = data_to_fit[factor_id].isin(problem_codes).mean() # Более простой способ посчитать долю
          total_count = data_to_fit[factor_id].isin(problem_codes).sum()
          print(f"  P({factor_id}>0) = {total_prob:.4f} ({total_count} из {total_incidents})")
      except Exception as e:
          print(f"  Ошибка при обработке {factor_id}: {e}")
    else:
        print(f"  Фактор '{factor_id}' отсутствует.")

print(f"\nУсловные вероятности Исхода при наличии Фактора риска (P(Исход | Фактор>0))")
for factor_id in factors_to_show:
    if factor_id in data_to_fit.columns:
      try:
          factor_config = next((f for f in config['factors'] if f['id'] == factor_id), None)
          if factor_config is None or 'states' not in factor_config:
               print(f"\nФактор: {factor_id} - Конфигурация не найдена. Пропускаем.")
               continue
          problem_codes = [s['code'] for s in factor_config['states'] if s['code'] > 0]
          if not problem_codes: continue
          print(f"\nФактор: {factor_id}")
          df_problem = data_to_fit[data_to_fit[factor_id].isin(problem_codes)]
          n_problem = len(df_problem)
          if n_problem > 0:
              print(f"  (Всего {n_problem} случаев с {factor_id} > 0)")
              outcome_counts_problem = df_problem[outcome_node].value_counts()
              outcome_probs_problem = df_problem[outcome_node].value_counts(normalize=True)
              for state_info in outcome_states:
                   code = state_info['code']
                   label = state_info.get('label', f'Состояние {code}')
                   prob = outcome_probs_problem.get(code, 0)
                   count = outcome_counts_problem.get(code, 0)
                   print(f"    P(Исход={code} - {label} | {factor_id} > 0) = {prob:.4f} ({count} из {n_problem})")
          else:
              print(f"  Нет случаев с {factor_id} > 0 для расчета.")
      except Exception as e:
           print(f"\nОшибка при расчете для {factor_id}: {e}")
    else:
         print(f"\nФактор: {factor_id} - Отсутствует.")

# --- Визуализация априорных вероятностей Факторов ---
apriori_factor_probs_data = {}
for factor_id in factors_to_show:
    if factor_id in data_to_fit.columns:
        factor_config = next((f for f in config['factors'] if f['id'] == factor_id), None)
        if factor_config and 'states' in factor_config:
            problem_codes = [s['code'] for s in factor_config['states'] if s['code'] > 0]
            if problem_codes:
                prob_gt_0 = data_to_fit[factor_id].isin(problem_codes).mean()
                apriori_factor_probs_data[factor_id] = prob_gt_0
if apriori_factor_probs_data:
    factor_names = list(apriori_factor_probs_data.keys())
    factor_values = list(apriori_factor_probs_data.values())
    plt.figure(figsize=(12, 7))
    plt.bar(factor_names, factor_values, color='skyblue')
    plt.xlabel("Факторы риска")
    plt.ylabel("P(Фактор > 0)")
    plt.title("Априорные вероятности 'проблемных' состояний факторов")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('apriori_factors_distribution.png', dpi=100)
    print("График априорного распределения факторов сохранен в 'apriori_factors_distribution.png'")
    # plt.show()
else:
    print("Нет данных для построения графика априорного распределения факторов.")
# --- Визуализация условных вероятностей Исхода для нескольких факторов ---
factors_for_conditional_plot = ['Tech_SystemFailure', 'Env_RunwayCondition', 'HF_Ground_Deicing', 'HF_DecisionError']
plot_data_conditional = {}
outcome_labels_for_plot = [s.get('label', f'Состояние {s["code"]}') for s in config['outcome_states']]

for factor_id in factors_for_conditional_plot:
    if factor_id in data_to_fit.columns:
        factor_config = next((f for f in config['factors'] if f['id'] == factor_id), None)
        if factor_config and 'states' in factor_config:
            problem_codes = [s['code'] for s in factor_config['states'] if s['code'] > 0]
            if not problem_codes: continue

            df_problem = data_to_fit[data_to_fit[factor_id].isin(problem_codes)]
            n_problem = len(df_problem)
            if n_problem > 0:
                outcome_probs_problem = df_problem[outcome_node].value_counts(normalize=True)
                factor_probs_list = []
                for state_info in config['outcome_states']: # Используем config для правильного порядка
                    code = state_info['code']
                    factor_probs_list.append(outcome_probs_problem.get(code, 0))
                plot_data_conditional[factor_id] = factor_probs_list

if plot_data_conditional:
    df_plot_conditional = pd.DataFrame(plot_data_conditional, index=outcome_labels_for_plot)
    df_plot_conditional.T.plot(kind='bar', figsize=(14, 8)) # Транспонируем для удобства
    plt.title("Условные вероятности Исхода P(Исход | Фактор>0)")
    plt.ylabel("Вероятность")
    plt.xlabel("Факторы риска (при условии 'проблемного' состояния)")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Исход")
    plt.tight_layout()
    plt.savefig('conditional_outcome_probabilities.png', dpi=100)
    print("График условных вероятностей исхода сохранен в 'conditional_outcome_probabilities.png'")
    # plt.show()
else:
    print("Нет данных для построения графика условных вероятностей исхода.")
# --- 6. Логический Вывод (Inference) ---
if model.cpds: # Проверяем, что модель обучена
    print("\n--- Логический Вывод (Примеры Запросов) ---")
    try:
        inference = VariableElimination(model)

        # Пример 1: Вероятность катастрофы (код 2) при плохой видимости и ошибке пилота
        evidence_case1 = {'Env_Visibility': 1, 'HF_SkillError': 1}
        # Проверим, существуют ли узлы и допустимы ли состояния
        valid_evidence1 = True
        for node, state in evidence_case1.items():
            if node not in model.nodes():
                print(f"ПРЕДУПРЕЖДЕНИЕ: Узел '{node}' для запроса 1 отсутствует в модели.")
                valid_evidence1 = False
                break
            if state not in state_names.get(node,[]):
                print(f"ПРЕДУПРЕЖДЕНИЕ: Состояние '{state}' для узла '{node}' недопустимо (разрешенные: {state_names.get(node,[])}).")
                valid_evidence1 = False
                break
        if valid_evidence1:
            try:
                query1 = inference.query(variables=[outcome_node], evidence=evidence_case1)
                print(f"\nВероятность исхода при Env_Visibility=1 и HF_SkillError=1:")
                print(query1)
                prob_catastrophe = query1.get_value(**{outcome_node: 2}) # Получаем вероятность Исхода=2
                print(f"  В частности, P(Катастрофа | Env_Visibility=1, HF_SkillError=1) = {prob_catastrophe:.4f}")
            except Exception as e_q1:
                 print(f"Ошибка при вычислении запроса 1: {e_q1}")
        else:
            print("Запрос 1 не может быть выполнен из-за некорректных узлов/состояний.")


        # Пример 2: Вероятность отказа техники при плохом ТО
        evidence_case2 = {'HF_Maint_Quality': 1}
        variable_case2 = 'Tech_SystemFailure'
        if variable_case2 in model.nodes() and 'HF_Maint_Quality' in model.nodes() and 1 in state_names.get('HF_Maint_Quality',[]):
            try:
                query2 = inference.query(variables=[variable_case2], evidence=evidence_case2)
                print(f"\nВероятность '{variable_case2}' при HF_Maint_Quality=1:")
                print(query2)
            except Exception as e_q2:
                print(f"Ошибка при вычислении запроса 2: {e_q2}")
        else:
             print(f"\nУзлы для запроса 2 ('{variable_case2}', 'HF_Maint_Quality') отсутствуют или состояние 1 недопустимо.")

        # Пример 3: Сравнение вероятности аварии (Исход=1) при разном CRM
        node_crm = 'HF_CRM'
        outcome_severity_code_for_accident = 1 # Код для Аварии
        if node_crm in model.nodes() and outcome_node in model.nodes() and \
           0 in state_names.get(node_crm,[]) and 2 in state_names.get(node_crm,[]) and \
           outcome_severity_code_for_accident in state_names.get(outcome_node,[]):
           try:
               prob_a_crm2 = inference.query(variables=[outcome_node], evidence={node_crm: 2}).get_value(**{outcome_node: outcome_severity_code_for_accident})
               prob_a_crm0 = inference.query(variables=[outcome_node], evidence={node_crm: 0}).get_value(**{outcome_node: outcome_severity_code_for_accident})
               print(f"\nСравнение вероятности Аварии (Исход={outcome_severity_code_for_accident}) при разном CRM:")
               print(f"  P(Авария | {node_crm}=2 (Низкое)) = {prob_a_crm2:.4f}")
               print(f"  P(Авария | {node_crm}=0 (Высокое)) = {prob_a_crm0:.4f}")
           except Exception as e_q3:
                print(f"Ошибка при вычислении запроса 3 (сравнение CRM): {e_q3}")
        else:
            print("\nУзлы или состояния для запроса 3 (сравнение CRM) некорректны/отсутствуют.")

    except Exception as e:
        print(f"\nОшибка при инициализации или выполнении логического вывода: {e}")

else:
    print("\nМодель не была успешно обучена, логический вывод не выполнен.")

try:
    with open(MODEL_FILE, 'wb') as f: # 'wb' - запись в бинарном режиме
        pickle.dump(model, f)
    print(f"Обученная модель сохранена в файл: {MODEL_FILE}")
except Exception as e:
    print(f"Ошибка при сохранении модели: {e}")

# --- 7. Визуализация ---
print("\n--- Попытка визуализации сети ---")
try:
    # Проверяем, что граф ацикличен перед попыткой layout
    if not nx.is_directed_acyclic_graph(model):
         print("Граф содержит циклы, невозможно использовать graphviz_layout.")
    else:
        graph = nx.DiGraph(model.edges()) # Создаем граф NetworkX из ребер модели
        # Добавляем узлы, которые могли остаться без связей
        graph.add_nodes_from(model.nodes())

        plt.figure(figsize=(24, 18)) # Увеличим размер для читаемости
        # Используем layout, который пытается показать иерархию
        pos = nx.drawing.nx_pydot.graphviz_layout(graph, prog='dot')

        nx.draw(graph, pos, with_labels=True, node_size=2500, node_color='lightblue',
                font_size=7, font_weight='bold', arrowsize=15, edge_color='gray')
        plt.title("Структура Реалистичной Байесовской Сети", fontsize=16)
        filename = "realistic_bn_structure.png"
        plt.savefig(filename, dpi=150) # Сохраняем с хорошим разрешением
        print(f"Граф сети сохранен в файл: {filename}")
        # plt.show() # Раскомментируйте, если хотите показать граф сразу

except ImportError:
    print("ПРЕДУПРЕЖДЕНИЕ: Для визуализации графа не установлены networkx и pygraphviz/pydot.")
    print("Попробуйте установить: pip install networkx pygraphviz (или python-graphviz и pydot)")
except Exception as e:
    print(f"Ошибка при визуализации графа: {e}")
    print("Возможно, Graphviz не установлен или не найден в PATH.")


print("\n--- Выполнение скрипта завершено ---")