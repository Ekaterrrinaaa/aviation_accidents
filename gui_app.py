import customtkinter as ctk
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import json
import predict_utils # Наш вспомогательный модуль
from PIL import Image, ImageTk
import os

# --- Глобальные переменные ---
config_data = None
trained_model = None
current_evidence = {}
survey_history = []

prediction_results_text = ""
influence_results_text = ""
influence_results_data_for_plot = [] # Измененное имя для ясности
predicted_probabilities_dict = {}

# --- Настройки CustomTkinter ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# --- Функции GUI ---

def load_dependencies():
    global config_data, trained_model
    config_data = predict_utils.load_config()
    trained_model = predict_utils.load_model()
    if not config_data:
        messagebox.showerror("Ошибка Загрузки", "Не удалось загрузить bn_config.json. Программа будет закрыта.")
        root.quit()
    if not trained_model:
        messagebox.showwarning("Внимание", f"Файл обученной модели '{predict_utils.MODEL_FILE}' не найден.\n"
                                           "Некоторые функции будут недоступны до обучения и сохранения модели (например, через основной скрипт BSD.py).")
    update_status()

def update_status():
    config_status = "Загружена" if config_data else "Ошибка"
    model_status = "Загружена" if trained_model and hasattr(trained_model, 'cpds') and trained_model.cpds else "Не загружена / Не обучена"
    status_label.configure(text=f"Конфигурация: {config_status}\nМодель: {model_status}")

def show_next_factor_in_survey(survey_window, factors, factor_index_var, factor_vars_dict):
    global survey_history

    current_factor_idx = factor_index_var.get()

    for widget in survey_window.winfo_children():
        widget.destroy()

    if current_factor_idx >= len(factors):
        for f_id, data in factor_vars_dict.items():
            try:
                selected_code = int(data['var'].get())
                if selected_code != -999: # -999 это "пропустить"
                    current_evidence[f_id] = selected_code
                elif f_id in current_evidence: # Если ранее был ответ, а теперь пропуск
                    del current_evidence[f_id]
            except ValueError: # Если значение не int (не должно быть, но на всякий случай)
                if f_id in current_evidence: del current_evidence[f_id]
        survey_window.destroy()
        messagebox.showinfo("Опрос завершен", "Свидетельства собраны и сохранены в 'incident_evidence.json'.")
        save_evidence_to_file(current_evidence)
        display_collected_evidence()
        return

    if current_factor_idx < 0:
        factor_index_var.set(0)
        show_next_factor_in_survey(survey_window, factors, factor_index_var, factor_vars_dict)
        return

    # Получаем индекс фактора из ИСХОДНОГО списка factors, используя историю
    actual_config_index = survey_history[current_factor_idx][2] if current_factor_idx < len(survey_history) else current_factor_idx


    factor = factors[actual_config_index]
    factor_id = factor['id']
    factor_name_ru = predict_utils.get_factor_name_by_id(factor_id, config_data) # Русское имя
    states = factor.get('states', [])

    header_frame = ctk.CTkFrame(survey_window, fg_color="transparent")
    header_frame.pack(pady=(15, 5), fill="x", padx=20)
    ctk.CTkLabel(header_frame, text=f"Фактор {current_factor_idx + 1} из {len(factors)}", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

    ctk.CTkLabel(survey_window, text=f"{factor_name_ru}", font=ctk.CTkFont(size=15), wraplength=550, justify="left").pack(pady=(5, 10), anchor="w", padx=20)
    ctk.CTkLabel(survey_window, text="Возможные состояния:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=20)

    if factor_id not in factor_vars_dict:
        factor_vars_dict[factor_id] = {'var': tk.StringVar(value="-999"), 'default_set': False}
    current_var_obj = factor_vars_dict[factor_id]['var']

    if not factor_vars_dict[factor_id]['default_set']:
        if factor_id in current_evidence:
            current_var_obj.set(str(current_evidence[factor_id]))
        else:
            current_var_obj.set("-999") # "Пропустить" по умолчанию
        factor_vars_dict[factor_id]['default_set'] = True

    radio_buttons_list = []
    for i, state in enumerate(states):
        label = state.get('label', f"Состояние {state['code']}")
        code = state['code']
        rb = ctk.CTkRadioButton(survey_window, text=f"{code}: {label}", variable=current_var_obj, value=str(code), font=ctk.CTkFont(size=13))
        rb.pack(anchor=tk.W, padx=40, pady=(3,1))
        radio_buttons_list.append(rb)
        # Привязка цифровых клавиш к Radiobutton
        if i < 9 : # 1-9
            survey_window.bind(str(i+1), lambda e, v=str(code): current_var_obj.set(v))
        elif i == 9: # 0 для 10-го состояния (если есть)
            survey_window.bind("0", lambda e, v=str(code): current_var_obj.set(v))


    rb_skip = ctk.CTkRadioButton(survey_window, text="Пропустить этот фактор (Нажмите Enter)", variable=current_var_obj, value="-999", font=ctk.CTkFont(size=13))
    rb_skip.pack(anchor=tk.W, padx=40, pady=(8,5))
    radio_buttons_list.append(rb_skip)

    nav_frame = ctk.CTkFrame(survey_window, fg_color="transparent")
    nav_frame.pack(pady=25, fill="x", side="bottom", padx=20)

    def go_back_action():
        global survey_history # Указываем, что работаем с глобальной survey_history

        if current_factor_idx > 0 : # Только если это не первый вопрос
            # Не сохраняем текущий ответ при возврате, значение будет восстановлено
            factor_index_var.set(current_factor_idx - 1)
            # Для предыдущего фактора сбросить default_set, чтобы его значение
            # корректно подгрузилось из current_evidence при перерисовке
            prev_hist_entry = survey_history[current_factor_idx - 1]
            prev_factor_id_hist = prev_hist_entry[0]
            if prev_factor_id_hist in factor_vars_dict:
                factor_vars_dict[prev_factor_id_hist]['default_set'] = False
            show_next_factor_in_survey(survey_window, factors, factor_index_var, factor_vars_dict)

    def go_next_action():
        global survey_history # Указываем, что работаем с глобальной survey_history
        selected_code_str = current_var_obj.get()
        try:
            selected_code = int(selected_code_str)
            if selected_code != -999:
                current_evidence[factor_id] = selected_code
            elif factor_id in current_evidence:
                del current_evidence[factor_id]
        except ValueError:
            if factor_id in current_evidence: del current_evidence[factor_id]

        if current_factor_idx == len(survey_history):
            survey_history.append((factor_id, selected_code_str, actual_config_index))
        elif current_factor_idx < len(survey_history):
            survey_history[current_factor_idx] = (factor_id, selected_code_str, actual_config_index)
            survey_history = survey_history[:current_factor_idx+1] # Обрезаем, если пошли по другому пути

        factor_vars_dict[factor_id]['default_set'] = True # Помечаем, что значение установлено (или пропущено)
        factor_index_var.set(current_factor_idx + 1)
        show_next_factor_in_survey(survey_window, factors, factor_index_var, factor_vars_dict)

    back_btn = ctk.CTkButton(nav_frame, text="<< Назад", command=go_back_action, width=120, height=35, font=ctk.CTkFont(size=13))
    back_btn.pack(side="left")
    if current_factor_idx == 0:
        back_btn.configure(state="disabled")

    next_btn = ctk.CTkButton(nav_frame, text="Далее >>", command=go_next_action, width=120, height=35, font=ctk.CTkFont(size=13))
    next_btn.pack(side="right")

    def key_press_handler_survey(event):
        key = event.keysym
        current_factor_state_codes = [s['code'] for s in states]
        if key.isdigit() and int(key) in current_factor_state_codes:
            current_var_obj.set(key)
        elif key == "Return":
            current_var_obj.set("-999")
            go_next_action()
        elif key == "Right":
            go_next_action()
        elif key == "Left" and current_factor_idx > 0:
            go_back_action()

    survey_window.bind("<KeyPress>", key_press_handler_survey)
    if radio_buttons_list:
        radio_buttons_list[0].focus_set()

def start_survey_wrapper():
    global current_evidence, survey_history
    current_evidence = {}
    survey_history = []
    output_text_area.configure(state="normal")
    output_text_area.delete('1.0', tk.END)
    output_text_area.configure(state="disabled")

    influence_text_area.configure(state="normal")
    influence_text_area.delete('1.0', tk.END)
    influence_text_area.configure(state="disabled")


    if not config_data or 'factors' not in config_data:
        messagebox.showerror("Ошибка", "Конфигурация факторов не загружена.")
        return

    factors = config_data['factors']
    survey_window = ctk.CTkToplevel(root)
    survey_window.title("Опрос по Факторам АП")
    survey_window.geometry("650x600") # Увеличил для читаемости
    survey_window.attributes("-topmost", True)
    survey_window.grab_set() # Делаем модальным

    factor_index_var = tk.IntVar(value=0)
    factor_vars_dict = {}

    show_next_factor_in_survey(survey_window, factors, factor_index_var, factor_vars_dict)

def save_evidence_to_file(evidence_dict, filename="incident_evidence.json"):
    data_to_save = {"evidence": evidence_dict}
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        print(f"Свидетельства сохранены в файл: {filename}")
    except Exception as e:
        print(f"Ошибка сохранения файла свидетельств: {e}")
        # messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить свидетельства: {e}") # Можно раскомментировать

def display_collected_evidence():
    output_text_area.configure(state="normal")
    output_text_area.delete('1.0', tk.END)
    if not current_evidence:
        output_text_area.insert(tk.END, "Свидетельства не введены или были очищены.")
        output_text_area.configure(state="disabled")
        return

    output_text_area.insert(tk.END, "--- Текущие Заданные Свидетельства ---\n")
    # Сортировка: сначала факторы с кодом > 0 (проблема), затем с кодом 0 (норма)
    sorted_evidence_items = sorted(current_evidence.items(), key=lambda item: (item[1] == 0, predict_utils.get_factor_name_by_id(item[0], config_data)))

    for factor_id, state_code in sorted_evidence_items:
        factor_name_ru = predict_utils.get_factor_name_by_id(factor_id, config_data)
        state_label_ru = str(state_code) # По умолчанию, если метка не найдена
        for factor_cfg in config_data.get('factors', []):
            if factor_cfg.get('id') == factor_id:
                for state_cfg in factor_cfg.get('states', []):
                    if state_cfg.get('code') == state_code:
                        state_label_ru = state_cfg.get('label', str(state_code))
                        break
                break
        output_text_area.insert(tk.END, f"  - {factor_name_ru}: {state_code} ({state_label_ru})\n")
    output_text_area.configure(state="disabled")

def calculate_probability():
    global prediction_results_text, influence_results_text, influence_results_data_for_plot, predicted_probabilities_dict, current_evidence

    if not trained_model or not hasattr(trained_model, 'cpds') or not trained_model.cpds:
        messagebox.showerror("Ошибка Модели", "Модель не загружена или не обучена. Запустите основной скрипт BSD.py для обучения и сохранения модели.")
        return

    if not current_evidence:
         try:
             with open("incident_evidence.json", 'r', encoding='utf-8') as f:
                 loaded_data = json.load(f)
                 current_evidence = loaded_data.get("evidence", {})
             if not current_evidence:
                 messagebox.showwarning("Нет Свидетельств", "Свидетельства не введены. Пожалуйста, пройдите опрос или убедитесь, что файл 'incident_evidence.json' содержит данные.")
                 return
             messagebox.showinfo("Свидетельства Загружены", "Свидетельства для расчета загружены из 'incident_evidence.json'.")
             display_collected_evidence()
         except FileNotFoundError:
             messagebox.showwarning("Нет Свидетельств", "Файл 'incident_evidence.json' не найден. Пожалуйста, пройдите опрос.")
             return
         except json.JSONDecodeError:
             messagebox.showerror("Ошибка Файла", "Файл 'incident_evidence.json' содержит ошибку. Проверьте его или пройдите опрос.")
             return

    output_text_area.configure(state="normal")
    output_text_area.delete('1.0', tk.END)
    influence_text_area.configure(state="normal")
    influence_text_area.delete('1.0', tk.END)

    prediction_results_text = ""
    influence_results_text = ""
    influence_results_data_for_plot = []
    predicted_probabilities_dict = {}

    pred_text, pred_dict = predict_utils.predict_outcome(trained_model, current_evidence, config_data)
    prediction_results_text = pred_text
    predicted_probabilities_dict = pred_dict
    output_text_area.insert(tk.END, prediction_results_text)

    infl_text, infl_data = predict_utils.assess_factor_influence(trained_model, current_evidence, config_data)
    influence_results_text = infl_text
    influence_results_data_for_plot = infl_data # infl_data уже содержит русские имена для графика
    influence_text_area.insert(tk.END, influence_results_text)

    output_text_area.configure(state="disabled")
    influence_text_area.configure(state="disabled")

    show_outcome_plot_button.configure(state=tk.NORMAL if predicted_probabilities_dict else tk.DISABLED)
    show_influence_plot_button.configure(state=tk.NORMAL if influence_results_data_for_plot else tk.DISABLED)
    save_results_button.configure(state=tk.NORMAL)

def show_plot_window(plot_function, data, title, filename):
    # ... (код show_plot_window остается как в предыдущем ответе) ...
    if not data:
        messagebox.showinfo("Нет данных", f"Нет данных для построения графика '{title}'.")
        return
    temp_filename = f"temp_plot_{filename}"
    plot_function(data, filename=temp_filename) # plot_function теперь из predict_utils
    if not os.path.exists(temp_filename):
         messagebox.showerror("Ошибка", f"Не удалось создать файл графика: {temp_filename}")
         return
    plot_window = ctk.CTkToplevel(root)
    plot_window.title(title)
    plot_window.geometry("850x650") # Немного больше для графиков
    plot_window.attributes("-topmost", True)
    plot_window.grab_set()
    try:
        img = Image.open(temp_filename)
        photo = ImageTk.PhotoImage(img)
        # Используем tk.Label, так как CTkLabel может иметь проблемы с PhotoImage в некоторых случаях
        img_label = tk.Label(plot_window, image=photo, bd=0)
        img_label.image = photo
        img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    except Exception as e:
        ctk.CTkLabel(plot_window, text=f"Ошибка отображения графика: {e}").pack(padx=10, pady=10)
    finally:
        # Попытка удалить временный файл
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError as e:
                print(f"Не удалось удалить временный файл {temp_filename}: {e}")


def save_results():
    # ... (код save_results остается как в предыдущем ответе) ...
    output_text_content = output_text_area.get("1.0", tk.END)
    influence_text_content = influence_text_area.get("1.0", tk.END)
    if not output_text_content.strip() and not influence_text_content.strip():
        messagebox.showwarning("Нет результатов", "Сначала рассчитайте вероятности.")
        return
    filepath = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        title="Сохранить результаты анализа",
        initialfile="Результаты_Анализа_АП.txt"
    )
    if not filepath: return
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output_text_content)
            f.write("\n\n")
            f.write(influence_text_content)
        messagebox.showinfo("Сохранено", f"Результаты сохранены в файл:\n{filepath}")
    except Exception as e:
        messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл: {e}")

def clear_all():
     global current_evidence, survey_history, prediction_results_text, influence_results_text, influence_results_data_for_plot, predicted_probabilities_dict
     # ... (код clear_all остается как в предыдущем ответе) ...
     current_evidence = {}
     survey_history.clear()
     prediction_results_text = ""
     influence_results_text = ""
     influence_results_data_for_plot = []
     predicted_probabilities_dict = {}
     output_text_area.configure(state="normal")
     output_text_area.delete('1.0', tk.END)
     output_text_area.configure(state="disabled")
     influence_text_area.configure(state="normal")
     influence_text_area.delete('1.0', tk.END)
     influence_text_area.configure(state="disabled")
     show_outcome_plot_button.configure(state=tk.DISABLED)
     show_influence_plot_button.configure(state=tk.DISABLED)
     save_results_button.configure(state=tk.DISABLED)
     try:
        if os.path.exists("incident_evidence.json"):
            os.remove("incident_evidence.json")
            print("Файл incident_evidence.json удален.")
     except OSError as e:
        print(f"Ошибка удаления incident_evidence.json: {e}")
     messagebox.showinfo("Очищено", "Все введенные свидетельства, результаты и файл 'incident_evidence.json' очищены.")


# --- Создание основного окна ---
root = ctk.CTk()
root.title("Система Анализа Рисков АП на Основе Байесовской Сети")
root.geometry("1100x800")

# --- Левая панель (меню) ---
left_panel = ctk.CTkFrame(root, width=320, corner_radius=10)
left_panel.pack(side="left", fill="y", padx=(10,5), pady=10)
left_panel.pack_propagate(False)

ctk.CTkLabel(left_panel, text="Анализ Рисков АП", font=ctk.CTkFont(size=26, weight="bold")).pack(pady=(30, 15), padx=20)

info_text = ("\n\n"
             "Порядок работы:\n"
             "1. Нажмите 'Ввести Свидетельства' для пошагового ввода известных факторов риска.\n"
             "2. Нажмите 'Рассчитать Вероятности' для получения прогноза и анализа влияния факторов на основе введенных или загруженных свидетельств.\n"
             "3. Для очистки используйте соответствующую кнопку.")
info_label = ctk.CTkLabel(left_panel, text=info_text, wraplength=280, justify="left", font=ctk.CTkFont(size=12))
info_label.pack(pady=20, padx=20, fill="x", expand=True) # expand=True чтобы текст занял доступное место

ctk.CTkButton(left_panel, text="Ввести Свидетельства (Опрос)", command=start_survey_wrapper, height=45, font=ctk.CTkFont(size=14)).pack(pady=10, padx=30, fill="x")
ctk.CTkButton(left_panel, text="Рассчитать Вероятности", command=calculate_probability, height=45, font=ctk.CTkFont(size=14)).pack(pady=10, padx=30, fill="x")
ctk.CTkButton(left_panel, text="Очистить Все", command=clear_all, height=45, fg_color="#777777", hover_color="#555555", font=ctk.CTkFont(size=14)).pack(pady=10, padx=30, fill="x")

status_label = ctk.CTkLabel(left_panel, text="Статус: Загрузка...", font=ctk.CTkFont(size=11), wraplength=280, justify="left")
status_label.pack(pady=(20,15), padx=20, side="bottom", fill="x")


# --- Правая панель (для вывода результатов) ---
right_panel = ctk.CTkFrame(root, corner_radius=0, fg_color="transparent")
right_panel.pack(side="right", fill="both", expand=True, padx=(5,10), pady=10)

ctk.CTkLabel(right_panel, text="Результаты Анализа", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(5,15))

results_container = ctk.CTkFrame(right_panel, fg_color="transparent")
results_container.pack(fill=tk.BOTH, expand=True)

# Фрейм для предсказания исхода
output_frame = ctk.CTkFrame(results_container, corner_radius=6)
output_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5), side=tk.TOP)

ctk.CTkLabel(output_frame, text="Предсказание Исхода:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="nw", padx=10, pady=(10,5))
output_text_area = ctk.CTkTextbox(output_frame, wrap=tk.WORD, height=180, font=("Courier New", 15), corner_radius=6, border_width=1, state="disabled")
output_text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

# Фрейм для ранжирования факторов
influence_frame = ctk.CTkFrame(results_container, corner_radius=6)
influence_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0), side=tk.TOP)

ctk.CTkLabel(influence_frame, text="Ранжирование Факторов Влияния:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="nw", padx=10, pady=(10,5))
influence_text_area = ctk.CTkTextbox(influence_frame, wrap=tk.WORD, font=("Courier New", 15), corner_radius=6, border_width=1, state="disabled")
influence_text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

# Кнопки графиков и сохранения под результатами
bottom_buttons_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
bottom_buttons_frame.pack(fill=tk.X, pady=(15,0))

show_outcome_plot_button = ctk.CTkButton(bottom_buttons_frame, text="График Исхода", state=tk.DISABLED,
                                         command=lambda: show_plot_window(predict_utils.plot_outcome_distribution,
                                                                          predicted_probabilities_dict,
                                                                          "Распределение Исхода",
                                                                          "predicted_outcome_dist.png"))
show_outcome_plot_button.pack(side=tk.LEFT, padx=(0,5))

show_influence_plot_button = ctk.CTkButton(bottom_buttons_frame, text="График Влияния Факторов", state=tk.DISABLED,
                                          command=lambda: show_plot_window(predict_utils.plot_factor_influence,
                                                                           influence_results_data_for_plot,
                                                                           "Влияние Факторов (Топ 10)",
                                                                           "factor_influence.png"))
show_influence_plot_button.pack(side=tk.LEFT, padx=5)

save_results_button = ctk.CTkButton(bottom_buttons_frame, text="Сохранить Результаты (Текст)", state=tk.DISABLED, command=save_results)
save_results_button.pack(side=tk.LEFT, padx=5)

# --- Запуск приложения ---
if __name__ == "__main__":
    load_dependencies()
    root.mainloop()