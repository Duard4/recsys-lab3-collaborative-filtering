# ЛР3 – Колаборативна фільтрація (елективні курси)

Вхід: датасет і кластери курсів із ЛР1 (`data/processed`, `data/lab1_results`), рекомендації та пули ЛР2 (`lab2_results/`).

Порядок запуску (Python 3.11; pandas, numpy, scipy, scikit-learn, matplotlib):

1. `python 01_tune.py` – відкладена вибірка, добір параметрів (λ, β, γ, k, тип схожості, поріг κ) → `results/tuning_*.csv`, `results/config.json`
2. `python 02_recommend.py` – підвибірки з 100 курсів і РЕК. 1–5 для активних та пасивних → `results/recommendations_lab3.csv`, `candidates_100.csv`
3. `python 03_compare.py` – чотири показники порівняння та допоміжні (`results/compare_*.csv`)
4. `python 04_figures.py` – рисунки (`figures/l3_*.png`)
5. `python 05_export_report_data.py` – `results/report_data.json`
6. `cd report && NODE_PATH=$(npm root -g) node build_report.js` – звіт `Lab3_Collaborative_Filtering_Report.docx` (потрібен npm-пакет `docx`)

`cf_lib.py` – підвибірка, Пірсон (повні вектори / ко-рейтингові позиції), IUF, прогноз, item-based.
