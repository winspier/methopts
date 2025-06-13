## Лабораторная работа. Методы Оптимизаций

## Требования

- **C++20**, **CMake**, **GoogleTest**, **Eigen3**, **Python 3.8+**, **nlohmann\_json**, **matplotlib**, **numpy**

### Сборка проекта

```bash
mkdir build && cd build
cmake ..
cmake --build . --parallel
```

### Запуск тестов

В папке сборки:

```bash
ctest --output-on-failure
```

---

### task1\_sgd\_constrained
   
Запустите скрипт `train_and_plot.py`, чтобы получить графики:
   ```bash
   cmake --build . --target train_and_plot_regression
   ```

### task2\_newton

Выполните демонстрацию и визуализацию Лагранжиана:

```bash
cmake --build . --target demo_newton
```

### task3\_advanced\_sgd

1. Соберите и запустите серию экспериментов (GD, Momentum, Adam):
   ```bash
   cmake --build . --target run_experiments
   ```
2. Постройте объединённый и раздельные графики:
   ```bash
   cmake --build . --target plot_convergence
   ```

Результаты: `output/convergence.csv`, `output/convergence.png`, а также `output/convergence_<Method>.png`.

### task4\_lbfgs

1. Запуск обучения с историей:
   ```bash
   cmake --build . --target run_lbfgs_with_history
   ```
2. Построение графиков:
   ```bash
   cmake --build . --target plot_lbfgs
   ```

Графики в `output/lbfgs_history.png` и `output/lbfgs_convergence.png`.

### task5\_branch\_and\_cut\_tsp

```bash
cmake --build . --target plot_tsp
```

- `output/solution.json` — оптимальный маршрут
- `output/tsp_plot.png` — визуализация тура

