import numpy as np
from mosek.fusion import Model, Domain, Expr, ObjectiveSense

def solve_dual(X, y, C, mu):
    """
    Решение двойственной задачи SFM по формуле (81)
    диссертации.

    Parameters
    ----------
    X : array-like, shape (N, n)
        Обучающие объекты, признаки расположены по столбцам.
    y : array-like, shape (N,)
        Метки классов {-1, +1}.
    C : float
        Параметр регуляризации, C > 0.
    mu : float
        Параметр селективности, mu >= 0.

    Returns
    -------
    dict
        Статус решения, lambda, xi и значение целевой функции.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X должна быть двумерной матрицей")

    N, n = X.shape

    if y.shape != (N,):
        raise ValueError("Длина y должна совпадать с числом строк X")

    if not np.all(np.isin(y, [-1.0, 1.0])):
        raise ValueError("Метки y должны принимать значения -1 и +1")

    if not np.isfinite(C) or C <= 0:
        raise ValueError("C должен быть конечным и положительным")

    if not np.isfinite(mu) or mu < 0:
        raise ValueError("mu должен быть конечным и неотрицательным")

    with Model("SFM_dual") as M:
        # 0 <= lambda_j <= C / 2
        lam = M.variable(
            "lambda",
            N,
            Domain.inRange(0.0, C / 2.0),
        )

        # xi_i >= 0
        xi = M.variable(
            "xi",
            n,
            Domain.greaterThan(0.0),
        )

        # Sum_j y_j * lambda_j = 0
        M.constraint(
            "equality",
            Expr.dot(y.tolist(), lam),
            Domain.equalsTo(0.0),
        )

        
        # xi_i >= z_i**2 - mu**2
        # Эквивалентно: xi_i + mu**2 >= z_i**2
        for i in range(n):
            coeff = y * X[:, i]
            z_i = Expr.dot(coeff.tolist(), lam)

            M.constraint(
                f"feature_{i}",
                Expr.vstack(
                    Expr.add(xi.index(i), float(mu**2)),
                    0.5,
                    z_i,
                ),
                Domain.inRotatedQCone(),
            )


        # Целевая функция по формуле (81):
        # maximize sum(lambda) + (!ПЛЮС!) 1/2 * sum(xi)
        # но тогда получается нет ограничения сверху и выгодно брать 
        # для максимума целевой функции большие значения xi 
        # и это не сходится с ручным выводом. Поэтому используется знак минус:
        # maximize sum(lambda) - 1/2 * sum(xi)
        objective = Expr.add(
            Expr.sum(lam),
            Expr.mul(-0.5, Expr.sum(xi)),
        )

        M.objective(
            "dual_objective",
            ObjectiveSense.Maximize,
            objective,
        )

        M.solve()

        status = str(M.getPrimalSolutionStatus())
        problem_status = str(M.getProblemStatus())

        result = {
            "status": status,
            "problem_status": problem_status,
            "lambda": None,
            "xi": None,
            "objective": None,
        }

        # Численные значения возвращаем только при
        # наличии допустимого первичного решения.
        try:
            result["lambda"] = np.asarray(
                lam.level(), dtype=float
            )
            result["xi"] = np.asarray(
                xi.level(), dtype=float
            )
            result["objective"] = float(
                M.primalObjValue()
            )
        except Exception:
            pass

        return result


def solve_final_lp(X, y, C, mu, lam, tol=1e-8):
    """
    Решение заключительной линейной задачи SFM
    по формуле (85) диссертации с учётом исправленных индексов
    и коэффициента при eta.

    Parameters
    ----------
    X : array-like, shape (N, n)
        Обучающие объекты, признаки расположены по столбцам.
    y : array-like, shape (N,)
        Метки классов {-1, +1}.
    C : float
        Параметр регуляризации, C > 0.
    mu : float
        Параметр селективности, mu >= 0.
    lam : array-like, shape (N,)
        Решение двойственной задачи (81).
    tol : float
        Допуск для определения множеств I+, I0, I-.

    Returns
    -------
    dict
        Статус решения, коэффициенты a, свободный член b,
        eta, slack-переменные delta и множества индексов.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X должна быть двумерной матрицей")

    N, n = X.shape

    if y.shape != (N,):
        raise ValueError("Длина y должна совпадать с числом строк X")

    if lam.shape != (N,):
        raise ValueError("Длина lam должна совпадать с числом строк X")

    if not np.all(np.isin(y, [-1.0, 1.0])):
        raise ValueError("Метки y должны принимать значения -1 и +1")

    if not np.isfinite(C) or C <= 0:
        raise ValueError("C должен быть конечным и положительным")

    if not np.isfinite(mu) or mu < 0:
        raise ValueError("mu должен быть конечным и неотрицательным")

    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol должен быть конечным и неотрицательным")

    # z_i = sum_j y_j * X[j, i] * lambda_j
    z = X.T.dot(y * lam)

    # Разбиение признаков с учётом численного допуска.
    diff = z**2 - mu**2

    I_plus = np.flatnonzero(diff > tol)
    I_zero = np.flatnonzero(np.abs(diff) <= tol)
    I_minus = np.flatnonzero(diff < -tol)

    # Проверяем, что каждый индекс попал ровно в одно множество
    assert np.unique(np.concatenate([I_plus, I_zero, I_minus])).size == n

    # Известный вклад признаков из I+:
    # sum_{i in I+} y_j * z_i * X[j, i]
    baseline = X[:, I_plus].dot(z[I_plus])

    # Коэффициенты при eta_i для i in I0:
    # y_j * z_i * X[j, i]
    eta_features = X[:, I_zero] * z[I_zero]

    n_zero = len(I_zero)

    with Model("SFM_final_LP") as M:
        # 0 <= eta_i <= 1, i in I0
        eta = M.variable(
            "eta",
            n_zero,
            Domain.inRange(0.0, 1.0),
        )

        # Свободный член b
        b = M.variable(
            "b",
            1,
            Domain.unbounded(),
        )

        # delta_j >= 0
        delta = M.variable(
            "delta",
            N,
            Domain.greaterThan(0.0),
        )

        # Ограничения:
        # sum_{i in I0} y_j * z_i * X[j, i] * eta_i + y_j * b + delta_j >= 
            # >= 1 - y_j * sum_{i in I+} z_i * X[j, i]
        for j in range(N):
            terms = [
                Expr.mul(float(y[j]), b.index(0)),
                delta.index(j),
            ]

            if n_zero > 0:
                coeff = y[j] * eta_features[j, :]
                terms.append(Expr.dot(coeff.tolist(), eta))

            lhs = Expr.add(terms)
            rhs = float(1.0 - y[j] * baseline[j])

            M.constraint(
                f"sample_{j}",
                lhs,
                Domain.greaterThan(rhs),
            )

        # Целевая функция:
        # 2 * mu**2 * sum(eta) + C * sum(delta) -> min
        objective = Expr.add(
            Expr.mul(2.0 * mu**2, Expr.sum(eta)),
            Expr.mul(C, Expr.sum(delta)),
        )

        M.objective(
            "final_objective",
            ObjectiveSense.Minimize,
            objective,
        )

        M.solve()

        status = str(M.getPrimalSolutionStatus())
        problem_status = str(M.getProblemStatus())

        result = {
            "status": status,
            "problem_status": problem_status,
            "a": None,
            "b": None,
            "eta": None,
            "delta": None,
            "I_plus": I_plus,
            "I_zero": I_zero,
            "I_minus": I_minus,
            "z": z,
            "objective": None,
        }

        try:
            eta_value = np.asarray(eta.level(), dtype=float)
            b_value = float(b.level()[0])
            delta_value = np.asarray(delta.level(), dtype=float)

            # Восстанавливаем вектор коэффициентов a.
            a = np.zeros(n)
            a[I_plus] = z[I_plus]
            a[I_zero] = eta_value * z[I_zero]
            # Для I_minus коэффициенты остаются нулевыми.

            result.update({
                "a": a,
                "b": b_value,
                "eta": eta_value,
                "delta": delta_value,
                "objective": float(M.primalObjValue()),
            })

        except Exception:
            pass

        return result


if __name__ == "__main__": 
    import preprocessing
    from sklearn.metrics import roc_auc_score, classification_report
    
    ## Parameters
    # Indices in Matlab start from 1
    # They are sequential numbers: first is 1, last is 66
    ind_electrodes = [16, 26, 27, 28, 30, 33, 34, 37, 39, 42, 46, 53, 60]
    MU = [0, 0.0025, 0.005, 0.0075, 0.01, 0.025] 
    C = 10
    ##

    # Loading raw data
    raw_signals, labels = preprocessing.load_matlab_data(
        matlab_file_path = "SFSVM/data/workspace_oliver.mat",
        classes_path = "SFSVM/data/classes.mat",
    )

    # Preprocessing
    signals = preprocessing.preprocess_raw_signals(
        raw_signals,
        thin = True,
        step = 11,
        smooth = True,
        window = 11,
        scale = False,
    )

    print("Raw signals:", raw_signals.shape)
    print("Preprocessed signals:", signals.shape)
    print("Labels:", labels.shape)    

    x_train, x_test, y_train, y_test = preprocessing.split_data(signals, labels, ind_electrodes)

    for _mu in MU:        
        print("\n=== NEW Experiment ===")
        print(f"\nSVM C = {C}")
        print(f"electrodes: {ind_electrodes}")
        print(f"mu = {_mu}")

        # 1. Solve the dual problem
        dual = solve_dual(X = x_train, y = y_train, C = C, mu = _mu)
        print("\nDual problem:")
        print("Status:", dual["status"])
        print("Problem status:", dual["problem_status"])
        print("Objective:", dual["objective"])

        # 2. Solve the final LP (85)
        lp = solve_final_lp(X = x_train, y = y_train, C = C, mu = _mu, lam = dual["lambda"])
        print("\nFinal LP:")
        print("Status:", lp["status"])
        print("Problem status:", lp["problem_status"])  
        print("Number of features:", x_train.shape[1])
        print("I+:", len(lp["I_plus"]))
        print("I0:", len(lp["I_zero"]))
        print("I-:", len(lp["I_minus"]))
        print("Number of selected features:", np.count_nonzero(lp["a"]))
        print("b:", lp["b"])

        # 3. Decision rule: f(x) = x * a + b
        predict_scores = np.dot(x_test, lp["a"]) + lp["b"]
        y_pred = np.where(predict_scores >= 0, 1, -1)

        # 4. Evaluation
        auc = roc_auc_score(y_test, predict_scores)

        print("\nClassification:")
        print("AUC:", f"{auc:.4f}")
        report = classification_report(y_test, y_pred, labels = [1, -1])
        print('\n'.join(report.splitlines()[:6])) 