"""集中定义候选模型的可审计选择规则。"""

from functools import cmp_to_key
import math


# 旧实验继续使用这组“固定容差字典序”指标，不能因新增 HERON 模式改变语义。
DEFAULT_SELECTION_RULE = (
    ("delivered_timeliness_raw_mean", "max", 1.0e-6),
    ("completion_rate_mean", "max", 1.0e-6),
    ("expiration_rate_mean", "min", 1.0e-6),
    ("delivered_data_mbit_mean", "max", 1.0e-3),
    ("load_balance_mean_per_task_mean", "max", 1.0e-9),
    ("rejected_subaction_rate_mean", "min", 1.0e-9),
)


def normalize_selection_rule(rule=None):
    """校验旧固定容差字典序规则。"""
    source = DEFAULT_SELECTION_RULE if rule is None else rule
    normalized = []
    for item in source:
        if isinstance(item, dict):
            metric, direction, tolerance = (
                item.get("metric"), item.get("direction"), item.get("tolerance")
            )
        else:
            try:
                metric, direction, tolerance = item
            except (TypeError, ValueError) as error:
                raise ValueError("模型选择规则项格式错误") from error
        if not isinstance(metric, str) or not metric:
            raise ValueError("模型选择指标名必须是非空字符串")
        if direction not in {"max", "min"}:
            raise ValueError("模型选择方向只能是max或min")
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("模型选择容差必须是非负有限数")
        normalized.append({"metric": metric, "direction": direction, "tolerance": tolerance})
    if not normalized:
        raise ValueError("模型选择规则不得为空")
    return normalized


def normalize_heron_hierarchy(hierarchy):
    """读取 HERON 启发式层级；配置写基础名，结果自动读取 mean/std。"""
    if not isinstance(hierarchy, list) or not hierarchy:
        raise ValueError("HERON层级规则不得为空")
    normalized = []
    for item in hierarchy:
        if not isinstance(item, dict):
            raise ValueError("HERON层级项必须是字典")
        metric, direction = item.get("metric"), item.get("direction")
        if not isinstance(metric, str) or not metric or metric.endswith(("_mean", "_std")):
            raise ValueError("HERON指标必须使用不带_mean/_std的基础名称")
        if direction not in {"max", "min"}:
            raise ValueError("HERON指标方向只能是max或min")
        entry = {"metric": metric, "direction": direction}
        for name in ("absolute_margin", "relative_margin", "std_scale"):
            value = float(item.get(name, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("HERON.{0}必须是非负有限数".format(name))
            entry[name] = value
        normalized.append(entry)
    return normalized


def normalize_best_model_rule(rule):
    """标准化两种模式，避免调用方散落判断配置结构。"""
    if not isinstance(rule, dict):
        # 为兼容早期直接传入 metrics 列表的调用。
        return {"mode": "tolerance_lexicographic", "metrics": normalize_selection_rule(rule)}
    mode = rule.get("mode", "tolerance_lexicographic")
    if mode == "tolerance_lexicographic":
        return {"mode": mode, "metrics": normalize_selection_rule(rule.get("metrics"))}
    if mode == "heron_hierarchical":
        return {"mode": mode, "hierarchy": normalize_heron_hierarchy(rule.get("hierarchy"))}
    raise ValueError("未知模型选择模式：{0}".format(mode))


def _finite_metric(result, name):
    try:
        value = float(result[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("模型选择缺少或无法解析指标：{0}".format(name)) from error
    if not math.isfinite(value):
        raise ValueError("模型选择指标包含NaN或Inf：{0}".format(name))
    return value


def compare_validation_results(candidate, incumbent, rule=None):
    """按旧容差字典序比较，返回 1（更优）、0（等价）或 -1（更差）。"""
    normalized = normalize_selection_rule(rule)
    # 先读取全部字段，保持旧实现“即使前项已决定也拒绝坏后项”的审计行为。
    validated = [(entry, _finite_metric(candidate, entry["metric"]), _finite_metric(incumbent, entry["metric"])) for entry in normalized]
    for entry, left, right in validated:
        difference = left - right
        if abs(difference) <= entry["tolerance"]:
            continue
        return 1 if (difference > 0.0) == (entry["direction"] == "max") else -1
    return 0


def _sample_count(result, metric):
    """读取聚合窗口样本数；旧单次验证结果兼容为一个样本。"""
    value = result.get(metric + "_sample_count", 1)
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("HERON样本数必须是正整数：{0}".format(metric)) from error
    if isinstance(value, bool) or count != value or count <= 0:
        raise ValueError("HERON样本数必须是正整数：{0}".format(metric))
    return count


def compare_heron_results(candidate, incumbent, hierarchy, candidate_id="", incumbent_id="", candidate_spec_id="", incumbent_spec_id=""):
    """逐层比较真实任务指标，并返回数值审计 trace。

    每个候选的均值来自稳定窗口；门槛使用均值的标准误而非原始标准差，
    因而不会因较宽的原始波动而过度把核心任务指标判为等价。
    """
    traces = []
    for level, entry in enumerate(normalize_heron_hierarchy(hierarchy), start=1):
        mean_name, std_name = entry["metric"] + "_mean", entry["metric"] + "_std"
        mean_a, mean_b = _finite_metric(candidate, mean_name), _finite_metric(incumbent, mean_name)
        std_a, std_b = _finite_metric(candidate, std_name), _finite_metric(incumbent, std_name)
        count_a = _sample_count(candidate, entry["metric"])
        count_b = _sample_count(incumbent, entry["metric"])
        standard_error_a = std_a / math.sqrt(count_a)
        standard_error_b = std_b / math.sqrt(count_b)
        pooled_standard_error = math.sqrt(
            standard_error_a * standard_error_a
            + standard_error_b * standard_error_b
        )
        delta = max(
            entry["absolute_margin"],
            entry["relative_margin"] * max(abs(mean_a), abs(mean_b), 1.0e-12),
            entry["std_scale"] * pooled_standard_error,
        )
        difference = mean_a - mean_b
        trace = {
            "candidate_a": candidate_id,
            "candidate_b": incumbent_id,
            "metric": entry["metric"],
            "level": level,
            "direction": entry["direction"],
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": std_a,
            "std_b": std_b,
            "sample_count_a": count_a,
            "sample_count_b": count_b,
            "standard_error_a": standard_error_a,
            "standard_error_b": standard_error_b,
            "pooled_standard_error": pooled_standard_error,
            "absolute_margin": entry["absolute_margin"],
            "relative_margin": entry["relative_margin"],
            "std_scale": entry["std_scale"],
            "effective_delta": delta,
            "difference": difference,
        }
        if abs(difference) <= delta:
            trace.update({"decision": "equivalent", "reason": "difference_within_dynamic_margin"})
            traces.append(trace)
            continue
        winner = 1 if (difference > 0.0) == (entry["direction"] == "max") else -1
        trace.update({"decision": "candidate_a_wins" if winner > 0 else "candidate_b_wins", "reason": "first_material_difference"})
        traces.append(trace)
        return winner, traces
    # 所有层级等价时只以稳定字符串作最后裁决，绝不引入随机性。
    left_key, right_key = (str(candidate_spec_id), str(candidate_id)), (str(incumbent_spec_id), str(incumbent_id))
    winner = 0 if left_key == right_key else (1 if left_key < right_key else -1)
    traces.append({"candidate_a": candidate_id, "candidate_b": incumbent_id, "metric": "stable_tie_breaker", "level": len(hierarchy) + 1, "direction": "min_lexicographic", "decision": "equivalent" if winner == 0 else ("candidate_a_wins" if winner > 0 else "candidate_b_wins"), "reason": "reward_spec_id_then_candidate_id", "candidate_a_key": left_key, "candidate_b_key": right_key})
    return winner, traces


def compare_records(candidate, incumbent, rule):
    """统一入口：同时返回胜负与可写入 JSON 的比较轨迹。"""
    normalized = normalize_best_model_rule(rule)
    if normalized["mode"] == "tolerance_lexicographic":
        result = compare_validation_results(candidate["validation"], incumbent["validation"], normalized["metrics"])
        return result, [{"candidate_a": candidate.get("candidate_id", ""), "candidate_b": incumbent.get("candidate_id", ""), "mode": normalized["mode"], "decision": result}]
    return compare_heron_results(candidate["validation"], incumbent["validation"], normalized["hierarchy"], candidate.get("candidate_id", ""), incumbent.get("candidate_id", ""), candidate.get("reward_spec_id", ""), incumbent.get("reward_spec_id", ""))


def is_better_validation_result(candidate, incumbent, rule=None):
    """供训练中 best checkpoint 使用；HERON 必须保留候选 ID 时由记录接口完成。"""
    if incumbent is None:
        if normalize_best_model_rule(rule)["mode"] == "heron_hierarchical":
            compare_heron_results(candidate, candidate, normalize_best_model_rule(rule)["hierarchy"])
        else:
            compare_validation_results(candidate, candidate, normalize_best_model_rule(rule)["metrics"])
        return True
    normalized = normalize_best_model_rule(rule)
    if normalized["mode"] == "tolerance_lexicographic":
        return compare_validation_results(candidate, incumbent, normalized["metrics"]) > 0
    return compare_heron_results(candidate, incumbent, normalized["hierarchy"])[0] > 0


def rank_validation_results(records, metrics_getter=None, rule=None, return_traces=False):
    """稳定排序记录；HERON 模式把相邻排名的决定依据保留为 trace。"""
    getter = metrics_getter or (lambda item: item)
    normalized = normalize_best_model_rule(rule)
    items = list(records)
    if normalized["mode"] == "tolerance_lexicographic":
        ranked = sorted(items, key=cmp_to_key(lambda left, right: -compare_validation_results(getter(left), getter(right), normalized["metrics"])))
    else:
        def compare(left, right):
            left_record = left if isinstance(left, dict) and "validation" in left else {"validation": getter(left)}
            right_record = right if isinstance(right, dict) and "validation" in right else {"validation": getter(right)}
            return -compare_records(left_record, right_record, normalized)[0]
        ranked = sorted(items, key=cmp_to_key(compare))
    traces = []
    for index in range(1, len(ranked)):
        left, right = ranked[index - 1], ranked[index]
        left_record = left if isinstance(left, dict) and "validation" in left else {"validation": getter(left)}
        right_record = right if isinstance(right, dict) and "validation" in right else {"validation": getter(right)}
        _, trace = compare_records(left_record, right_record, normalized)
        traces.append({"higher_ranked": left_record.get("candidate_id"), "lower_ranked": right_record.get("candidate_id"), "comparison": trace})
    return (ranked, traces) if return_traces else ranked
