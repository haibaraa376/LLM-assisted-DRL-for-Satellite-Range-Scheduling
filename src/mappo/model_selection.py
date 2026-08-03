"""集中定义基线模型选择的容差字典序规则。"""

from functools import cmp_to_key
import math


DEFAULT_SELECTION_RULE = (
    ("delivered_timeliness_raw_mean", "max", 1.0e-6),
    ("completion_rate_mean", "max", 1.0e-6),
    ("expiration_rate_mean", "min", 1.0e-6),
    ("delivered_data_mbit_mean", "max", 1.0e-3),
    ("load_balance_mean_per_task_mean", "max", 1.0e-9),
    ("rejected_subaction_rate_mean", "min", 1.0e-9),
)


def normalize_selection_rule(rule=None):
    """校验并标准化选择规则，未知方向或非法容差立即报错。"""
    source = DEFAULT_SELECTION_RULE if rule is None else rule
    normalized = []
    for item in source:
        if isinstance(item, dict):
            metric = item.get("metric")
            direction = item.get("direction")
            tolerance = item.get("tolerance")
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
        normalized.append(
            {"metric": metric, "direction": direction, "tolerance": tolerance}
        )
    if not normalized:
        raise ValueError("模型选择规则不得为空")
    return normalized


def compare_validation_results(candidate, incumbent, rule=None):
    """按容差字典序比较，返回1（更优）、0（等价）或-1（更差）。"""
    normalized = normalize_selection_rule(rule)
    validated = []
    for entry in normalized:
        metric = entry["metric"]
        try:
            left = float(candidate[metric])
            right = float(incumbent[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("模型选择缺少或无法解析指标：{0}".format(metric)) from error
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("模型选择指标包含NaN或Inf：{0}".format(metric))
        validated.append((entry, left, right))
    for entry, left, right in validated:
        difference = left - right
        if abs(difference) <= entry["tolerance"]:
            continue
        if entry["direction"] == "max":
            return 1 if difference > 0.0 else -1
        return 1 if difference < 0.0 else -1
    return 0


def is_better_validation_result(candidate, incumbent, rule=None):
    """判断候选是否严格优于已有Best；空Best总是接受。"""
    if incumbent is None:
        # 即使没有Best也先验证候选字段，避免坏指标进入Checkpoint。
        compare_validation_results(candidate, candidate, rule)
        return True
    return compare_validation_results(candidate, incumbent, rule) > 0


def rank_validation_results(records, metrics_getter=None, rule=None):
    """稳定排序记录；规则等价的记录保持输入顺序。"""
    getter = metrics_getter or (lambda item: item)

    def compare(left, right):
        # Python排序要求负数表示left在前，因此对“更优”结果取反。
        return -compare_validation_results(getter(left), getter(right), rule)

    return sorted(list(records), key=cmp_to_key(compare))
