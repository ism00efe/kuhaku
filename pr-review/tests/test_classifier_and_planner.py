from pr_review.analyzer.classifier import Classifier
from pr_review.analyzer.deterministic import DeterministicAnalyzer
from pr_review.config import load_config
from pr_review.models import PRContext, RepoMetadata
from pr_review.plan.planner import Planner
from pr_review.providers.selector import ModelSelector


def _pr(diff, title="feat: add modulo"):
    return PRContext(
        title=title, body="adds a modulo helper", base_ref="main", head_ref="f",
        base_sha="0", head_sha="1", diff=diff,
    )


def test_heuristic_plan_without_llm(sample_diff):
    cfg = load_config(overrides={"classifier_enabled": False})
    analysis = DeterministicAnalyzer().analyze(_pr(sample_diff), RepoMetadata())
    plan = Classifier(cfg, ModelSelector(cfg)).plan(analysis, RepoMetadata(), _pr(sample_diff))
    assert plan.source == "heuristic"
    axes = {e.axis for e in plan.active()}
    assert "correctness" in axes
    assert "structure" in axes  # dependency file changed


def test_planner_builds_tasks_and_narrows_context(sample_diff):
    cfg = load_config(overrides={"classifier_enabled": False, "force_provider": "mock"})
    analysis = DeterministicAnalyzer().analyze(_pr(sample_diff), RepoMetadata())
    plan = Classifier(cfg, ModelSelector(cfg)).plan(analysis, RepoMetadata(), _pr(sample_diff))
    tasks = Planner(cfg).build(plan)
    assert tasks
    by_axis = {t.axis: t for t in tasks}
    # scope never pulls callers even if the depth would allow it
    if "scope" in by_axis:
        assert by_axis["scope"].context_spec.callers_usages is False


def test_mock_planner_returns_valid_plan(sample_diff):
    cfg = load_config(overrides={"force_provider": "mock"})
    analysis = DeterministicAnalyzer().analyze(_pr(sample_diff), RepoMetadata())
    plan = Classifier(cfg, ModelSelector(cfg)).plan(analysis, RepoMetadata(), _pr(sample_diff))
    assert plan.source in ("llm", "heuristic")
    assert plan.active()
