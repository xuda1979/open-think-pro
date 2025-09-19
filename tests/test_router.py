from hairf.modules import ChainOfThoughtReasoner, ReflectiveCritic, SelfConsistencyReasoner
from hairf.router import AdaptiveRouter, RouterRule
from hairf.types import Query


def test_compute_optimal_router_allocates_more_budget_for_harder_queries():
    router = AdaptiveRouter()
    modules = [ChainOfThoughtReasoner(), SelfConsistencyReasoner(), ReflectiveCritic()]
    for idx, module in enumerate(modules, start=1):
        router.register(RouterRule(module=module, priority=idx * 5))

    easy_query = Query(text="Summarize the benefits of exercise.")
    hard_query = Query(
        text=(
            "Develop a step-by-step 7-day optimization plan that balances "
            "resource allocation, includes numeric milestones, and "
            "specifies verification checks?"
        ),
        task_type="planning",
    )

    easy_decision = router.route(easy_query)
    hard_decision = router.route(hard_query)

    assert hard_decision.difficulty >= easy_decision.difficulty
    assert hard_decision.budget >= easy_decision.budget >= 1
    assert len(hard_decision.selected_modules) >= len(easy_decision.selected_modules)
    assert hard_decision.allocation
    for module_name in hard_decision.selected_modules:
        assert module_name in hard_decision.allocation
