from hairf.dcmn import DynamicContextualMemoryNetwork
from hairf.types import Query


def test_memory_retrieval_orders_by_salience_and_overlap():
    network = DynamicContextualMemoryNetwork()
    network.ingest("math", "Remember to expand binomials carefully", boost=0.2)
    network.ingest("poetry", "Consider metaphors and rhythm")
    query = Query(text="How to expand a binomial expression?")
    nodes = network.retrieve(query, top_k=1)
    assert nodes
    assert nodes[0].key == "math"
