from hsgbdh.hypergraph import HyperedgeReasoner

def test_basic_hyperedge():
    """
    Test simple A -> B (represented as A => {B})
    """
    model = HyperedgeReasoner()
    model.add_hyperedge('A', {'B'})
    
    result = model.query('A', 'B')
    assert result['reachable'] == True
    assert not result['exception']
    assert len(result['paths']) == 1

def test_join_composition():
    """
    Test {A, B} => C
    """
    model = HyperedgeReasoner()
    model.compose_join({'A', 'B'}, {'C'})
    
    # Not reachable from A alone
    result_a = model.query('A', 'C')
    assert result_a['reachable'] == False
    
    # Reachable from A if we add A -> B first? 
    # Actually query(source, target) starts with knowledge={source}
    # To test Join, we need a way to reach BOTH A and B.
    
    model.add_hyperedge('Start', {'A', 'B'})
    result_start = model.query('Start', 'C')
    assert result_start['reachable'] == True
    assert any('A' in p['path'] and 'B' in p['path'] for p in result_start['paths'])

def test_exception_reasoning():
    """
    Prove that hyperedges handle exceptions.
    """
    model = HyperedgeReasoner()
    
    # General rule: Birds fly
    model.add_hyperedge(
        source='Bird',
        targets={'Fly', 'HasFeathers', 'LayEggs'}
    )
    
    # Specific exception: Penguins don't fly
    model.add_hyperedge(
        source='Penguin',
        targets={'Bird', '¬Fly', 'Swim'}
    )
    
    # Query: Does a penguin fly?
    result = model.query('Penguin', 'Fly')
    
    # EXPECTED: Conflict detected!
    assert result['reachable'] == True # It is reachable via Bird
    assert result['exception'] == True # BUT it's an exception/conflict
    assert len(result['conflicts']) > 0
    
    # Check conflict details
    conflict = result['conflicts'][0]
    assert conflict['statement'] == 'Fly'
    
    # One path implies Fly (via Bird), another path implies ¬Fly (directly from Penguin)
    conclusions = set()
    for p in conflict['paths']:
        for node in p['path']:
            conclusions.add(node)
            
    assert 'Fly' in conclusions
    assert '¬Fly' in conclusions

if __name__ == "__main__":
    # Simple runner if pytest not available or needed quickly
    test_basic_hyperedge()
    test_join_composition()
    test_exception_reasoning()
    print("All tests passed!")
