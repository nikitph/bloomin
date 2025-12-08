from experiment import CurvatureExperiment

def run_glove():
    # Define GloVe config
    models = {
        'glove': {
            'model': 'glove-wiki-gigaword-100',
            'source': 'gensim',
            'year': 2014,
            'dim': 100,
            'arch': 'GloVe',
            'domain': 'Wiki',
            'corpus_type': 'words', 
            'n_samples': 10000
        }
    }
    
    # Initialize experiment
    exp = CurvatureExperiment(output_dir='./results')
    
    # Run
    exp.run_all_models(models, n_triangles=2000)

if __name__ == "__main__":
    run_glove()
