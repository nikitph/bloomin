from experiment import CurvatureExperiment

def run_word2vec():
    # Define Word2Vec config
    models = {
        'word2vec': {
            'model': 'word2vec-google-news-300',
            'source': 'gensim',
            'year': 2013,
            'dim': 300,
            'arch': 'Skip-Gram',
            'domain': 'News',
            'corpus_type': 'words', # Word2Vec embeds words, not sentences
            'n_samples': 10000
        }
    }
    
    # Initialize experiment
    exp = CurvatureExperiment(output_dir='./results')
    
    # Run
    exp.run_all_models(models, n_triangles=2000)

if __name__ == "__main__":
    run_word2vec()
