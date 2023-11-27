from common_imports import *

# download the counterfact dataset if it is not present
ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
COUNTERFACT_PATH = ROOT_PATH / 'counterfact.json'
STATS_ROOT = ROOT_PATH / 'stats/gpt2-xl/wikipedia_stats/'
GPT2_XL_N_LAYERS = 48


def setup_counterfact():
    if not COUNTERFACT_PATH.exists():
        print('Downloading counterfact dataset...')
        url = 'https://rome.baulab.info/data/dsets/counterfact.json'
        # make wget not be verbose
        os.system(f'wget -q {url} -O {COUNTERFACT_PATH}')
        print('Done.')



def setup_stats():
    fname = 'transformer.h.{layer}.mlp.c_proj_float32_mom2_100000.npz'
    for layer in range(GPT2_XL_N_LAYERS):
        path = STATS_ROOT / fname.format(layer=layer)
        if not path.exists():
            print('Downloading stats for layer', layer)
            url = f'https://rome.baulab.info/data/stats/gpt2-xl/wikipedia_stats/{fname.format(layer=layer)}'
            os.system(f'wget -q {url} -O {path}')
            print('Done.')
    
def get_covariance_path(layer: int) -> Path:
    return STATS_ROOT / f'transformer.h.{layer}.mlp.c_proj_float32_mom2_100000.npz'