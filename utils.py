import numpy as np
import datetime

def cur_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stress_message(message, extra_newline=False):
    print('{2}{0}\n{1}\n{0}{2}'.format('='*len(message), message, '\n' if extra_newline else ''))


def shuffle_data(a, b):
    ''' Shuffles 2 np arrays with same length together '''
    assert len(a) == len(b)                 # Sanity check
    random_state = np.random.get_state()    # Store random state s.t. 2 shuffles are the same
    np.random.shuffle(a)
    np.random.set_state(random_state)
    np.random.shuffle(b)
    np.random.seed()    # Re-seed generator
