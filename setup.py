'''

    Hands free setup

'''

import os

def main():
    if os.path.exists('data'):
        if not os.path.exists('data/images'):
            os.mkdir('data/images')
            if not os.path.exists('data/images/testing'):
                os.mkdir('data/images/testing')
            if not os.path.exists('data/images/training'):
                os.mkdir('data/images/training')
            if not os.path.exists('data/images/validation'):
                os.mkdir('data/images/validation')
        if not os.path.exists('data/labels'):
            os.mkdir('data/labels')
            if not os.path.exists('data/labels/testing'):
                os.mkdir('data/labels/testing')
            if not os.path.exists('data/labels/training'):
                os.mkdir('data/labels/training')
            if not os.path.exists('data/labels/validation'):
                os.mkdir('data/labels/validation')

    print('setup complete')
    
if __name__ == '__main__':
    main()