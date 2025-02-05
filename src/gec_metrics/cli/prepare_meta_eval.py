import os
import subprocess
from pathlib import Path

def main():
    print(__file__)
    path = Path(os.path.dirname(__file__)) / '../meta_eval/prepare_meta_eval.sh'
    subprocess.run([
        'bash',
        path
    ])
    
if __name__ == '__main__':
    main()