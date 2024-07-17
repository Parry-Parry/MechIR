
from .util import load_yaml
import subprocess as sp
import logging
import json

def execute(config_path : str, default_path : str = None):
    executions = load_yaml(config_path)
    if default_path is not None: defaults = load_yaml(default_path)
    for k, cfg in executions.items():
        if default_path is not None: cfg['args'].update(defaults)  
        logging.info('\n'.join([f'EXECUTION NAME: {k}', 'ARGS:', json.dumps(cfg['args'], indent=2)]))
        cmd = ['python', '-m', cfg['script']]
        for arg, val in cfg['args'].items():
            cmd.append(f'--{arg}')
            if val is not None:
                if type(val) == list:
                    cmd.append(' '.join(val))
                    continue
                cmd.append(str(val))
        sp.run(cmd)
    
    return f'Completed {len(executions)} executions.'

def cli():
    import sys 
    args = sys.argv[1:]
    if len(args) > 1: execute(args[0], args[1])
    else: execute(args[0])

if __name__ == '__main__':
    cli()
