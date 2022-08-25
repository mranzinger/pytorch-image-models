#!/usr/bin/python3

import argparse
import getpass
import subprocess
import os
import platform
import sys
import shutil
from typing import List

HOSTNAME = platform.node()


def launch_job(output_dir, name, args, rest_args: List[str]):
    print("Creating output dir...")
    os.makedirs(output_dir, exist_ok=True)
    print("Done")

    source_dir = os.path.dirname(__file__)
    if not source_dir:
        source_dir = os.getcwd()
    print('Source directory:', source_dir)

    dest_dir = os.path.join(output_dir, 'source')

    work_dir = dest_dir

    if args.resume:
        assert os.path.exists(dest_dir)
    else:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)

        if not args.no_copy_source:
            print('Copying source code to "{}"...'.format(dest_dir))
            shutil.copytree(source_dir, dest_dir, symlinks=True,
                ignore=shutil.ignore_patterns('synthetic_data', 'logs', '.git*',
                                            '__pycache__', '.vscode', 'metrics.csv',
                                            '*.mov', '*.sqsh')
            )
            print('Done')

            proc = subprocess.run(['bash', 'git_branch.sh'], stdout=subprocess.PIPE)

            git_info = proc.stdout.decode('utf-8').strip()

            git_dir = os.path.join(dest_dir, 'git-info')
            os.makedirs(git_dir, exist_ok=True)

            with open(os.path.join(git_dir, 'revision.txt'), 'w') as fd:
                fd.write(git_info)
                fd.write('\n')
                print('Wrote git revision info to "{}".'.format(fd.name))

            proc = subprocess.run(['git', '--no-pager', 'diff'], stdout=subprocess.PIPE)

            git_diff = proc.stdout.decode('utf-8').strip()

            with open(os.path.join(git_dir, 'delta.diff'), 'w') as fd:
                fd.write(git_diff)
                fd.write('\n')
                print('Wrote git diff to "{}".'.format(fd.name))
        else:
            work_dir = source_dir

    try:
        # If we're training, then inject the distributed launch code
        python_idx = rest_args.index('python')
        train_idx = rest_args.index('train.py')

        # Python raises errors if those values aren't present, so by reaching
        # this point, we know they're both there
        if 'torch.distributed.launch' not in rest_args:
            dist_args = [
                '-m', 'torch.distributed.launch',
                '--nproc_per_node', '$SUBMIT_GPUS',
                '--master_addr', '$MASTER_ADDR',
                '--master_port', '$MASTER_PORT',
                '--nnodes', '$NUM_NODES',
                '--node_rank', '$NODE_RANK',
            ]
            rest_args[python_idx+1:python_idx+1] = dist_args
    except:
        pass


    command_args = rest_args + ['--output', output_dir]

    if args.resume:
        command_args = command_args + ['--resume']

    command_args = subprocess.list2cmdline(command_args)
    print('command: {}'.format(command_args))

    with open('docker_image', 'r') as fd:
        docker_image = fd.read().strip().splitlines()
        docker_image = [line for line in docker_image if not line.startswith('#')]
        docker_image = ''.join(docker_image)

    print('docker image: {}'.format(docker_image))

    num_gpus = args.gpus
    if num_gpus == 0:
        if 'batch_dgx2' in args.partition:
            num_gpus = 16
        else:
            num_gpus = 8

    duration = args.duration or 8

    submit_args = [
        'submit_job',
        '--gpu', str(num_gpus),
        '--nodes', str(args.nodes),
        '--partition', args.partition,
        '--workdir', work_dir,
        '--logroot', os.path.join(work_dir, 'logs'),
        '--image', docker_image,
        '--duration', str(duration),
        '--autoresume_timer', str(duration * 60 - 30),
        # '--setenv', 'CUDA_LAUNCH_BLOCKING=1',
        '-c', command_args
    ]

    if args.interactive:
        submit_args.insert(1, '--interactive')

    # Ensure 32g machines on Reno
    if HOSTNAME == 'draco-rno-login-0001' and not args.interactive:
        submit_args.insert(1, 'dgx1,gpu_32gb')
        submit_args.insert(1, '--constraint')

    if HOSTNAME != 'draco-aws-login-01' and not HOSTNAME.startswith('ip-'):
        submit_args.insert(1, os.environ['MOUNTS'])
        submit_args.insert(1, '--mounts')

    if name:
        submit_args.insert(1, name)
        submit_args.insert(1, '--name')
    else:
        submit_args.insert(1, '--coolname')

    login_node = args.login_node
    if login_node is not None:
        usr_name = getpass.getuser()
        submit_args.insert(1, f'{usr_name}@{login_node}')
        submit_args.insert(1, '--submit_proxy')

    if args.dependency:
        submit_args.insert(1, f'--dependency=afterany:{args.dependency}')

    print('submit_args:', submit_args)

    popen = subprocess.Popen(submit_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line, end="")
    popen.stdout.close()
    return_code = popen.wait()

    return return_code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Utility to launch jobs on the cluster.")
    parser.add_argument('-r', '--output', required=True,
                        help="Path to the output directory")
    parser.add_argument('--interactive', default=False, action='store_true',
                        help="Run the job in interactive mode")
    parser.add_argument('--resume', default=False, action='store_true',
                        help="Resume the job with the source in the output dir.")
    parser.add_argument('--partition', default='', type=str,
                        help='Which partition to launch the job on. One of: batch_16GB, batch_32GB, batch_dgx2_singlenode')
    parser.add_argument('--gpus', default=0, type=int, help='The number of GPUs to allocate')
    parser.add_argument('-p', '--num_parallel', default=1, type=int,
                        help='The number of runs with the given configuration to launch.')
    parser.add_argument('--p_ctr', default=0, type=int,
                        help='The starting point for the parallel counter')
    parser.add_argument('--nodes', default=1, type=int,
                        help='The number of nodes to allocate for the job')
    parser.add_argument('--duration', default=None, type=int,
                        help='The duration of the job. Default is 8 hours.')
    parser.add_argument('--name', default=None, type=str,
                        help='The name to give the jobs')
    parser.add_argument('--login_node', default=None, type=str,
                        help='Launch the job via proxy to the login_node')
    parser.add_argument('--no_copy_source', default=False, action='store_true',
                        help='Don\'t copy over the source tree. Instead, use the current directory.')
    parser.add_argument('--dependency', default=None, type=str,
                        help='(Optional) A job-id that this job is dependent upon')

    # Get the output directory without fussing with the rest
    args, rest_args = parser.parse_known_args()

    partition = args.partition
    if not partition and not args.interactive:
        if HOSTNAME == 'draco-aws-login-01' or HOSTNAME.startswith('ip-'):
            partition = 'batch'
            if not args.duration:
                args.duration = 4
        elif HOSTNAME == 'draco-rno-login-0001' or HOSTNAME.startswith('rno'):
            partition = 'batch_dgx1'
            # partition = 'batch_dgx1_m3'
    args.partition = partition

    if not args.login_node:
        if HOSTNAME.startswith('rno'):
            args.login_node = 'draco-rno-login-0002'
        elif HOSTNAME.startswith('ip-'):
            args.login_node = 'draco-aws-login-01'

    name = args.name
    if name is None:
        output_dir = os.environ['OUTPUT_DIR']
        resdir = args.output
        if resdir.startswith(output_dir):
            resdir = resdir[len(output_dir):]
            if resdir.startswith('/'):
                resdir = resdir[1:]

        rs_comps: list = resdir.split('/')[1:]

        name = '-'.join(rs_comps)

    output_dir = args.output
    names = []
    if args.num_parallel == 1 and args.p_ctr == 0:
        output_dir = [output_dir]
        names.append(name)
    else:
        names = [f'{name}_{i + args.p_ctr}' if name is not None else None for i in range(args.num_parallel)]
        output_dir = [os.path.join(output_dir, f'run_{i + args.p_ctr}') for i in range(args.num_parallel)]

    for d, n in zip(output_dir, names):
        launch_job(d, n, args, rest_args)
