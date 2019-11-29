import shutil
import errno
import os
import subprocess
import argparse
import threading
import time
import datetime
import sys
import io

QUEUE_DIR = 'jobs/'

class Job():
    def __init__(self, code_dir=None, data_dir=None, cmd=None, job_dir=None, gpus=None, n_runs=1, time=None, status=None, name=None):

        # check data types and values
        if code_dir is not None: assert isinstance(code_dir, str)
        if data_dir is not None:assert isinstance(data_dir, str)
        if cmd is not None:assert isinstance(cmd, str)
        if job_dir is not None: assert isinstance(job_dir, str)
        if gpus is not None: assert isinstance(gpus, list)
        assert isinstance(n_runs, int)
        assert n_runs > 0

        # save values
        self.code_dir = os.path.abspath(code_dir) if code_dir is not None else None
        self.data_dir = os.path.abspath(data_dir) if data_dir is not None else None
        self.cmd = cmd
        self.gpus = gpus
        self.job_dir = os.path.abspath(job_dir) if job_dir is not None else None
        self.n_runs = n_runs
        self.time = time
        self.status = status
        self.name = name


class JobQueue():
    '''Queue designed to handle a list of jobs. Thread-safe'''
    def __init__(self, queue_dir):
        self._lock = threading.Lock()
        self.jobs = {}
        #self.finished_jobs = {}
        self.queue_dir = queue_dir

    def update(self):
        """Update jobs list by reading all jobs from queue directory"""
        with self._lock:
            self.jobs = {}
            for job_name in os.listdir(self.queue_dir):
                job_dir = os.path.join(self.queue_dir, job_name)
                if os.path.isdir(job_dir):
                    try:
                        curr_job = self._read_job(job_dir)
                        self.jobs[job_name] = curr_job
                        #if curr_job.status == 'finished':
                        #    self.finished_jobs[job_name] = curr_job
                    except FileNotFoundError:
                        print('Failed to read job of name: %s' % job_name)

    def _write_job(self, job):
        # override a previous job
        if os.path.exists(job.job_dir):
            shutil.rmtree(job.job_dir)
        os.makedirs(job.job_dir)

        # within job directory, create snapshot directory and store code
        snapshot_dir = os.path.join(job.job_dir, 'snapshot/')
        data_dir_name = os.path.split(job.data_dir)[-1]
        copy(job.code_dir, snapshot_dir, ignore=[data_dir_name])

        if job.gpus is not None:
            gpu_str_list = ''.join([str(gpu) + ',' for gpu in job.gpus])
        else:
            gpu_str_list = 'none'

        # write config
        config_file = os.path.join(job.job_dir, 'config.txt')
        with open(config_file, 'w') as f:
            f.write('code_dir: %s\n' % job.code_dir)
            f.write('data_dir: %s\n' % job.data_dir)
            f.write('cmd: %s\n' % job.cmd)
            f.write('time: %s\n' % job.time)
            f.write('gpus: %s\n' % gpu_str_list)

        self._write_status(job)

    def _read_job(self, job_dir):
        job = Job()
        job.job_dir = job_dir
        job.name = os.path.split(job_dir)[-1]
        config_file = os.path.join(job_dir, 'config.txt')
        with open(config_file, 'r') as f:
            job.code_dir = f.readline().replace('code_dir:', '').strip()
            job.data_dir = f.readline().replace('data_dir:', '').strip()
            job.cmd = f.readline().strip().replace('cmd:', '')
            job.time = f.readline().replace('time:', '').strip()

            gpus = f.readline().replace('gpus:', '').strip()

            if gpus == 'none':
                job.gpus = None
            else:
                gpus = gpus.split(',')
                gpus = [int(gpu) for gpu in gpus if gpu != '']
                job.gpus = gpus

        self._read_status(job)

        return job

    def _read_status(self, job):
        status_file = os.path.join(job.job_dir, 'status.txt')
        with open(status_file, 'r') as f:
            job.status = f.readline().replace('status:', '').strip()

    def read_status(self, job):
        with self._lock:
            self._read_status(job)
            return job.status

    def write_status(self, job, status):
        with self._lock:
            job.status = status
            self._write_status(job)

    def _write_status(self, job):
        status_file = os.path.join(job.job_dir, 'status.txt')
        with open(status_file, 'w') as f:
            f.write('status: %s\n' % job.status)

    def add_job(self, job_name, code_dir, data_dir, cmd, job_time=None, gpus=None):

        # convert to absolute paths
        code_dir = os.path.expanduser(code_dir)
        code_dir = os.path.abspath(code_dir)
        data_dir = os.path.expanduser(data_dir)
        data_dir = os.path.abspath(data_dir)

        if job_name is None:
            job_name = 'job_' + str(time.time())

        if job_time is None:
            job_time = str(datetime.datetime.now())

        with self._lock:
            # we copy code directory to temp folder, and add sym link to data directory
            job_dir = os.path.join(self.queue_dir, job_name)

            job = Job(code_dir=code_dir, data_dir=data_dir, name=job_name,
                      cmd=cmd, gpus=gpus, job_dir=job_dir, time=job_time, status='ready')

            self._write_job(job)

        return job

    def load_job(self, job_name):

        with self._lock:
            job_dir = os.path.join(self.queue_dir, job_name)

            if not os.path.exists(job_dir):
                raise ValueError('Job does not exist with that name!')

            job = self._read_job(job_dir)

            #print('Code dir: %s' % job.code_dir)
            #print('Data dir: %s' % job.data_dir)
            #print('Job dir: %s' % job.job_dir)
            #print('Job cmd: %s' % job.cmd)

        return job

    def rm_job(self, job):
        if isinstance(job, str):
            job_dir = os.path.join(self.queue_dir, job)
        else:
            job_dir = job.job_dir
        shutil.rmtree(job_dir)

    def oldest(self):
        sorted_jobs = sorted([self.jobs[name] for name in self.jobs if self.jobs[name].status == 'ready'],
                             key=lambda job: job.time)
        if len(sorted_jobs) > 0:
            return sorted_jobs[0]
        else:
            return None


def ignore_function(ignore):
    def _ignore_(path, names):
        ignored_names = []
        for ignored_name in ignore:
            if ignored_name in names:
                ignored_names.append(ignored_name)
        return set(ignored_names)
    return _ignore_


def copy(src, dest, ignore=None):
    ignore = ignore if ignore is not None else []
    try:
        shutil.copytree(src, dest, ignore=ignore_function(ignore))
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def example_job_specs():
    code_dir = '/home/ddonahue/GPUQueue/example_project'
    data_dir = '/home/ddonahue/GPUQueue/example_project/data'
    cmd = '/home/ddonahue/torch/bin/python example_program.py'
    gpus = []
    return code_dir, data_dir, cmd, gpus

def load_job_specs(opt):
    if opt.code is not None and opt.data is not None and opt.cmd is not None:

        # we convert string input to list
        if opt.gpus is not None:
            gpus = opt.gpus.split(',')
            gpus = [int(gpu.strip()) for gpu in gpus if gpu != '']
        else:
            gpus = None

        return opt.code, opt.data, opt.cmd, gpus
    else:
        return None


class JobRunner:
    def __init__(self, queue):
        self.queue = queue

    def start(self):
        print('Starting job runner...')
        while True:
            # update queue with all available jobs

            self.queue.update()
            next_job = self.queue.oldest()

            time.sleep(1)

            if next_job is None:
                continue

            print('Running job: %s' % next_job.name)
            self.run_job(next_job)



    def run_job(self, job):
        '''
        Run job using specified configuration. Does not lock queue access.
        :param job:
        :return:
        '''
        # create symbolic link to data directory
        data_dir_name = os.path.split(job.data_dir)[-1]
        snapshot_dir = os.path.join(job.job_dir, 'snapshot/')
        sl_path = os.path.join(snapshot_dir, data_dir_name)
        if not os.path.exists(sl_path):
            # only create symlink if it doesn't exist
            os.symlink(job.data_dir, sl_path)

        # run process
        cmd_tokens = job.cmd

        run_env = os.environ.copy()

        # set available gpus
        if job.gpus is not None:
            cuda_spec = ''.join([str(gpu) + ',' for gpu in job.gpus])
            run_env['CUDA_VISIBLE_DEVICES'] = cuda_spec

        try:
            output_file = os.path.join(job.job_dir, 'output.txt')
            with open(output_file, 'w') as f:
                print('Run subprocess: %s' % str(cmd_tokens))
                process = subprocess.Popen(cmd_tokens, stdout=subprocess.PIPE, cwd=snapshot_dir, env=run_env,
                                           stderr=subprocess.PIPE, shell=True)
                for line in process.stdout:
                    line = line.decode('utf-8')
                    sys.stdout.write(line)
                    f.write(line)
                process.wait()

            print('Job finished. Waiting for next job.')
            self.queue.write_status(job, 'finished')


        except Exception as e:
            self.queue.write_status(job, 'crashed')
            print('Exception: %s' % str(e))
            print('Job crashed. Moving on...')

        # write standard output to file


def parse_input_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('action', nargs='?')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--code', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--cmd', action='append', nargs='+', type=str, default=None)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--start', action='store_true', default=False)
    parser.add_argument('--status', type=str, default=None)
    #parser.add_argument('--name', type=str, default=None)

    opt = parser.parse_args()

    # if opt.cmd is not None:
    #     opt.cmd = ';'.join([' '.join(cmd) for cmd in opt.cmd])
    #     print(opt.cmd)

    return opt


if __name__ == '__main__':
    opt = parse_input_arguments()


    queue = JobQueue(QUEUE_DIR)

    if opt.action == 'add':
        specs = load_job_specs(opt)
        if specs is not None:
            print('Loading specs from command line')
            code_dir, data_dir, cmd, gpus = specs
        else:
            print('Loading example job specs (no name provided)!')
            code_dir, data_dir, cmd, gpus = example_job_specs()

        print('Adding job to queue')
        job = queue.add_job(opt.name, code_dir=code_dir, data_dir=data_dir, cmd=cmd, gpus=gpus)

        print('Job name: %s' % job.name)
        print('Code dir: %s' % job.code_dir)
        print('Data dir: %s' % job.data_dir)
        print('Job dir: %s' % job.job_dir)
        print('Job cmd: %s' % job.cmd)

    if opt.action == 'rm' or opt.action == 'remove':
        if opt.name is not None:
            queue.rm_job(opt.name)
        else:
            queue.update()
            selected_jobs = [queue.jobs[job_name] for job_name in queue.jobs]

            if opt.status is not None:
                if opt.status == 'all':
                    rm_by_status = None
                else:
                    rm_by_status = opt.status  # we only remove jobs with this status

                if rm_by_status is not None:
                    selected_jobs = [job for job in selected_jobs if job.status == rm_by_status]

                print('Removing jobs:')
                for job in selected_jobs:
                    print(job.name)
                    queue.rm_job(job)
            else:
                print('Must specify [--status=ready, finished, crashed] or [--name=job_name] to remove')

    # if opt.action == 'rm-finished':
    #     if opt.name is not None:
    #         queue.rm_job(opt.name)
    #     else:
    #         queue.update()
    #         for job_name in queue.finished_jobs:
    #             queue.rm_job(job_name)

    if opt.action == 'run':
        print('Running job')
        jr = JobRunner(queue)
        job = queue.load_job(opt.name)
        jr.run_job(job)

    if opt.action == 'list' or opt.action == 'ls':
        queue.update()

        selected_jobs = [queue.jobs[job_name] for job_name in queue.jobs]

        ls_by_status = 'ready'  # if opt.status isn't set, we by default only list ready jobs
        if opt.status is not None:
            if opt.status == 'all':
                ls_by_status = None
            else:
                ls_by_status = opt.status  # we only remove jobs with this status

        if ls_by_status is not None:
            selected_jobs = [job for job in selected_jobs if job.status == ls_by_status]

        if ls_by_status is None:
            print('Listing all jobs')
        else:
            print('Listing jobs of type: %s' % ls_by_status)

        if len(selected_jobs) == 0:
            print('No jobs to list')
        else:
            print('%-*s  %-*s  %s' % (20, 'Status', 30, 'Time Submitted', 'Job Name'))
            print('%-*s  %-*s  %s' % (20, '-----', 30, '-----', '-----'))
            for job_name in queue.jobs:
                job = queue.jobs[job_name]
                print('%-*s  %-*s  %s' % (20, job.status, 30, job.time, job.name))

    # if opt.action == 'list-finished':
    #     queue.update()
    #
    #     if len(queue.finished_jobs) > 0:
    #         print('Finished jobs:')
    #     else:
    #         print('No jobs finished')
    #
    #     for job_name in queue.finished_jobs:
    #         job = queue.finished_jobs[job_name]
    #         print('%s\t%s' % (job.time, job_name))

    if opt.action == 'start' or opt.action == 's':
        runner = JobRunner(queue)
        runner.start()

    #print('Updating queue jobs')
    #queue.update()
    #print(queue.jobs)

    #first_job = queue.oldest()

    #print('Running job')
    #output = run_job(job=first_job)

    # after running job we remove it
    #print('Removing job')
    #rm_job(job)

    if opt.action == 'print' or opt.action == 'p':
        print('########## PROCESS OUTPUT ###########')
        print()
        job = queue.load_job(opt.name)
        output = open(os.path.join(job.job_dir, 'output.txt'), 'r').read()

        print(output)

    # when a gpu is available, we set cuda visible devices to that gpu and run

    # save standard output to a file

    # when job finishes, we start the next job in the queue