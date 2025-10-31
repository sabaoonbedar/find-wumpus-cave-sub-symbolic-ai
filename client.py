"""
A Python implementation of the AISysProj server protocol.

If you want to do the protocol implementation yourself,
there is also a simplified client implementation, which should
be much easier to understand.

You can find client implementations here:
https://aisysprojserver.readthedocs.io/en/latest/clients.html
"""

import abc
import dataclasses
import json
import logging
import multiprocessing
import time
from multiprocessing import Process
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TypedDict, Optional, Callable, Any, Literal

import requests as requests_lib

logger = logging.getLogger(__name__)

# type info (not using e.g. pydantic to keep dependencies minimal)
AgentConfig = TypedDict('AgentConfig', {'agent': str, 'env': str, 'url': str, 'pwd': str})
Action = TypedDict('Action', {'run': str, 'act_no': int, 'action': Any})
ActionRequest = TypedDict('ActionRequest', {'run': str, 'act_no': int, 'percept': Any})
Message = TypedDict('Message', {'type': Literal['info', 'warning', 'error'], 'content': str, 'run': Optional[str]})
ServerResponse = TypedDict(
    'ServerResponse',
    {
        'action_requests': list[ActionRequest],
        'active_runs': list[str],
        'messages': list[Message],
        'finished_runs': dict[str, Any],
    }
)


def get_run_url(agent_config: AgentConfig, run_id: str) -> str:
    url = agent_config['url']
    if not url.endswith('/'):
        url += '/'
    return url + f'run/{agent_config["env"]}/{run_id}'


def _handle_response(response) -> Optional[ServerResponse]:
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 503:
        logger.warning('Server is busy - retrying in 3 seconds')
        time.sleep(3)
        return None
    else:  # in other cases, retrying does not help (authentication problems, etc.)
        logger.error(f'Status code {response.status_code}.')
        j = response.json()
        logger.error(f'{j["errorname"]}: {j["description"]}')
        logger.error('Stopping.')
        response.raise_for_status()
        return None     # unreachable, but mypy doesn't know that


def send_request(
        config: AgentConfig,
        actions: list[Action],
        *,
        to_abandon: Optional[list[str]] = None,
        parallel_runs: bool = True
) -> ServerResponse:
    while True:  # retry until success
        logger.debug(f'Sending request with {len(actions) or "no"} actions: {actions}')
        base_url = config['url']
        if not base_url.endswith('/'):
            base_url += '/'
        response = requests_lib.put(f'{base_url}act/{config["env"]}', json={
            'protocol_version': 1,
            'agent': config['agent'],
            'pwd': config['pwd'],
            'actions': actions,
            'to_abandon': to_abandon or [],
            'parallel_runs': parallel_runs,
            'client': 'py-client-v1',
        })
        result = _handle_response(response)
        if result is not None:
            return result


@dataclasses.dataclass(frozen=True)
class RequestInfo:
    run_url: str
    action_number: int
    run_id: str


class _RunTracker:
    def __init__(self):
        self.number_of_new_runs_finished: int = 0
        self.old_runs: Optional[set[str]] = None
        self.ongoing_runs: set[str] = set()

    def update(self, response: ServerResponse):
        if self.old_runs is None:
            self.old_runs = set(response['active_runs'])
            for rq in response['action_requests']:
                if rq['act_no'] == 0:  # not actually old
                    self.old_runs.remove(rq['run'])

        ongoing_runs = set(response['active_runs'])
        self.number_of_new_runs_finished += len(
                self.ongoing_runs - self.old_runs - ongoing_runs
        )
        self.ongoing_runs = ongoing_runs


class RequestProcessor(abc.ABC):
    @abc.abstractmethod
    def process_requests(self, requests: list[tuple[Any, RequestInfo]], counter: _RunTracker) -> list[Action]:
        pass

    def close(self):
        pass

    def on_new_run(self, run_id: str):
        logger.info(f'Starting new run ({run_id})')

    def on_finished_run(self, run_id: str, url: str, outcome: Any):
        logger.info(f'Finished run {run_id} with outcome {json.dumps(outcome)}')
        logger.info(f'You can view the run at {url}')

    def on_message(self, message: Message):
        {
            'info': logger.info,
            'warning': logger.warning,
            'error': logger.error,
        }[message['type']](
            f'run {message["run"]}: {message["content"]}' if message['run'] is not None else message['content']
        )


class SimpleRequestProcessor(RequestProcessor):
    def __init__(self, action_function: Callable[[Any, RequestInfo], Any], processes: int = 1):
        self.action_function = action_function
        self.pool = None
        if processes > 1:
            self.pool = multiprocessing.Pool(processes=processes)

    def process_requests(self, requests: list[tuple[Any, RequestInfo]], counter: _RunTracker) -> list[Action]:
        if self.pool is None:
            return [
                {
                    'run': request_info.run_id,
                    'act_no': request_info.action_number,
                    'action': self.action_function(percept, request_info)
                } for percept, request_info in requests
            ]
        else:
            return [
                {
                    'run': request_info.run_id,
                    'act_no': request_info.action_number,
                    'action': action
                } for action, (_percept, request_info) in zip(
                    self.pool.starmap(self.action_function, requests),
                    requests
                )
            ]

    def close(self):
        if self.pool is not None:
            self.pool.terminate()


class Agent(abc.ABC):
    def __init__(self, run_id: str, agent_config: AgentConfig):
        self.__run_id = run_id
        self.__agent_config = agent_config

    @abc.abstractmethod
    def get_action(self, percept: Any, request_info: RequestInfo) -> Any:
        raise NotImplementedError()

    def on_finish(self, outcome: Any):
        logger.info(f'Finished run {self.__run_id} with outcome {json.dumps(outcome)}')
        logger.info(f'You can view the run at {self.get_run_url()}')

    def on_message(self, content: str, type: str):
        {
            'info': logger.info,
            'warning': logger.warning,
            'error': logger.error,
        }[type](f'Message for run {self.__run_id}: {content}')

    def get_run_url(self) -> str:
        return get_run_url(self.__agent_config, self.__run_id)

    @classmethod
    def run(
            cls,
            agent_config_file: str | Path | AgentConfig,
            *,
            parallel_runs: bool = True,
            multiprocessing: bool = False,
            abandon_old_runs: bool = False,
            run_limit: Optional[int] = None,
    ):
        agent_config = _get_agent_config(agent_config_file)
        request_processor: RequestProcessor
        if multiprocessing:
            request_processor = MultiProcessAgentRequestProcessor(cls, agent_config)
        else:
            request_processor = SequentialAgentRequestProcessor(cls, agent_config)
        _run(agent_config, request_processor, parallel_runs=parallel_runs,
             abandon_old_runs=abandon_old_runs, run_limit=run_limit)


class SequentialAgentRequestProcessor(RequestProcessor):
    def __init__(self, agent_class: type[Agent], agent_config: AgentConfig):
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.agents: dict[str, Agent] = {}

    def process_requests(self, requests: list[tuple[Any, RequestInfo]], counter: _RunTracker) -> list[Action]:
        actions: list[Action] = []
        for percept, request_info in requests:
            if request_info.run_id not in self.agents:
                self.agents[request_info.run_id] = self.agent_class(request_info.run_id, self.agent_config)

            actions.append({
                'run': request_info.run_id,
                'act_no': request_info.action_number,
                'action': self.agents[request_info.run_id].get_action(percept, request_info)
            })

        for run_id in list(self.agents.keys()):
            if run_id not in counter.ongoing_runs:
                del self.agents[run_id]

        return actions

    def on_finished_run(self, run_id: str, url: str, outcome: Any):
        if run_id in self.agents:
            self.agents[run_id].on_finish(outcome)
        else:
            super().on_finished_run(run_id, url, outcome)

    def on_message(self, message: Message):
        if message['run'] in self.agents:
            self.agents[message['run']].on_message(message['content'], message['type'])
        else:
            super().on_message(message)


class AgentProcess:
    def __init__(self, agent_class: type[Agent]):
        self.conn, conn = multiprocessing.Pipe(duplex=True)
        self.process = Process(target=self._run, args=(conn, agent_class))
        self.process.start()

    def new_run(self, run_id: str, agent_config: AgentConfig):
        self.send_command('new_run', run_id, agent_config)

    def finish_run(self, outcome: Any):
        self.send_command('finish_run', outcome)

    def send_message(self, content: str, type: str):
        self.send_command('message', content, type)

    def send_action_request(self, percept: Any, request_info: RequestInfo):
        self.send_command('get_action', percept, request_info)

    def get_response(self):
        return self.conn.recv()

    def stop(self):
        self.send_command('stop')
        self.process.join()

    def send_command(self, command: str, *args):
        self.conn.send((command, *args))

    def _run(self, conn: Connection, agent_class: type[Agent]):
        agent: Optional[Agent] = None
        while True:
            match conn.recv():
                case ('new_run', run_id, agent_config):
                    agent = agent_class(run_id, agent_config)
                case ('finish_run', outcome):
                    agent.on_finish(outcome)
                    agent = None
                case ('message', content, type):
                    agent.on_message(content, type)
                case ('get_action', percept, request_info):
                    conn.send(agent.get_action(percept, request_info))
                case ('stop',):
                    break


class MultiProcessAgentRequestProcessor(RequestProcessor):
    def __init__(self, agent_class: type[Agent], agent_config: AgentConfig):
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.assigned_processes: dict[str, AgentProcess] = {}
        self.unassigned_processes: list[AgentProcess] = []

    def process_requests(self, requests: list[tuple[Any, RequestInfo]], counter: _RunTracker) -> list[Action]:
        # unassign processes for finished runs
        for run_id, proc in list(self.assigned_processes.items()):
            if run_id not in counter.ongoing_runs:
                del self.assigned_processes[run_id]
                self.unassigned_processes.append(proc)

        actions: list[Action] = []
        for percept, request_info in requests:
            if request_info.run_id not in self.assigned_processes:
                if self.unassigned_processes:
                    process = self.unassigned_processes.pop()
                else:
                    process = AgentProcess(self.agent_class)
                process.new_run(request_info.run_id, self.agent_config)
                self.assigned_processes[request_info.run_id] = process

            self.assigned_processes[request_info.run_id].send_action_request(percept, request_info)

        for percept, request_info in requests:
            actions.append({
                'run': request_info.run_id,
                'act_no': request_info.action_number,
                'action': self.assigned_processes[request_info.run_id].get_response()
            })

        return actions

    def on_finished_run(self, run_id: str, url: str, outcome: Any):
        if run_id in self.assigned_processes:
            self.assigned_processes[run_id].finish_run(outcome)
            self.unassigned_processes.append(self.assigned_processes[run_id])
            del self.assigned_processes[run_id]
        else:
            super().on_finished_run(run_id, url, outcome)

    def on_message(self, message: Message):
        if message['run'] in self.assigned_processes:
            self.assigned_processes[message['run']].send_message(message['content'], message['type'])
        else:
            super().on_message(message)

    def close(self):
        for proc in self.assigned_processes.values():
            proc.stop()
        for proc in self.unassigned_processes:
            proc.stop()


def _run(
        agent_config: AgentConfig,
        request_processor: RequestProcessor,
        *,
        parallel_runs: bool = True,
        run_limit: Optional[int] = None,
        abandon_old_runs: bool = False,
):
    counter = _RunTracker()

    actions_to_send: list[Action] = []
    to_abandon: list[str] = []

    try:
        while True:
            if to_abandon:
                logger.info(f'Abandoning {len(to_abandon)} old runs: {", ".join(to_abandon)}')
            response = send_request(agent_config, actions_to_send, parallel_runs=parallel_runs, to_abandon=to_abandon)
            to_abandon = []

            for message in response['messages']:
                request_processor.on_message(message)

            counter.update(response)

            if run_limit is not None and counter.number_of_new_runs_finished >= run_limit:
                logger.info(f'Stopping after {run_limit} runs.')
                break

            requests = []
            for ar in response['action_requests']:
                if abandon_old_runs and counter.old_runs and ar['run'] in counter.old_runs:
                    to_abandon.append(ar['run'])
                    continue
                requests.append(
                    (ar['percept'], RequestInfo(get_run_url(agent_config, ar['run']), ar['act_no'], ar['run']))
                )

            for r in requests:
                if r[1].action_number == 0:
                    request_processor.on_new_run(r[1].run_id)

            for run_id, outcome in response['finished_runs'].items():
                request_processor.on_finished_run(run_id, get_run_url(agent_config, run_id), outcome)

            actions_to_send = request_processor.process_requests(requests, counter)

    finally:
        request_processor.close()
        logger.info('Finished.')


def _get_agent_config(agent_config_file: str | Path | AgentConfig) -> AgentConfig:
    if isinstance(agent_config_file, (str, Path)):
        agent_config = json.loads(Path(agent_config_file).read_text())
    elif isinstance(agent_config_file, dict):
        agent_config = agent_config_file
    else:
        raise ValueError('Invalid agent_config_file')
    return agent_config


def run(
        agent_config_file: str | Path | AgentConfig,
        agent: Callable[[Any, RequestInfo], Any],
        *,
        parallel_runs: bool = True,
        processes: int = 1,
        run_limit: Optional[int] = None,
        abandon_old_runs: bool = False,
):
    _run(
        _get_agent_config(agent_config_file),
        SimpleRequestProcessor(agent, processes=processes),
        parallel_runs=parallel_runs,
        run_limit=run_limit,
        abandon_old_runs=abandon_old_runs
    )
