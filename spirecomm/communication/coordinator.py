import os
import sys
import io
import queue
import threading
import json
import collections
import time

from spirecomm.spire.game import Game
from spirecomm.spire.screen import ScreenType
from spirecomm.communication.action import Action, StartGameAction

_STDOUT_WRITE_LOCK = threading.Lock()


def _coordinator_log_path():
    raw = os.environ.get("SPIRECOMM_COORDINATOR_LOG")
    return raw if raw else None


def _append_coordinator_log(event: str, payload: str = ""):
    path = _coordinator_log_path()
    if not path:
        return
    line = f"{time.time():.6f} {event}"
    if payload:
        line += f" {payload}"
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def _emit_stdout_line(output: str):
    line = f"{output}\n"
    try:
        fileno = sys.stdout.fileno()
    except (AttributeError, OSError, io.UnsupportedOperation, ValueError):
        fileno = None
    if fileno is not None:
        os.write(fileno, line.encode("utf-8"))
        return
    sys.stdout.write(line)
    sys.stdout.flush()


def read_stdin(input_queue):
    """Read lines from stdin and write them to a queue

    :param input_queue: A queue, to which lines from stdin will be written
    :type input_queue: queue.Queue
    :return: None
    """
    while True:
        stdin_input = sys.stdin.readline()
        if stdin_input == "":
            time.sleep(0.05)
            continue
        stdin_input = stdin_input.rstrip("\n")
        if stdin_input.endswith("\r"):
            stdin_input = stdin_input[:-1]
        _append_coordinator_log("read_stdin", repr(stdin_input))
        input_queue.put(stdin_input)


def write_stdout(output_queue):
    """Read lines from a queue and write them to stdout

    :param output_queue: A queue, from which this function will receive lines of text
    :type output_queue: queue.Queue
    :return: None
    """
    while True:
        output = output_queue.get()
        _append_coordinator_log("write_stdout", repr(output))
        try:
            with _STDOUT_WRITE_LOCK:
                _emit_stdout_line(output)
            _append_coordinator_log("write_stdout_done", repr(output))
        except OSError as exc:
            _append_coordinator_log("write_stdout_error", repr(exc))
            raise


class Coordinator:
    """An object to coordinate communication with Slay the Spire"""

    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.input_thread = threading.Thread(target=read_stdin, args=(self.input_queue,))
        self.output_thread = threading.Thread(target=write_stdout, args=(self.output_queue,))
        self.input_thread.daemon = True
        self.input_thread.start()
        self.output_thread.daemon = True
        self.output_thread.start()
        self.action_queue = collections.deque()
        self.state_change_callback = None
        self.out_of_game_callback = None
        self.error_callback = None
        self.game_is_ready = False
        self.stop_after_run = False
        self.in_game = False
        self.last_game_state = None
        self.last_error = None
        self.last_raw_message = None
        self.last_communication_state = None
        self.last_communication_source = None
        self.last_raw_game_state = None
        self.raw_message_sequence = 0
        self.raw_message_callback = None

    def _ingest_communication_state(self, communication_state, *, raw_message=None, source="pipe"):
        self.last_raw_message = raw_message
        self.last_communication_state = communication_state
        self.last_communication_source = source
        self.last_raw_game_state = communication_state.get("game_state")
        self.raw_message_sequence += 1
        if self.raw_message_callback is not None:
            self.raw_message_callback(
                {
                    "sequence": self.raw_message_sequence,
                    "raw_message": raw_message,
                    "communication_state": communication_state,
                    "source": source,
                }
            )
        self.last_error = communication_state.get("error", None)
        self.game_is_ready = communication_state.get("ready_for_command")
        if self.last_error is None:
            self.in_game = communication_state.get("in_game")
            if self.in_game:
                self.last_game_state = Game.from_json(
                    communication_state.get("game_state"),
                    communication_state.get("available_commands"),
                )

    def ingest_communication_state(self, communication_state, *, raw_message=None, source="tap"):
        self._ingest_communication_state(communication_state, raw_message=raw_message, source=source)

    def signal_ready(self):
        """Indicate to Communication Mod that setup is complete

        Must be used once, before any other commands can be sent.
        :return: None
        """
        self.send_message("ready")

    def send_message(self, message):
        """Send a command to Communication Mod and start waiting for a response

        :param message: the message to send
        :type message: str
        :return: None
        """
        _append_coordinator_log("send_message", repr(message))
        self.output_queue.put(message)
        self.game_is_ready = False

    def send_message_immediate(self, message):
        """Synchronously send a command to Communication Mod.

        Replay bridge probes sometimes need to write a command immediately from
        the main thread instead of enqueueing it behind the coordinator's async
        output loop.
        """
        _append_coordinator_log("send_message_immediate", repr(message))
        with _STDOUT_WRITE_LOCK:
            _emit_stdout_line(message)
        _append_coordinator_log("send_message_immediate_done", repr(message))
        self.game_is_ready = False

    def add_action_to_queue(self, action):
        """Queue an action to perform when ready

        :param action: the action to queue
        :type action: Action
        :return: None
        """
        if action is None:
            from spirecomm.communication.action import StateAction
            action = StateAction()
        self.action_queue.append(action)

    def clear_actions(self):
        """Remove all actions from the action queue

        :return: None
        """
        self.action_queue.clear()

    def execute_next_action(self):
        """Immediately execute the next action in the action queue

        :return: None
        """
        action = self.action_queue.popleft()
        action.execute(self)

    def execute_next_action_if_ready(self):
        """Immediately execute the next action in the action queue, if ready to do so

        :return: None
        """
        if len(self.action_queue) > 0 and self.action_queue[0].can_be_executed(self):
            self.execute_next_action()

    def register_state_change_callback(self, new_callback):
        """Register a function to be called when a message is received from Communication Mod

        :param new_callback: the function to call
        :type new_callback: function(game_state: Game) -> Action
        :return: None
        """
        self.state_change_callback = new_callback

    def register_command_error_callback(self, new_callback):
        """Register a function to be called when an error is received from Communication Mod

        :param new_callback: the function to call
        :type new_callback: function(error: str) -> Action
        :return: None
        """
        self.error_callback = new_callback

    def register_out_of_game_callback(self, new_callback):
        """Register a function to be called when Communication Mod indicates we are in the main menu

        :param new_callback: the function to call
        :type new_callback: function() -> Action
        :return: None
        """
        self.out_of_game_callback = new_callback

    def register_raw_message_callback(self, new_callback):
        """Register a function to receive every raw Communication Mod payload."""
        self.raw_message_callback = new_callback

    def get_next_raw_message(self, block=False, timeout=None):
        """Get the next message from Communication Mod as a string

        :param block: set to True to wait for the next message
        :type block: bool
        :return: the message from Communication Mod
        :rtype: str
        """
        if not block and timeout is None and self.input_queue.empty():
            return None
        try:
            should_block = block or timeout is not None
            if timeout is None:
                return self.input_queue.get(should_block)
            return self.input_queue.get(should_block, timeout=timeout)
        except queue.Empty:
            return None

    def receive_game_state_update(self, block=False, perform_callbacks=True, timeout=None):
        """Using the next message from Communication Mod, update the stored game state

        :param block: set to True to wait for the next message
        :type block: bool
        :param perform_callbacks: set to True to perform callbacks based on the new game state
        :type perform_callbacks: bool
        :return: whether a message was received
        """
        message = self.get_next_raw_message(block, timeout=timeout)
        if message is not None:
            communication_state = json.loads(message)
            self._ingest_communication_state(communication_state, raw_message=message, source="pipe")
            if perform_callbacks:
                if self.last_error is not None:
                    self.action_queue.clear()
                    if self.last_error.startswith("Invalid command:"):
                        from spirecomm.communication.action import StateAction
                        new_action = StateAction()
                    else:
                        new_action = self.error_callback(self.last_error)
                    self.add_action_to_queue(new_action)
                elif self.in_game:
                    if len(self.action_queue) == 0 and perform_callbacks:
                        new_action = self.state_change_callback(self.last_game_state)
                        self.add_action_to_queue(new_action)
                elif self.stop_after_run:
                    self.clear_actions()
                else:
                    new_action = self.out_of_game_callback()
                    self.add_action_to_queue(new_action)
            return True
        return False

    def run(self):
        """Start executing actions forever

        :return: None
        """
        while True:
            self.execute_next_action_if_ready()
            self.receive_game_state_update(perform_callbacks=True)

    def wait_for_command_state(self, block=True, max_updates=500):
        """Wait until Communication Mod reports a fresh command-ready state or error."""
        updates = 0
        while updates < max_updates:
            received = self.receive_game_state_update(block=block, perform_callbacks=False)
            if not received:
                if not block:
                    break
                continue
            updates += 1
            if self.last_error is not None:
                return True
            if self.game_is_ready:
                return True
        return False

    def send_and_wait(self, message, max_updates=500):
        """Send a raw message, then wait for the next command-ready state."""
        self.send_message(message)
        return self.wait_for_command_state(block=True, max_updates=max_updates)

    def play_one_game(self, player_class, ascension_level=0, seed=None):
        """

        :param player_class: the class to play
        :type player_class: PlayerClass
        :param ascension_level: the ascension level to use
        :type ascension_level: int
        :param seed: the alphanumeric seed to use
        :type seed: str
        :return: True if the game was a victory, else False
        :rtype: bool
        """
        self.clear_actions()
        while not self.game_is_ready:
            self.receive_game_state_update(block=True, perform_callbacks=False)
        if not self.in_game:
            StartGameAction(player_class, ascension_level, seed).execute(self)
            self.receive_game_state_update(block=True)
        while self.in_game:
            self.execute_next_action_if_ready()
            self.receive_game_state_update()
        if self.last_game_state.screen_type == ScreenType.GAME_OVER:
            return self.last_game_state.screen.victory
        else:
            return False
