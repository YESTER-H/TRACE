# SPDX-License-Identifier: BSD-3-Clause
"""Durable process diagnostics; no environment, PPO or RNG operations.

Kit fast shutdown can terminate the process inside close(), before Python prints
an uncaught exception. Report first and pass the exit status explicitly. A
close-end event exists in-process only if close returns; an external supervisor
may record observed process exit instead (and must label it as such).
"""
from __future__ import annotations

import functools
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from datetime import datetime, timezone


class _SignalShutdownProxy:
    """Redirect only a signal handler's raw Kit shutdown to status-aware close.

Isaac Sim 6.0.1's SIGINT handler calls SimulationApp._app.shutdown() directly,
which otherwise exits 0 before the wrapping handler can regain control. The
original handler is still invoked; non-shutdown attributes are delegated.
This proxy is installed only while handling a signal, never during training.
"""

    def __init__(self, kit_app, diagnostic):
        self._kit_app = kit_app
        self._diagnostic = diagnostic

    def __getattr__(self, name):
        return getattr(self._kit_app, name)

    def shutdown(self):
        return self._diagnostic.app.close(exit_code=self._diagnostic.exit_code)


class TrainingDiagnostics:
    def __init__(self, app):
        self.app = app
        self.runner = None
        self.expected_final_iteration = None
        self.last_checkpoint_iteration = None
        self.normal_completed = False
        self.exit_code = 0
        self.path = None
        self._fd = None
        self._pending = []
        self._reported_exception = None
        self._original_close = app.close
        if 'exit_code' not in inspect.signature(self._original_close).parameters:
            raise RuntimeError('SimulationApp.close must support explicit exit_code')
        app.close = self.close
        sys.excepthook = self._excepthook
        self.original_signal_handlers = {}
        for name in ('SIGINT', 'SIGTERM', 'SIGABRT', 'SIGSEGV'):
            number = getattr(signal, name, None)
            if number is not None:
                previous = signal.getsignal(number)
                self.original_signal_handlers[number] = previous
                signal.signal(number, self._signal_wrapper(number, previous))
        self.emit('diagnostics_installed', stderr=True,
                  close_signature=str(inspect.signature(self._original_close)),
                  fast_shutdown=app.config.get('fast_shutdown'),
                  original_handlers={signal.Signals(n).name: getattr(h, '__qualname__', str(h))
                                     for n, h in self.original_signal_handlers.items()})

    def set_log_dir(self, directory):
        path = Path(directory) / 'termination_diagnostic.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        for record in self._pending:
            self._write(record)
        self._pending.clear()

    def _write(self, record):
        payload = (json.dumps(record, ensure_ascii=False, default=str) + '\n').encode()
        # One append/write avoids buffered stream locks in Python signal handlers.
        written = os.write(self._fd, payload)
        if written != len(payload):
            raise OSError('Short termination diagnostic write')
        os.fsync(self._fd)

    def emit(self, event, *, stderr=False, **fields):
        record = {'timestamp': datetime.now(timezone.utc).isoformat(),
                  'pid': os.getpid(), 'event': event,
                  'iteration': getattr(self.runner, 'current_learning_iteration', None),
                  'last_checkpoint_iteration': self.last_checkpoint_iteration,
                  **fields}
        if self._fd is None:
            self._pending.append(record)
        else:
            self._write(record)
        if stderr:
            print('[TRAIN-DIAGNOSTIC] ' + json.dumps(record, ensure_ascii=False, default=str),
                  file=sys.stderr, flush=True)
        return record

    def bind_runner(self, runner, iterations, **conditions):
        self.runner = runner
        start = int(runner.current_learning_iteration)
        self.expected_final_iteration = start + int(iterations) - 1
        self.emit('training_start', stderr=True, start_iteration=start,
                  max_iterations=int(iterations), expected_final_iteration=self.expected_final_iteration,
                  **conditions)
        original_save = runner.save

        @functools.wraps(original_save)
        def save(*args, **kwargs):
            original_save(*args, **kwargs)  # Exactly one original call, with unchanged arguments/content.
            self.last_checkpoint_iteration = int(runner.current_learning_iteration)
            fields = {'checkpoint': str(args[0] if args else kwargs.get('path'))}
            # Cheap allocator queries only near the previous shutdown; no CUDA synchronization/RNG.
            if 5650 <= self.last_checkpoint_iteration <= 5750:
                import torch
                if str(runner.device).startswith('cuda'):
                    fields['cuda_allocated_bytes'] = torch.cuda.memory_allocated(runner.device)
                    fields['cuda_reserved_bytes'] = torch.cuda.memory_reserved(runner.device)
            self.emit('checkpoint_saved', **fields)

        runner.save = save

    def verify_completion(self):
        actual = getattr(self.runner, 'current_learning_iteration', None)
        if self.expected_final_iteration is None or actual != self.expected_final_iteration:
            raise RuntimeError(f'Incomplete training: last iteration {actual}, '
                               f'expected {self.expected_final_iteration}')

    def _ensure_failure_file(self):
        if self._fd is None:
            self.set_log_dir(Path('/tmp') / f'cyclo_training_failure_{os.getpid()}')

    def _report_exception(self, kind, value, tb):
        if self._reported_exception is value:
            return
        self._reported_exception = value
        self._ensure_failure_file()
        active_iteration = None
        cursor = tb
        while cursor is not None:
            if cursor.tb_frame.f_code.co_name == 'learn':
                active_iteration = cursor.tb_frame.f_locals.get('it')
            cursor = cursor.tb_next
        self.emit('caught_exception', stderr=True, exception_type=kind.__name__,
                  exception_message=str(value), traceback=''.join(traceback.format_exception(kind, value, tb)),
                  active_iteration=active_iteration, exit_code=self.exit_code)
        traceback.print_exception(kind, value, tb, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()

    def _exception_code(self, value):
        if self.exit_code:
            return self.exit_code
        if isinstance(value, KeyboardInterrupt):
            return 128 + signal.SIGINT
        if isinstance(value, SystemExit) and isinstance(value.code, int) and value.code != 0:
            return value.code
        return 1  # Includes premature SystemExit(0).

    def _excepthook(self, kind, value, tb):
        self.exit_code = self._exception_code(value)
        self._report_exception(kind, value, tb)
        self.close(exit_code=self.exit_code)
        raise SystemExit(self.exit_code)

    def _signal_wrapper(self, number, original):
        def handler(received, frame):
            self.exit_code = 128 + received
            self._ensure_failure_file()
            self.emit('received_signal', stderr=True, signal_number=received,
                      signal_name=signal.Signals(received).name, exit_code=self.exit_code,
                      original_handler=getattr(original, '__qualname__', str(original)))
            kit_app = self.app._app
            self.app._app = _SignalShutdownProxy(kit_app, self)
            try:
                if callable(original):
                    original(received, frame)
                else:
                    self.close(exit_code=self.exit_code)
            finally:
                self.app._app = kit_app
            # Covers a returning original handler or non-fast-shutdown runtime.
            raise SystemExit(self.exit_code)
        return handler

    def close(self, wait_for_replicator=True, skip_cleanup=False, exit_code=0):
        status = self.exit_code or exit_code or (0 if self.normal_completed else 1)
        self.exit_code = status
        if status:
            self._ensure_failure_file()
        self.emit('simulation_app_close_begin', stderr=True, exit_code=status,
                  normal_completed=self.normal_completed, caller_stack=''.join(traceback.format_stack()))
        sys.stdout.flush()
        sys.stderr.flush()
        self._original_close(wait_for_replicator=wait_for_replicator, skip_cleanup=skip_cleanup, exit_code=status)
        self.emit('simulation_app_close_end', stderr=True, exit_code=status, close_returned=True)

    def run(self, callback):
        try:
            callback()
            self.verify_completion()
        except BaseException as error:
            self.exit_code = self._exception_code(error)
            self._report_exception(type(error), error, error.__traceback__)
            try:
                self.close(exit_code=self.exit_code)
            except BaseException as close_error:
                self._report_exception(type(close_error), close_error, close_error.__traceback__)
            raise SystemExit(self.exit_code)
        else:
            self.normal_completed = True
            self.emit('normal_training_complete', stderr=True,
                      expected_final_iteration=self.expected_final_iteration, exit_code=0)
            self.close(exit_code=0)
