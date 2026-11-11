# SPDX-License-Identifier: BSD-3-Clause
"""Play stages and status-aware shutdown, reusing proven training diagnostics.

Only process/logging state is touched; no policy, environment or RNG operations.
Play diagnostics are always written outside the checkpoint's training directory.
"""
import json
import sys
import tempfile

from training_diagnostics import TrainingDiagnostics


class PlayDiagnostics(TrainingDiagnostics):
    def __init__(self, app, directory=None):
        self.phase = "app_launched"
        super().__init__(app)
        self.directory = directory or tempfile.mkdtemp(prefix="cyclo_play_")
        self.set_log_dir(self.directory)
        self.stage("app_launched", diagnostic_directory=self.directory)

    def emit(self, event, *, stderr=False, **fields):
        record = super().emit(event, stderr=False, play_phase=self.phase, **fields)
        if stderr:
            print("[PLAY-DIAGNOSTIC] " + json.dumps(record, ensure_ascii=False, default=str),
                  file=sys.stderr, flush=True)
        return record

    def stage(self, phase, **fields):
        self.phase = phase
        self.emit("play_stage", stderr=True, **fields)

    def invoke(self, callback, *args, **kwargs):
        # Report the original exception inside Hydra as well, before Hydra can
        # replace it with SystemExit and before Kit's fast shutdown is entered.
        try:
            return callback(*args, **kwargs)
        except BaseException as error:
            self.exit_code = self._exception_code(error)
            self._report_exception(type(error), error, error.__traceback__)
            raise

    def run(self, callback):
        try:
            self.invoke(callback)
        except BaseException as error:
            self.exit_code = self._exception_code(error)
            try:
                self.close(exit_code=self.exit_code)
            except BaseException as close_error:
                self._report_exception(type(close_error), close_error, close_error.__traceback__)
            raise SystemExit(self.exit_code)
        else:
            self.normal_completed = True
            self.emit("normal_play_complete", stderr=True, exit_code=0)
            self.close(exit_code=0)
