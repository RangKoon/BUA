# logged_agent.py
import json
import time
from datetime import datetime
from browser_use import Agent
from browser_use.llm.messages import BaseMessage
from browser_use.agent.views import AgentOutput

class LoggedAgent(Agent):
    def __init__(self, *args, log_path="training_data.jsonl", **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self._step_start_time: float | None = None
        self._step_index: int = 0
        self._last_llm_end: float | None = None
        self._last_action_exec_s: float | None = None

    def _log_training_data(self, input_messages: list[BaseMessage], parsed: AgentOutput, timing: dict):
        try:
            data = {
                "step": self._step_index,
                "timestamp": datetime.now().isoformat(),
                "timing": timing,
                "input": [m.model_dump() for m in input_messages],
                "output": parsed.model_dump()
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to log training data: {e}")

    async def step(self, *args, **kwargs):
        self._step_start_time = time.time()
        result = await super().step(*args, **kwargs)
        if self._last_llm_end is not None:
            self._last_action_exec_s = round(time.time() - self._last_llm_end, 3)
        return result

    async def get_model_output(self, input_messages):
        llm_start = time.time()
        parsed = await super().get_model_output(input_messages)
        llm_end = time.time()

        browser_state_s = round(llm_start - self._step_start_time, 3) if self._step_start_time else None
        step_total = round(llm_end - self._step_start_time, 3) if self._step_start_time else None

        timing = {
            "step_total_s": step_total,
            "prev_action_exec_s": self._last_action_exec_s,
            "browser_state_s": browser_state_s,
            "llm_inference_s": round(llm_end - llm_start, 3),
        }

        self._last_llm_end = llm_end
        self._log_training_data(input_messages, parsed, timing)
        self._step_index += 1
        return parsed

    async def run(self, *args, **kwargs):
        self._step_start_time = time.time()
        self._step_index = 0
        self._last_llm_end = None
        self._last_action_exec_s = None
        return await super().run(*args, **kwargs)