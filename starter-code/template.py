"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
import sys
import unicodedata
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
Nếu công cụ gặp lỗi hai lần, dừng và trả lời thông báo lỗi.
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def query(self, user_input: str) -> dict:
        return {
            "status": "success",
            "answer": "Tôi không có kết nối dữ liệu để tra cứu chuyến bay hoặc thời tiết. "
                      "Vui lòng kiểm tra với hãng hàng không và dịch vụ thời tiết.",
            "tool_calls": [],
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
        return "".join(char for char in text if not unicodedata.combining(char))

    def _plan(self, user_input: str):
        """Deterministic planner for the lab; no external LLM is required."""
        text = self._normalize(user_input)
        if any(word in text for word in ("chinh sach", "doi tra", "hoan ve")):
            return [], ("Chính sách đổi trả vé Vinpearl phụ thuộc điều kiện vé. "
                        "Vui lòng liên hệ nơi đặt vé để xác nhận phí và điều kiện áp dụng.")
        cities = re.findall(r"\b(?:han|sgn|dad)\b", text)
        if not cities:
            aliases = {"ha noi": "han", "ho chi minh": "sgn", "sai gon": "sgn", "da nang": "dad"}
            cities = [code for _, code in sorted(
                (text.index(name), code) for name, code in aliases.items() if name in text
            )]
        flight = any(word in text for word in ("chuyen bay", "ve", "flight"))
        weather = any(word in text for word in ("thoi tiet", "mac gi", "weather"))
        if flight and len(cities) < 2:
            return [], "Vui lòng cung cấp mã sân bay đi và đến (HAN, SGN hoặc DAD)."
        if weather and not cities:
            return [], "Vui lòng cung cấp thành phố cần tra cứu thời tiết."
        actions = []
        if flight:
            budget = re.search(r"(?:duoi|toi da|gia|under)\s*(\d+(?:[.,]\d+)?)\s*(trieu|million)?", text)
            max_price = 5000000
            if budget:
                max_price = int(float(budget[1].replace(",", ".")) * (1000000 if budget[2] else 1))
            actions.append({"name": "get_flight_info", "args": {
                "origin": cities[0].upper(), "destination": cities[1].upper(), "max_price": max_price,
            }})
        if weather:
            actions.append({"name": "get_weather_forecast", "args": {"city_code": cities[-1].upper()}})
        return actions, "Tôi có thể hỗ trợ tra cứu chuyến bay và thời tiết theo dữ liệu bài lab."

    @staticmethod
    def _execute_action(action_json: str):
        try:
            action = json.loads(action_json)
        except (json.JSONDecodeError, TypeError):
            return {"error": "Invalid JSON format"}
        if not isinstance(action, dict) or not isinstance(action.get("name"), str) or not isinstance(action.get("args"), dict):
            return {"error": "Invalid action: expected name and args"}
        name = action["name"].strip().lower()
        if name not in TOOL_MAP:
            return {"error": f"Unknown tool: {name}"}
        try:
            return TOOL_MAP[name](**action["args"])
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _summarize(observations):
        parts = []
        for name, result in observations:
            if isinstance(result, dict) and "error" in result:
                parts.append(f"Không thể tra cứu: {result['error']}")
            elif name == "get_flight_info":
                parts.append("\n".join(
                    f"{fl['flight_number']} ({fl['airline']}): {fl['origin']} → {fl['destination']}, "
                    f"{fl['departure_time']}, {fl['price_vnd']:,} VND." for fl in result
                ) or "Không tìm thấy chuyến bay phù hợp ngân sách.")
            else:
                parts.append(f"{result['city']}: {result['temperature_c']}°C, {result['condition']}. "
                             f"{result['recommendation']}")
        return "\n".join(parts)

    def run(self, user_input: str) -> dict:
        self.trace = []
        actions, fallback = self._plan(user_input)
        observations = []
        iteration = 0
        action_index = 0
        errors = 0
        while iteration < self.max_iterations:
            iteration += 1
            entry = {"iteration": iteration}
            if action_index < len(actions):
                action = actions[action_index]
                observation = self._execute_action(json.dumps(action))
                entry.update(thought=f"Tra cứu dữ liệu bằng {action['name']}.",
                             action=action, observation=observation)
                self.trace.append(entry)
                failed = isinstance(observation, dict) and "error" in observation
                if failed:
                    errors += 1
                    if errors < 2:
                        continue
                    observations.append((action["name"], observation))
                else:
                    observations.append((action["name"], observation))
                    action_index += 1
                if not failed and len(actions) > 1:
                    continue
            else:
                entry.update(thought="Tổng hợp câu trả lời từ dữ liệu đã thu thập.",
                             action=None, observation=None)
                self.trace.append(entry)
            answer = self._summarize(observations) if observations else fallback
            entry["final_answer"] = answer
            return {"status": "completed", "answer": answer,
                    "iterations": iteration, "trace": self.trace}
        return {"status": "max_iterations_reached",
                "answer": "Không thể hoàn thành trong số bước tối đa.",
                "iterations": iteration, "trace": self.trace}

def main():
    # Support Vietnamese output on Windows terminals and redirected output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
