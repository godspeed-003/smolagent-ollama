#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mimetypes
import os
import re
import shutil
from typing import Optional
import threading
import time

class AgentManager:
    def __init__(self, agent):
        self.agent = agent
        self.current_thread = None
        self.stop_event = threading.Event()

    def reset(self):
        if self.current_thread and self.current_thread.is_alive():
            self.stop_event.set()
            self.current_thread.join(timeout=0.5)
        self.stop_event = threading.Event()
        self.current_thread = None

    def interact_with_agent(self, prompt, messages):
        import gradio as gr
        self.reset()

        def get_message_content(msg):
            if hasattr(msg, 'content'):
                return msg.content
            elif isinstance(msg, dict) and 'content' in msg:
                return msg['content']
            return None

        # Expanded: catch more natural greetings and bypass agent pipeline
        greeting_prefixes = ("hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon", "yo", "what's up")
        prompt_clean = prompt.strip().lower()
        if any(prompt_clean.startswith(greet) for greet in greeting_prefixes):
            if not messages or get_message_content(messages[-1]) != prompt:
                messages.append(gr.ChatMessage(role="user", content=prompt))
            content = "Hello! How can I help you today?"
            messages.append(gr.ChatMessage(role="assistant", content=content))
            yield messages
            return

        if not messages or get_message_content(messages[-1]) != prompt:
            messages.append(gr.ChatMessage(role="user", content=prompt))

        result_queue = []
        exception_holder = [None]

        def agent_worker():
            try:
                for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
                    if self.stop_event.is_set():
                        break
                    result_queue.append(msg)
            except Exception as e:
                exception_holder[0] = e

        self.current_thread = threading.Thread(target=agent_worker)
        self.current_thread.start()

        last_yielded = 0
        while self.current_thread.is_alive() or last_yielded < len(result_queue):
            if self.stop_event.is_set():
                messages.append(gr.ChatMessage(role="assistant", content="⏹️ Stopped, new message received."))
                yield messages
                break
            while last_yielded < len(result_queue):
                msg = result_queue[last_yielded]
                last_yielded += 1
                messages.append(msg)
                yield messages
            time.sleep(0.05)
        if exception_holder[0] is not None:
            messages.append(gr.ChatMessage(role="assistant", content=f"⚠️ Agent error: {exception_holder[0]}"))
            yield messages

        messages.clear()
        if hasattr(self.agent, 'reset') and callable(getattr(self.agent, 'reset')):
            self.agent.reset()
        if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear') and callable(getattr(self.agent.memory, 'clear')):
            self.agent.memory.clear()
        if hasattr(self.agent, 'history') and hasattr(self.agent.history, 'clear') and callable(getattr(self.agent.history, 'clear')):
            self.agent.history.clear()
        return

# Import tool functions/objects from app.py
from app import get_current_time_in_timezone, web_search_tool, visit_webpage_tool, image_generation_tool, my_custom_tool

from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available


def pull_messages_from_step(
    step_log: MemoryStep,
):
    """Extract ChatMessage objects from agent steps with proper nesting, suppressing redundant/empty code blocks."""
    import gradio as gr
    import re
    
    # Use the is_similar function from GradioUI.interact_with_agent
    def is_similar(a, b):
        import re
        def normalize(s):
            return re.sub(r'\W+', '', s.lower().strip())
        return normalize(a) == normalize(b)

    last_output = None
    redundant_count = 0
    REDUNDANT_LIMIT = 3

    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
            model_output = model_output.strip()
            # Suppress empty outputs
            if model_output:
                # Suppress redundant outputs
                if last_output and is_similar(model_output, last_output):
                    redundant_count += 1
                    if redundant_count > REDUNDANT_LIMIT:
                        yield gr.ChatMessage(role="assistant", content="⚠️ Repeated output suppressed.")
                        return
                else:
                    redundant_count = 0
                    last_output = model_output
                    yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            # Only wrap in code block if content is non-empty and looks like code
            def looks_like_code(text):
                # Heuristic: contains newlines, indentation, or starts with 'def', 'class', or import
                if not text:
                    return False
                code_keywords = ("def ", "class ", "import ", "for ", "while ", "if ", "elif ", "else:")
                return (any(text.lstrip().startswith(kw) for kw in code_keywords) or
                        '\n' in text or
                        re.search(r"[{}=]", text))

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(r"```.*?\n", "", content)
                content = re.sub(r"\s*<end_code>\s*", "", content)
                content = content.strip()
                if content and looks_like_code(content):
                    if not content.startswith("```python"):
                        content = f"```python\n{content}\n```"
                # Suppress empty or non-code outputs
                elif not content:
                    content = None
            # Suppress redundant tool outputs
            if content and last_output and is_similar(content, last_output):
                redundant_count += 1
                if redundant_count > REDUNDANT_LIMIT:
                    yield gr.ChatMessage(role="assistant", content="⚠️ Repeated tool output suppressed.")
                    return
            elif content:
                redundant_count = 0
                last_output = content
                parent_message_tool = gr.ChatMessage(
                    role="assistant",
                    content=content,
                    metadata={
                        "title": f"🛠️ Used tool {first_tool_call.name}",
                        "id": parent_id,
                        "status": "pending",
                    },
                )
                yield parent_message_tool

            # Nesting execution logs under the tool call if they exist
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"{log_content}",
                        metadata={"title": "📝 Execution Logs", "parent_id": parent_id, "status": "done"},
                    )

            # Nesting any errors under the tool call
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={"title": "💥 Error", "parent_id": parent_id, "status": "done"},
                )

            # Update parent message metadata to done status without yielding a new message
            # parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = (
                f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            )
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        step_footnote = f"""<span style=\"color: #bbbbc2; font-size: 12px;\">{step_footnote}</span> """
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        yield gr.ChatMessage(role="assistant", content="-----")


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr

    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count") and agent.model.last_input_token_count is not None:
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, ActionStep):
                if agent.model.last_input_token_count is not None:
                    step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    # Only show the actual answer, not the repr of FinalAnswerStep or similar wrappers
    def extract_final_answer(ans):
        # For objects like FinalAnswerStep(final_answer='...'), extract the string
        if hasattr(ans, 'final_answer'):
            return getattr(ans, 'final_answer')
        if hasattr(ans, 'to_string') and callable(ans.to_string):
            return ans.to_string()
        if isinstance(ans, str):
            return ans
        return str(ans)

    answer_text = extract_final_answer(final_answer)
    if isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": answer_text, "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": answer_text, "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(role="assistant", content=answer_text)

class GradioUI:
    """A one-line interface to launch your agent in Gradio"""
    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent_manager = AgentManager(agent)
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages):
        # Delegate to AgentManager
        yield from self.agent_manager.interact_with_agent(prompt, messages)

    def launch(self, **kwargs):
        import gradio as gr

        # Cancel any running agent thread
        if agent_thread is not None and agent_thread.is_alive():
            agent_stop_event.set()
            agent_thread.join(timeout=1)
        agent_stop_event = threading.Event()

        # Helper for UI message extraction
        def get_message_content(msg):
            if hasattr(msg, 'content'):
                return msg.content
            elif isinstance(msg, dict) and 'content' in msg:
                return msg['content']
            return None

        # Expanded: catch more natural greetings and bypass agent pipeline
        greeting_prefixes = ("hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon", "yo", "what's up")
        prompt_clean = prompt.strip().lower()
        if any(prompt_clean.startswith(greet) for greet in greeting_prefixes):
            if not messages or get_message_content(messages[-1]) != prompt:
                messages.append(gr.ChatMessage(role="user", content=prompt))
            # Friendly, simple reply
            content = "Hello! How can I help you today?"
            messages.append(gr.ChatMessage(role="assistant", content=content))
            yield messages
            return

        if not messages or get_message_content(messages[-1]) != prompt:
            messages.append(gr.ChatMessage(role="user", content=prompt))

        # --- Actual agent execution in a background thread with cancellation ---
        result_queue = []
        exception_holder = [None]

        def agent_worker():
            try:
                # Use the same logic as before, but check agent_stop_event periodically
                for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
                    if agent_stop_event.is_set():
                        break
                    result_queue.append(msg)
            except Exception as e:
                exception_holder[0] = e

        agent_thread = threading.Thread(target=agent_worker)
        agent_thread.start()

        # Poll results and yield to UI
        last_yielded = 0
        while agent_thread.is_alive() or last_yielded < len(result_queue):
            if agent_stop_event.is_set():
                messages.append(gr.ChatMessage(role="assistant", content="⏹️ Stopped, new message received."))
                yield messages
                break
            # Yield new messages as they come in
            while last_yielded < len(result_queue):
                msg = result_queue[last_yielded]
                last_yielded += 1
                messages.append(msg)
                yield messages
            time.sleep(0.05)
        # If there was an exception, yield it
        if exception_holder[0] is not None:
            messages.append(gr.ChatMessage(role="assistant", content=f"⚠️ Agent error: {exception_holder[0]}"))
            yield messages

        # After agent completes or errors, reset messages so user can keep chatting
        messages.clear()
        if hasattr(self.agent, 'reset') and callable(getattr(self.agent, 'reset')):
            self.agent.reset()
        if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear') and callable(getattr(self.agent.memory, 'clear')):
            self.agent.memory.clear()
        if hasattr(self.agent, 'history') and hasattr(self.agent.history, 'clear') and callable(getattr(self.agent.history, 'clear')):
            self.agent.history.clear()
        return

        # Web search tool (force for product/info queries)
        search_patterns = [
            r"search for (.+)",
            r"look up (.+)",
            r"find on the web (.+)",
            r"web search (.+)",
            r"google (.+)",
            r"duckduckgo (.+)",
            r"review (.+)",
            r"compare (.+)",
            r"buy (.+)",
            r"price of (.+)",
            r"flipkart (.+)",
            r"amazon (.+)",
            r"should I get (.+)",
            r"is (.+) worth it",
            r"pros and cons of (.+)",
            r"best (.+)",
        ]
        forced_search_tool = False
        for pattern in search_patterns:
            match = re.search(pattern, prompt.strip(), re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                answer = web_search_tool(query)
                # Synthesize the answer using the LLM
                synthesis_prompt = (
                    f"Based on these search results, answer the user's question as a helpful assistant. "
                    f"Do NOT output code unless specifically asked. "
                    f"Question: {prompt}\n"
                    f"Search Results: {answer}"
                )
                synthesized = self.agent.model(synthesis_prompt)
                synthesized_content = getattr(synthesized, 'content', str(synthesized))
                messages.append(gr.ChatMessage(role="assistant", content=synthesized_content))
                yield messages
                forced_search_tool = True
                break
        if forced_search_tool:
            # After agent completes or errors, reset messages so user can keep chatting
            messages.clear()
            # Robustly reset agent memory/context after every response
            if hasattr(self.agent, 'reset') and callable(getattr(self.agent, 'reset')):
                self.agent.reset()
            if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear') and callable(getattr(self.agent.memory, 'clear')):
                self.agent.memory.clear()
            if hasattr(self.agent, 'history') and hasattr(self.agent.history, 'clear') and callable(getattr(self.agent.history, 'clear')):
                self.agent.history.clear()
            return

        # --- Retry/loop protection and code parsing robustness ---
        # (Wrap the agent call in a retry loop; if repeated error or output, break and show fallback message)
        MAX_AGENT_RETRIES = 2
        last_agent_output = None
        retry_count = 0
        parse_error_count = 0
        # --- Strict one-active-generation enforcement ---
        if not hasattr(self, 'active_generation_id'):
            self.active_generation_id = 0
        self.active_generation_id += 1
        my_generation_id = self.active_generation_id
        last_output = None
        repeat_count = 0
        MAX_REPEAT = 3
        def is_similar(a, b):
            # Simple similarity: ignore case, whitespace, punctuation differences
            import re
            def normalize(s):
                return re.sub(r'\W+', '', s.lower().strip())
            return normalize(a) == normalize(b)
        try:
            for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
                # If a new message arrived (newer generation started), stop this one immediately
                if my_generation_id != self.active_generation_id:
                    messages.append(gr.ChatMessage(role="assistant", content="⏹️ Stopped, new message received."))
                    yield messages
                    break
                msg_content = getattr(msg, 'content', str(msg))
                user_wants_code = any(
                    kw in prompt.lower() for kw in ["code", "python", "script", "show code", "give code"]
                )
                # If code block and not requested, treat as parse error
                if msg_content.strip().startswith("```") and not user_wants_code:
                    retry_count += 1
                    if retry_count >= MAX_AGENT_RETRIES:
                        messages.append(gr.ChatMessage(role="assistant", content="⚠️ Sorry, the agent kept outputting code. Please rephrase your question or try again."))
                        yield messages
                        break
                    continue
                # If error message about code parsing and user didn't ask for code, treat as valid plain text
                elif "Error in code parsing" in msg_content and not user_wants_code:
                    parse_error_count += 1
                    if parse_error_count >= MAX_AGENT_RETRIES:
                        messages.append(gr.ChatMessage(role="assistant", content="Sorry, I couldn't parse code because your query doesn't require code. Here's the information in plain text. If you want code, please specify in your question."))
                        yield messages
                        break
                    continue
                else:
                    retry_count = 0
                    parse_error_count = 0
                # Suppress repeated/similar outputs
                if last_output is not None and is_similar(msg_content, last_output):
                    repeat_count += 1
                    if repeat_count >= MAX_REPEAT:
                        messages.append(gr.ChatMessage(role="assistant", content="⚠️ Agent stopped due to repeated similar answers. If you want more details, please rephrase your question."))
                        yield messages
                        break
                    continue
                else:
                    repeat_count = 0
                last_output = msg_content
                # Only append code if user wants code, otherwise filter out code blocks
                if msg_content.strip().startswith("```") and not user_wants_code:
                    continue
                messages.append(msg)
                yield messages
            yield messages
        except Exception as e:
            import gradio as gr
            error_message = f"⚠️ Error during agent execution: {str(e)}\n\nYou can continue chatting or try a different query."
            messages.append(gr.ChatMessage(role="assistant", content=error_message))
            yield messages

        # --- ADVANCED: True process-level cancellation ---
        # To truly kill a long-running agent/tool/LLM call, you would need to run each call in a subprocess
        # and forcibly terminate the process on new user input. This is a complex change and not implemented here.

        # After agent completes or errors, reset messages so user can keep chatting
        messages.clear()
        # Robustly reset agent memory/context after every response
        if hasattr(self.agent, 'reset') and callable(getattr(self.agent, 'reset')):
            self.agent.reset()
        if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear') and callable(getattr(self.agent.memory, 'clear')):
            self.agent.memory.clear()
        if hasattr(self.agent, 'history') and hasattr(self.agent.history, 'clear') and callable(getattr(self.agent.history, 'clear')):
            self.agent.history.clear()

        # Visit webpage tool
        visit_patterns = [
            r"visit (https?://[^ ]+)",
            r"open (https?://[^ ]+)",
            r"read (https?://[^ ]+)"
        ]
        for pattern in visit_patterns:
            match = re.search(pattern, prompt.strip(), re.IGNORECASE)
            if match:
                url = match.group(1).strip()
                scraped_content = visit_webpage_tool.forward(url)
                # Always pass scraped content through LLM with strict prompt
                llm_prompt = (
                    f"You are a helpful assistant. The user asked: '{prompt}'.\n"
                    f"Below is the content of the website they want information from.\n"
                    f"Extract and show only the information relevant to the user's question. "
                    f"Do NOT output code unless the user specifically asked for code. "
                    f"If the answer is not present, say so.\n"
                    f"---\n"
                    f"Website Content:\n{scraped_content}\n---"
                )
                # Retry/guardrails: up to 2 attempts to get a non-code answer unless code is requested
                MAX_LLM_RETRIES = 2
                user_wants_code = any(
                    kw in prompt.lower() for kw in ["code", "python", "script", "show code", "give code"]
                )
                for attempt in range(MAX_LLM_RETRIES):
                    llm_response = self.agent.model(llm_prompt)
                    llm_content = getattr(llm_response, 'content', str(llm_response)).strip()
                    # Post-processing: If code block and not explicitly requested, re-prompt or filter
                    if llm_content.startswith("```") and not user_wants_code:
                        if attempt < MAX_LLM_RETRIES - 1:
                            llm_prompt += "\nREMINDER: Do NOT output code. Only answer in plain text."
                            continue
                        else:
                            llm_content = "⚠️ Sorry, the assistant kept outputting code. Please rephrase your question or try again."
                    messages.append(gr.ChatMessage(role="assistant", content=llm_content))
                    yield messages
                    return
                return

        # Image generation tool
        image_patterns = [
            r"draw (.+)",
            r"generate image of (.+)",
            r"create picture of (.+)",
            r"make an image of (.+)",
            r"image of (.+)"
        ]
        for pattern in image_patterns:
            match = re.search(pattern, prompt.strip(), re.IGNORECASE)
            if match:
                prompt_img = match.group(1).strip()
                answer = image_generation_tool(prompt_img)
                messages.append(gr.ChatMessage(role="assistant", content=f"```python\n{answer}\n```"))
                yield messages
                return

        # Fun fact tool
        fun_fact_patterns = [
            r"fun fact",
            r"tell me something interesting",
            r"surprise me",
            r"random fact"
        ]
        if any(re.search(pattern, prompt.strip(), re.IGNORECASE) for pattern in fun_fact_patterns):
            answer = my_custom_tool()
            messages.append(gr.ChatMessage(role="assistant", content=f"```python\n{answer}\n```"))
            yield messages
            return

        # Only append user message if not already present (should never happen here, but for safety)
        def get_message_content(msg):
            if hasattr(msg, 'content'):
                return msg.content
            elif isinstance(msg, dict) and 'content' in msg:
                return msg['content']
            return None
        if not messages or get_message_content(messages[-1]) != prompt:
            messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        try:
            for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
                messages.append(msg)
                yield messages
            yield messages
        except Exception as e:
            import gradio as gr
            error_message = f"⚠️ Error during agent execution: {str(e)}\n\nYou can continue chatting or try a different query."
            messages.append(gr.ChatMessage(role="assistant", content=error_message))
            yield messages
        # After agent completes or errors, reset messages so user can keep chatting
        messages.clear()
        # Robustly reset agent memory/context after every response
        if hasattr(self.agent, 'reset') and callable(getattr(self.agent, 'reset')):
            self.agent.reset()
        if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear') and callable(getattr(self.agent.memory, 'clear')):
            self.agent.memory.clear()
        if hasattr(self.agent, 'history') and hasattr(self.agent.history, 'clear') and callable(getattr(self.agent.history, 'clear')):
            self.agent.history.clear()

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
        )

    def launch(self, **kwargs):
        import gradio as gr

        with gr.Blocks(fill_height=True) as demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png",
                ),
                resizeable=True,
                scale=1,
            )
            # If an upload folder is provided, enable the upload feature
            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(self.interact_with_agent, [stored_messages, chatbot], [chatbot])

        demo.launch(debug=True, share=True, **kwargs)


if __name__ == "__main__":
    from app import agent
    GradioUI(agent).launch()

__all__ = ["stream_to_gradio", "GradioUI"]