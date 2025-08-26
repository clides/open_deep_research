"""Main LangGraph implementation for the Codebase Consulting agent."""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from code_consultant.configuration import (
    Configuration,
)
from code_consultant.prompts import (
    clarify_with_user_instructions,
    compress_analysis_simple_human_message,
    compress_analysis_system_prompt,
    final_report_generation_prompt,
    lead_consultant_prompt,
    analyst_system_prompt,
    transform_messages_into_task_prompt,
)
from code_consultant.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ExecuteTask,
    TaskComplete,
    AnalystOutputState,
    AnalystState,
    TaskBrief,
    SupervisorState,
)
from code_consultant.utils import (
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
    get_model_and_provider,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "model_provider"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_task_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the task scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with the task. If clarification is disabled or not needed, it proceeds directly to the task.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to task brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to the task
        return Command(goto="write_task_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    actual_model, model_provider = get_model_and_provider(configurable.analyst_model)
    
    model_config = {
        "model": actual_model,
        "max_tokens": configurable.analyst_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.analyst_model, config),
        "tags": ["langsmith:nostream"]
    }
    if model_provider:
        model_config["model_provider"] = model_provider
    
    # Configure model with structured output and retry logic
    clarification_model = (  
        configurable_model
        .with_structured_output(ClarifyWithUser)  
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)  
        .with_config(model_config)
    )    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to task with verification message
        return Command(
            goto="write_task_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_task_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    """Transform user messages into a structured task brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused task brief
    that will guide the supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor with initialized context
    """
    # Step 1: Set up the model for structured output
    configurable = Configuration.from_runnable_config(config)
    actual_model, model_provider = get_model_and_provider(configurable.analyst_model)
    
    model_config = {
        "model": actual_model,
        "max_tokens": configurable.analyst_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.analyst_model, config),
        "tags": ["langsmith:nostream"]
    }
    if model_provider:
        model_config["model_provider"] = model_provider
    
    # Configure model for structured task brief generation
    model = (  
        configurable_model
        .with_structured_output(TaskBrief)  
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)  
        .with_config(model_config)
    ) 

    # Step 2: Generate structured task brief from user messages
    prompt_content = transform_messages_into_task_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with task brief and instructions
    supervisor_system_prompt = lead_consultant_prompt.format(
        date=get_today_str(),
        max_concurrent_tasks=configurable.max_concurrent_tasks,
        max_iterations=configurable.max_iterations,
        supervisor_mcp_instructions=configurable.mcp_prompt or ""
    )

    if configurable.mcp_prompt:  
        supervisor_system_prompt += configurable.mcp_prompt
    
    return Command(
        goto="supervisor", 
        update={
            "task_brief": response.task_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.task_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead supervisor that plans strategy and delegates to analysts.
    
    The supervisor analyzes the task brief and decides how to break down the work
    into manageable tasks. It can use think_tool for strategic planning, ExecuteTask
    to delegate tasks to sub-agents, or TaskComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and task context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    actual_model, model_provider = get_model_and_provider(configurable.analyst_model)
    
    model_config = {
        "model": actual_model,
        "max_tokens": configurable.analyst_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.analyst_model, config),
        "tags": ["langsmith:nostream"]
    }
    if model_provider:
        model_config["model_provider"] = model_provider
    
    # Available tools: task delegation, completion signaling, and strategic thinking
    supervisor_tools = [ExecuteTask, TaskComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    model = (  
        configurable_model
        .bind_tools(supervisor_tools, tool_choice="auto")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)  
        .with_config(model_config)
    ) 

    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "iterations": state.get("iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including task delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ExecuteTask - Delegates tasks to analysts
    3. TaskComplete - Signals completion of the main task
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with limits and model settings
        
    Returns:
        Command to either continue supervision loop or end the process
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    iterations = state.get("iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for the process
    exceeded_allowed_iterations = iterations > configurable.max_iterations
    no_tool_calls = not most_recent_message.tool_calls
    task_complete_tool_call = any(
        tool_call["name"] == "TaskComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or task_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "task_brief": state.get("task_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ExecuteTask)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    if think_tool_calls:
        print(f"Supervisor is calling think_tool")
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ExecuteTask calls (task delegation)
    execute_task_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ExecuteTask"
    ]
    if execute_task_calls:
        print(f"Supervisor is calling ExecuteTask with tasks: {[call['args']['task_description'] for call in execute_task_calls]}")
    
    if execute_task_calls:
        try:
            # Limit concurrent tasks to prevent resource exhaustion
            allowed_execute_task_calls = execute_task_calls[:configurable.max_concurrent_tasks]
            overflow_execute_task_calls = execute_task_calls[configurable.max_concurrent_tasks:]
            
            # Execute analysis tasks in parallel
            analysis_tasks = [
                analyst_subgraph.ainvoke({
                    "analyst_messages": [
                        HumanMessage(content=tool_call["args"]["task_description"])
                    ],
                    "task_description": tool_call["args"]["task_description"]
                }, config) 
                for tool_call in allowed_execute_task_calls
            ]
            
            tool_results = await asyncio.gather(*analysis_tasks)
            
            # Create tool messages with analysis results
            for observation, tool_call in zip(tool_results, allowed_execute_task_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_analysis", "Error synthesizing analysis report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow task calls with error messages
            for overflow_call in overflow_execute_task_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this task as you have already exceeded the maximum number of concurrent tasks. Please try again with {configurable.max_concurrent_tasks} or fewer tasks.",
                    name="ExecuteTask",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all analysis results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle execution errors by ending the process
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "task_brief": state.get("task_brief", "")
                }
            )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages task delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for task management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def analyst(state: AnalystState, config: RunnableConfig) -> Command[Literal["analyst_tools"]]:
    """Individual analyst that conducts focused analysis on specific topics.
    
    This analyst is given a specific task by the supervisor and uses
    available tools (e.g., extract_relevant_file_content) to gather comprehensive information.
    It can use think_tool for strategic planning between tool calls.
    
    Args:
        state: Current analyst state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to analyst_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    analyst_messages = state.get("analyst_messages", [])
    
    # Get all available tools
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct analysis: Please configure your tools."
        )
    
    actual_model, model_provider = get_model_and_provider(configurable.analyst_model)
    
    # Step 2: Configure the analyst model with tools
    model_config = {
        "model": actual_model,
        "max_tokens": configurable.analyst_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.analyst_model, config),
        "tags": ["langsmith:nostream"]
    }
    if model_provider:
        model_config["model_provider"] = model_provider
    
    # Prepare system prompt with MCP context if available
    analyst_prompt = analyst_system_prompt.format(  
        mcp_prompt=configurable.mcp_prompt or "",   
        date=get_today_str()  
    )
    
    # Configure model with tools, retry logic, and settings
    model = (  
        configurable_model
        .bind_tools(tools)  
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)  
        .with_config(model_config)
    )    

    # Step 3: Generate analyst response with system context
    messages = [SystemMessage(content=analyst_prompt)] + analyst_messages
    response = await model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="analyst_tools",
        update={
            "analyst_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def analyst_tools(state: AnalystState, config: RunnableConfig) -> Command[Literal["analyst", "compress_analysis"]]:
    """Execute tools called by the analyst.
    
    This function handles various types of analyst tool calls:
    1. think_tool - Strategic reflection that continues the analysis
    2. Other tools (e.g., extract_relevant_file_content) - Information gathering
    
    Args:
        state: Current analyst state with messages and iteration count
        config: Runtime configuration with limits and tool settings
        
    Returns:
        Command to either continue analysis loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    analyst_messages = state.get("analyst_messages", [])
    most_recent_message = analyst_messages[-1]
    
    if not most_recent_message.tool_calls:
        return Command(goto="compress_analysis")
    
    # Step 2: Handle tool calls
    tools = await get_all_tools(config)
    tools_by_name = {tool.name: tool for tool in tools}
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    print(f"Analyst is calling tools: {[call['name'] for call in tool_calls]}")
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=str(observation),
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    
    if exceeded_iterations:
        # End analysis and proceed to compression
        return Command(
            goto="compress_analysis",
            update={"analyst_messages": tool_outputs}
        )
    
    # Continue analysis loop with tool results
    return Command(
        goto="analyst",
        update={"analyst_messages": tool_outputs}
    )

async def compress_analysis(state: AnalystState, config: RunnableConfig):
    """Compress and synthesize analysis findings into a concise, structured summary.
    
    This function takes all the analysis findings, tool outputs, and AI messages from
    an analyst's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current analyst state with accumulated analysis messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed analysis summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)  
      
    compression_params = {  
        "max_tokens": configurable.compression_model_max_tokens,  
        "api_key": get_api_key_for_model(configurable.compression_model, config),  
        "tags": ["langsmith:nostream"]
    }  
    
    actual_model, model_provider = get_model_and_provider(configurable.compression_model)
      
    synthesizer_model = configurable_model.with_config({
        "model": actual_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    if model_provider:
        synthesizer_model = synthesizer_model.with_config({"model_provider": model_provider})

    # Step 2: Prepare messages for compression
    analyst_messages = state.get("analyst_messages", [])
    
    # Add instruction to switch from analysis mode to compression mode
    analyst_messages.append(HumanMessage(content=compress_analysis_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_analysis_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + analyst_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(analyst_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_analysis": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.analyst_model):
                analyst_messages = remove_up_to_last_ai_message(analyst_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(analyst_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_analysis": "Error synthesizing analysis report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Analyst Subgraph Construction
# Creates individual analyst workflow for conducting focused analysis on specific topics
analyst_builder = StateGraph(
    AnalystState, 
    output=AnalystOutputState, 
    config_schema=Configuration
)

# Add analyst nodes for analysis execution and compression
analyst_builder.add_node("analyst", analyst)                 # Main analyst logic
analyst_builder.add_node("analyst_tools", analyst_tools)     # Tool execution handler
analyst_builder.add_node("compress_analysis", compress_analysis)   # Analysis compression

# Define analyst workflow edges
analyst_builder.add_edge(START, "analyst")           # Entry point to analyst
analyst_builder.add_edge("compress_analysis", END)      # Exit point after compression

# Compile analyst subgraph for parallel execution by supervisor
analyst_subgraph = analyst_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive analysis report with retry logic for token limits.
    
    This function takes all collected analysis findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing analysis findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract analysis findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    actual_model, model_provider = get_model_and_provider(configurable.final_report_model)
    
    writer_model_config = {
        "model": actual_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    if model_provider:
        writer_model_config["model_provider"] = model_provider
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all analysis context
            final_report_prompt = final_report_generation_prompt.format(
                task_brief=state.get("task_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
 
            # Return successful report generation
            return {
                "final_report": final_report.content, 
                "messages": [AIMessage(content=final_report.content)],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in code_consultant/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Code Consultant Graph Construction
# Creates the complete codebase consulting workflow from user input to final report
code_consultant = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete analysis process
code_consultant.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
code_consultant.add_node("write_task_brief", write_task_brief)     # Task planning phase
code_consultant.add_node("supervisor", supervisor_subgraph)       # Analysis execution phase
code_consultant.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
code_consultant.add_edge(START, "clarify_with_user")                       # Entry point
code_consultant.add_edge("supervisor", "final_report_generation") # Analysis to report
code_consultant.add_edge("final_report_generation", END)                   # Final exit point