
#!/usr/bin/env python3
"""
Super-Alita Web UI - Streamlit Interface

Professional web interface for Super-Alita agent interactions,
enhanced consensus testing, and REUG v9.7 operational monitoring.

Template Application:
- appName: "Super-Alita"
- interfaceType: "Agent Interaction Dashboard"
- features: ["consensus_testing", "reug_monitoring", "prompt_optimization", "real_time_chat"]
- defaultModel: "llama3.2:3b"
- theme: "professional_dark"
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Super-Alita AI Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yaboyshades/super-alita",
        "Report a bug": "https://github.com/yaboyshades/super-alita/issues",
        "About": "Super-Alita v0.9.7 - Advanced AI Agent System with Enhanced Consensus",
    },
)

# Custom CSS for professional appearance
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }

    .status-success {
        color: #28a745;
        font-weight: bold;
    }

    .status-error {
        color: #dc3545;
        font-weight: bold;
    }

    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }

    .consensus-result {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "consensus_history" not in st.session_state:
    st.session_state.consensus_history = []
if "system_status" not in st.session_state:
    st.session_state.system_status = {"healthy": False, "last_check": None}

# Configuration
BASE_URL = "http://127.0.0.1:8081"  # Default Super-Alita port


class SuperAlitaAPI:
    """API client for Super-Alita backend."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def check_health(self) -> dict[str, Any]:
        """Check system health."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/healthz")
                if response.status_code == 200:
                    return {"healthy": True, "data": response.json()}
                else:
                    return {"healthy": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def get_tools_catalog(self) -> dict[str, Any]:
        """Get available tools catalog."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/tools/catalog")
                if response.status_code == 200:
                    return {"success": True, "tools": response.json()}
                else:
                    return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def consensus_request(
        self, prompt: str, method: str = "weighted_vote", num_samples: int = 3, **kwargs
    ) -> dict[str, Any]:
        """Make enhanced consensus request."""
        try:
            payload = {
                "prompt": prompt,
                "method": method,
                "num_samples": num_samples,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 256),
                "confidence_threshold": kwargs.get("confidence_threshold", 0.7),
                "temperature_range": kwargs.get("temperature_range", 0.2),
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/ability/execute/deepconf_consensus", json=payload
                )
                if response.status_code == 200:
                    return {"success": True, "result": response.json()}
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "details": response.text,
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Initialize API client
api = SuperAlitaAPI(BASE_URL)


def main_header():
    """Render main header."""
    st.markdown(
        '<h1 class="main-header">🤖 Super-Alita AI Agent Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


def sidebar_configuration():
    """Render sidebar configuration."""
    st.sidebar.markdown("## ⚙️ Configuration")

    # Model selection
    model = st.sidebar.selectbox(
        "LLM Model",
        ["llama3.2:3b", "gpt-4", "claude-3", "mistral-7b"],
        index=0,
        help="Select the language model for consensus operations",
    )

    # Consensus method
    consensus_method = st.sidebar.selectbox(
        "Consensus Method",
        [
            "weighted_vote",
            "confidence_based",
            "semantic_similarity",
            "ensemble_ranking",
            "simple_vote",
        ],
        index=0,
        help="Choose the consensus aggregation algorithm",
    )

    # Advanced parameters
    st.sidebar.markdown("### Advanced Parameters")
    num_samples = st.sidebar.slider("Number of Samples", 1, 10, 3)
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 256, 50)

    return {
        "model": model,
        "consensus_method": consensus_method,
        "num_samples": num_samples,
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "max_tokens": max_tokens,
    }


async def check_system_status():
    """Check and update system status."""
    health_check = await api.check_health()
    tools_check = await api.get_tools_catalog()

    st.session_state.system_status = {
        "healthy": health_check.get("healthy", False),
        "health_data": health_check.get("data", {}),
        "tools_available": tools_check.get("success", False),
        "tools_count": len(tools_check.get("tools", [])),
        "last_check": datetime.now(),
    }


def render_system_status():
    """Render system status panel."""
    st.markdown("## 📊 System Status")

    if st.button("🔄 Refresh Status", type="secondary"):
        with st.spinner("Checking system health..."):
            asyncio.run(check_system_status())

    if st.session_state.system_status["last_check"]:
        status = st.session_state.system_status

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if status["healthy"]:
                st.markdown(
                    '<p class="status-success">✅ System Healthy</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="status-error">❌ System Offline</p>',
                    unsafe_allow_html=True,
                )

        with col2:
            if status.get("tools_available", False):
                st.markdown(
                    f'<p class="status-success">🛠️ {status["tools_count"]} Tools Available</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="status-warning">⚠️ Tools Unavailable</p>',
                    unsafe_allow_html=True,
                )

        with col3:
            st.markdown(f"🕐 Last Check: {status['last_check'].strftime('%H:%M:%S')}")

        with col4:
            if status.get("health_data"):
                components = status["health_data"].get("components", {})
                healthy_components = sum(
                    1 for c in components.values() if c.get("status") == "ok"
                )
                total_components = len(components)
                st.markdown(f"🔧 Components: {healthy_components}/{total_components}")


def render_consensus_interface(config: dict[str, Any]):
    """Render enhanced consensus interface."""
    st.markdown("## 🧠 Enhanced Consensus Testing")

    # Prompt input
    user_prompt = st.text_area(
        "Enter your prompt for consensus analysis:",
        placeholder="e.g., 'What are the key principles of responsible AI development?'",
        height=100,
        help="Enter a question or prompt that you want multiple AI models to answer and reach consensus on.",
    )

    # Quick prompt examples
    st.markdown("### 💡 Quick Examples")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🤔 Ask about AI Ethics"):
            user_prompt = "What are the most important ethical considerations when developing AI systems?"
            st.rerun()

    with col2:
        if st.button("🔬 Technical Question"):
            user_prompt = "Explain the difference between supervised and unsupervised machine learning."
            st.rerun()

    with col3:
        if st.button("🌍 Current Events"):
            user_prompt = "What are the potential benefits and risks of renewable energy adoption?"
            st.rerun()

    # Execute consensus
    if st.button(
        "🚀 Generate Consensus", type="primary", disabled=not user_prompt.strip()
    ):
        if user_prompt.strip():
            with st.spinner(
                f"Generating consensus using {config['consensus_method']}..."
            ):
                result = asyncio.run(
                    api.consensus_request(
                        prompt=user_prompt,
                        method=config["consensus_method"],
                        num_samples=config["num_samples"],
                        temperature=config["temperature"],
                        confidence_threshold=config["confidence_threshold"],
                        max_tokens=config["max_tokens"],
                    )
                )

                if result["success"]:
                    consensus_data = result["result"]

                    # Store in history
                    st.session_state.consensus_history.append(
                        {
                            "timestamp": datetime.now(),
                            "prompt": user_prompt,
                            "method": config["consensus_method"],
                            "result": consensus_data,
                            "config": config.copy(),
                        }
                    )

                    # Display result
                    render_consensus_result(consensus_data, config)
                else:
                    st.error(f"❌ Consensus failed: {result['error']}")
                    if "details" in result:
                        st.code(result["details"])


def render_consensus_result(result: dict[str, Any], config: dict[str, Any]):
    """Render consensus result with detailed analysis."""
    st.markdown("### 🎯 Consensus Result")

    # Main result
    consensus_text = result.get("consensus_text", "No consensus text available")
    confidence = result.get("consensus_confidence", 0.0)
    method = result.get("aggregation_method", config["consensus_method"])

    st.markdown(
        f"""
    <div class="consensus-result">
        <h4>📝 Consensus Response:</h4>
        <p>{consensus_text}</p>
        <br>
        <strong>📊 Confidence Score:</strong> {confidence:.3f}<br>
        <strong>🔧 Method:</strong> {method}<br>
        <strong>📈 Individual Responses:</strong> {len(result.get('individual_responses', []))}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Detailed analysis
    with st.expander("📊 Detailed Analysis"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Individual Responses:**")
            for i, response in enumerate(result.get("individual_responses", []), 1):
                st.markdown(f"**Response {i}:** {response[:100]}...")

        with col2:
            st.markdown("**Metadata:**")
            metadata = result.get("metadata", {})
            for key, value in metadata.items():
                if isinstance(value, dict):
                    st.json(value)
                else:
                    st.markdown(f"**{key}:** {value}")

    # Confidence visualization
    if confidence > 0:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Consensus Confidence"},
                gauge={
                    "axis": {"range": [None, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 0.3], "color": "lightgray"},
                        {"range": [0.3, 0.7], "color": "yellow"},
                        {"range": [0.7, 1], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": config["confidence_threshold"],
                    },
                },
            )
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_consensus_history():
    """Render consensus history and analytics."""
    st.markdown("## 📈 Consensus History & Analytics")

    if not st.session_state.consensus_history:
        st.info(
            "No consensus history yet. Try generating some consensus results above!"
        )
        return

    # History table
    history_data = []
    for item in st.session_state.consensus_history:
        history_data.append(
            {
                "Timestamp": item["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "Method": item["method"],
                "Confidence": f"{item['result'].get('consensus_confidence', 0):.3f}",
                "Samples": len(item["result"].get("individual_responses", [])),
                "Prompt": (
                    item["prompt"][:50] + "..."
                    if len(item["prompt"]) > 50
                    else item["prompt"]
                ),
            }
        )

    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True)

    # Analytics charts
    if len(st.session_state.consensus_history) > 1:
        col1, col2 = st.columns(2)

        with col1:
            # Confidence over time
            timestamps = [
                item["timestamp"] for item in st.session_state.consensus_history
            ]
            confidences = [
                item["result"].get("consensus_confidence", 0)
                for item in st.session_state.consensus_history
            ]

            fig = px.line(x=timestamps, y=confidences, title="Confidence Over Time")
            fig.update_layout(xaxis_title="Time", yaxis_title="Confidence Score")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Method usage distribution
            methods = [item["method"] for item in st.session_state.consensus_history]
            method_counts = pd.Series(methods).value_counts()

            fig = px.pie(
                values=method_counts.values,
                names=method_counts.index,
                title="Consensus Method Usage",
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application function."""
    main_header()

    # Sidebar configuration
    config = sidebar_configuration()

    # System status check on first load
    if st.session_state.system_status["last_check"] is None:
        with st.spinner("Checking system status..."):
            asyncio.run(check_system_status())

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🧠 Consensus Testing", "📊 Analytics"])

    with tab1:
        render_system_status()

        # Quick stats
        if st.session_state.consensus_history:
            st.markdown("## 📋 Quick Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Requests", len(st.session_state.consensus_history))

            with col2:
                avg_confidence = sum(
                    item["result"].get("consensus_confidence", 0)
                    for item in st.session_state.consensus_history
                ) / len(st.session_state.consensus_history)
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")

            with col3:
                methods_used = len(
                    set(item["method"] for item in st.session_state.consensus_history)
                )
                st.metric("Methods Used", methods_used)

            with col4:
                latest_result = st.session_state.consensus_history[-1]
                st.metric(
                    "Latest Confidence",
                    f"{latest_result['result'].get('consensus_confidence', 0):.3f}",
                )

    with tab2:
        render_consensus_interface(config)

    with tab3:
        render_consensus_history()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Super-Alita v0.9.7 - Advanced AI Agent System | "
        "<a href='https://github.com/yaboyshades/super-alita'>GitHub</a>"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
