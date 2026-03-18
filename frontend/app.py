import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Omnimart CX Assistant", page_icon="🛍️", layout="centered")
st.title("🛍️ Omnimart CX Assistant")
st.markdown("Ask me about company policies, products, shipping, and more!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Sources"):
                for src in message["sources"]:
                    st.caption(f"Source: {src['source']}")
                    st.write(src['text'])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json={"query": prompt}, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            answer = data["answer"]
            status = data["status"]
            latency = data["latency_ms"]
            sources = data.get("sources", [])
            
            # Format output based on status
            if status == "success":
                display_text = f"{answer}\n\n*(Latency: {latency/1000:.2f}s)*"
            else:
                display_text = f"**[{status.upper()}]** {answer}\n\n*(Latency: {latency/1000:.2f}s)*"
                
            message_placeholder.markdown(display_text)
            
            if sources:
                with st.expander("View Sources"):
                    for src in sources:
                        st.caption(f"Source: {src['source']}")
                        st.write(src['text'])
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": display_text,
                "sources": sources
            })
            
        except requests.exceptions.ConnectionError:
            error_msg = ("**Error**: Cannot connect to backend API. Please make sure the "
                         "FastAPI server is running on localhost:8000.")
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except Exception as e:
            error_msg = f"**Error**: An error occurred: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
