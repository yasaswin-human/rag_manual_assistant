import streamlit as st
import requests
import json

st.set_page_config(page_title="Manual Assistant", layout="wide")

st.title("🛠️ Industrial Manual Intelligence System")
st.markdown("Ask questions about your machine manuals. The system will provide grounded answers with citations.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Manual")
    uploaded_file = st.file_uploader("Choose a PDF manual", type="pdf")
    if uploaded_file is not None:
        if st.button("Index Manual"):
            # Send to backend
            files = {"file": uploaded_file}
            response = requests.post("http://localhost:8000/index", files=files)
            if response.status_code == 200:
                st.success("Manual indexed successfully!")
            else:
                st.error(f"Error: {response.text}")

    st.header("Status")
    status_resp = requests.get("http://localhost:8000/status")
    if status_resp.status_code == 200:
        status = status_resp.json()
        st.write(f"Indexed: {status['indexed']}")
        st.write(f"Documents: {status.get('num_docs', 0)}")
    else:
        st.write("Backend not reachable.")

# Query input
query = st.text_input("Ask a question about the machine:", placeholder="e.g., Hydraulic pressure dropping after 20 minutes")
if st.button("Ask") and query:
    with st.spinner("Retrieving and generating answer..."):
        resp = requests.post("http://localhost:8000/query", json={"query": query})
        if resp.status_code == 200:
            data = resp.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Sources")
            for source in data["sources"]:
                meta = source["metadata"]
                st.markdown(f"**📄 {meta.get('source', 'Unknown')}**  |  **Page {meta.get('page', 'N/A')}**  |  Section: {meta.get('section', 'N/A')}")
                with st.expander("See excerpt"):
                    st.write(source["text"])
            st.info(f"Confidence: {data['confidence']}")
            if data["is_insufficient"]:
                st.warning("⚠️ The manual may not contain sufficient information for this query.")
        else:
            st.error(f"Error: {resp.text}")