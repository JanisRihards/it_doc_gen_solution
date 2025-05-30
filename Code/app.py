import os
import json
import re
import shutil

import streamlit as st

from input_module import process_pdf, process_json
from processing_module import get_vector_collection, add_documents_to_vectorstore, call_llm

# === Streamlit UI ===

st.set_page_config(page_title="RAG Doc Generator")
st.sidebar.title("📥 Upload and Process Files")

uploaded_context_pdfs = st.sidebar.file_uploader(
    "Upload Context PDFs", type=["pdf"], accept_multiple_files=True
)
uploaded_instructions_json = st.sidebar.file_uploader(
    "Upload Instructions JSON", type=["json"]
)
uploaded_structure_json = st.sidebar.file_uploader(
    "Upload Structure JSON", type=["json"]
)

if st.sidebar.button("Process"):
    col = get_vector_collection()
    existing_ids = col.get().get("ids", [])
    if existing_ids:
        col.delete(ids=existing_ids)
        st.info("Cleared vector database for fresh testing.")
    else:
        st.info("Vector database already empty.")

    if uploaded_context_pdfs:
        for f in uploaded_context_pdfs:
            splits = process_pdf(f)
            add_documents_to_vectorstore(splits, f.name, "context_pdf")

    if uploaded_instructions_json:
        raw_instr = process_json(uploaded_instructions_json)
        if raw_instr is not None:
            st.session_state["instructions_data"] = raw_instr

    if uploaded_structure_json:
        raw_struct = process_json(uploaded_structure_json)
        if raw_struct is not None:
            st.session_state["output_template"] = raw_struct

if st.sidebar.button("Clear Database"):
    db_path = "./demo-rag-chroma"
    try:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            # re-create empty collection
            _ = get_vector_collection()
            st.sidebar.success("✔️ Chroma database directory removed.")
        else:
            st.sidebar.info("Chroma database directory not found; nothing to clear.")
    except Exception as e:
        st.sidebar.error(f"Failed to remove Chroma database directory: {e}")

st.header("Generate Artefact from Document")
user_prompt = st.text_area("User Prompt (additional comments):")

if st.button("Generate"):
    instructions_data = st.session_state.get("instructions_data")
    output_template = st.session_state.get("output_template")
    if not instructions_data or not output_template:
        st.error("Please upload both Instructions JSON and Structure JSON before generating.")
    else:
        # extract llm_prompt_instructions
        if isinstance(instructions_data, dict) and "llm_prompt_instructions" in instructions_data:
            instr_payload = instructions_data["llm_prompt_instructions"]
            parent_id = instructions_data.get("instruction_id", "instructions")
        elif isinstance(instructions_data, list):
            parent_id = "instructions"
            instr_payload = next(
                (item["llm_prompt_instructions"] for item in instructions_data
                 if isinstance(item, dict) and "llm_prompt_instructions" in item),
                None
            )
        else:
            st.error("Invalid instructions format.")
            st.stop()

        steps_list = instr_payload.get("steps", []) or []
        goal_text = instr_payload.get("goal", "") or ""
        final_results = {}
        col = get_vector_collection()

        for step_index, step_instruction in enumerate(steps_list):
            # skip final formatting step
            if step_index == len(steps_list) - 1 and "Format" in str(step_instruction):
                break

            step_text = re.sub(
                r":contentReference\[oaicite:\d+\]\{index=\d+\}", "",
                str(step_instruction).replace("\u200b", "").strip()
            )
            instruction_prompt = f"{goal_text}\n{step_text}" if goal_text else step_text

            # retrieve context chunks
            query_res = col.query(
                query_texts=[step_text],
                where={"type": "context_pdf"},
                n_results=5
            )
            chunks = []
            for docs in query_res.get("documents", []):
                chunks.extend(docs if isinstance(docs, list) else [docs])
            context_text = "\n".join(chunks)

            # add previous output & user prompt
            user_additions = ""
            if final_results:
                user_additions += "Previous results:\n" + json.dumps(final_results, ensure_ascii=False, indent=2) + "\n\n"
            if user_prompt:
                user_additions += user_prompt

            # call LLM
            example_structure_str = json.dumps(output_template, ensure_ascii=False)
            llm_response = "".join(call_llm(context_text, instruction_prompt, user_additions, example_structure_str))

            # extract JSON
            resp = llm_response.strip()
            if resp.startswith("```"):
                resp = resp.strip("```").lstrip("json\n")
            json_content = None
            if resp:
                start, end = resp.find("{"), resp.rfind("}")
                if start != -1 and end != -1:
                    try:
                        json_content = json.loads(resp[start : end + 1])
                    except json.JSONDecodeError:
                        pass

            # save and merge
            if isinstance(json_content, dict) and json_content:
                col.upsert(
                    documents=[json.dumps(json_content, ensure_ascii=False)],
                    metadatas=[{
                        "type": "result_json",
                        "source": "step_result",
                        "instruction_id": parent_id,
                        "step": step_index + 1
                    }],
                    ids=[f"result_step_{step_index+1}"]
                )
                final_results.update(json_content)

        # assemble final output
        if isinstance(output_template, dict):
            output_data = {
                key: final_results.get(key, template_val)
                for key, template_val in output_template.items()
            }
        else:
            output_data = final_results or output_template

        st.text_area("Generated Output", value=json.dumps(output_data, ensure_ascii=False, indent=2), height=300)
