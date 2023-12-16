import streamlit as st
from langchain.chains import LLMChain, SequentialChain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import Replicate
from langchain.prompts import (
    PromptTemplate,
)
from openai.embeddings_utils import cosine_similarity

from utils import get_pdf_text


def get_cv_pre_screening(selected_model):
    # Init LLM model
    llm = Replicate(
        model=selected_model,
        model_kwargs={
            "temperature": 0.01,
            "max_new_tokens": 500,
            "system_prompt": "You are a senior machine learning engineer who answers very accurate and professionally",
        },
    )

    # Creating session variables
    if (
        "matched_score" not in st.session_state
        and "analyzed_red_flags" not in st.session_state
    ):
        # init responses
        st.session_state["submit"] = ""
        st.session_state["jd_content"] = ""
        st.session_state["pdf_cv"] = ""
        st.session_state["matched_score"] = ""
        st.session_state["analyzed_red_flags"] = ""

    # Start UI
    col1, col2 = st.columns(2)
    col1.header("Job description")
    jd_content = col1.text_area("Please paste the JD here...", key="1")
    col2.header("Upload CV")
    pdf_cv = col2.file_uploader("Upload candidates CV:", type="pdf", key="2")
    submit = st.button("Help me to analyze")

    if st.session_state["submit"] != "":
        submit = st.session_state["submit"]

    if st.session_state["jd_content"] != "":
        jd_content = st.session_state["jd_content"]

    if st.session_state["pdf_cv"] != "":
        pdf_cv = st.session_state["pdf_cv"]

    if submit and jd_content and pdf_cv:
        st.session_state["submit"] = submit

        # Extract JD content
        print("JD content:", jd_content)
        st.session_state["jd_content"] = jd_content

        # extract CV content
        candidates_cv = get_pdf_text(pdf_cv)
        print("CV content:", candidates_cv)
        st.session_state["pdf_cv"] = pdf_cv

        if st.session_state["matched_score"] == "":
            with st.spinner("Wait a second..."):
                # Embedding JD and CV
                embeddings = SentenceTransformerEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                jd_embedding = embeddings.embed_query(jd_content)
                cv_embedding = embeddings.embed_query(candidates_cv)

                st.session_state["matched_score"] = cosine_similarity(
                    jd_embedding, cv_embedding
                )

        st.subheader(f"Matched score: {st.session_state['matched_score'] * 100:.2f}%")

        # JD summarization chain
        template = """
            Can you summarize the following job description ```{jd_content}```\n
            Answer the users question as best as possible.\n
            {format_instructions}
        """
        format_instructions = "The output should be under 200 words with bullet points."

        prompt = PromptTemplate(
            input_variables=["jd_content"],
            template=template,
            partial_variables={"format_instructions": format_instructions},
        )
        chain_analyze_jd = LLMChain(llm=llm, prompt=prompt, output_key="analyzed_jd")

        # CV analytics chain
        template = """
            Can you summarize the following candidate ```{candidates_cv}```\n
            Answer the users question as best as possible.\n
            {format_instructions}
        """

        format_instructions = """
            The output should be under 200 words with bullet points:
            - companies:  # list of companies name
            - projects:  # list projects are working on
            - roles:  # list roles in these projects
            - responsibilities:  # list of responsibilities in these projects
            - tech_stacks:  # list of tech stacks in these projects
        """

        prompt = PromptTemplate(
            input_variables=["candidates_cv"],
            template=template,
            partial_variables={"format_instructions": format_instructions},
        )
        chain_analyze_cv = LLMChain(llm=llm, prompt=prompt, output_key="analyzed_cv")

        # Red flags detection chain
        template = """
            Can you detect red flags in this cv?  ```{analyzed_cv}```\n
            Given job description ```{analyzed_jd}```
            Answer the users question as best as possible.\n
            {format_instructions}
        """

        format_instructions = """
            The output should be under 200 words with bullet points:
            - classifications: # list of red flags priorities (Minor, Major, Critical)
            - issues: # list of potential issues with the candidate's qualifications or work experience
            - explainations: # list of detailed explanation of candidate's qualifications or work experience
            - next_steps: # list of next processing steps according to your discipline
        }
        """

        prompt = PromptTemplate(
            input_variables=["analyzed_cv", "analyzed_jd"],
            template=template,
            partial_variables={"format_instructions": format_instructions},
        )
        chain_red_flags = LLMChain(
            llm=llm, prompt=prompt, output_key="analyzed_red_flags"
        )

        # Chain overall analytics
        if st.session_state["analyzed_red_flags"] == "":
            with st.spinner("Wait a second..."):
                overall_analyze_chain = SequentialChain(
                    chains=[chain_analyze_cv, chain_analyze_jd, chain_red_flags],
                    input_variables=["candidates_cv", "jd_content"],
                    output_variables=[
                        "analyzed_cv",
                        "analyzed_jd",
                        "analyzed_red_flags",
                    ],
                    verbose=True,
                )

                response = overall_analyze_chain(
                    {"candidates_cv": candidates_cv, "jd_content": jd_content}
                )

                # caching outputs
                st.session_state["candidates_cv"] = response["candidates_cv"]
                st.session_state["jd_content"] = response["jd_content"]
                st.session_state["analyzed_jd"] = response["analyzed_jd"]
                st.session_state["analyzed_cv"] = response["analyzed_cv"]
                st.session_state["analyzed_red_flags"] = response["analyzed_red_flags"]

        st.header("JD analysis")
        with st.expander("👀"):
            st.markdown(st.session_state["analyzed_jd"])

        st.header("CV analysis")
        with st.expander("👀"):
            st.markdown(st.session_state["analyzed_cv"])

        st.header("Red flags analysis")
        with st.expander("👀"):
            st.markdown(st.session_state["analyzed_red_flags"])

        st.success("Hope I was able to save your time ❤️")
