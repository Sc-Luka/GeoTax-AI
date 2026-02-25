import streamlit as st
from utils import buildRetriver
from model import load_model_and_tokenizer

st.set_page_config(page_title="AI ასისტენტი",
                   page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #121212; color: white; }
    
    .header-container { text-align: center; padding: 10px; }
    .main-title { color: #4ade80; font-size: 36px; font-weight: bold; }
    .github-link { color: #3b82f6 !important; text-decoration: none; font-size: 14px; }
    
    .source-text {
        font-size: 0.8em;
        color: #a0a0a0;
        border-top: 1px solid #333;
        margin-top: 15px;
        padding-top: 10px;
    }
    .source-link { color: #4ade80 !important; text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="header-container">
        <div class="main-title"> 🤖 AI ასისტენტი 🤖 </div>
        <a class="github-link" href="https://github.com/Sc-Luka" target="_blank">GitHub: https://github.com/Sc-Luka</a>
    </div>
""", unsafe_allow_html=True)

st.divider()


@st.cache_resource
def get_ai_resources():
    tokenizer, model, embeddings, device = load_model_and_tokenizer()
    retriever = buildRetriver(embeddings)
    return tokenizer, model, embeddings, device, retriever


with st.status("⏳ იტვირთება AI მოდელები... გთხოვთ დაიცადოთ", expanded=False) as status:
    tokenizer, model, embeddings, device, retriever = get_ai_resources()
    status.update(label="AI მოდელები წარმატებით ჩაიტვირთა!",
                  state="complete", expanded=False)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
            "content": "გამარჯობა მე ვარ AI ასისტენტი! რით შემიძლია დაგეხმაროთ?"}
    ]


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


query = st.chat_input("ჩაწერეთ თქვენი შეკითხვა აქ...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("ვამუშავებ..."):
            docs = retriever.invoke(f"query: {query}")
            context_text = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""<|im_start|>system
                    შენ ხარ დამხმარე, რომელიც პასუხობს მხოლოდ მოცემული კონტექსტის გამოყენებით. 
                    პასუხი დაწერე გამართული წინადადებებით. არ გამოიყენო ჩამონათვალი, თუ ტექსტში პირდაპირ არ წერია. 
                    თუ ინფორმაცია არ არის, უპასუხე: 'არ ვიცი'.
                    <|im_end|>
                    <|im_start|>user
                    კონტექსტი:
                    {context_text}

                    კითხვა:
                    {query}<|im_end|>
                    <|im_start|>assistant
                    """
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]

            output = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_tokens = output[0][input_length:]
            response = tokenizer.decode(
                generated_tokens, skip_special_tokens=True).strip()

            source_text = (
                "წყარო: საინფორმაციო და მეთოდოლოგიური ჰაბზე განთავსებული დოკუმენტების მიხედვით "
                "(საგადასახადო და საბაჟო ადმინისტრირებასთან დაკავშირებული დოკუმენტები და ინფორმაცია ერთ სივრცეში) "
                "ლინკი: [infohub.rs.ge](https://infohub.rs.ge/ka)"
            )

            response = response.replace("```", "").strip()
            if "არ ვიცი" in response:
                full_display = "ბოდიში, მოცემულ დოკუმენტებში ეს ინფორმაცია ვერ მოიძებნა."
            else:
                full_display = f"{response}\n\n{source_text}"

            st.markdown(full_display, unsafe_allow_html=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_display}
            )
